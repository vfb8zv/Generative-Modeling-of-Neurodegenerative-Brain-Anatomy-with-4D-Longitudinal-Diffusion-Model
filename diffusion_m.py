import torch
from torch import nn
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm
from einops_exts import check_shape
from torch.utils import data
import numpy as np
from velocity_loader import get_deformed

torch.manual_seed(42)
torch.cuda.manual_seed(42) 

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

class DVIT_Diffusion(nn.Module):
    def __init__(
        self,
        denoiser,
        channels=3,
        num_frames=3,
        image_size=32,
        timesteps = 1000,
        sampling_timesteps = 200,
        ddim_sampling_eta=0.
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoiser = denoiser
        self.num_frames = num_frames
        self.sampling_timesteps = sampling_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        
        self.smoothness = Grad('l2', loss_mult=1)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def p_sample_train(self, x_t, t, noise = None):
        return (x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * noise) / extract(self.sqrt_alphas_cumprod, t, x_t.shape)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped 

    def p_mean_variance(self, x, t, cond_c, cond_i, cond_a, cond_scale = 1.):
        noise_pred = self.denoiser(x, t, cond_c, cond_i, cond_a)
        x_recon = self.predict_start_from_noise(x, t=t, noise = noise_pred)

        clip_denoised = True
        if clip_denoised==True:
            s = 5
            x_recon = x_recon.clamp(-s, s)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    
    @torch.inference_mode()
    def p_sample(self, x, t, cond_c, cond_i, cond_a, cond_scale = 1.):
        b, *_, device = *x.shape, x.device
        #with torch.no_grad():
        model_mean, _, model_log_variance = self.p_mean_variance(x,t, cond_c, cond_i, cond_a, cond_scale = cond_scale)
        noise = torch.randn_like(x)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond_c, cond_i, cond_a, cond_scale = 1., sample_ts= 1000):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        self.num_timesteps= sample_ts 

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond_c, cond_i, cond_a, cond_scale = cond_scale)
            print("During sampling max value is: ", (img).max())
            print("During sampling min value is: ", (img).min())
            
        # img = unnormalize_to_zero_to_one(img)
        unnorm_img = self.unnormalize(img)
        print(f'\033[92mmaximum value of the unnormalized velocity field is: {unnorm_img.max()}\033[0m')
        print(f'\033[92mminimum value of the unnormalized velocity field is: {unnorm_img.min()}\033[0m')
        return unnorm_img

    @torch.inference_mode() 
    def sample(self, cond_c, cond_i, cond_a, cond_scale = 1., batch_size = 1, sample_ts=1000):
        device = next(self.denoiser.parameters()).device

        batch_size = cond_i.shape[0] if exists(cond_i) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop((batch_size, channels, num_frames, image_size, image_size, image_size), cond_c, cond_i, cond_a, cond_scale = cond_scale, sample_ts=sample_ts)
    
    def unnormalize(self, z_norm, scale=2):
        clipped = torch.clip(z_norm, -0.9999, 0.9999)
        z_unnorm = scale * torch.arctanh(clipped)
        return z_unnorm
    
    def p_losses(self, x_start, t, cond_c, cond_i, cond_a, im, step, noise = None, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.denoiser(x_noisy, t, cond_c, cond_i, cond_a)  ## pass tensor of classes as cond_c and resized gradient of source obtained from dataloader as cond_i

        
        diff_loss = 10*(F.l1_loss(noise, x_recon))
        
        reconstructed_x = self.p_sample_train(x_noisy, t, x_recon)
        sim_loss = F.mse_loss(x_start, reconstructed_x, reduction="mean")
        
        smoothness_loss = self.smoothness.loss(None, reconstructed_x)
        loss = diff_loss 
        
        if step%100 == 0:
            reconstructed_x = self.unnormalize(reconstructed_x, scale=1)
            x_start_gt = self.unnormalize(x_start, scale=1)
            z_all = get_deformed(reconstructed_x[0].unsqueeze(0).cuda(), im[0,0,0,0].unsqueeze(0).unsqueeze(0).cuda())
            pred_all = []
            pred_v = []
            pred_vs = []
            for i in z_all:
                pred_all.append(i[0])  # transformed source
                pred_v.append(i[1])    # displacement field
                pred_vs.append(i[2])   # velocity field

            z_all_gt = get_deformed(x_start_gt[0].unsqueeze(0).cuda(), im[0,0,0,0].unsqueeze(0).unsqueeze(0).cuda())
            pred_all_gt = []
            for i in z_all_gt:
                pred_all_gt.append(i[0])  # transformed source
                
        return loss, diff_loss, sim_loss, smoothness_loss

    def forward(self, x, cond_c, cond_i, cond_a, im, step, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        # check_shape(x, 'b c f h w l', c = self.channels, f = self.num_frames, h = img_size, w = img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, cond_c, cond_i, cond_a, im, step, *args, **kwargs)

class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()
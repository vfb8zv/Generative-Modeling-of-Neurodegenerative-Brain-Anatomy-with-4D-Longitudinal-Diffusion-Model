import os
import numpy as np
import torch
from dataloader_pc import get_data
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from velocity_loader import get_deformed, init_velocity_loader
from diffusion_m import DVIT_Diffusion
from ldm_train import DVIT
from dvit import DViT
import os 
import argparse
import glob
from datetime import datetime
from tqdm import tqdm

# Predictor-Corrector Sampling Implementation
class PredictorCorrectorSampler:
    """
    Predictor-Corrector sampling for diffusion models.
    Predictor: DDPM ancestral sampling
    Corrector: Langevin dynamics
    """
    
    def __init__(self, diffusion_model, corrector_steps=1, corrector_step_size=1e-5, 
                 corrector_snr=0.16, target_snr=0.16):
        """
        Args:
            diffusion_model: Your DVIT_Diffusion model
            corrector_steps: Number of Langevin steps per timestep
            corrector_step_size: Step size for Langevin dynamics
            corrector_snr: Signal-to-noise ratio for corrector
            target_snr: Target SNR for adaptive step sizing
        """
        self.diffusion = diffusion_model
        self.corrector_steps = corrector_steps
        self.corrector_step_size = corrector_step_size
        self.corrector_snr = corrector_snr
        self.target_snr = target_snr
    
    def get_score_fn(self, x, t, cond_c, cond_i, cond_a):
        """
        Compute score function (gradient of log probability) from noise prediction.
        Score = -noise_pred / sigma_t
        """
        # Get noise prediction from your denoiser
        with torch.enable_grad():
            x.requires_grad_(True)
            noise_pred = self.diffusion.denoiser(x, t, cond_c, cond_i, cond_a)
            
        # Convert noise prediction to score
        sigma_t = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
        sigma_t = sigma_t.view(-1, *([1] * (x.ndim - 1)))
        score = -noise_pred / sigma_t
        
        return score
    
    def langevin_corrector_step(self, x, t, cond_c, cond_i, cond_a, step_size=None):
        """
        Single Langevin dynamics corrector step.
        x_{t+1} = x_t + step_size * score(x_t) + sqrt(2 * step_size) * noise
        """
        if step_size is None:
            step_size = self.corrector_step_size
            
        # Compute score function
        score = self.get_score_fn(x, t, cond_c, cond_i, cond_a)
        
        # Langevin update
        noise = torch.randn_like(x)
        x_corrected = x + step_size * score + torch.sqrt(2 * step_size) * noise
        
        return x_corrected
    
    def adaptive_step_size(self, x, t, cond_c, cond_i, cond_a):
        """
        Compute adaptive step size based on SNR.
        """
        # Get current noise level
        sigma_t = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
        sigma_t = sigma_t.view(-1, *([1] * (x.ndim - 1)))
        
        # Compute score magnitude
        score = self.get_score_fn(x, t, cond_c, cond_i, cond_a)
        score_norm = torch.norm(score.flatten(start_dim=1), dim=1, keepdim=True)
        score_norm = score_norm.view(-1, *([1] * (x.ndim - 1)))
        
        # Adaptive step size based on target SNR
        step_size = 2 * (self.target_snr * sigma_t / score_norm) ** 2
        step_size = torch.clamp(step_size, min=1e-6, max=1e-3)
        
        return step_size
    
    @torch.inference_mode()
    def ddpm_predictor_step(self, x, t, cond_c, cond_i, cond_a):
        """
        DDPM predictor step (ancestral sampling).
        This is your existing p_sample method.
        """
        return self.diffusion.p_sample(x, t, cond_c, cond_i, cond_a)
    
    @torch.inference_mode()
    def pc_sample_step(self, x, t, cond_c, cond_i, cond_a, use_adaptive_step=True):
        """
        Combined predictor-corrector step.
        """
        # Predictor step (DDPM)
        x_pred = self.ddpm_predictor_step(x, t, cond_c, cond_i, cond_a)
        
        # Corrector steps (Langevin)
        x_corrected = x_pred
        for _ in range(self.corrector_steps):
            if use_adaptive_step:
                step_size = self.adaptive_step_size(x_corrected, t, cond_c, cond_i, cond_a)
            else:
                step_size = self.corrector_step_size
                
            x_corrected = self.langevin_corrector_step(x_corrected, t, cond_c, cond_i, cond_a, step_size)
        
        return x_pred, x_corrected
    
    @torch.inference_mode()
    def pc_sample_loop(self, cond_c, cond_i, cond_a, shape=None, use_corrector=True):
        """
        Full predictor-corrector sampling loop.
        """
        device = self.diffusion.betas.device
        b = cond_c.shape[0] if cond_c is not None else 1
        
        if shape is None:
            image_size = self.diffusion.image_size
            channels = self.diffusion.channels
            num_frames = self.diffusion.num_frames
            shape = (b, channels, num_frames, image_size, image_size, image_size)
        
        # Initialize with noise
        img = torch.randn(shape, device=device)
        
        intermediates = []
        
        # Sampling loop
        for i in tqdm(reversed(range(0, self.diffusion.num_timesteps)), 
                     desc='PC sampling loop', total=self.diffusion.num_timesteps):
            
            t = torch.full((b,), i, device=device, dtype=torch.long)
            
            if use_corrector and i > 0:  # Don't use corrector at final step
                # Predictor-Corrector step
                img_pred, img = self.pc_sample_step(
                    img, t, cond_c, cond_i, cond_a, use_adaptive_step=True
                )
            else:
                # Predictor only step
                img = self.ddpm_predictor_step(img, t, cond_c, cond_i, cond_a)

        unnorm_img = self.diffusion.unnormalize(img)
        print(f'\033[92mFinal PC sampling - max: {unnorm_img.max():.6f}, min: {unnorm_img.min():.6f}\033[0m')

        return unnorm_img
        
    @torch.inference_mode()
    def sample(self, cond_c, cond_i, cond_a, **kwargs):
        """
        Main sampling interface compatible with your existing code.
        """
        return self.pc_sample_loop(cond_c, cond_i, cond_a, **kwargs)


def add_pc_sampling_to_dvit(diffusion_model, corrector_steps=1, corrector_step_size=1e-5):
    """
    Add predictor-corrector sampling to your existing DVIT_Diffusion model.
    """
    # Create PC sampler
    pc_sampler = PredictorCorrectorSampler(
        diffusion_model=diffusion_model,
        corrector_steps=corrector_steps,
        corrector_step_size=corrector_step_size,
        target_snr=0.16
    )
    
    # Add PC sampling method to the diffusion model
    def pc_sample(self, cond_c, cond_i, cond_a, corrector_steps=1, corrector_step_size=1e-5, 
                  use_corrector=True, **kwargs):
        """
        Predictor-Corrector sampling method for DVIT_Diffusion.
        """
        # Update corrector parameters if provided
        pc_sampler.corrector_steps = corrector_steps
        pc_sampler.corrector_step_size = corrector_step_size
        
        return pc_sampler.sample(
            cond_c=cond_c, 
            cond_i=cond_i, 
            cond_a=cond_a,
            use_corrector=use_corrector,
            **kwargs
        )
    
    # Bind the method to the class
    diffusion_model.pc_sample = pc_sample.__get__(diffusion_model, type(diffusion_model))
    
    return diffusion_model


def get_latest_checkpoint(checkpoint_dir='./results/models/', pattern='*_z_genmodel_*.pt'):
    """
    Find the latest checkpoint file based on modification time.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        pattern: File pattern to match checkpoint files
    
    Returns:
        Latest model name (without .pt extension) or None if no files found
    """
    # Get all checkpoint files matching the pattern
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir} with pattern {pattern}")
        return None
    
    # Sort by modification time (most recent first)
    latest_file = max(checkpoint_files, key=os.path.getmtime)
    
    # Extract just the filename without path and extension
    model_name = os.path.splitext(os.path.basename(latest_file))[0]
    
    # Print info about the selected checkpoint
    mod_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
    print(f"Latest checkpoint: {model_name}")
    print(f"Modified: {mod_time}")
    
    return model_name


def get_slices(vol, axis=0, idx=64):
    if axis == 0:
        return vol[idx, :, :]
    elif axis == 1:
        return vol[:, idx, :]
    elif axis == 2:
        return vol[:, :, idx]


# Initialize data loader
datal = None
dl = None
save_dir_pc = None

def test(step, use_pc_sampling=True, corrector_steps=1, corrector_step_size=1e-5):
    """
    Test function with PC sampling integration.
    
    Args:
        step: Current test step
        use_pc_sampling: Whether to use PC sampling (default: True)
        corrector_steps: Number of Langevin corrector steps
        corrector_step_size: Step size for Langevin dynamics
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42) 

    log_dir = './results/'

    batch_size = 1
    img_size = 32
    channels = 3
    num_frames = 3
    
    global dl, save_dir_pc
    
    # Model parameters 
    patch_size = 4 
    embed_dim = 384  
    num_heads = 6  
    depth = 12  

    denoiser = DViT(patch_size=patch_size,
                    embed_dim=embed_dim, 
                    num_patches=None,
                    img_size=img_size,
                    channels=channels,
                    num_heads=num_heads, 
                    mlp_ratio=4.0,
                    num_classes=3,
                    depth=depth,
                    num_frames=num_frames,
                    out_channels=channels).cuda()

    diffusion = DVIT_Diffusion(
        denoiser,
        channels=channels,
        image_size=img_size,    
        timesteps=1000,
        num_frames=3   
    ).cuda()
    
    # Add PC sampling capability to the diffusion model
    if use_pc_sampling:
        diffusion = add_pc_sampling_to_dvit(
            diffusion, 
            corrector_steps=corrector_steps,
            corrector_step_size=corrector_step_size
        )
        print(f"PC Sampling enabled with {corrector_steps} corrector steps and step size {corrector_step_size}")
    else:
        print("Using regular DDPM sampling")
    
    # Automatically get the latest checkpoint
    latest_model_name = get_latest_checkpoint()
    print(f"Loading the latest model: {latest_model_name}")

    if latest_model_name is None:
        print(f"Using fallback model: {latest_model_name}")

    lcgd_model = DVIT(
            diffusion, 
            datal, 
            train_batch_size=batch_size,  
            train_lr=1e-4,  
            model_name=latest_model_name,
            train_num_steps=100000,
            finetune=latest_model_name,
            batch_normed=True
        )

    lcgd_model.load()
    lcgd_model.model.eval()

    # Initialize velocity loader
    init_velocity_loader(local_rank=0) 

    print(f"\n{'='*50}")
    print(f"Running Test Step {step}")
    print(f"{'='*50}")
    
    
    os.makedirs(save_dir_pc, exist_ok=True)
    
    print(len(dl))

    for i in range(len(dl)):
        z, cond_c, cond_i, cond_a, im, ptid = next(dl)
        # Use PC sampling or regular sampling based on flag
        if use_pc_sampling:
            print(f"Sample {i+1}: Running PC sampling...")
            z_sample = diffusion.pc_sample(
                cond_c, cond_i, cond_a,
                corrector_steps=corrector_steps,
                corrector_step_size=corrector_step_size,
                use_corrector=True
            )
        else:
            print(f"Sample {i+1}: Running regular DDPM sampling...")
            z_sample = diffusion.sample(cond_c, cond_i, cond_a)
        

        # Save with the same filename as ground truth
        sample_filename = ptid[0] 
        save_path = os.path.join(save_dir_pc, sample_filename)

        im = im.squeeze()
        if len(im.size())==3:
            im = im.unsqueeze(0)
        z_all = get_deformed(z_sample, im[0].unsqueeze(0).unsqueeze(0).cuda())

        pred_all = [im[0]]
        for i in z_all:
            pred_all.append(i[0].squeeze().detach().cpu())  # transformed source
        sample_fin_img = torch.stack(pred_all, dim=0)
        torch.save(sample_fin_img, save_path)
        
        
        phi_save_path = os.path.join(save_dir_pc, 'phi_'+sample_filename)
        pred_phi = []
        for i in z_all:
            pred_phi.append(i[1].squeeze().detach().cpu())  # transformation
        sample_fin_phi = torch.stack(pred_phi, dim=0)
        torch.save(sample_fin_phi, phi_save_path)
        print(sample_fin_img.size(), sample_fin_phi.size())
        
        print(f"Saved PC-sampled volumes to {save_path}")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='./dataset/', help="Data DIR with all volumes")
    parser.add_argument("--save_dir", type=str, default='./samples/', help="DIR to save samples (sub DIR will be automatically defined)")
    
    args = parser.parse_args()
    global datal, dl, save_dir_pc
    
   
    datal = get_data(path=args.dataset_path, b=1, test_flag=True)
    dl = iter(datal)
    save_dir_pc = args.save_dir

    print("Starting PC Sampling Tests...")
    

    for i in range(1):
        configs = [
            # (step, use_pc_sampling, corrector_steps, corrector_step_size)
            (1, True, 2, 1e-5),    # PC with 2 corrector steps  
        ]
        
        for step, use_pc, c_steps, c_step_size in configs:
            print(f"\n{'='*80}")
            if use_pc:
                print(f"TEST {step}: PC Sampling (corrector_steps={c_steps}, step_size={c_step_size})")
            else:
                print(f"TEST {step}: Regular DDPM Sampling")
            print(f"{'='*80}")
            
            test(step=step, 
                use_pc_sampling=use_pc, 
                corrector_steps=c_steps, 
                corrector_step_size=c_step_size)
        
        print("All PC sampling tests completed! Check the 'sampling_visualization' folder for results.")
        print("Compare the results between different configurations to see the effect of PC sampling.")


if __name__ == '__main__':
    main()
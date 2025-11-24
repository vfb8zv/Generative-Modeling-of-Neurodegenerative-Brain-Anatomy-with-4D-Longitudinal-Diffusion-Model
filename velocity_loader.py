import os
import torch
import velocity as vt


# from dataset import get_data
import warnings
warnings.filterwarnings('ignore')

os.environ['NEURITE_BACKEND'] = 'pytorch'

inshape = [128]*3

out_channels= 3

enc_nf =  [16, 32, 32, 16]
dec_nf = [16, 32, 32, 16, 16, out_channels]

nb_gpus = 1
device = 'cuda'

def load_velocity_model(local_rank):
    device = torch.device(f'cuda:{local_rank}')

    model = vt.networks.DIGIT_reg(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        int_downsize=8,
        att_based=False,
        use_skip=True
    ).to(device)
    latest_reg = max([f for f in os.listdir('log_results/models/') if f.endswith('.pt')], key=lambda f: os.path.getmtime(os.path.join('log_results/models/', f)))
    checkpoint = torch.load(os.path.join('log_results/models/', latest_reg), map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    return model, device

today_x='22_04_2025'

model = None
device = None

def init_velocity_loader(local_rank):
    global model, device
    model, device = load_velocity_model(local_rank)
    latest_reg = max([f for f in os.listdir('log_results/models/') if f.endswith('.pt')], key=lambda f: os.path.getmtime(os.path.join('log_results/models/', f)))
    m=torch.load(os.path.join('log_results/models/', latest_reg))
    model.load_state_dict(m)

def tanh_normalize(data, scale=1.0):
    return torch.tanh(data / scale)

def tanh_unnormalize(normalized_data, scale=1.0):
    # Clip to prevent numerical issues at boundaries
    clipped = torch.clip(normalized_data, -0.9999, 0.9999)
    return scale * torch.arctanh(clipped)

def get_velocity(src, tgt):
    z=[]
    src = src.to(device)
    tgt = tgt.to(device)
    with torch.no_grad():
        for _ in range(3):   #3
            _, _, z_, _ = model(src, tgt[:,:,_,...])
            z_ = z_.squeeze()

            z_norm = tanh_normalize(z_, scale=1)
            z_unnorm = tanh_unnormalize(z_norm, scale=1)
            recovery_error = torch.abs(z_ - z_unnorm)
            z.append(z_norm)

    z = torch.stack(z, dim=1)
    return z
    
def get_deformed(x, src):
    z=[]
    src = src.to(device)
    with torch.no_grad():
        for _ in range(3):
            z.append(model.get_full_deform(x[:,:,_,...], src))
    return z

def get_gt(src, tgt):
    z=[]
    tgt = tgt.to(device)
    with torch.no_grad():
        for _ in range(3):
            z.append(model(src, tgt[:,:,_,...]))
    return z
    
def get_src_def(src, phi):
    src = src.to(device).unsqueeze(0)
    phi = phi.to(device).unsqueeze(0)
    
    with torch.no_grad():
        z = model.transformer(src,phi)
    return z.detach().cpu().squeeze()
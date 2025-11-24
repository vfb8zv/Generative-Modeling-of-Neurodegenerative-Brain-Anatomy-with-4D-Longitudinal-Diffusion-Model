import os
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from dataloader import get_data
from diffusion_m import DVIT_Diffusion
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from velocity_loader import init_velocity_loader
from ldm_train import DVIT
from dvit import DViT
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


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

log_dir = 'results/'  #'./log_results/'
models_dir = os.path.join(log_dir, 'models')

batch_size=3  #64,24
img_size = 32 #16
channels = 3
num_frames = 3

def init_distributed():
    dist.init_process_group(backend='nccl')  # Use NCCL for GPU communication
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    return local_rank

local_rank = init_distributed()  # Initialize process group
init_velocity_loader(local_rank)  # <- new line
is_main_process = local_rank == 0  # Flag for main process
cudnn.enabled = True
cudnn.benchmark = True

torch.manual_seed(42)
torch.cuda.manual_seed(42) 

datal = get_data(b=batch_size, test_flag=False,  large_def_split = 0.8, class_split = [0.2, 0.35, 0.45], balanced=False)  ## Set 'balanced' to True when complete dataset is available

## change parameters based on reported value
patch_size = 4 
embed_dim = 384  
num_heads = 6  
depth = 12  

denoiser = DViT(patch_size=patch_size,   ## ablation
                embed_dim=embed_dim,  ## ablation
                num_patches =None,
                img_size=img_size,
                channels=channels,
                num_heads=num_heads,  ## ablation
                mlp_ratio=4.0,
                num_classes=3,
                depth=depth,   ## ablation
                num_frames=num_frames,
                out_channels = channels).cuda()   ## ablation


diffusion = DVIT_Diffusion(
    denoiser,
    channels= channels,
    image_size = img_size,    
    timesteps = 1000,
    num_frames=3   
).cuda()

# Automatically get the latest checkpoint
latest_checkpoint = get_latest_checkpoint()

if latest_checkpoint:
    print(f"Loading from latest checkpoint: {latest_checkpoint}")
    finetune_name = latest_checkpoint
else:
    print("No checkpoint found, starting from scratch")
    finetune_name = ''

# Wrap only the model (not the trainer) with DDP
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    diffusion = DDP(diffusion, device_ids=[local_rank], output_device=local_rank)

lcgd_model = DVIT(
        diffusion, 
        datal, 
        train_batch_size = batch_size,  
        train_lr = 1e-4,  
        model_name = f'z_genmodel_{patch_size}_{embed_dim}_{num_heads}_{depth}',
        train_num_steps = 1000000, 
        finetune = finetune_name,
        batch_normed = True,
        results_folder = log_dir
    )

lcgd_model.train()
import os
import numpy as np
import torch
import glob
# from DIGIT import unet_ldm
from torch.utils.data import Dataset
import einops
import pandas as pd

import velocity as vt
# from DIGIT import ldm_train
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

colors_list= list(mcolors.CSS4_COLORS.keys())

from datetime import datetime
todays = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")[:10]
print(str(todays))
timestep = 3 # excluding the source

os.makedirs('log_results/models', exist_ok=True)
os.makedirs('log_results/losses', exist_ok=True)


def std_img(tens):
    t_ = (tens-tens.min())/(tens.max()-tens.min())
    return t_

def find_latest_checkpoint():
    """Find the latest checkpoint file"""
    checkpoint_pattern = './log_results/models/regae_*.pt'
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None, 0
    
    # Sort by modification time and get the latest
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    
    # Extract epoch number from saved losses file
    checkpoint_date = latest_checkpoint.split('regae_')[-1].split('.pt')[0]
    loss_file = f'./log_results/losses/regae_{checkpoint_date}.txt'
    
    start_epoch = 0
    if os.path.exists(loss_file):
        try:
            saved_losses = torch.load(loss_file)
            start_epoch = len(saved_losses)  # Number of steps completed
            print(f"Found {len(saved_losses)} saved loss values")
        except:
            print("Could not load loss history")
    
    return latest_checkpoint, start_epoch

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer state from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        
        # If checkpoint is just state_dict (your current saving format)
        if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
            model_state_dict = model.state_dict()
            checkpoint_state_dict = checkpoint
            
            # Check for missing keys
            missing_keys = set(model_state_dict.keys()) - set(checkpoint_state_dict.keys())
            unexpected_keys = set(checkpoint_state_dict.keys()) - set(model_state_dict.keys())
            
            if missing_keys:
                print(f"Missing keys in checkpoint: {missing_keys}")
                print("Initializing missing keys with model's current values...")
                
                # Keep the model's initialized values for missing keys
                for key in missing_keys:
                    checkpoint_state_dict[key] = model_state_dict[key]
            
            if unexpected_keys:
                print(f"Unexpected keys in checkpoint: {unexpected_keys}")
                # Remove unexpected keys
                for key in unexpected_keys:
                    del checkpoint_state_dict[key]
            
            # Load the updated state dict
            model.load_state_dict(checkpoint_state_dict)
            print("Loaded model state_dict with missing keys initialized")
            return True
        
        # If checkpoint contains model and optimizer states
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = model.state_dict()
            checkpoint_state_dict = checkpoint['model_state_dict']
            
            # Handle missing keys
            missing_keys = set(model_state_dict.keys()) - set(checkpoint_state_dict.keys())
            if missing_keys:
                print(f"Missing keys in checkpoint: {missing_keys}")
                print("Initializing missing keys with model's current values...")
                for key in missing_keys:
                    checkpoint_state_dict[key] = model_state_dict[key]
            
            model.load_state_dict(checkpoint_state_dict)
            
            if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Loaded model and optimizer state_dict")
                except:
                    print("Loaded model state_dict only (optimizer state incompatible)")
            else:
                print("Loaded model state_dict only")
            return True
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

class DataLoading(Dataset):
    def __init__(self, test_flag=False):
        self.test_flag = test_flag
        
        if self.test_flag==True:
            self.typ = 'test'

            self.files = ([f for f in os.listdir('./dataset/ad_split/test/') if f.endswith('.pt')] + 
                          [f for f in os.listdir('./dataset/cn_split/test/') if f.endswith('.pt')] +
                          [f for f in os.listdir('./dataset/mci_split/test/') if f.endswith('.pt')])
            
        else:
            self.typ = 'train'
            self.files = ([f for f in os.listdir('./dataset/ad_split/train/') if f.endswith('.pt')] + 
                          [f for f in os.listdir('./dataset/cn_split/train/') if f.endswith('.pt')] +
                          [f for f in os.listdir('./dataset/mci_split/train/') if f.endswith('.pt')])
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f_name = self.files[idx]
            
        if 'ad' in f_name:
            split = 'ad_split'
        elif 'cn' in f_name:
            split = 'cn_split'
        else:
            split = 'mci_split'
            
        im = torch.load(os.path.join('./dataset/', split, self.typ, f_name))
        for i in range(4):
            im[i] = std_img(im[i])
            assert im[i].min()==0 and im[i].max()==1
        src = im[0].unsqueeze(0).repeat(3,1,1,1).unsqueeze(1)
        tgt = im[1:].unsqueeze(1)
        assert src.size()==tgt.size()
                             
        return src, tgt, f_name
            
        
def get_data(test_flag=False, b=32):
    ds = DataLoading(test_flag=test_flag)
    datal= torch.utils.data.DataLoader(ds, batch_size=b, shuffle=True)
    return datal

def plot_slices(source, target, y_pred, slice_idx=None, save_dir='visualization', step=0):
    """
    Plot source, target, prediction across 3 volumes and 3 views.
    
    Args:
        source: [3,128,128,128] tensor - source image
        target: [3,128,128,128] tensor - target image  
        y_pred: [3,128,128,128] tensor - prediction
        slice_idx: slice index (default: middle slice)
        save_dir: save directory
        step: step number for filename
    """
    def process_tensor(tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        return tensor  # shape [3, D, H, W]

    source = process_tensor(source)
    target = process_tensor(target)
    y_pred = process_tensor(y_pred)

    if slice_idx is None:
        slice_idx = source.shape[1] // 2

    titles = ['Volume 0', 'Volume 1', 'Volume 2']
    views = ['Axial', 'Coronal', 'Sagittal']
    view_slices = [
        lambda vol: vol[:, :, slice_idx],   # Axial (Z)
        lambda vol: vol[:, slice_idx, :],   # Coronal (Y)
        lambda vol: vol[slice_idx, :, :]    # Sagittal (X)
    ]

    # Set up figure
    fig, axes = plt.subplots(3, 3 * 3, figsize=(15, 9))  # 3 rows (views), 9 columns (3 images × 3 volumes)

    for v in range(3):  # For each volume index
        for col, img in enumerate([source[v], target[v], y_pred[v]]):  # For source, target, prediction
            for row in range(3):  # For axial, coronal, sagittal
                ax = axes[row, 3*v + col]
                ax.imshow(view_slices[row](img), cmap='gray')
                ax.axis('off')

                # Label columns on top
                if row == 0:
                    label = ['Source', 'Target', 'Prediction'][col]
                    ax.set_title(f'{label} ({titles[v]})', fontsize=10)

                # Label rows on side
                if col == 0 and v == 0:
                    ax.set_ylabel(views[row], fontweight='bold')

    plt.suptitle(f'Registration Results - Step {step}', fontsize=14)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/registration_step_{step}.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved registration plot for step {step}.")


def evaluate_model(model, checkpoint_path=None, save_dir='log_results/eval_all_visualizations', batch_size=1, timestep=3, device='cuda'):
    """
    Evaluate the registration model on the test dataset.
    
    Args:
        model: Trained model to evaluate (already on device).
        checkpoint_path: Optional path to a saved model checkpoint.
        batch_size: Batch size for test DataLoader.
        timestep: Number of volumes to visualize.
        device: 'cuda' or 'cpu'
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Load test dataset
    test_loader = get_data(test_flag=True, b=1)

    mse_loss_fn = vt.losses.MSE().loss
    grad_loss_fn = vt.losses.Grad().loss

    mse_total = []
    grad_total = []

    with torch.no_grad():
        for idx, (source, target, file_name) in enumerate(test_loader):
            source = rearrange(source, 'b f c h w l -> (b f) c h w l').to(device).float()
            target = rearrange(target, 'b f c h w l -> (b f) c h w l').to(device).float()

            y_pred, flow, velocity, _ = model(source, target)

            mse = mse_loss_fn(target, y_pred).item()
            grad = grad_loss_fn(velocity, velocity).item()

            mse_total.append(mse)
            grad_total.append(grad)

            # Save visualization for first batch
            #if idx == 0:
            plot_slices(
                source[0:timestep].squeeze(1),
                target[0:timestep].squeeze(1),
                y_pred[0:timestep].squeeze(1),
                step=file_name,
                save_dir=save_dir
            )

    print("Evaluation Results:")
    print(f"Average MSE Loss:  {np.mean(mse_total):.4e}")
    print(f"Average Grad Loss: {np.mean(grad_total):.4e}")

    model.train()

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

model = vt.networks.DIGIT_reg(
    inshape=inshape,
    nb_unet_features=[enc_nf, dec_nf]).cuda()

batch_size= 2
datal = get_data(b=batch_size, test_flag=False) 
generator = iter(datal)

EPOCHS = 1500
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-14)
losses = [vt.losses.MSE().loss, vt.losses.Grad().loss]
weights = [1, 1e-2]  #1e-2

# Load checkpoint if available
latest_checkpoint, steps_completed = find_latest_checkpoint()
start_epoch = 0
saving_loss = []

if latest_checkpoint:
    success = load_checkpoint(model, optimizer, latest_checkpoint)
    if success:
        # Load existing loss history
        checkpoint_date = latest_checkpoint.split('regae_')[-1].split('.pt')[0]
        loss_file = f'./log_results/losses/regae_{checkpoint_date}.txt'
        if os.path.exists(loss_file):
            try:
                saving_loss = torch.load(loss_file)
                # Estimate start epoch from steps (approximate)
                steps_per_epoch = len(generator)
                start_epoch = len(saving_loss) // steps_per_epoch
                print(f"Resuming from epoch {start_epoch}, step {len(saving_loss)}")
            except:
                print("Could not load loss history, starting fresh")
        
        # Use the same checkpoint date for continued training
        todays = checkpoint_date
    else:
        print("Failed to load checkpoint, starting from scratch")
else:
    print("No checkpoint found, starting training from scratch")

model.train()

from einops import rearrange
reg_error = []

for epoch in range(start_epoch, EPOCHS):
    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(len(generator)):
        # generate inputs (and true outputs) and convert them to tensors
        try:
            source, target, filename = next(generator)
        except:
            generator = iter(datal)
            source, target, filename = next(generator)
            
        source = rearrange(source, 'b f c h w l -> (b f) c h w l')
        target = rearrange(target, 'b f c h w l -> (b f) c h w l')
        
        loss=0
        
        inputs = [source.cuda().float(), target.cuda().float()]

        y_pred, flow_field, velocity, x= model(*inputs)

        curr_loss = losses[0](target.to(device), y_pred)
        loss += curr_loss
        loss+= weights[-1]*losses[1](velocity, velocity)

        epoch_loss.append([curr_loss.item()])
        epoch_total_loss.append(loss.item())
        saving_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #evaluate_model(model)
        reg_error.append(np.mean(epoch_total_loss, axis=0))

    if epoch % 1 == 0:
        epoch_info = 'Epoch %d/%d' % (epoch + 1, EPOCHS)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        
        if epoch % 10 == 0:
            # Save model checkpoint
            torch.save(model.state_dict(), './log_results/models/regae_'+str(todays)+'_'+str(epoch)+'.pt')
            torch.save(saving_loss, './log_results/losses/regae_'+str(todays)+'.txt')
            print("New model is being saved")
        
        if epoch % 5 == 0:
            evaluate_model(model)
        
        print(' - '.join((epoch_info, loss_info)), flush=True)
        
        plot_slices(source[0:timestep].squeeze(1), target[0:timestep].squeeze(1), y_pred[0:timestep].squeeze(1), step=epoch, save_dir='log_results/visualization')

    # final model save
    torch.save(model.state_dict(), './log_results/models/regae_'+str(todays)+'.pt')
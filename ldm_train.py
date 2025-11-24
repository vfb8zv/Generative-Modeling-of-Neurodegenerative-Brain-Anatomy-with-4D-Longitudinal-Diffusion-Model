import torch, os
import numpy as np
import random
from scipy import ndimage
from torch.optim import Adam
from torch.utils import data
import numpy as np
import gc
from pathlib import Path
import time

from datetime import datetime

torch.manual_seed(42)
torch.cuda.manual_seed(42) 


def cycle(dl):
    while True:
        for data in dl:
            yield data
    

class DVIT(object):
    def __init__(
        self,
        diffusion_model, 
        dataset,
        *,
        train_batch_size = 3,
        train_lr = 1e-4,
        train_num_steps = 100000,
        results_folder = 'results/',
        model_name = 'gen_model',
        finetune = '',
        batch_normed=False
    ):
        super().__init__()
        self.model = diffusion_model
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_main_process = self.local_rank == 0
        self.ds= dataset
        self.bn = batch_normed
        if len(finetune)>0:
            self.finetune = finetune +'.pt'
        else:
            self.finetune = finetune

        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps

        self.dl = iter(self.ds)

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        self.max_saved_models = 5
        self.step = 0
        self.todays = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        self.model_name = model_name +'.pt'
        self.loss_name = model_name +'_loss' + '.pt'
        self.results_folder = results_folder

        if len(self.finetune)>0:
            checkpoint = torch.load(os.path.join(self.results_folder,'models' , self.finetune))   
            state_dict = checkpoint['model_state_dict']
            try:
                model_state_dict = self.model.state_dict()

                filtered_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].size() == state_dict[k].size()}
                print('Found ', len(filtered_dict), 'matching layers.')
                model_state_dict.update(filtered_dict)
                del filtered_dict
                gc.collect()
                self.model.load_state_dict(model_state_dict)
                self.step=0
                print('Pre-trained model loaded at provided checkpoint: ', self.finetune)
            except:
                print('Invalid Finetune Checkpoint.')
            self.model.load_state_dict(model_state_dict)
        else:
            print('No pre-trained model provided')
        

    def load(self):
        checkpoint = torch.load(os.path.join(self.results_folder,'models' , self.finetune))   
        if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                new_state_dict[k.replace("module.", "")] = v
            self.model.load_state_dict(new_state_dict)
            print(f'loaded the model {self.finetune}')
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def train(self):
        
        loss_list=[]
        local_step = self.step
        global_step = self.local_rank + local_step * torch.distributed.get_world_size()
        os.makedirs(os.path.join(self.results_folder, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.results_folder, 'losses'), exist_ok=True)
        
        while self.step < self.train_num_steps:
            step_start_time = time.time()
            try:
                z, cond_c, cond_i, cond_a, im, ptid = next(self.dl)
            except:
                self.dl= iter(self.ds)
                z, cond_c, cond_i, cond_a, im, ptid = next(self.dl)
            
            if self.bn:
                loss, diffusion_loss, sim_loss, smoothness_loss = self.model(z, cond_c, cond_i, cond_a, im, global_step)
            else:
                loss = self.model(z, cond_c, cond_i, cond_a)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            current_lr = self.opt.param_groups[0]['lr']
            
            # Calculate step time
            step_time = time.time() - step_start_time
            it_per_sec = 1.0 / step_time if step_time > 0 else 0

            self.step += 1
            local_step += 1
            global_step = self.local_rank + local_step * torch.distributed.get_world_size()

            # Save model and print progress
            if self.is_main_process and global_step != 0 and global_step % 10 == 0:  
                model_save_path = os.path.join(self.results_folder, 'models', 
                                                self.todays[:-8] + f'{self.step}_{self.model_name}')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'step': self.step,
                    'lr': current_lr
                }, model_save_path)
                print("New model is saved at: ", model_save_path)

                # Manage model history: delete oldest if more than threshold
                result_folder = model_save_path.split("/")
                result_folder = os.path.dirname(model_save_path)
                model_files = sorted(
                    [os.path.join(result_folder, f) for f in os.listdir(result_folder) if f.endswith('.pt')],
                    key=os.path.getmtime  # Sort by modification time
                )

                if len(model_files) > self.max_saved_models:
                    num_to_delete = len(model_files) - self.max_saved_models
                    for old_model in model_files[:num_to_delete]:
                        try:
                            os.remove(old_model)
                            print(f"Deleted old model: {old_model}")
                        except Exception as e:
                            print(f"Failed to delete model {old_model}: {e}")

            if global_step != 0:
                print(f'\033[91mStep {global_step}: Loss: {loss.item():.6f} | '
                f'Diffusion Loss: {diffusion_loss.item():.6f} | '
                f'Similarity Loss: {sim_loss.item():.6f} | '
                f'Smoothness Loss: {smoothness_loss.item():.6f} | '
                f'{it_per_sec:.2f} it/sec | '
                f'Progress: {self.step/self.train_num_steps*100:.1f}%\033[0m')
                
            loss_list.append(loss.item())
            # Save loss every 10 steps
            if self.step != 0 and self.step % 10 == 0:
                loss_save_path = os.path.join(self.results_folder, 'losses', 
                                            self.todays[:-8] + f'{self.step}_{self.loss_name}')
                torch.save(loss_list, loss_save_path)

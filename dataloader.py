import torch, os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from scipy import ndimage
import velocity_loader
from torch.utils.data import Sampler
import random


torch.manual_seed(42)
torch.cuda.manual_seed(42) 

def resize_volume(img, ex=32):
    pr = img.shape[-1]
    img = ndimage.zoom(img, (1, ex/pr, ex/pr, ex/pr),order=1)
    return img

def std_img(tens):
    t_ = (tens-tens.min())/(tens.max()-tens.min())
    return t_

class DataLoading(Dataset):
    def __init__(self, path, test_flag=False):
        self.pth = path
        self.files = []
        self.test_flag = test_flag
        self.typ = 'test' if test_flag else 'train'

        self.df1 = pd.read_csv('dataset/MNI_data_DX_4f.csv')[['ptid', 'age_list']]
        self.df2 = pd.read_csv('dataset/MNI_88.csv')[['ptid', 'age_list']]
        self.df = pd.concat([self.df1, self.df2], axis=0)
        
        self.files_df1 = []
        self.files_df2 = []
                
        label_val = {'ad_split': 1, 'cn_split': 0, 'mci_split': 2}
        self.index_labels_df1 = []; self.index_labels_df2 = []
                    
        for cl in ['ad_split', 'cn_split', 'mci_split']:
            folder = os.path.join(self.pth, cl, self.typ)
            for f in os.listdir(folder):
                ptid = f.split('_', 1)[-1].rsplit('.', 1)[0]
                if ptid in self.df1['ptid'].values:
                    self.files_df1.append((cl, f))
                    self.index_labels_df1.append(label_val[cl])
                elif ptid in self.df2['ptid'].values:
                    self.files_df2.append((cl, f))
                    self.index_labels_df2.append(label_val[cl])
                

        self.files = self.files_df1 + self.files_df2
        self.index_source = [0] * len(self.files_df1) + [1] * len(self.files_df2)  # 0=df1, 1=df2
       

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cl, fname = self.files[idx]
        im = torch.load(os.path.join(self.pth, cl, self.typ, fname))
        im = std_img(im.unsqueeze(0).unsqueeze(0))

        label = {'ad_split': 1, 'cn_split': 0, 'mci_split': 2}[cl]
        pid = fname.split('_', 1)[-1].rsplit('.', 1)[0]
        age_list = self.df[self.df['ptid'] == pid]['age_list'].values[0]
        age_list = torch.from_numpy(np.fromstring(age_list.strip('[]'), sep=' '))

        z = velocity_loader.get_velocity(im[:, :, 0, ...].cuda(), im[:, :, 1:, ...].cuda())
        grad_src = np.array(np.gradient(im[0, 0, 0, ...].numpy()))
        grad_src = resize_volume(grad_src, ex=z.shape[-1])
        grad_src = std_img(torch.from_numpy(grad_src)).cuda()

        return z, torch.tensor(int(label)).cuda(), grad_src, age_list.cuda(), im, fname

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, large_def_split, class_split):
        self.batch_size = batch_size
        self.df1_indices = [i for i, src in enumerate(dataset.index_source) if src == 0]
        self.df2_indices = [i for i, src in enumerate(dataset.index_source) if src == 1]
                
        self.df1_cn_indices = [i for ci, i in enumerate(self.df1_indices) if dataset.index_labels_df1[ci] == 0]
        self.df1_ad_indices = [i for ci, i in enumerate(self.df1_indices) if dataset.index_labels_df1[ci] == 1]
        self.df1_mci_indices = [i for  ci, i in enumerate(self.df1_indices) if dataset.index_labels_df1[ci] == 2]

        self.df2_cn_indices = [i for ci, i in enumerate(self.df2_indices) if dataset.index_labels_df2[ci] == 0]
        self.df2_ad_indices = [i for ci, i in enumerate(self.df2_indices) if dataset.index_labels_df2[ci] == 1]
        self.df2_mci_indices = [i for ci, i in enumerate(self.df2_indices) if dataset.index_labels_df2[ci] == 2]
        
        n_df2_total = int(round(self.batch_size * large_def_split))
        self.n_df2_cn = max(0, int(round(n_df2_total * class_split[0])))
        self.n_df2_ad = max(0, int(round(n_df2_total * class_split[1])))
        self.n_df2_mci = max(0, int(round(n_df2_total * class_split[2])))

        n_df1_total = self.batch_size - n_df2_total
        self.n_df1_cn = max(0, int(round(n_df1_total * class_split[0])))
        self.n_df1_ad = max(0, int(round(n_df1_total * class_split[1])))
        self.n_df1_mci = max(0, int(round(n_df1_total * class_split[2])))
        
        print(self.n_df2_cn, self.n_df2_ad, self.n_df2_mci, self.n_df1_cn, self.n_df1_ad, self.n_df1_mci)
        print(len(self.df2_cn_indices), len(self.df2_ad_indices), len(self.df2_mci_indices), len(self.df1_cn_indices), len(self.df1_ad_indices), len(self.df1_mci_indices))
        
        max_batches_per_category=[]
        if self.n_df1_cn > 0: max_batches_per_category.append(len(self.df1_cn_indices) // self.n_df1_cn)
        if self.n_df1_ad > 0: max_batches_per_category.append(len(self.df1_ad_indices) // self.n_df1_ad)
        if self.n_df1_mci > 0: max_batches_per_category.append(len(self.df1_mci_indices) // self.n_df1_mci)

        if self.n_df2_cn > 0: max_batches_per_category.append(len(self.df2_cn_indices) // self.n_df2_cn)
        if self.n_df2_ad > 0: max_batches_per_category.append(len(self.df2_ad_indices) // self.n_df2_ad)
        if self.n_df2_mci > 0: max_batches_per_category.append(len(self.df2_mci_indices) // self.n_df2_mci)
        
        self.num_batches = min(max_batches_per_category)

    def __iter__(self):
        random.shuffle(self.df1_cn_indices)
        random.shuffle(self.df1_ad_indices)
        random.shuffle(self.df1_mci_indices)
        random.shuffle(self.df2_cn_indices)
        random.shuffle(self.df2_ad_indices)
        random.shuffle(self.df2_mci_indices)

        df1_cn_iter = iter(self.df1_cn_indices)
        df1_ad_iter = iter(self.df1_ad_indices)
        df1_mci_iter = iter(self.df1_mci_indices)
        df2_cn_iter = iter(self.df2_cn_indices)
        df2_ad_iter = iter(self.df2_ad_indices)
        df2_mci_iter = iter(self.df2_mci_indices)

        for _ in range(self.num_batches):
            batch = []
            try:
                batch += [next(df1_cn_iter) for _ in range(self.n_df1_cn)]
                batch += [next(df1_ad_iter) for _ in range(self.n_df1_ad)]
                batch += [next(df1_mci_iter) for _ in range(self.n_df1_mci)]

                batch += [next(df2_cn_iter) for _ in range(self.n_df2_cn)]
                batch += [next(df2_ad_iter) for _ in range(self.n_df2_ad)]
                batch += [next(df2_mci_iter) for _ in range(self.n_df2_mci)]

            except StopIteration:
                break
            random.shuffle(batch)
            yield batch


def get_data(path='./dataset/', b=1, test_flag=False, large_def_split = 0.7, class_split = [0.2, 0.4, 0.4], balanced=True):  ## [CN, AD, MCI]
    ds = DataLoading(path, test_flag)
    if test_flag or not balanced:
        return torch.utils.data.DataLoader(ds, batch_size=b, shuffle=False)
    else:
        sampler = BalancedBatchSampler(ds, batch_size=b, large_def_split = large_def_split, class_split = class_split)
        return torch.utils.data.DataLoader(ds, batch_sampler=sampler)



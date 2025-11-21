#!/usr/bin/env python3

"""
UNetConvNext model for Project03
Daniel Villarruel-Yanez (2025.11.13)
"""

import os
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import DataLoader

from the_well.data import WellDataset
from the_well.benchmark.models.unet_convnext import UNetConvNext

import logging
import argparse
#from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class CNextUNetbaseline(nn.Module):
    """
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
               initial_dimension: int,
               up_down_blocks: int,
               blocks_per_stage: int,
               bottleneck_blocks: int,
               spatial_resolution: tuple,
               spatial_dims: int = 2):
         
         super().__init__()
         
         self.model = UNetConvNext(
             dim_in=in_channels,
             dim_out=out_channels,
             n_spatial_dims=spatial_dims,
             spatial_resolution=spatial_resolution,             stages=up_down_blocks,
             blocks_per_stage=blocks_per_stage,
             blocks_at_neck=bottleneck_blocks,
             init_features=initial_dimension
             )
         
    def forward(self, x):
        return self.model(x)

class ModelTrainer:
    def __init__(self, path, model, device, n_workers):
        self.path = path
        self.model = model

        if isinstance(device, str):
            self.device = torch.device('cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        self.n_workers = int(n_workers) if n_workers is not None else 0
        self.F = 11
        self.mu = None
        self.sigma = None
        self.model_instance = None

    def _loader(self, split, n_steps_input, n_steps_output):
        dataset = WellDataset(
            well_base_path = self.path,
            well_dataset_name = 'active_matter',
            well_split_name = split,
            n_steps_input = n_steps_input,
            n_steps_output = n_steps_output,
            use_normalization = False
        )

        return dataset
    
    def _setup(self, n_input, n_output):
        logging.info('Setting up model and normalisation')

        train_dataset = self._loader('train', n_input, n_output)

        max_samples = min(20, len(train_dataset))
        sample_idx  = np.linspace(0, len(train_dataset) - 1, num=max_samples).astype(int)

        mu_sum = torch.zeros(self.F)
        sigma_sum = torch.zeros(self.F)
        count = 0

        for idx in sample_idx:
            x = train_dataset[idx]['input_fields']
            x = x.reshape(-1, self.F)

            mu_sum += x.mean(dim=0)
            sigma_sum += x.std(dim=0)
            count += 1

        self.mu = (mu_sum / count).to(self.device)
        self.sigma = (sigma_sum / count).to(self.device)
        self.sigma[self.sigma == 0] = 1.0

        #xs = []
        
        #for i in range(0, 200, 50):
             #x = train_dataset[i]["input_fields"]
             #if isinstance(x, np.ndarray):
                #x = torch.from_numpy(x)
             #xs.append(x)
             
        #xs = torch.stack(xs)

        #self.mu = xs.reshape(-1, self.F).mean(dim=0).to(self.device)
        #self.sigma = xs.reshape(-1, self.F).std(dim=0).to(self.device)
        #self.sigma[self.sigma == 0.0] = 1.0

        logging.info('Normalisation stats calculated!')

        in_channels = n_input * self.F
        out_channels = n_output * self.F
        grid = (256, 256)

        self.model_instance = self.model(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_resolution=grid,
            initial_dimension=42,
            up_down_blocks=4,
            blocks_per_stage=2,
            bottleneck_blocks=1
            ).to(self.device)
    
    def _preprocess(self, x):
        if self.mu is None:
            raise RuntimeError('Call _setup first!')
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(self.device)
        return (x - self.mu) / self.sigma
    
    def _postprocess(self, y):
        if self.mu is None:
            raise RuntimeError('Call _setup first!')
        
        mu_b = self.mu.view(1, self.F, 1, 1)
        sigma_b = self.sigma.view(1, self.F, 1, 1)

        return (y * sigma_b) + mu_b
    
    def train(self, batch, epochs, lr, patience, n_input, n_output):
        
        self._setup(n_input, n_output)
        optimizer = torch.optim.Adam(self.model_instance.parameters(), lr=lr)

        train_dataset = self._loader('train', n_input, n_output)
        valid_dataset  = self._loader('valid', n_input, n_output)

        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=self.n_workers)
        val_loader = DataLoader(valid_dataset, batch_size=batch, shuffle=False, num_workers=self.n_workers)

        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        patience_count = 0

        for epoch in range(epochs):
            train_loss = 0.0
            self.model_instance.train()
            print(f'EPOCH {epoch + 1} / {epochs}')

            for batch in (bar := tqdm(train_loader)):
                x = batch['input_fields']
                x = self._preprocess(x)
                x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

                y = batch['output_fields']
                y = self._preprocess(y)
                y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")

                fx = self.model_instance(x)
                
                loss = criterion(fx, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bar.set_postfix(loss=loss.item())
                train_loss += loss.item()

            train_loss /= max(1, len(train_loader))
            train_losses.append(train_loss)

            self.model_instance.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in (bar := tqdm(val_loader)):
                    x = batch['input_fields']
                    x = self._preprocess(x)
                    x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                    
                    y = batch['output_fields']
                    y = self._preprocess(y)
                    y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")
                    
                    fx = self.model_instance(x)

                    loss = criterion(fx, y)
                    bar.set_postfix(loss=loss.item())
                    val_loss += loss.item()

            val_loss /= max(1, len(val_loader))
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0

                save_dir = './outputs/'
                os.makedirs(save_dir, exist_ok=True)
                #Path('checkpoints'); save_dir.mkdir(exist_ok=True, parents=True)
                torch.save(self.model_instance.state_dict(), save_dir + 'best_model.pth')
                logging.info(f"Saved best model to {save_dir + 'best_model.pth'}")
            else:
                patience_count += 1

            if patience_count >= patience:
                print('Early stop triggered')
                break

        return train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(
        prog = 'UNetConvNext baseline model trainer',
        description = 'Model trainer for the active_matter dataset of The Well'
    )

    parser.add_argument('path', type=str, help='Path to active matter dataset')
    parser.add_argument('-n', '--num', type=int, help='Number of processors')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    args = parser.parse_args()
    if not os.path.isdir(args.path):
        logging.error(f'Path to active_matter dataset not found: {args.path}')
        return
    
    path = args.path
    n_workers = args.num
    device = args.device
    
    trainer = ModelTrainer(path, CNextUNetbaseline, device, n_workers)
    train, valid = trainer.train(batch=4, epochs=156, lr=5e-3, patience=5, n_input=4, n_output=1)

    print('TRAINING successful')

    outfile = 'losses.dat'
    with open(outfile, 'w') as f:
        f.write('epoch train_loss valid_loss\n')
        maxlen = max(len(train), len(valid))
        for i in range(maxlen):
            t = train[i] if i < len(train) else 0
            v = valid[i] if i < len(valid) else 0
            f.write(f'{i+1} {t} {v}\n')

    print('losses.dat file located in ./outputs')

if __name__ == '__main__':
    main()
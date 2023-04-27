import sys
sys.path.append('/work/jiaqi/Diff2D3D')
import torch
import torch_geometric as pyg
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import os, time, yaml, json, argparse
from tqdm.auto import tqdm
from datetime import datetime

from models.diffusion import Diffusion2D3D
from utils.common import get_logger, get_new_log_dir, count_parameters, seed_all


def train(model, config):
    pass

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('config', type=str, default='./configs/train_diffusion.yml')

if __name__ == '__main__':
    main()


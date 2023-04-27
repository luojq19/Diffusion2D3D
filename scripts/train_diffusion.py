import sys
sys.path.append('/work/jiaqi/Diff2D3D')
import torch
import torch_geometric as pyg
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import os, time, yaml, json, argparse, shutil
from tqdm.auto import tqdm
from datetime import datetime

from models.diffusion import Diffusion2D3D
from utils import common


def train(model, config):
    pass

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('config', type=str, default='./configs/train_diffusion.yml')
    parser.add_argument('--log_dir', type=str, default='./logs_diffusion')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tag', type=str, default='')
    
    args = parser.parse_args()
    
    # Load configs
    config = common.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    common.seed_all(config.train.seed)

    # Logging
    log_dir = common.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = common.get_logger('train', log_dir)
    writer = SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))
    
    # datasets
    
    
    
    # train
    

if __name__ == '__main__':
    main()


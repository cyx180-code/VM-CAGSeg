import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import CAGDatasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet
from models.vmunet.cagsegmamba import CAGSegMamba

from engine_cag import *
import os
import sys

from utils import *
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")
import argparse
from ema_pytorch import EMA


def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    resume_model = config.pretrained_path
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)


    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()


    print(f'#----------Preparing dataset{config.phase}----------#')
    test_dataset = CAGDatasets(config.data_path, config, split=config.phase)
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)





    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network.startswith('vmunet'):
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from(logger=logger)
    elif config.network.startswith('cagsegmamba'):
        model = CAGSegMamba(
            # num_classes=model_cfg['num_classes'],
            # input_channels=model_cfg['input_channels'],
            # depths=model_cfg['depths'],
            # depths_decoder=model_cfg['depths_decoder'],
            # drop_path_rate=model_cfg['drop_path_rate'],
            # load_ckpt_path=model_cfg['load_ckpt_path'],
            # vepe=model_cfg['vepe'],
            **model_cfg
        )
        model.load_from(logger=logger)
    else: raise Exception('network in not right!')
    model = model.cuda()

    if config.ema:
        ema = EMA(model, beta=config.alpha, update_after_step=100, update_every=1)

    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion

    checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
    best_ep = checkpoint['epoch']
    print(f'#----------Testing epoch:{best_ep}----------#')
    model_val = None
    if config.ema:
        model_val = ema.ema_model
        ema.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        model_val = model
        model.load_state_dict(checkpoint['model_state_dict'])
    # best_weight = torch.load(resume_model, map_location=torch.device('cpu'))
    # model.load_state_dict(best_weight)
    loss = test_one_epoch(
            test_loader,
            model_val,
            criterion,
            logger,
            config,
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Coronary dataset to mmsegmentation format')
    parser.add_argument('-c', '--config', help='config', default='configs/cagsegmamba_vepe_b2.py')
    parser.add_argument('-gpu', '--gpu_id', help='gpu id', default='0')
    parser.add_argument('-phase', '--phase', help='phase test/val', default='test')
    # parser.add_argument('-split', '--split', help='split of dataset', default='test')
    # parser.add_argument('-closs', '--conn_loss', help='conn_loss_rate', action='store_true')
    # parser.add_argument('-closs_rate', '--conn_loss_rate', help='conn_loss_rate', default=1.0)
    # parser.add_argument('-vepe', '--vepe', help='vepe version', default='vepe_b3')  # off, b3, b2
    args = parser.parse_args()
    return args

def update_config_with_args(config, args):
    config.gpu_id = args.gpu_id
    config.phase = args.phase
    config.work_dir = 'results/' + config.network + '_' + config.phase + '_' + config.datasets + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '/'
    # config.split = args.split
    # config.network = os.path.basename(args.config).split('-')[0]
    # if args.conn_loss:
    #     config.criterion = ConnectivityLoss(wb=1.0, wd=1.0, wc=float(args.conn_loss_rate))
    #     config.network = config.network + "_closs" + args.conn_loss_rate
    # config.model_config['vepe'] = args.vepe
    # if args.vepe != "vepe_off":
    #     config.network = config.network + "_" + args.vepe

import importlib.util

def load_config_from_file(filepath):
    module_name = os.path.basename(args.config).split('.')[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

if __name__ == '__main__':
    args = parse_args()
    print(args)
    config = load_config_from_file(args.config).setting_config
    update_config_with_args(config, args)
    print(config.network, config.gpu_id, config.criterion)
    main(config)

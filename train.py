import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import CAGDatasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet
from models.vmunet.cagsegmamba import CAGSegMamba
# from models.cagsegnet.cagsegnet import CAGSegNet

from engine_cag import *
import os
import sys

from utils import *


import warnings
warnings.filterwarnings("ignore")
import argparse
from datetime import datetime
from ema_pytorch import EMA


def main(config):

    print('#----------Creating logger----------#')
    # config.work_dir = config.get_work_dir()
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()





    print('#----------Preparing dataset----------#')
    train_dataset = CAGDatasets(config.data_path, config, split="train")
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True)
    val_dataset = CAGDatasets(config.data_path, config, split="val")
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    test_dataset = CAGDatasets(config.data_path, config, split="test")
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)





    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    ema = None
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
    # elif config.network.startswith('cagsegnet'):
    #     model = CAGSegNet(
    #         num_classes=model_cfg['num_classes'],
    #         in_chans=model_cfg['input_channels'],
    #         depths=model_cfg['depths'],
    #         depths_decoder=model_cfg['depths_decoder'],
    #         drop_path_rate=model_cfg['drop_path_rate'],
    #         load_ckpt_path=model_cfg['load_ckpt_path'],
    #     )
    #     model.load_from(logger=logger)
    elif config.network.startswith('cagsegmamba'):
        model = CAGSegMamba(
            # num_classes=model_cfg['num_classes'],
            # input_channels=model_cfg['input_channels'],
            # depths=model_cfg['depths'],
            # depths_decoder=model_cfg['depths_decoder'],
            # drop_path_rate=model_cfg['drop_path_rate'],
            # load_ckpt_path=model_cfg['load_ckpt_path'],
            # vepe=model_cfg['vepe'],
            # neck=model_cfg['neck'],
            # skip_attention=model_cfg['skip_attention'],
            **model_cfg,
        )
        model.load_from(logger=logger)
    else: raise Exception('network in not right!')
    model = model.cuda()

    if config.ema:
        print(f"create ema beta:{config.ema_alpha} after_step:10 update_every: 1")
        ema = EMA(model, beta=config.ema_alpha, update_after_step=10, update_every=1)

    cal_params_flops(model, 256, logger)





    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    max_dsc = 0
    max_iou = 0





    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        if config.ema:
            ema.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
        max_dsc, max_iou = checkpoint['max_dsc'], checkpoint['max_iou']
        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)




    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            ema
        )
        if epoch % config.val_interval == 0:
            model_val = ema.ema_model if config.ema else model
            loss, dsc, IoU = val_one_epoch(
                val_loader,
                model_val,
                criterion,
                epoch,
                logger,
                config
            )

            # if loss < min_loss:
            #     torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            #     min_loss = loss
            #     min_epoch = epoch
            if dsc > max_dsc:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_model_state_dict': ema.ema_model.state_dict() if config.ema else None,
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(checkpoint_dir, 'best.pth'))
                max_dsc = dsc
                min_epoch = epoch

            torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'loss': loss,
                    'model_state_dict': model.state_dict(),
                    'ema_model_state_dict': ema.ema_model.state_dict() if config.ema else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'max_dsc': max_dsc,
                    'max_iou': max_iou,
                }, os.path.join(checkpoint_dir, 'latest.pth')) 

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        best_epoch = checkpoint['epoch']
        print(f'#----------Testing epoch:{best_epoch}----------#')
        if config.ema:
            model_val = ema.ema_model
            ema.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        else:
            model_val = model
            model.load_state_dict(checkpoint['model_state_dict'])
        # best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        # model.load_state_dict(best_weight)
        loss = test_one_epoch(
                test_loader,
                model_val,
                criterion,
                logger,
                config,
            )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            # os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-dsc{max_dsc:.4f}.pth')
        )      

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Coronary dataset to mmsegmentation format')
    parser.add_argument('-c', '--config', help='config', default='configs/cagsegmamba_vepe_b2.py')
    parser.add_argument('-gpu', '--gpu_id', help='gpu id', default='0')
    parser.add_argument('-phase', '--phase', help='phase', default='train')
    # parser.add_argument('-ckpt_dir', '--ckpt_dir', help='checkpoint work directory')
    # parser.add_argument('-closs', '--conn_loss', help='conn_loss_rate', action='store_true')
    # parser.add_argument('-closs_rate', '--conn_loss_rate', help='conn_loss_rate', default=1.0)
    # parser.add_argument('-vepe', '--vepe', help='vepe version', default='vepe_b3')  # off, b3, b2
    args = parser.parse_args()
    return args

def update_config_with_args(config, args):
    config.gpu_id = args.gpu_id
    config.phase = args.phase
    if hasattr(config, "ckpt_dir"):
        config.work_dir = config.ckpt_dir
    else:
        config.work_dir = 'results/' + config.network + '_' + config.phase + '_' + config.datasets + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '/'
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


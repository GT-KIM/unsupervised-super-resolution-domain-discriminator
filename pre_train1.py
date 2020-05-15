from datetime import datetime
import gc
import logging
import os
import argparse
import tensorflow as tf
import numpy as np
from glob import glob

from pretrain_generator import pretrain_generator1 , test_pretrain_generator
from train_module import Network, Loss, Optimizer
from utils import create_dirs, log, normalize_images, save_image, load_npz_data, load_and_save_data, generate_batch, generate_testset
from utils import build_filter, apply_bicubic_downsample

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description='')
# ----- vanilla ------
# About Data
parser.add_argument('--data_dir', dest='data_dir', default='/sdc1/NTIRE2020/task1/', help='path of the dataset')
parser.add_argument('--crop', dest='crop', default=True, help='patch width')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=64, help='patch height')
parser.add_argument('--stride', dest='stride', type=int, default=64, help='patch stride')

# About Network
parser.add_argument('--scale_SR', dest='scale_SR', default=4, help='the scale of super-resolution')
parser.add_argument('--num_repeat_RRDB', dest='num_repeat_RRDB', type=int, default=15, help='the number of RRDB blocks')
parser.add_argument('--residual_scaling', dest='residual_scaling', type=float, default=0.2, help='residual scaling parameter')
parser.add_argument('--initialization_random_seed', dest='initialization_random_seed', default=111, help='random_seed')
parser.add_argument('--perceptual_loss', dest='perceptual_loss', default='VGG19', help='the part of loss function. "VGG19" or "pixel-wise"')
parser.add_argument('--gan_loss_type', dest='gan_loss_type', default='MaGAN', help='the type of GAN loss functions. "RaGAN" or "GAN"')

# About training
parser.add_argument('--num_iter', dest='num_iter', type=int, default=50000, help='The number of iterations')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Mini-batch size')
parser.add_argument('--channel', dest='channel', type=int, default=3, help='Number of input/output image channel')
parser.add_argument('--pretrain_generator', dest='pretrain_generator', type=bool, default=True, help='Whether to pretrain generator')
parser.add_argument('--pretrain_learning_rate', dest='pretrain_learning_rate', type=float, default=2e-4, help='learning rate for pretrain')
parser.add_argument('--pretrain_lr_decay_step', dest='pretrain_lr_decay_step', type=float, default=20000, help='decay by every n iteration')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_initialize_scale', dest='weight_initialize_scale', type=float, default=0.1, help='scale to multiply after MSRA initialization')
parser.add_argument('--HR_image_size', dest='HR_image_size', type=int, default=128, help='Image width and height of LR image. This is should be 1/4 of HR_image_size exactly')
parser.add_argument('--LR_image_size', dest='LR_image_size', type=int, default=32, help='Image width and height of LR image.')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=1e-12, help='used in loss function')
parser.add_argument('--gan_loss_coeff', dest='gan_loss_coeff', type=float, default=0.1, help='used in perceptual loss')
parser.add_argument('--content_loss_coeff', dest='content_loss_coeff', type=float, default=0.01, help='used in content loss')

# About log
parser.add_argument('--logging', dest='logging', type=bool, default=True, help='whether to record training log')
parser.add_argument('--train_sample_save_freq', dest='train_sample_save_freq', type=int, default=500, help='save samples during training every n iteration')
parser.add_argument('--train_ckpt_save_freq', dest='train_ckpt_save_freq', type=int, default=500, help='save checkpoint during training every n iteration')
parser.add_argument('--train_summary_save_freq', dest='train_summary_save_freq', type=int, default=200, help='save summary during training every n iteration')

# GPU setting
parser.add_argument('--gpu_dev_num', dest='gpu_dev_num', type=str, default='0,1', help='Which GPU to use for multi-GPUs')

# ----- my args -----
parser.add_argument('--image_batch_size', dest='image_batch_size', type=int, default=64, help='Mini-batch size')
parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='Total Epochs')

parser.add_argument('--logdir', dest='logdir', type=str, default='./log', help='log directory')
parser.add_argument('--pre_train_checkpoint_dir', dest='pre_train_checkpoint_dir', type=str, default='./pre_train_checkpoint', help='pre-train checkpoint directory')
parser.add_argument('--pre_valid_result_dir', dest='pre_valid_result_dir', default='/sdc1/NTIRE2020/pre_result1/valid_result', help='output directory during training')
parser.add_argument('--pre_valid_LR_result_dir', dest='pre_valid_LR_result_dir', default='/sdc1/NTIRE2020/pre_result1/valid_LR_result', help='output directory during training')
parser.add_argument('--pre_train_result_dir', dest='pre_train_result_dir', default='./sdc1/NTIRE2020/pre_result1/pre_train_result', help='output directory during training')
parser.add_argument('--pre_raw_result_dir', dest='pre_raw_result_dir', default='./sdc1/NTIRE2020/pre_result1/pre_raw_result', help='output directory during training')

args = parser.parse_args()

def set_logger(args):
    """set logger for training recordinbg"""
    if args.logging:
        logfile = '{0}/training_logfile_{1}.log'.format(args.logdir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        formatter = '%(levelname)s:%(asctime)s:%(message)s'
        logging.basicConfig(level=logging.INFO, filename=logfile, format=formatter, datefmt='%Y-%m-%d %I:%M:%S')
        return True
    else:
        print('No logging is set')
        return False


def main():
    # make dirs
    target_dirs = [args.pre_train_checkpoint_dir,args.logdir,
                   args.pre_train_result_dir, args.pre_raw_result_dir, args.pre_valid_result_dir,
                   args.pre_valid_LR_result_dir]
    create_dirs(target_dirs)

    # set logger
    logflag = set_logger(args)
    log(logflag, 'Training script start', 'info')

    # set logger
    logflag = set_logger(args)
    log(logflag, 'Training script start', 'info')

    # pre-train generator with pixel-wise loss and save the trained model
    if args.pretrain_generator:
        pretrain_generator1(args, logflag)
        #test_pretrain_generator(args, logflag)
        tf.reset_default_graph()
        gc.collect()
    else:
        log(logflag, 'Pre-train : Pre-train skips and an existing trained model will be used', 'info')

if __name__ == '__main__':
    main()
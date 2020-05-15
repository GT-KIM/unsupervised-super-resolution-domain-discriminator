from datetime import datetime
import gc
import logging
import math
import os
import argparse
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
from glob import glob

from train_module import Network, Loss, Optimizer
from utils import create_dirs, log, normalize_images, save_image, load_npz_data, load_and_save_data, generate_batch, generate_testset
from utils import build_filter, apply_bicubic_downsample
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='')
# ----- vanilla ------
parser.add_argument('--data_dir', dest='data_dir', default='./data/track1/', help='path of the dataset')
parser.add_argument('--test_result_dir', dest='test_result_dir', default='./test_result', help='output directory during training')
#parser.add_argument('--test_LR_result_dir', dest='test_LR_result_dir', default='./test_LR_result', help='output directory during training')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='./checkpoint_track1', help='checkpoint directory')

# About Data
parser.add_argument('--crop', dest='crop', default=True, help='patch width')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=64, help='patch height')
parser.add_argument('--stride', dest='stride', type=int, default=64, help='patch stride')
parser.add_argument('--data_augmentation', dest='data_augmentation',  default=True, help='')

# About Network
parser.add_argument('--scale_SR', dest='scale_SR', default=4, help='the scale of super-resolution')
parser.add_argument('--num_repeat_RRDB', dest='num_repeat_RRDB', type=int, default=15, help='the number of RRDB blocks')
parser.add_argument('--residual_scaling', dest='residual_scaling', type=float, default=0.2, help='residual scaling parameter')
parser.add_argument('--initialization_random_seed', dest='initialization_random_seed', default=111, help='random_seed')
parser.add_argument('--perceptual_loss', dest='perceptual_loss', default='VGG19', help='the part of loss function. "VGG19" or "pixel-wise"')
parser.add_argument('--gan_loss_type', dest='gan_loss_type', default='MaGAN', help='the type of GAN loss functions. "RaGAN" or "GAN"')

# About training
parser.add_argument('--num_iter', dest='num_iter', type=int, default=50000, help='The number of iterations')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='Mini-batch size')
parser.add_argument('--channel', dest='channel', type=int, default=3, help='Number of input/output image channel')
parser.add_argument('--pretrain_generator', dest='pretrain_generator', type=bool, default=False, help='Whether to pretrain generator')
parser.add_argument('--pretrain_learning_rate', dest='pretrain_learning_rate', type=float, default=2e-4, help='learning rate for pretrain')
parser.add_argument('--pretrain_lr_decay_step', dest='pretrain_lr_decay_step', type=float, default=20000, help='decay by every n iteration')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_initialize_scale', dest='weight_initialize_scale', type=float, default=0.1, help='scale to multiply after MSRA initialization')
parser.add_argument('--HR_image_size', dest='HR_image_size', type=int, default=128, help='Image width and height of LR image. This is should be 1/4 of HR_image_size exactly')
parser.add_argument('--LR_image_size', dest='LR_image_size', type=int, default=32, help='Image width and height of LR image.')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=1e-12, help='used in loss function')
parser.add_argument('--gan_loss_coeff', dest='gan_loss_coeff', type=float, default=1.0, help='used in perceptual loss')
parser.add_argument('--content_loss_coeff', dest='content_loss_coeff', type=float, default=0.01, help='used in content loss')

# About log
parser.add_argument('--logging', dest='logging', type=bool, default=True, help='whether to record training log')
parser.add_argument('--train_sample_save_freq', dest='train_sample_save_freq', type=int, default=100, help='save samples during training every n iteration')
parser.add_argument('--train_ckpt_save_freq', dest='train_ckpt_save_freq', type=int, default=100, help='save checkpoint during training every n iteration')
parser.add_argument('--train_summary_save_freq', dest='train_summary_save_freq', type=int, default=200, help='save summary during training every n iteration')
parser.add_argument('--pre_train_checkpoint_dir', dest='pre_train_checkpoint_dir', type=str, default='./pre_train_checkpoint', help='pre-train checkpoint directory')
parser.add_argument('--logdir', dest='logdir', type=str, default='./log_test', help='log directory')

# GPU setting
parser.add_argument('--gpu_dev_num', dest='gpu_dev_num', type=str, default='0', help='Which GPU to use for multi-GPUs')

# ----- my args -----
parser.add_argument('--image_batch_size', dest='image_batch_size', type=int, default=64, help='Mini-batch size')
parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='Total Epochs')

args = parser.parse_args()

def set_logger(args):
    """set logger for training recording"""
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
    target_dirs = [args.logdir,args.test_result_dir]#., args.test_LR_result_dir]
    create_dirs(target_dirs)

    # set logger
    logflag = set_logger(args)
    log(logflag, 'Test script start', 'info')

    NLR_data = tf.placeholder(tf.float32, shape=[None, None, None, args.channel],
                              name='NLR_input')
    CLR_data = tf.placeholder(tf.float32, shape=[None, None, None, args.channel],
                              name='CLR_input')
    NHR_data = tf.placeholder(tf.float32, shape=[None, None, None, args.channel],
                              name='NHR_input')
    CHR_data = tf.placeholder(tf.float32, shape=[None, None, None, args.channel],
                              name='CHR_input')

    # build Generator and Discriminator
    network = Network(args, NLR_data=NLR_data, CLR_data=CLR_data, NHR_data=NHR_data, CHR_data=CHR_data, is_test =True)
    CLR_C1, NLR_C1, CLR_C2, CHR_C3, NLR_C3, CHR_C4, CLR_I1, CHR_I1, CHR_I2 = network.train_generator()

    # define optimizers
    global_iter = tf.Variable(0, trainable=False)

    gc.collect()

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            visible_device_list=args.gpu_dev_num
        )
    )

    # Start Session
    with tf.Session(config=config) as sess:
        log(logflag, 'Start Session', 'info')

        sess.run(tf.global_variables_initializer())
        sess.run(global_iter.initializer)

        saver = tf.train.Saver(max_to_keep=10)
        saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))


        validpathLR = np.sort(
            np.asarray(glob(os.path.join(args.data_dir, '*.png'))))
        validpathHR = np.sort(
            np.asarray(glob(os.path.join(args.data_dir, '*.png'))))
        import time

        avgtime = 0
        for valid_i in range(100) :
            validLR, validHR = generate_testset(validpathLR[valid_i],
                                                validpathHR[valid_i],
                                                args)
            name = validpathLR[valid_i].split('/')[-1]
            validLR = np.transpose(validLR[:, :, :, np.newaxis], (3, 0, 1, 2))
            validHR = np.transpose(validHR[:, :, :, np.newaxis], (3, 0, 1, 2))
            starttime = time.time()
            valid_out, valid_out_LR = sess.run([CHR_C3, CLR_C1],
                                               feed_dict={NLR_data: validLR,
                                                          NHR_data: validHR,
                                                          CLR_data: validLR,
                                                          CHR_data: validHR})

            validLR = np.rot90(validLR, 1, axes=(1, 2))
            validHR = np.rot90(validHR, 1, axes=(1, 2))
            valid_out90, valid_out_LR90 = sess.run([CHR_C3, CLR_C1],
                                                   feed_dict={NLR_data: validLR,
                                                              NHR_data: validHR,
                                                              CLR_data: validLR,
                                                              CHR_data: validHR})
            valid_out += np.rot90(valid_out90, 3, axes=(1, 2))
            valid_out_LR += np.rot90(valid_out_LR90, 3, axes=(1, 2))

            validLR = np.rot90(validLR, 1, axes=(1, 2))
            validHR = np.rot90(validHR, 1, axes=(1, 2))
            valid_out90, valid_out_LR90 = sess.run([CHR_C3, CLR_C1],
                                                   feed_dict={NLR_data: validLR,
                                                              NHR_data: validHR,
                                                              CLR_data: validLR,
                                                              CHR_data: validHR})
            valid_out += np.rot90(valid_out90, 2, axes=(1, 2))
            valid_out_LR += np.rot90(valid_out_LR90, 2, axes=(1, 2))

            validLR = np.rot90(validLR, 1, axes=(1, 2))
            validHR = np.rot90(validHR, 1, axes=(1, 2))
            valid_out90, valid_out_LR90 = sess.run([CHR_C3, CLR_C1],
                                                   feed_dict={NLR_data: validLR,
                                                              NHR_data: validHR,
                                                              CLR_data: validLR,
                                                              CHR_data: validHR})
            valid_out += np.rot90(valid_out90, 1, axes=(1, 2))
            valid_out_LR += np.rot90(valid_out_LR90, 1, axes=(1, 2))

            validLR = np.rot90(validLR, 1, axes=(1, 2))
            validHR = np.rot90(validHR, 1, axes=(1, 2))

            validLR = validLR[:,::-1, :, :]
            validHR = validHR[:,::-1, :, :]

            valid_out90, valid_out_LR90 = sess.run([CHR_C3, CLR_C1],
                                                   feed_dict={NLR_data: validLR,
                                                              NHR_data: validHR,
                                                              CLR_data: validLR,
                                                              CHR_data: validHR})
            valid_out += valid_out90[:,::-1,:,:]
            valid_out_LR += valid_out_LR90[:,::-1,:,:]

            validLR = np.rot90(validLR, 1, axes=(1, 2))
            validHR = np.rot90(validHR, 1, axes=(1, 2))
            valid_out90, valid_out_LR90 = sess.run([CHR_C3, CLR_C1],
                                                   feed_dict={NLR_data: validLR,
                                                              NHR_data: validHR,
                                                              CLR_data: validLR,
                                                              CHR_data: validHR})
            valid_out90 = np.rot90(valid_out90, 3, axes=(1, 2))
            valid_out_LR90 = np.rot90(valid_out_LR90, 3, axes=(1, 2))

            valid_out += valid_out90[:,::-1,:,:]
            valid_out_LR += valid_out_LR90[:,::-1,:,:]

            validLR = np.rot90(validLR, 1, axes=(1, 2))
            validHR = np.rot90(validHR, 1, axes=(1, 2))
            valid_out90, valid_out_LR90 = sess.run([CHR_C3, CLR_C1],
                                                   feed_dict={NLR_data: validLR,
                                                              NHR_data: validHR,
                                                              CLR_data: validLR,
                                                              CHR_data: validHR})
            valid_out90 = np.rot90(valid_out90, 2, axes=(1, 2))
            valid_out_LR90 = np.rot90(valid_out_LR90, 2, axes=(1, 2))

            valid_out += valid_out90[:,::-1,:,:]
            valid_out_LR += valid_out_LR90[:,::-1,:,:]

            validLR = np.rot90(validLR, 1, axes=(1, 2))
            validHR = np.rot90(validHR, 1, axes=(1, 2))
            valid_out90, valid_out_LR90 = sess.run([CHR_C3, CLR_C1],
                                                   feed_dict={NLR_data: validLR,
                                                              NHR_data: validHR,
                                                              CLR_data: validLR,
                                                              CHR_data: validHR})
            valid_out90 = np.rot90(valid_out90, 1, axes=(1, 2))
            valid_out_LR90 = np.rot90(valid_out_LR90, 1, axes=(1, 2))

            valid_out += valid_out90[:,::-1,:,:]
            valid_out_LR += valid_out_LR90[:,::-1,:,:]

            valid_out /= 8.
            valid_out_LR /= 8.
            currtime = time.time() - starttime
            print("time : %fs"%(currtime))
            avgtime += currtime / 100
            save_image(args, valid_out, 'test', name, save_max_num=1)
            #save_image(args, valid_out_LR, 'test_LR', valid_i, save_max_num=5)
        print("avg. time : %f"%avgtime)

if __name__ == '__main__':
    main()
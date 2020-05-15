from datetime import datetime
import gc
import logging
import math
import os
import argparse
import tensorflow as tf
import numpy as np
from glob import glob
from train_module import Network, Loss, Optimizer
from utils import create_dirs, log, save_image, generate_batch, generate_testset
from ops import load_vgg19_weight
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

parser = argparse.ArgumentParser(description='')

# directory
parser.add_argument('--data_dir', dest='data_dir', default='/sdc1/NTIRE2020/task1/', help='path of the dataset')
parser.add_argument('--train_result_dir', dest='train_result_dir', default='/sdc1/NTIRE2020/result/train_result', help='output directory during training')
parser.add_argument('--train_LR_result_dir', dest='train_LR_result_dir', default='/sdc1/NTIRE2020/result/train_LR_result', help='output directory during training')
parser.add_argument('--valid_result_dir', dest='valid_result_dir', default='/sdc1/NTIRE2020/result/valid_result', help='output directory during training')
parser.add_argument('--valid_LR_result_dir', dest='valid_LR_result_dir', default='/sdc1/NTIRE2020/result/valid_LR_result', help='output directory during training')

# About Data
parser.add_argument('--crop', dest='crop', default=True, help='patch width')
parser.add_argument('--crop_size', dest='crop_size', type=int, default=56, help='patch height')
parser.add_argument('--stride', dest='stride', type=int, default=56, help='patch stride')
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
parser.add_argument('--pretrain_lr_decay_step', dest='pretrain_lr_decay_step', type=float, default=100000, help='decay by every n iteration')
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
parser.add_argument('--pre_train_checkpoint_dir1', dest='pre_train_checkpoint_dir1', type=str, default='./pre_train_checkpoint', help='pre-train checkpoint directory')
parser.add_argument('--pre_train_checkpoint_dir2', dest='pre_train_checkpoint_dir2', type=str, default='./pre_train_checkpoint2', help='pre-train checkpoint directory')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='./checkpoint', help='checkpoint directory')
parser.add_argument('--logdir', dest='logdir', type=str, default='./log3', help='log directory')

# GPU setting
parser.add_argument('--gpu_dev_num', dest='gpu_dev_num', type=str, default='0,1,2', help='Which GPU to use for multi-GPUs')

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
    target_dirs = [args.checkpoint_dir, args.logdir,
                   args.train_result_dir, args.train_LR_result_dir,
                   args.valid_result_dir, args.valid_LR_result_dir]
    create_dirs(target_dirs)

    # set logger
    logflag = set_logger(args)
    log(logflag, 'Training script start', 'info')

    NLR_data = tf.placeholder(tf.float32, shape=[None, None, None, args.channel],
                             name='NLR_input')
    CLR_data = tf.placeholder(tf.float32, shape=[None, None, None, args.channel],
                             name='CLR_input')
    NHR_data = tf.placeholder(tf.float32, shape=[None, None, None, args.channel],
                             name='NHR_input')
    CHR_data = tf.placeholder(tf.float32, shape=[None, None, None, args.channel],
                             name='CHR_input')

    # build Generator and Discriminator
    network = Network(args, NLR_data=NLR_data,CLR_data=CLR_data, NHR_data=NHR_data,CHR_data=CHR_data)
    CLR_C1, NLR_C1, CLR_C2, CHR_C3, NLR_C3, CHR_C4, CLR_I1, CHR_I1, CHR_I2 = network.train_generator()
    D_out, Y_out, C_out = network.train_discriminator(CLR_C1, CHR_C3)

    # build loss function
    loss = Loss()
    gen_loss, g_gen_loss, dis_loss, Y_dis_loss, C_dis_loss = loss.gan_loss(args, NLR_data, CLR_data, NHR_data, CHR_data,
                 CLR_C1, NLR_C1, CLR_C2, CHR_C3, NLR_C3, CHR_C4, CLR_I1, CHR_I1, CHR_I2,
                 D_out, Y_out, C_out)

    # define optimizers
    global_iter = tf.Variable(0, trainable=False)
    dis_var, dis_optimizer, gen_var, gen_optimizer, Y_dis_optimizer, C_dis_optimizer = Optimizer().gan_optimizer(
        args, global_iter, dis_loss, gen_loss, Y_dis_loss, C_dis_loss)

    # build summary writer
    tr_summary = tf.summary.merge(loss.add_summary_writer())
    fetches = {'dis_optimizer': dis_optimizer,'Y_dis_optimizer': Y_dis_optimizer,
               'C_dis_optimizer': C_dis_optimizer, 'gen_optimizer': gen_optimizer,
               'dis_loss': dis_loss,'Y_dis_loss': Y_dis_loss,'C_dis_loss': C_dis_loss,
               'gen_loss': gen_loss, 'g_gen_loss' : g_gen_loss,
               'CHR_out': CHR_C3, 'CLR_out' : CLR_C1, 'summary' : tr_summary}
    """
    fetches = {'dis_optimizer': dis_optimizer,'dis_optimizerLR': dis_optimizerLR, 'gen_optimizer': gen_optimizer,
               'dis_loss': dis_loss, 'gen_loss': gen_loss,
               'CHR_out': CHR_C3, 'CLR_out' : CLR_C1, 'summary' : tr_summary}
    """

    gc.collect()

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            visible_device_list=args.gpu_dev_num
        )
    )

    # Start Session
    with tf.Session(config=config) as sess:
        log(logflag, 'Training ESRGAN starts', 'info')

        sess.run(tf.global_variables_initializer())
        sess.run(global_iter.initializer)

        writer = tf.summary.FileWriter(args.logdir, graph=sess.graph)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator/Generator1')\
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator/Generator2')\
                 + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator')

        pre_saver = tf.train.Saver(var_list=var_list)
        pre_saver.restore(sess, tf.train.latest_checkpoint(args.pre_train_checkpoint_dir1))

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator/SR')

        pre_saver = tf.train.Saver(var_list=var_list)
        pre_saver.restore(sess, tf.train.latest_checkpoint(args.pre_train_checkpoint_dir2))

        if args.perceptual_loss == 'VGG19':
            sess.run(load_vgg19_weight(args))

        saver = tf.train.Saver(max_to_keep=10)
        saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint_dir))

        _datapathNLR = np.sort(np.asarray(glob(os.path.join(args.data_dir + '/source/train_HR_aug/x4/', '*.png'))))
        _datapathNHR = np.sort(np.asarray(glob(os.path.join(args.data_dir + '/source/train_HR_aug/x4/', '*.png'))))
        _datapathCLR = np.sort(np.asarray(glob(os.path.join(args.data_dir + '/target/train_LR_aug/x4/', '*.png'))))
        _datapathCHR = np.sort(np.asarray(glob(os.path.join(args.data_dir + '/target/train_HR_aug/x4/', '*.png'))))
        idxLR = np.random.permutation(len(_datapathNLR))
        datapathNLR = _datapathNLR[idxLR]
        datapathNHR = _datapathNHR[idxLR]
        idxHR = np.random.permutation(len(_datapathCHR))
        datapathCLR = _datapathCLR[idxHR]
        datapathCHR = _datapathCHR[idxHR]

        epoch = 0
        counterNLR = 0
        counterCLR = 0

        g_loss = 0.0
        d_loss = 0.0
        steps = 0
        psnr_max = 20
        while True:
            if counterNLR >= len(_datapathNLR):
                log(logflag, 'Train Epoch: {0} g.loss : {1} d.loss : {2}'.format(
                    epoch, g_loss / steps, d_loss / steps), 'info')
                idx = np.random.permutation(len(_datapathNLR))
                datapathNLR = _datapathNLR[idx]
                datapathNHR = _datapathNHR[idx]
                counterNLR = 0
                g_loss = 0.0
                d_loss = 0.0
                steps = 0
                epoch += 1
                if epoch == 200:
                    break
            if counterCLR >= len(_datapathCHR):
                idx = np.random.permutation(len(_datapathCHR))
                datapathCHR = _datapathCHR[idx]
                datapathCLR = _datapathCLR[idx]
                counterCLR = 0

            dataNLR, dataNHR, dataCLR, dataCHR = generate_batch(datapathNLR[counterNLR:counterNLR + args.image_batch_size],
                                            datapathNHR[counterNLR:counterNLR + args.image_batch_size],
                                            datapathCLR[counterCLR:counterCLR + args.image_batch_size],
                                            datapathCHR[counterCLR:counterCLR + args.image_batch_size],
                                            args)

            counterNLR += args.image_batch_size
            counterCLR += args.image_batch_size

            for iteration in range(0, dataCLR.shape[0], args.batch_size):

                _CHR_data = dataCHR[iteration:iteration + args.batch_size]
                _CLR_data = dataCLR[iteration:iteration + args.batch_size]
                _NLR_data = dataNLR[iteration:iteration + args.batch_size]
                _NHR_data = dataNHR[iteration:iteration + args.batch_size]

                feed_dict = {
                    CHR_data: _CHR_data,
                    CLR_data : _CLR_data,
                    NLR_data: _NLR_data,
                    NHR_data: _NHR_data,
                }
                # update weights
                result = sess.run(fetches=fetches, feed_dict=feed_dict)
                current_iter = tf.train.global_step(sess, global_iter)

                g_loss += result['gen_loss']
                d_loss += result['dis_loss']
                steps += 1

                # save summary every n iter
                if current_iter % args.train_summary_save_freq == 0:
                    writer.add_summary(result['summary'], global_step=current_iter)

                # save samples every n iter
                if current_iter % 100 == 0 :
                    log(logflag,
                        'Mymodel iteration : {0}, gen_loss : {1}, dis_loss : {2},'
                        ' Y_dis_loss {3} C_dis_loss {4} g_gen_loss {5}'.format(current_iter,
                                                                         result['gen_loss'],
                                                                         result['dis_loss'],
                                                                         result['Y_dis_loss'],
                                                                         result['C_dis_loss'],
                                                                         result['g_gen_loss']),
                        'info')
                if current_iter % args.train_sample_save_freq == 0:
                    validpathLR = np.sort(
                        np.asarray(glob(os.path.join(args.data_dir + '/validation/x/', '*.png'))))
                    validpathHR = np.sort(
                        np.asarray(glob(os.path.join(args.data_dir + '/validation/y/', '*.png'))))
                    psnr_avg = 0.0
                    for valid_ii in range(100) :
                        #valid_i = np.random.randint(100)
                        validLR, validHR = generate_testset(validpathLR[valid_ii],
                                                        validpathHR[valid_ii],
                                                        args)

                        validLR = np.transpose(validLR[:,:,:,np.newaxis],(3,0,1,2))
                        validHR = np.transpose(validHR[:,:,:,np.newaxis],(3,0,1,2))
                        valid_out, valid_out_LR = sess.run([CHR_C3, CLR_C1],
                                                           feed_dict = {NLR_data : validLR,
                                                                        NHR_data: validHR,
                                                                        CLR_data : validLR,
                                                                        CHR_data : validHR})

                        validLR = np.rot90(validLR, 1, axes=(1,2))
                        validHR = np.rot90(validHR, 1, axes=(1,2))
                        valid_out90, valid_out_LR90 = sess.run([CHR_C3, CLR_C1],
                                                           feed_dict = {NLR_data : validLR,
                                                                        NHR_data: validHR,
                                                                        CLR_data : validLR,
                                                                        CHR_data : validHR})
                        valid_out += np.rot90(valid_out90, 3, axes=(1,2))
                        valid_out_LR += np.rot90(valid_out_LR90, 3, axes=(1,2))

                        validLR = np.rot90(validLR, 1, axes=(1,2))
                        validHR = np.rot90(validHR, 1, axes=(1,2))
                        valid_out90, valid_out_LR90 = sess.run([CHR_C3, CLR_C1],
                                                           feed_dict = {NLR_data : validLR,
                                                                        NHR_data: validHR,
                                                                        CLR_data : validLR,
                                                                        CHR_data : validHR})
                        valid_out += np.rot90(valid_out90, 2, axes=(1,2))
                        valid_out_LR += np.rot90(valid_out_LR90, 2, axes=(1,2))

                        validLR = np.rot90(validLR, 1, axes=(1,2))
                        validHR = np.rot90(validHR, 1, axes=(1,2))
                        valid_out90, valid_out_LR90 = sess.run([CHR_C3, CLR_C1],
                                                           feed_dict = {NLR_data : validLR,
                                                                        NHR_data: validHR,
                                                                        CLR_data : validLR,
                                                                        CHR_data : validHR})
                        valid_out += np.rot90(valid_out90, 1, axes=(1,2))
                        valid_out_LR += np.rot90(valid_out_LR90, 1, axes=(1,2))

                        valid_out /=4.
                        valid_out_LR /=4.

                        from utils import de_normalize_image
                        validHR = np.rot90(validHR, 1, axes=(1, 2))
                        F = de_normalize_image(validHR) / 255.
                        G = de_normalize_image(valid_out) / 255.

                        E = F-G
                        N = np.size(E)
                        PSNR = 10*np.log10( N / np.sum(E ** 2))
                        print(PSNR)
                        psnr_avg += PSNR / 100
                        if valid_ii < 5 :
                            save_image(args, valid_out, 'valid', current_iter+valid_ii, save_max_num=5)
                            save_image(args, valid_out_LR, 'valid_LR', current_iter + valid_ii, save_max_num=5)
                    if psnr_avg > psnr_max :
                        print("max psnr : %f" %psnr_avg)
                        psnr_max = psnr_avg
                        saver.save(sess, os.path.join('./best_checkpoint', 'gen'), global_step=current_iter)
                    #save_image(args, result['gen_HR'], 'My_train', current_iter, save_max_num=5)
                    #save_image(args, result['gen_out_LR'], 'My_train_LR', current_iter, save_max_num=5)
                # save checkpoints
                if current_iter % args.train_ckpt_save_freq == 0:
                    saver.save(sess, os.path.join(args.checkpoint_dir, 'gen'), global_step=current_iter)

        writer.close()
        log(logflag, 'Training ESRGAN end', 'info')
        log(logflag, 'Training script end', 'info')


if __name__ == '__main__':
    main()
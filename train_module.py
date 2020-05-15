from collections import OrderedDict

import tensorflow as tf
from network import Generator, Discriminator, Perceptual_VGG19
import numpy as np
from scipy import signal
def rgb2gray(rgb):
    _rgb = (rgb + 1) / 2
    gray = tf.image.rgb_to_grayscale(rgb)
    gray = gray * 2 - 1
    return gray

def gaussian_kernel(size, mean, std) :
    k = signal.gaussian(size, std=std).reshape(size,1)
    k = np.outer(k, k)[...,np.newaxis]
    k = k / np.sum(k)
    k = np.concatenate((k,k,k), axis=2)
    return tf.constant(k, dtype=tf.float32, shape=(size,size,3,1))

    #d = tf.distributions.Normal(mean, std)
    #vals = d.prob(tf.range(start=-size, limit=size+1, dtype=tf.float32))
    #gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    #return tf.constant(gauss_kernel / tf.reduce_sum(gauss_kernel))

def gauss_blur(image) :
    image = (image + 1) / 2
    gauss_kernel = gaussian_kernel(7, 0., 3.)
    blur = tf.nn.conv2d(image, gauss_kernel, strides=[1,1,1,1], padding='SAME')
    blur = blur * 2 - 1
    return blur

class Network(object):
    """class to build networks"""
    def __init__(self, args, NLR_data=None, NHR_data=None, CLR_data = None, CHR_data=None, mask_data = None, is_test = False):
        self.args = args
        self.NLR_data = NLR_data
        self.NHR_data = NHR_data
        self.CLR_data = CLR_data
        self.CHR_data = CHR_data
        self.mask_data = mask_data
        self.is_test = is_test

    def pretrain_generator_LR(self):
        with tf.device("/gpu:1"):
            with tf.name_scope('generator'):
                with tf.variable_scope('generator', reuse=False):
                    CLR_N = Generator(self.args).build_G1(self.NLR_data)
                    NLR_NC = Generator(self.args).build_G2(CLR_N)
                with tf.variable_scope('generator', reuse=True):
                    NLR_C = Generator(self.args).build_G2(self.CLR_data)
                    CLR_CN = Generator(self.args).build_G1(NLR_C)
                with tf.variable_scope('generator', reuse=True):
                    CLR_C = Generator(self.args).build_G1(self.CLR_data)

        return CLR_N, NLR_NC, NLR_C, CLR_CN, CLR_C

    def pretrain_discriminator_LR(self, CLR_out):
        with tf.device("/gpu:1"):
            discriminatorLR = Discriminator(self.args)
            with tf.name_scope('real_discriminator'):
                with tf.variable_scope('discriminator', reuse=False):
                    dis_out_realLR = discriminatorLR.build(self.CLR_data)
            with tf.name_scope('fake_discriminator'):
                with tf.variable_scope('discriminator', reuse=True):
                    dis_out_fakeLR = discriminatorLR.build(CLR_out)
            with tf.name_scope('noisy_discriminator'):
                with tf.variable_scope('discriminator', reuse=True):
                    dis_out_noisyLR = discriminatorLR.build(self.NLR_data)

        Y_CLR_out = rgb2gray(CLR_out)
        Y_CLR_data = rgb2gray(self.CLR_data)
        Y_NLR_data = rgb2gray(self.NLR_data)
        Y_out = list()
        with tf.device("/gpu:1"):
            Y_discriminator = Discriminator(self.args)
            with tf.name_scope('Y_real_discriminator'):
                with tf.variable_scope('Y_discriminator', reuse=False):
                    Y_dis_out_realLR = Y_discriminator.build(Y_CLR_data)
                    Y_out.append(Y_dis_out_realLR)
            with tf.name_scope('Y_fake_discriminator'):
                with tf.variable_scope('Y_discriminator', reuse=True):
                    Y_dis_out_fakeLR = Y_discriminator.build(Y_CLR_out)
                    Y_out.append(Y_dis_out_fakeLR)
            with tf.name_scope('Y_noisy_discriminator'):
                with tf.variable_scope('Y_discriminator', reuse=True):
                    Y_dis_out_noisyLR = Y_discriminator.build(Y_NLR_data)
                    Y_out.append(Y_dis_out_noisyLR)

        C_CLR_out = gauss_blur(CLR_out)
        C_CLR_data = gauss_blur(self.CLR_data)
        C_NLR_data = gauss_blur(self.NLR_data)
        C_out = list()
        with tf.device("/gpu:1"):
            C_discriminator = Discriminator(self.args)
            with tf.name_scope('C_real_discriminator'):
                with tf.variable_scope('C_discriminator', reuse=False):
                    C_dis_out_realLR = C_discriminator.build(C_CLR_data)
                    C_out.append(C_dis_out_realLR)
            with tf.name_scope('C_fake_discriminator'):
                with tf.variable_scope('C_discriminator', reuse=True):
                    C_dis_out_fakeLR = C_discriminator.build(C_CLR_out)
                    C_out.append(C_dis_out_fakeLR)
            with tf.name_scope('C_noisy_discriminator'):
                with tf.variable_scope('C_discriminator', reuse=True):
                    C_dis_out_noisyLR = C_discriminator.build(C_NLR_data)
                    C_out.append(C_dis_out_noisyLR)

        return dis_out_realLR, dis_out_fakeLR, dis_out_noisyLR, Y_out, C_out

    def pretrain_generator_SR(self):
        with tf.device("/gpu:0"):
            with tf.name_scope('generator'):
                with tf.variable_scope('generator', reuse=False):
                    CHR = Generator(self.args).build_SR(self.CLR_data)

        return CHR

    def train_generator(self):
        if self.is_test:
            dev = "0"
        else:
            dev = "1"
        generator = Generator(self.args)
        with tf.name_scope('generator'):
            # cycle Nl -> Cl -> Nl
            with tf.variable_scope('generator', reuse=False):
                with tf.device("/gpu:0"):
                    CLR_C1 = generator.build_G1(self.NLR_data)
                    NLR_C1 = generator.build_G2(CLR_C1)

            # cycle Cl -> Nl -> Cl
            with tf.variable_scope('generator', reuse=True):
                with tf.device("/gpu:0"):
                    NLR_C2 = generator.build_G2(self.CLR_data)
                    CLR_C2 = generator.build_G1(NLR_C2)

            # cycle Nl -> Ch -> Nl
            with tf.variable_scope('generator', reuse=False):
                with tf.device("/gpu:"+dev):
                    CHR_C3 = generator.build_SR(CLR_C1)
                with tf.device("/gpu:"+dev):
                    NLR_C3 = generator.build_G3(CHR_C3)

            # cycle Ch -> Nl -> Ch
            with tf.variable_scope('generator', reuse=True):
                with tf.device("/gpu:"+dev):
                    NLR_C4 = generator.build_G3(self.CHR_data)
                with tf.device("/gpu:0"):
                    CHR_C4 = generator.build_G1(NLR_C4)
                with tf.device("/gpu:"+dev):
                    CHR_C4 = generator.build_SR(CHR_C4)

            # Identity
            with tf.variable_scope('generator', reuse=True):
                with tf.device("/gpu:0"):
                    CLR_I1 = generator.build_G1(self.CLR_data)
                with tf.device("/gpu:"+dev):
                    CHR_I1 = generator.build_SR(CLR_I1)
                    CHR_I2 = generator.build_SR(self.CLR_data)

        return CLR_C1, NLR_C1, CLR_C2, CHR_C3, NLR_C3, CHR_C4, CLR_I1, CHR_I1, CHR_I2

    def train_discriminator(self, CLR_out, CHR_out):
        D_out = list()
        with tf.device("/gpu:2"):
            discriminator = Discriminator(self.args)
            with tf.variable_scope('discriminator', reuse=False):
                dis_out_real = discriminator.build(self.CHR_data)
                D_out.append(dis_out_real)
            with tf.variable_scope('discriminator', reuse=True):
                dis_out_fake = discriminator.build(CHR_out)
                D_out.append(dis_out_fake)
            with tf.variable_scope('discriminator', reuse=True):
                dis_out_noisy = discriminator.build(self.NHR_data)
                D_out.append(dis_out_noisy)
            with tf.variable_scope('discriminator', reuse=True):
                dis_out_fakeLR = discriminator.build(CLR_out)
                D_out.append(dis_out_fakeLR)
            with tf.variable_scope('discriminator', reuse=True):
                dis_out_noisyLR = discriminator.build(self.NLR_data)
                D_out.append(dis_out_noisyLR)
        Y_CHR_out = rgb2gray(CHR_out)
        Y_CHR_data = rgb2gray(self.CHR_data)
        Y_NHR_data = rgb2gray(self.NHR_data)
        Y_CLR_out = rgb2gray(CLR_out)
        Y_NLR_data = rgb2gray(self.NLR_data)
        Y_out = list()

        with tf.device("/gpu:2"):
            Y_discriminator = Discriminator(self.args)
            with tf.variable_scope('Y_discriminator', reuse=False):
                Y_dis_out_real = Y_discriminator.build(Y_CHR_data)
                Y_out.append(Y_dis_out_real)
            with tf.variable_scope('Y_discriminator', reuse=True):
                Y_dis_out_fake = Y_discriminator.build(Y_CHR_out)
                Y_out.append(Y_dis_out_fake)
            with tf.variable_scope('Y_discriminator', reuse=True):
                Y_dis_out_noisy = Y_discriminator.build(Y_NHR_data)
                Y_out.append(Y_dis_out_noisy)
            with tf.variable_scope('Y_discriminator', reuse=True):
                Y_dis_out_fakeLR = Y_discriminator.build(Y_CLR_out)
                Y_out.append(Y_dis_out_fakeLR)
            with tf.variable_scope('Y_discriminator', reuse=True):
                Y_dis_out_noisyLR = Y_discriminator.build(Y_NLR_data)
                Y_out.append(Y_dis_out_noisyLR)

        C_CHR_out = gauss_blur(CHR_out)
        C_CHR_data = gauss_blur(self.CHR_data)
        C_NHR_data = gauss_blur(self.NHR_data)
        C_CLR_out = gauss_blur(CLR_out)
        C_NLR_data = gauss_blur(self.NLR_data)
        C_out = list()
        with tf.device("/gpu:2"):
            C_discriminator = Discriminator(self.args)
            with tf.variable_scope('C_discriminator', reuse=False):
                C_dis_out_real = C_discriminator.build(C_CHR_data)
                C_out.append(C_dis_out_real)
            with tf.variable_scope('C_discriminator', reuse=True):
                C_dis_out_fake = C_discriminator.build(C_CHR_out)
                C_out.append(C_dis_out_fake)
            with tf.variable_scope('C_discriminator', reuse=True):
                C_dis_out_noisy = C_discriminator.build(C_NHR_data)
                C_out.append(C_dis_out_noisy)
            with tf.variable_scope('C_discriminator', reuse=True):
                C_dis_out_fakeLR = C_discriminator.build(C_CLR_out)
                C_out.append(C_dis_out_fakeLR)
            with tf.variable_scope('C_discriminator', reuse=True):
                C_dis_out_noisyLR = C_discriminator.build(C_NLR_data)
                C_out.append(C_dis_out_noisyLR)
        return D_out, Y_out, C_out

class Loss(object):
    """class to build loss functions"""
    def __init__(self):
        self.summary_target = OrderedDict()

    def pretrain_loss(self, args, NLR, CLR, NLR_NC, CLR_CN, CLR_C, dis_out_realLR, dis_out_fakeLR, dis_out_noisyLR, Y_out, C_out):
        with tf.name_scope('loss_function'):
            with tf.variable_scope('loss_generator') :
                pre_gen_N = tf.reduce_mean(tf.abs(NLR_NC - NLR))
                vgg_out_gen, vgg_out_hr = self._perceptual_vgg19_loss(NLR, NLR_NC)
                pre_gen_N += tf.reduce_mean(tf.square(vgg_out_gen - vgg_out_hr))
                pre_gen_N += 2 - tf.reduce_mean(tf.image.ssim(NLR, NLR_NC, 2.0))

                pre_gen_C = tf.reduce_mean(tf.abs(CLR_CN - CLR))
                vgg_out_gen, vgg_out_hr = self.__perceptual_vgg19_loss(CLR, CLR_CN)
                pre_gen_C += tf.reduce_mean(tf.reduce_mean(tf.square(vgg_out_gen - vgg_out_hr), axis=3))
                pre_gen_C += 2 - tf.reduce_mean(tf.image.ssim(CLR, CLR_CN, 2.0))

                pre_identity = tf.reduce_mean(tf.abs(CLR_C - CLR))
                vgg_out_gen, vgg_out_hr = self.__perceptual_vgg19_loss(CLR, CLR_C)
                pre_identity += tf.reduce_mean(tf.square(vgg_out_gen - vgg_out_hr))
                pre_identity += 2 - tf.reduce_mean(tf.image.ssim(CLR, CLR_C, 2.0))

            with tf.variable_scope('loss_generator'):
                if args.gan_loss_type == 'GAN':
                    g_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_fakeLR,
                                                                labels=tf.ones_like(dis_out_fakeLR)))
                if args.gan_loss_type == 'MaGAN':
                    g_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_fakeLR,
                                                                labels=tf.ones_like(dis_out_fakeLR)))
                    g_loss_fake += tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_out[1],
                                                                labels=tf.ones_like(Y_out[1])))
                    g_loss_fake += tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=C_out[1],
                                                                labels=tf.ones_like(C_out[1])))

            gen_loss = pre_gen_N + pre_gen_C + pre_identity + 1e-2 *g_loss_fake

            with tf.variable_scope('loss_discriminator'):
                if args.gan_loss_type == 'GAN':
                    d_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_realLR,
                                                                labels=tf.ones_like(dis_out_realLR)))
                    d_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_fakeLR,
                                                                labels=tf.zeros_like(dis_out_fakeLR)))
                    d_loss_noisy = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_noisyLR,
                                                                labels=tf.zeros_like(dis_out_noisyLR)))
                    dis_loss = d_loss_real + d_loss_fake + d_loss_noisy

                if args.gan_loss_type == 'MaGAN':
                    d_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_realLR,
                                                                labels=tf.ones_like(dis_out_realLR)))
                    d_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_fakeLR,
                                                                labels=tf.zeros_like(dis_out_fakeLR)))
                    d_loss_noisy = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_out_noisyLR,
                                                                labels=tf.zeros_like(dis_out_noisyLR)))
                    Y_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_out[0],
                                                                labels=tf.ones_like(Y_out[0])))
                    Y_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_out[1],
                                                                labels=tf.zeros_like(Y_out[1])))
                    Y_loss_noisy = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_out[2],
                                                                labels=tf.zeros_like(Y_out[2])))
                    C_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=C_out[0],
                                                                labels=tf.ones_like(C_out[0])))
                    C_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=C_out[1],
                                                                labels=tf.zeros_like(C_out[1])))
                    C_loss_noisy = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=C_out[2],
                                                                labels=tf.zeros_like(C_out[2])))

                    Y_loss = Y_loss_real + Y_loss_fake + Y_loss_noisy
                    C_loss = C_loss_real + C_loss_fake + C_loss_noisy
                    dis_loss = d_loss_real + d_loss_fake + d_loss_noisy

        return gen_loss, dis_loss, Y_loss, C_loss

    def pretrain_loss2(self, CHR_data, CHR):
        with tf.name_scope('loss_function'):
            with tf.variable_scope('pixel-wise_loss') :
                pre_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(CHR - CHR_data), axis=3))

        return pre_loss

    def _perceptual_vgg19_loss(self, HR_data, gen_out):
        with tf.device("/gpu:1"):
            with tf.name_scope('perceptual_vgg19_HR'):
                with tf.variable_scope('perceptual_vgg19', reuse=False):
                    vgg_out_hr = Perceptual_VGG19().build(HR_data)

            with tf.name_scope('perceptual_vgg19_Gen'):
                with tf.variable_scope('perceptual_vgg19', reuse=True):
                    vgg_out_gen = Perceptual_VGG19().build(gen_out)

            return vgg_out_hr, vgg_out_gen
    def __perceptual_vgg19_loss(self, HR_data, gen_out):
        with tf.device("/gpu:1"):
            with tf.name_scope('perceptual_vgg19_HR'):
                with tf.variable_scope('perceptual_vgg19', reuse=True):
                    vgg_out_hr = Perceptual_VGG19().build(HR_data)

            with tf.name_scope('perceptual_vgg19_Gen'):
                with tf.variable_scope('perceptual_vgg19', reuse=True):
                    vgg_out_gen = Perceptual_VGG19().build(gen_out)

            return vgg_out_hr, vgg_out_gen



    def gan_loss(self, FLAGS, NLR_data, CLR_data, NHR_data, CHR_data,
                 CLR_C1, NLR_C1, CLR_C2, CHR_C3, NLR_C3, CHR_C4, CLR_I1, CHR_I1, CHR_I2,
                 D_out, Y_out, C_out):
        with tf.name_scope('loss_function'):
            with tf.variable_scope('loss_generator'):
                w_gen = 1e-3
                if FLAGS.gan_loss_type == 'GAN':
                    gen_loss = 1e-2 *tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_out[3],
                                                                labels=tf.ones_like(D_out[3])))
                    gen_loss += 1e-2 *tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_out[1],
                                                                labels=tf.ones_like(D_out[1])))

                elif FLAGS.gan_loss_type == 'MaGAN':
                    gen_loss = 2*w_gen *tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_out[1],
                                                                labels=tf.ones_like(D_out[1])))
                    gen_loss += 2*w_gen * tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_out[1],
                                                                labels=tf.ones_like(Y_out[1])))
                    gen_loss += w_gen * tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=C_out[1],
                                                                labels=tf.ones_like(C_out[1])))
                    gen_loss += 2*w_gen * tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_out[3],
                                                                labels=tf.ones_like(D_out[3])))
                    gen_loss += 2*w_gen * tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_out[3],
                                                                labels=tf.ones_like(Y_out[3])))
                    gen_loss += w_gen * tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=C_out[3],
                                                                labels=tf.ones_like(C_out[3])))
                else:
                    raise ValueError('Unknown GAN loss function type')

                w_l1 = 1
                w_vgg = 1
                w_ssim = 1

                # cycle 1 loss
                g_loss_cycle = w_l1*tf.reduce_mean(tf.reduce_mean(tf.abs(NLR_C1 - NLR_data), axis=[1,2,3]))
                vgg_out_gen, vgg_out_hr = self._perceptual_vgg19_loss(NLR_data, NLR_C1)
                g_loss_cycle += w_vgg*tf.reduce_mean(tf.reduce_mean(tf.square(vgg_out_gen - vgg_out_hr), axis=3))
                g_loss_cycle += w_ssim*(2 - tf.reduce_mean(tf.image.ssim(NLR_data, NLR_C1, 2.0)))

                # cycle 2 loss
                g_loss_cycle += w_l1*tf.reduce_mean(tf.reduce_mean(tf.abs(CLR_C2 - CLR_data), axis=[1,2,3]))
                vgg_out_gen, vgg_out_hr = self.__perceptual_vgg19_loss(CLR_data, CLR_C2)
                g_loss_cycle += w_vgg*tf.reduce_mean(tf.reduce_mean(tf.square(vgg_out_gen - vgg_out_hr), axis=3))
                g_loss_cycle += w_ssim*(2 - tf.reduce_mean(tf.image.ssim(CLR_data, CLR_C2, 2.0)))

                # cycle 3 loss
                g_loss_cycle += w_l1*tf.reduce_mean(tf.reduce_mean(tf.abs(NLR_C3 - NLR_data), axis=[1,2,3]))
                vgg_out_gen, vgg_out_hr = self.__perceptual_vgg19_loss(NLR_data, NLR_C3)
                g_loss_cycle += w_vgg*tf.reduce_mean(tf.reduce_mean(tf.square(vgg_out_gen - vgg_out_hr), axis=3))
                g_loss_cycle += w_ssim*(2 - tf.reduce_mean(tf.image.ssim(NLR_data, NLR_C3, 2.0)))

                # cycle 4 loss
                g_loss_cycle += w_l1*tf.reduce_mean(tf.reduce_mean(tf.abs(CHR_C4 - CHR_data), axis=[1,2,3]))
                vgg_out_gen, vgg_out_hr = self.__perceptual_vgg19_loss(CHR_data, CHR_C4)
                g_loss_cycle += w_vgg*tf.reduce_mean(tf.reduce_mean(tf.square(vgg_out_gen - vgg_out_hr), axis=3))
                g_loss_cycle += w_ssim*(2 - tf.reduce_mean(tf.image.ssim(CHR_data, CHR_C4, 2.0)))

                # Identity 1 Loss
                g_loss_identity = w_l1*tf.reduce_mean(tf.reduce_mean(tf.abs(CLR_I1 - CLR_data), axis=[1,2,3]))
                vgg_out_gen, vgg_out_hr = self.__perceptual_vgg19_loss(CLR_data, CLR_I1)
                g_loss_identity += w_vgg*tf.reduce_mean(tf.reduce_mean(tf.square(vgg_out_gen - vgg_out_hr), axis=3))
                g_loss_identity += w_ssim*(2 - tf.reduce_mean(tf.image.ssim(CLR_data, CLR_I1, 2.0)))

                # Identity 2 Loss
                g_loss_identity += w_l1*tf.reduce_mean(tf.reduce_mean(tf.abs(CHR_I1 - CHR_data), axis=[1,2,3]))
                vgg_out_gen, vgg_out_hr = self.__perceptual_vgg19_loss(CHR_data, CHR_I1)
                g_loss_identity += w_vgg*tf.reduce_mean(tf.reduce_mean(tf.square(vgg_out_gen - vgg_out_hr), axis=3))
                g_loss_identity += w_ssim*(2 - tf.reduce_mean(tf.image.ssim(CHR_data, CHR_I1, 2.0)))

                # Identity 3 Loss
                g_loss_identity += w_l1*tf.reduce_mean(tf.reduce_mean(tf.abs(CHR_I2 - CHR_data), axis=[1,2,3]))
                vgg_out_gen, vgg_out_hr = self.__perceptual_vgg19_loss(CHR_data, CHR_I2)
                g_loss_identity += w_vgg*tf.reduce_mean(tf.reduce_mean(tf.square(vgg_out_gen - vgg_out_hr), axis=3))
                g_loss_identity += w_ssim*(2 - tf.reduce_mean(tf.image.ssim(CHR_data, CHR_I2, 2.0)))

                g_gen_loss = tf.identity(gen_loss)
                gen_loss += 2*g_loss_cycle
                gen_loss += g_loss_identity


            with tf.variable_scope('loss_discriminator'):
                if FLAGS.gan_loss_type == 'GAN':
                    d_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_out[0],
                                                                labels=tf.ones_like(D_out[0])))
                    d_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_out[1],
                                                                labels=tf.zeros_like(D_out[1])))
                    d_loss_noisy = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_out[2],
                                                                labels=tf.zeros_like(D_out[2])))
                    dis_loss = d_loss_real + d_loss_fake + d_loss_noisy
                elif FLAGS.gan_loss_type == 'MaGAN':
                    d_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_out[0],
                                                                labels=tf.ones_like(D_out[0])))
                    d_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_out[1],
                                                                labels=tf.zeros_like(D_out[1])))
                    d_loss_noisy = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_out[2],
                                                                labels=tf.zeros_like(D_out[2])))
                    d_loss_fakeLR = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_out[3],
                                                                labels=tf.zeros_like(D_out[3])))
                    d_loss_noisyLR = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_out[4],
                                                                labels=tf.zeros_like(D_out[4])))
                    dis_loss = 2 *d_loss_real + d_loss_fake + d_loss_noisy + d_loss_fakeLR + d_loss_noisyLR

                    Y_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_out[0],
                                                                labels=tf.ones_like(Y_out[0])))
                    Y_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_out[1],
                                                                labels=tf.zeros_like(Y_out[1])))
                    Y_loss_noisy = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_out[2],
                                                                labels=tf.zeros_like(Y_out[2])))
                    Y_loss_fakeLR = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_out[3],
                                                                labels=tf.zeros_like(Y_out[3])))
                    Y_loss_noisyLR = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_out[4],
                                                                labels=tf.zeros_like(Y_out[4])))
                    Y_dis_loss = 2 * Y_loss_real + Y_loss_fake + Y_loss_noisy + Y_loss_fakeLR + Y_loss_noisyLR

                    C_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=C_out[0],
                                                                labels=tf.ones_like(C_out[0])))
                    C_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=C_out[1],
                                                                labels=tf.zeros_like(C_out[1])))
                    C_loss_noisy = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=C_out[2],
                                                                labels=tf.zeros_like(C_out[2])))
                    C_loss_fakeLR = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=C_out[3],
                                                                labels=tf.zeros_like(C_out[3])))
                    C_loss_noisyLR = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=C_out[4],
                                                                labels=tf.zeros_like(C_out[4])))

                    C_dis_loss = 2 * C_loss_real + C_loss_fake + C_loss_noisy + C_loss_fakeLR + C_loss_noisyLR

                else:
                    raise ValueError('Unknown GAN loss function type')

            self.summary_target['generator_loss'] = gen_loss
            self.summary_target['discriminator_loss'] = dis_loss
        return gen_loss, g_gen_loss, dis_loss, Y_dis_loss, C_dis_loss

    def add_summary_writer(self):
        return [tf.summary.scalar(key, value) for key, value in self.summary_target.items()]


class Optimizer(object):
    """class to build optimizers"""
    @staticmethod
    def pretrain_optimizer(FLAGS, global_iter, pre_gen_loss, pre_dis_loss, Y_loss, C_loss):
        learning_rate = tf.train.exponential_decay(FLAGS.pretrain_learning_rate, global_iter,
                                                   FLAGS.pretrain_lr_decay_step, 0.5, staircase=True)

        with tf.name_scope('optimizer'):
            with tf.variable_scope('optimizer_discriminator'):
                dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                pre_dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=pre_dis_loss,
                                                                                             var_list=dis_var)
            with tf.name_scope('optimizer'):
                with tf.variable_scope('Y_optimizer_discriminator'):
                    dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Y_discriminator')
                    Y_dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=Y_loss,
                                                                                                   var_list=dis_var)
            with tf.name_scope('optimizer'):
                with tf.variable_scope('C_optimizer_discriminator'):
                    dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='C_discriminator')
                    C_dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=C_loss,
                                                                                                   var_list=dis_var)

            with tf.variable_scope('optimizer_generator'):
                pre_gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                pre_gen_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=pre_gen_loss,
                                                                                                 global_step=global_iter,
                                                                                                 var_list=pre_gen_var)

        return pre_gen_var, pre_gen_optimizer, pre_dis_optimizer, Y_dis_optimizer, C_dis_optimizer
    @staticmethod
    def pretrain_optimizer2(FLAGS, global_iter, pre_gen_loss):
        learning_rate = tf.train.exponential_decay(FLAGS.pretrain_learning_rate, global_iter,
                                                   FLAGS.pretrain_lr_decay_step, 0.5, staircase=True)

        with tf.name_scope('optimizer'):
            with tf.variable_scope('optimizer_generator'):
                pre_gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                pre_gen_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=pre_gen_loss,
                                                                                                 global_step=global_iter,
                                                                                                 var_list=pre_gen_var)

        return pre_gen_var, pre_gen_optimizer

    @staticmethod
    def gan_optimizer(FLAGS, global_iter, dis_loss, gen_loss, Y_dis_loss, C_dis_loss):
        boundaries = [100000, 200000, 300000, 400000]
        values = [FLAGS.learning_rate, FLAGS.learning_rate * 0.5, FLAGS.learning_rate * 0.5 ** 2,
                  FLAGS.learning_rate * 0.5 ** 3, FLAGS.learning_rate * 0.5 ** 4]
        learning_rate = tf.train.piecewise_constant(global_iter, boundaries, values)
        with tf.name_scope('optimizer'):
            with tf.variable_scope('optimizer_discriminator'):
                dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                       beta1=0.5).minimize(loss=dis_loss, var_list=dis_var)
        with tf.name_scope('optimizer'):
            with tf.variable_scope('Y_optimizer_discriminator'):
                dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Y_discriminator')
                Y_dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=Y_dis_loss,
                                                                                             var_list=dis_var)
        with tf.name_scope('optimizer'):
            with tf.variable_scope('C_optimizer_discriminator'):
                dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='C_discriminator')
                C_dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=C_dis_loss,
                                                                                             var_list=dis_var)
        with tf.name_scope('optimizer'):
            with tf.variable_scope('optimizer_generator'):
                with tf.control_dependencies([dis_optimizer]):
                    gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                    gen_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(loss=gen_loss,
                                                                                                 global_step=global_iter,
                                                                                                 var_list=gen_var)


        return dis_var, dis_optimizer, gen_var, gen_optimizer, Y_dis_optimizer, C_dis_optimizer
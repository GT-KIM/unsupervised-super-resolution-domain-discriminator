import tensorflow as tf
from ops import batch_instance_norm, instance_norm
class Generator(object) :
    def __init__(self, args) :
        self.channel= args.channel
        self.n_filter = 64
        self.inc_filter = 32
        self.num_repeat_RRDB = args.num_repeat_RRDB
        self.residual_scaling = args.residual_scaling
        self.init_kernel = tf.initializers.he_normal(seed=args.initialization_random_seed)

    def _conv_RRDB(self, x, out_channel, num=None, activate=True) :
        with tf.variable_scope('block{0}'.format(num)) :
            x =tf.layers.conv2d(x, out_channel, 3, 1, padding='same', kernel_initializer=self.init_kernel, name='conv')
            if activate :
                #x = instance_norm(x, name='RRDB_IN' + str(num)) #0218 pre-train2 학습
                x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')
        return x

    def _denseBlock(self, x, num=None) :
        with tf.variable_scope('DenseBlock_sub{0}'.format(num)) :
            x1 = self._conv_RRDB(x, self.inc_filter, 0)
            x2 = self._conv_RRDB(tf.concat([x, x1], axis=3), self.inc_filter, 1)
            x3 = self._conv_RRDB(tf.concat([x, x1, x2], axis=3), self.inc_filter, 2)
            x4 = self._conv_RRDB(tf.concat([x, x1, x2, x3], axis=3), self.inc_filter, 3)
            x5 = self._conv_RRDB(tf.concat([x, x1, x2, x3, x4], axis=3), self.n_filter, 4, activate=False)
        return x5 * self.residual_scaling

    def _RRDB(self, x, num=None) :
        with tf.variable_scope('RRDB_sub{0}'.format(num)) :
            x_branch = tf.identity(x)

            x_branch += self._denseBlock(x_branch, 0)
            x_branch += self._denseBlock(x_branch, 1)
            x_branch += self._denseBlock(x_branch, 2)
        return x + x_branch * self.residual_scaling

    def _upsampling_layer(self, x, num=None) :
        x = tf.layers.conv2d_transpose(x, self.n_filter, 3, 2, padding='same', name='upsample_{0}'.format(num))
        x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')

        return x

    def _upsampling_layer_v2(self, x, num=None) :
        x = tf.image.resize_images(x, [tf.shape(x)[1]*2, tf.shape(x)[2]*2], align_corners=True)
        x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
        x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='valid', name='upsample_{0}'.format(num))
        x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')

        return x

    def _upsampling_layer_v3(self, x, num=None) :
        x = tf.layers.conv2d(x, 64, 3, 1, padding='same', name='upsamplef_{0}'.format(num))
        x = tf.nn.depth_to_space(x, 2, name='pixel_shuffle_{0}'.format(num))
        x = tf.layers.conv2d(x, 64, 3, 1, padding='same', name='upsampleb_{0}'.format(num))
        x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')

        return x

    def _upsampling_layer_v4(self, x, num=None) :
        x_b = tf.identity(x)
        x_b = tf.image.resize_images(x_b, [tf.shape(x_b)[1]*2, tf.shape(x_b)[2]*2], align_corners=True)
        x_b = tf.layers.conv2d(x_b, self.n_filter, 3, 1, padding='same', name='upsampleb_{0}'.format(num))
        x_b = tf.nn.leaky_relu(x_b, alpha=0.2, name='leakyReLU')

        x = tf.layers.conv2d_transpose(x, self.n_filter, 3, 2, padding='same', name='upsamplet_{0}'.format(num))
        x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')

        return 0.1 * x + x_b

    def style_pool(self, x, name) :
        with tf.variable_scope(name) :
            x_avg, x_std = tf.nn.moments(x, axes=[1,2])
            x_std = tf.abs(tf.sqrt(x_std + 1e-12))
            x_avg = tf.expand_dims(x_avg, axis=2)
            x_std = tf.expand_dims(x_std, axis=2)
            x_feature = tf.concat((x_avg, x_std), axis=2)
            _, width, channel = x_feature.get_shape().as_list()
            x_weight = tf.get_variable("W", shape=[width, channel], initializer=self.init_kernel)
            x_feature = tf.reduce_sum(x_feature * x_weight, axis=2)
            x_feature = tf.expand_dims(x_feature, axis=1)
            x_feature = tf.expand_dims(x_feature, axis=2)
            #x_feature = instance_norm(x_feature)
            x_feature = tf.nn.sigmoid(x_feature)

            x = x * x_feature
        return x

    def _mask_layer(self, x, i) :
        x_mask = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel)
        #x_mask = tf.layers.BatchNormalization(name='batch_norm_0')(x_mask)
        x_mask = tf.nn.leaky_relu(x_mask, alpha=0.2)
        #x_mask = self.style_pool(x_mask, "style0_{0}".format(i))

        x_mask = tf.layers.conv2d(x_mask, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel)
        #x_mask = tf.layers.BatchNormalization(name='batch_norm_1')(x_mask)
        x_mask = tf.nn.leaky_relu(x_mask, alpha=0.2)
        #x_mask = self.style_pool(x_mask, "style1_{0}".format(i))
        x_mask = x + 0.2 * x_mask
        return x_mask

    def build_G1(self, NLR) :
        with tf.variable_scope('Generator1') :
            with tf.variable_scope('first_conv') :
                x = tf.layers.conv2d(NLR, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv0')
                x = tf.nn.leaky_relu(x, alpha=0.2)
                x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv1')
                x = tf.nn.leaky_relu(x, alpha=0.2)
                x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv2')
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('mask_conv') :
                x = self._mask_layer(x, 0)
                x = self._mask_layer(x, 1)
                x = self._mask_layer(x, 2)
                x = self._mask_layer(x, 3)
            with tf.variable_scope('last_conv') :
                x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv0')
                x = tf.nn.leaky_relu(x, alpha=0.2)
                CLR = tf.layers.conv2d(x, self.channel, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv1')
        return CLR

    def build_G2(self, CLR) :
        with tf.variable_scope('Generator2') :
            with tf.variable_scope('first_conv') :
                x = tf.layers.conv2d(CLR, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv0')
                x = tf.nn.leaky_relu(x, alpha=0.2)
                x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv1')
                x = tf.nn.leaky_relu(x, alpha=0.2)
                x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv2')
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('mask_conv') :
                x = self._mask_layer(x, 0)
                x = self._mask_layer(x, 1)
                x = self._mask_layer(x, 2)
                x = self._mask_layer(x, 3)
            with tf.variable_scope('last_conv') :
                x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv0')
                x = tf.nn.leaky_relu(x, alpha=0.2)
                NLR = tf.layers.conv2d(x, self.channel, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv1')
        return NLR

    def build_G3(self, CHR) :
        with tf.variable_scope('Generator3') :
            with tf.variable_scope('first_conv') :
                x = tf.layers.conv2d(CHR, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv0')
                x = tf.nn.leaky_relu(x, alpha=0.2)
                x = tf.layers.conv2d(x, self.n_filter, 3, 2, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv1')
                x = tf.nn.leaky_relu(x, alpha=0.2)
                x = tf.layers.conv2d(x, self.n_filter, 3, 2, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv2')
                x = tf.nn.leaky_relu(x, alpha=0.2)
            with tf.variable_scope('mask_conv') :
                x = self._mask_layer(x, 0)
                x = self._mask_layer(x, 1)
                x = self._mask_layer(x, 2)
                x = self._mask_layer(x, 3)
            with tf.variable_scope('last_conv') :
                x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv0')
                x = tf.nn.leaky_relu(x, alpha=0.2)
                NLR = tf.layers.conv2d(x, self.channel, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv1')
        return NLR

    def build_SR(self, CLR) :
        with tf.variable_scope('SR') :
            with tf.variable_scope('first_conv') :
                x = tf.layers.conv2d(CLR, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv')
                x = tf.nn.leaky_relu(x, alpha=0.2)

            with tf.variable_scope('RRDB'):
                x_branch = tf.identity(x)
                x_branch = tf.layers.conv2d(x_branch, self.n_filter, 3, 1, padding='same',
                                            kernel_initializer=self.init_kernel, name='conv')
                x_branch = tf.nn.leaky_relu(x_branch, alpha=0.2, name='leakyReLU')
                for i in range(self.num_repeat_RRDB):
                    x_branch = self._RRDB(x_branch, i)

                x_branch = tf.layers.conv2d(x_branch, self.n_filter, 3, 1, padding='same',
                                            kernel_initializer=self.init_kernel, name='trunk_conv')
                x += x_branch

            with tf.variable_scope('Upsampling'):
                x = self._upsampling_layer_v4(x, 1)
                x = self._upsampling_layer_v4(x, 2)

            with tf.variable_scope('last_conv'):
                x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv_1')
                x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')
                CHR = tf.layers.conv2d(x, self.channel, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                     name='conv_2')
        # x = tf.nn.tanh(x)
        return CHR

    def build_downsample(self, x) :
        with tf.variable_scope('mask_conv') :
            x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                 name='conv')
            x = tf.nn.leaky_relu(x, alpha=0.2)

        with tf.variable_scope('RRDB') :
            x_branch = tf.identity(x)
            x_branch =tf.layers.conv2d(x_branch, self.n_filter, 3, 1, padding='same',
                                       kernel_initializer=self.init_kernel, name='conv')
            x_branch = tf.nn.leaky_relu(x_branch, alpha=0.2, name='leakyReLU')
            for i in range(self.num_repeat_RRDB) :
                x_branch = self._RRDB(x_branch, i)

            x_branch = tf.layers.conv2d(x_branch, self.n_filter, 3, 1, padding='same',
                                        kernel_initializer=self.init_kernel,name='trunk_conv')
        x += x_branch

        with tf.variable_scope('Upsampling') :
            x = self._upsampling_layer_v2(x, 1)
            x = self._upsampling_layer_v2(x, 2)

        with tf.variable_scope('last_conv') :
            x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                 name='conv_1')
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU')
            x = tf.layers.conv2d(x, self.channel, 3, 1, padding='same', kernel_initializer=self.init_kernel,
                                 name='conv_2')
            #x = tf.nn.tanh(x)
        return x

class Discriminator(object) :
    def __init__(self, args) :
        self.channel = args.channel
        self.n_filter = 64
        self.init_kernel =  tf.initializers.he_normal(seed=args.initialization_random_seed)

    def _conv_block(self, x, out_channel, num=None):
        with tf.variable_scope('block_{0}'.format(num)):
            x = tf.layers.conv2d(x, out_channel, 3, 1, padding='same', use_bias=False,
                                 kernel_initializer=self.init_kernel, name='conv_1')
            #x = tf.layers.BatchNormalization(name='batch_norm_1')(x)
            x = batch_instance_norm(x, name='dconvBIN1_'+str(num))
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_1')

            x = tf.layers.conv2d(x, out_channel, 4, 2, padding='same', use_bias=False,
                                 kernel_initializer=self.init_kernel, name='conv_2')
            #x = tf.layers.BatchNormalization(name='batch_norm_2')(x)
            x = batch_instance_norm(x, name='dconvBIN2_'+str(num))
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_2')

            return x

    def build(self, x):
        with tf.variable_scope('first_conv'):
            x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', use_bias=False,
                                 kernel_initializer=self.init_kernel, name='conv_1')
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_1')
            x = tf.layers.conv2d(x, self.n_filter, 4, 2, padding='same', use_bias=False,
                                 kernel_initializer=self.init_kernel, name='conv_2')
            #x = tf.layers.BatchNormalization(name='batch_norm_1')(x)
            x = batch_instance_norm(x, name='BIN1')
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_2')

        with tf.variable_scope('conv_block'):
            x = self._conv_block(x, self.n_filter * 1, 0)
            x = self._conv_block(x, self.n_filter * 1, 1)
            x = self._conv_block(x, self.n_filter * 2, 2)
            x = self._conv_block(x, self.n_filter * 4, 3)

        with tf.variable_scope('full_connected'):
            x = tf.reduce_mean(x, axis=(1,2))
            #x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 100, name='fully_connected_1')
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_1')
            x = tf.layers.dense(x, 1, name='fully_connected_2')

        return x

class Discriminator_color(object) :
    def __init__(self, args) :
        self.channel = args.channel
        self.n_filter = 64
        self.init_kernel =  tf.initializers.he_normal(seed=args.initialization_random_seed)

    def _conv_block(self, x, out_channel, num=None):
        with tf.variable_scope('block_{0}'.format(num)):
            x = tf.layers.conv2d(x, out_channel, 3, 1, padding='same', use_bias=False,
                                 kernel_initializer=self.init_kernel, name='conv_1')
            x = tf.layers.BatchNormalization(name='batch_norm_1')(x)
            #x = instance_norm(x, name='dconvIN1_'+str(num))
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_1')

            x = tf.layers.conv2d(x, out_channel, 4, 2, padding='same', use_bias=False,
                                 kernel_initializer=self.init_kernel, name='conv_2')
            x = tf.layers.BatchNormalization(name='batch_norm_2')(x)
            #x = instance_norm(x, name='dconvIN2_'+str(num))
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_2')

            return x

    def build(self, x):
        with tf.variable_scope('first_conv'):
            x = tf.layers.conv2d(x, self.n_filter, 3, 1, padding='same', use_bias=False,
                                 kernel_initializer=self.init_kernel, name='conv_1')
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_1')
            x = tf.layers.conv2d(x, self.n_filter, 4, 2, padding='same', use_bias=False,
                                 kernel_initializer=self.init_kernel, name='conv_2')
            x = tf.layers.BatchNormalization(name='batch_norm_1')(x)
            #x = instance_norm(x, name='dconvIN0')
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_2')

        with tf.variable_scope('conv_block'):
            x = self._conv_block(x, self.n_filter , 0)
            x = self._conv_block(x, self.n_filter, 1)
            x = self._conv_block(x, self.n_filter * 2, 2)
            x = self._conv_block(x, self.n_filter * 4, 3)

        with tf.variable_scope('full_connected'):
            x = tf.reduce_mean(x, axis=(1,2))
            #x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 100, name='fully_connected_1')
            x = tf.nn.leaky_relu(x, alpha=0.2, name='leakyReLU_1')
            x = tf.layers.dense(x, 1, name='fully_connected_2')

        return x

class Perceptual_VGG19(object):
    """the definition of VGG19. This network is used for constructing perceptual loss"""
    @staticmethod
    def build(x):
        # Block 1
        x = tf.layers.conv2d(x, 64, (3, 3), activation='relu', padding='same', name='block1_conv1')
        x = tf.layers.conv2d(x, 64, (3, 3), activation='relu', padding='same', name='block1_conv2')
        x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block1_pool')

        # Block 2
        x = tf.layers.conv2d(x, 128, (3, 3), activation='relu', padding='same', name='block2_conv1')
        x = tf.layers.conv2d(x, 128, (3, 3), activation='relu', padding='same', name='block2_conv2')
        x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block2_pool')

        # Block 3
        x = tf.layers.conv2d(x, 256, (3, 3), activation='relu', padding='same', name='block3_conv1')
        x = tf.layers.conv2d(x, 256, (3, 3), activation='relu', padding='same', name='block3_conv2')
        x = tf.layers.conv2d(x, 256, (3, 3), activation='relu', padding='same', name='block3_conv3')
        x = tf.layers.conv2d(x, 256, (3, 3), activation='relu', padding='same', name='block3_conv4')
        x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block3_pool')

        # Block 4
        x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv1')
        x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv2')
        x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv3')
        x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv4')
        x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block4_pool')

        # Block 5
        x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block5_conv1')
        x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block5_conv2')
        x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block5_conv3')
        x = tf.layers.conv2d(x, 512, (3, 3), activation=None, padding='same', name='block5_conv4')

        return x

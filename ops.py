from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def batch_instance_norm(input, name="batch_instance_norm"):
    with tf.variable_scope(name):
        ch = input.get_shape()[-1]
        epsilon = 1e-5

        batch_mean, batch_sigma = tf.nn.moments(input, axes=[0,1,2], keep_dims=True)
        x_batch = (input - batch_mean) / (tf.sqrt(batch_sigma + epsilon))

        ins_mean, ins_sigma = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        x_ins = (input - ins_mean) / (tf.sqrt(ins_sigma + epsilon))

        rho = tf.get_variable("rho", [ch], initializer=tf.constant_initializer(1.0),
                              constraint=lambda x : tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        gamma = tf.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        x_hat = rho * x_batch + (1 - rho) * x_ins
        x_hat = x_hat * gamma + beta

        return x_hat

def scale_initialization(weights, FLAGS):
    return [tf.assign(weight, weight * FLAGS.weight_initialize_scale) for weight in weights]


def _transfer_vgg19_weight(FLAGS, weight_dict):
    from_model = VGG19(include_top=False, weights='imagenet', input_tensor=None,
                       input_shape=(FLAGS.HR_image_size, FLAGS.HR_image_size, FLAGS.channel))

    fetch_weight = []

    for layer in from_model.layers:
        if 'conv' in layer.name:
            W, b = layer.get_weights()

            fetch_weight.append(
                tf.assign(weight_dict['loss_generator/perceptual_vgg19/{}/kernel'.format(layer.name)], W)
            )
            fetch_weight.append(
                tf.assign(weight_dict['loss_generator/perceptual_vgg19/{}/bias'.format(layer.name)], b)
            )

    return fetch_weight


def load_vgg19_weight(FLAGS):
    vgg_weight = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='loss_generator/perceptual_vgg19')

    assert len(vgg_weight) > 0, 'No VGG19 weight was collected. The target scope might be wrong.'

    weight_dict = {}
    for weight in vgg_weight:
        weight_dict[weight.name.rsplit(':', 1)[0]] = weight

    return _transfer_vgg19_weight(FLAGS, weight_dict)


def extract_weight(network_vars):
    weight_dict = OrderedDict()

    for weight in network_vars:
        weight_dict[weight.name] = weight.eval()

    return weight_dict


def interpolate_weight(FLAGS, pretrain_weight):
    fetch_weight = []
    alpha = FLAGS.interpolation_param

    for name, pre_weight in pretrain_weight.items():
        esrgan_weight = tf.get_default_graph().get_tensor_by_name(name)

        assert pre_weight.shape == esrgan_weight.shape, 'The shape of weights does not match'

        fetch_weight.append(tf.assign(esrgan_weight, (1 - alpha) * pre_weight + alpha * esrgan_weight))

    return fetch_weight
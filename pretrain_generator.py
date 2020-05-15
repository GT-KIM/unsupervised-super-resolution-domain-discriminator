import gc
import os
import math
import time
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from glob import glob
from ops import scale_initialization
from train_module import Network, Loss, Optimizer
from utils import log, normalize_images, save_image, generate_testset, generate_pretrain_batch, generate_pretrain_batch2
from ops import load_vgg19_weight

def pretrain_generator1(args, logflag):
    """pre-train Low Resolution CycleGAN"""
    log(logflag, 'Pre-train : Process start', 'info')

    NLR_data = tf.placeholder(tf.float32, shape=[None, None, None, args.channel],
                              name='NLR_input')
    CLR_data = tf.placeholder(tf.float32, shape=[None, None, None, args.channel],
                             name='CLR_input')

    # build Generator
    network = Network(args,NLR_data= NLR_data,CLR_data=CLR_data)
    CLR_N, NLR_NC, NLR_C, CLR_CN, CLR_C = network.pretrain_generator_LR()
    dis_out_realLR, dis_out_fakeLR, dis_out_noisyLR, Y_out, C_out = network.pretrain_discriminator_LR(CLR_N)
    print("Built generator!")

    # build loss function
    loss = Loss()
    gen_loss, dis_loss, Y_loss, C_loss = loss.pretrain_loss(args, NLR_data, CLR_data, NLR_NC, CLR_CN, CLR_C, dis_out_realLR,
                                            dis_out_fakeLR, dis_out_noisyLR, Y_out, C_out)
    print("Built loss function!")

    # build optimizer
    global_iter = tf.Variable(0, trainable=False)
    pre_gen_var, pre_gen_optimizer, pre_dis_optimizer, Y_optimizer, C_optimizer = \
        Optimizer().pretrain_optimizer(args, global_iter, gen_loss, dis_loss, Y_loss, C_loss)
    print("Built optimizer!")

    # build summary writer
    #pre_summary = tf.summary.merge(loss.add_summary_writer())

    fetches = {'pre_gen_loss': gen_loss, 'pre_dis_loss' : dis_loss, 'Y_loss' : Y_loss, 'C_loss' : C_loss,
               'pre_gen_optimizer': pre_gen_optimizer,
               'pre_dis_optimizer': pre_dis_optimizer,
               'Y_optimizer' : Y_optimizer,
               'C_optimizer' : C_optimizer,
               'gen_HR': CLR_N}

    gc.collect()

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            visible_device_list=args.gpu_dev_num
        )
    )

    saver = tf.train.Saver(max_to_keep=10)

    # Start session
    with tf.Session(config=config) as sess:
        log(logflag, 'Pre-train : Training starts', 'info')

        sess.run(tf.global_variables_initializer())
        sess.run(global_iter.initializer)
        sess.run(scale_initialization(pre_gen_var, args))
        saver.restore(sess, tf.train.latest_checkpoint(args.pre_train_checkpoint_dir))
        if args.perceptual_loss == 'VGG19':
            sess.run(load_vgg19_weight(args))
        writer = tf.summary.FileWriter(args.logdir, graph=sess.graph, filename_suffix='pre-train')

        _datapathNLR = np.sort(np.asarray(glob(os.path.join(args.data_dir + '/source/train_HR_aug/x4/', '*.png'))))
        _datapathCLR = np.sort(np.asarray(glob(os.path.join(args.data_dir + '/target/train_HR_aug/x4/', '*.png'))))
        idxN = np.random.permutation(len(_datapathNLR))
        idxC = np.random.permutation(len(_datapathCLR))
        datapathNLR = _datapathNLR[idxN]
        datapathCLR = _datapathCLR[idxC]

        epoch = 0
        counter = 0

        log(logflag, 'Pre-train Epoch: {0}'.format(epoch), 'info')
        start_time = time.time()
        #while counter <= args.num_iter:
        loss = 0.0
        steps = 0
        while True:
            lr = args.pretrain_learning_rate
            if counter >= len(_datapathCLR) - args.image_batch_size:
                log(logflag, 'Pre-train Epoch: {0} avg.loss : {1}'.format(epoch, loss / steps), 'info')
                idx = np.random.permutation(len(_datapathCLR))
                datapathNLR = _datapathNLR[idx]
                datapathCLR = _datapathCLR[idx]
                counter = 0
                loss = 0.0
                steps = 0
                epoch += 1
                if epoch == 200 :
                    break
            dataNLR, dataCLR = generate_pretrain_batch(datapathNLR[counter:counter + args.image_batch_size],
                                            datapathCLR[counter:counter + args.image_batch_size],
                                            args)
            counter += args.image_batch_size
            for iteration in range(0, dataNLR.shape[0], args.batch_size) :
                _NLR_data = dataNLR[iteration:iteration + args.batch_size]
                _CLR_data = dataCLR[iteration:iteration + args.batch_size]
                feed_dict = {
                    NLR_data: _NLR_data,
                    CLR_data: _CLR_data,
                }
                # update weights
                result = sess.run(fetches=fetches, feed_dict=feed_dict)
                current_iter = tf.train.global_step(sess, global_iter)
                loss += result['pre_gen_loss']
                steps += 1

                # save samples every n iter
                if current_iter % args.train_sample_save_freq == 0 :
                    validpathLR = np.sort(
                        np.asarray(glob(os.path.join(args.data_dir + '/validation/x/', '*.png'))))
                    validpathHR = np.sort(
                        np.asarray(glob(os.path.join(args.data_dir + '/validation/y/', '*.png'))))
                    for valid_i in range(5):
                        validLR, validHR = generate_testset(validpathLR[valid_i],
                                                            validpathHR[valid_i],
                                                            args)
                        validLR = np.transpose(validLR[:, :, :, np.newaxis], (3, 0, 1, 2))
                        validHR = np.transpose(validHR[:, :, :, np.newaxis], (3, 0, 1, 2))
                        valid_out = sess.run(CLR_N,feed_dict={NLR_data: validLR,
                                                                      CLR_data: validLR,
                                                                      })
                        save_image(args, valid_out, 'pre-valid', current_iter + valid_i, save_max_num=5)
                    save_image(args, result['gen_HR'], 'pre-train', current_iter, save_max_num=5)
                    save_image(args, _NLR_data, 'pre-raw', current_iter, save_max_num=5)
                if current_iter % 10 == 0 :
                    log(logflag,
                        'Pre-train iteration : {0}, pre_gen_loss : {1}, pre_dis_loss : {2} Y_loss : {3}, C_loss : {4}'.format(
                            current_iter, result['pre_gen_loss'], result['pre_dis_loss'], result['Y_loss'], result['C_loss']),
                        'info')

                # save checkpoint
                if current_iter % args.train_ckpt_save_freq == 0:
                    saver.save(sess, os.path.join(args.pre_train_checkpoint_dir, 'pre_gen'), global_step=current_iter)


    writer.close()
    log(logflag, 'Pre-train : Process end', 'info')


def pretrain_generator2(args, logflag):
    """pre-train Low Resolution CycleGAN"""
    log(logflag, 'Pre-train : Process start', 'info')
    """
    v_list = list()
    with tf.Session() as sess :
        saver = tf.train.import_meta_graph(args.pre_train_checkpoint_dir + "/pre_gen-831000.meta")
        saver.restore(sess=sess, save_path = args.pre_train_checkpoint_dir + "/pre_gen-831000")
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) :
            v_list.append(v.name)
    tf.reset_default_graph()
    """
    g = tf.Graph()
    with g.as_default() as graph:
        CLR_data = tf.placeholder(tf.float32, shape=[None, None, None, args.channel],
                                  name='CLR_input')
        CHR_data = tf.placeholder(tf.float32, shape=[None, None, None, args.channel],
                                 name='CHR_input')
        # build Generator
        network = Network(args,CLR_data= CLR_data,CHR_data=CHR_data)
        CHR = network.pretrain_generator_SR()
        print("Built generator!")

        # build loss function
        loss = Loss()
        gen_loss = loss.pretrain_loss2(CHR_data, CHR)
        print("Built loss function!")

        # build optimizer
        global_iter = tf.Variable(0, trainable=False)
        pre_gen_var, pre_gen_optimizer= Optimizer().pretrain_optimizer2(args, global_iter, gen_loss)
        print("Built optimizer!")

    # build summary writer
    #pre_summary = tf.summary.merge(loss.add_summary_writer())

    fetches = {'pre_gen_loss': gen_loss,
               'pre_gen_optimizer': pre_gen_optimizer,
               'CHR_out': CHR}
    gc.collect()

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            visible_device_list=args.gpu_dev_num
        )
    )

    # Start session
    with tf.Session(config=config, graph=g) as sess:
        log(logflag, 'Pre-train : Training starts', 'info')

        sess.run(tf.global_variables_initializer())
        sess.run(global_iter.initializer)
        sess.run(scale_initialization(pre_gen_var, args))
        #var_list = [g.get_tensor_by_name('%s' % name) for name in v_list]
        #saver = tf.train.Saver(max_to_keep=10, var_list=var_list)
        #saver.restore(sess, tf.train.latest_checkpoint(args.pre_train_checkpoint_dir))
        saver = tf.train.Saver(max_to_keep=10)
        saver.restore(sess, tf.train.latest_checkpoint(args.pre_train_checkpoint_dir))

        writer = tf.summary.FileWriter(args.logdir, graph=sess.graph, filename_suffix='pre-train')

        _datapathCLR = np.sort(np.asarray(glob(os.path.join(args.data_dir + '/target/train_LR_aug/x4/', '*.png'))))
        _datapathCHR = np.sort(np.asarray(glob(os.path.join(args.data_dir + '/target/train_HR_aug/x4/', '*.png'))))
        idx = np.random.permutation(len(_datapathCHR))
        datapathCLR = _datapathCLR[idx]
        datapathCHR = _datapathCHR[idx]

        epoch = 0
        counter = 0

        log(logflag, 'Pre-train Epoch: {0}'.format(epoch), 'info')
        start_time = time.time()
        loss = 0.0
        steps = 0
        while True:
            lr = args.pretrain_learning_rate
            if counter >= len(_datapathCLR) - args.image_batch_size:
                log(logflag, 'Pre-train Epoch: {0} avg.loss : {1}'.format(epoch, loss / steps), 'info')
                idx = np.random.permutation(len(_datapathCLR))
                datapathCLR = _datapathCLR[idx]
                datapathCHR = _datapathCHR[idx]
                counter = 0
                loss = 0.0
                steps = 0
                epoch += 1
                if epoch == 200 :
                    break
            dataCLR, dataCHR = generate_pretrain_batch2(datapathCLR[counter:counter + args.image_batch_size],
                                            datapathCHR[counter:counter + args.image_batch_size],
                                            args)
            counter += args.image_batch_size
            #if current_iter > args.num_iter:
            #    break
            for iteration in range(0, dataCLR.shape[0], args.batch_size) :
                _CLR_data = dataCLR[iteration:iteration + args.batch_size]
                _CHR_data = dataCHR[iteration:iteration + args.batch_size]
                feed_dict = {
                    CLR_data: _CLR_data,
                    CHR_data: _CHR_data,
                }
                # update weights
                result = sess.run(fetches=fetches, feed_dict=feed_dict)
                current_iter = tf.train.global_step(sess, global_iter)
                loss += result['pre_gen_loss']
                steps += 1

                # save summary every n iter
                #if current_iter % args.train_summary_save_freq == 0:
                #    writer.add_summary(result['summary'], global_step=current_iter)

                # save samples every n iter
                if current_iter % args.train_sample_save_freq == 0:
                    validpathLR = np.sort(
                        np.asarray(glob(os.path.join(args.data_dir + '/validation/x/', '*.png'))))
                    validpathHR = np.sort(
                        np.asarray(glob(os.path.join(args.data_dir + '/validation/y/', '*.png'))))
                    for valid_i in range(5):
                        validLR, validHR = generate_testset(validpathLR[valid_i],
                                                            validpathHR[valid_i],
                                                            args)
                        validLR = np.transpose(validLR[:, :, :, np.newaxis], (3, 0, 1, 2))
                        validHR = np.transpose(validHR[:, :, :, np.newaxis], (3, 0, 1, 2))
                        valid_out = sess.run(CHR,feed_dict={CLR_data: validLR,
                                                            CHR_data: validHR})
                        save_image(args, valid_out, 'pre-valid', current_iter + valid_i, save_max_num=5)

                    save_image(args, result['CHR_out'], 'pre-train', current_iter, save_max_num=5)
                    save_image(args, _CHR_data, 'pre-raw', current_iter, save_max_num=5)
                if current_iter % 10 == 0 :
                    log(logflag,
                        'Pre-train iteration : {0}, pre_gen_loss : {1}'.format(
                            current_iter, result['pre_gen_loss']),
                        'info')
                # save checkpoint
                if current_iter % args.train_ckpt_save_freq == 0:
                    saver.save(sess, os.path.join(args.pre_train_checkpoint_dir, 'pre_gen'), global_step=current_iter)


    writer.close()
    log(logflag, 'Pre-train : Process end', 'info')

def test_pretrain_generator(args, logflag):
    """pre-train deep network as initialization weights of ESRGAN Generator"""
    log(logflag, 'Pre-test : Process start', 'info')

    LR_data = tf.placeholder(tf.float32, shape=[None, None, None, args.channel],
                             name='LR_input')
    HR_data = tf.placeholder(tf.float32, shape=[None, None, None, args.channel],
                             name='HR_input')

    # build Generator
    network = Network(args, LR_data)
    pre_gen_out = network.generator()

    # build loss function
    loss = Loss()
    pre_gen_loss = loss.pretrain_loss(pre_gen_out, HR_data)

    # build optimizer
    global_iter = tf.Variable(0, trainable=False)
    pre_gen_var, pre_gen_optimizer = Optimizer().pretrain_optimizer(args, global_iter, pre_gen_loss)

    # build summary writer
    pre_summary = tf.summary.merge(loss.add_summary_writer())

    fetches = {'gen_HR': pre_gen_out, 'summary': pre_summary}

    gc.collect()

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            visible_device_list=args.gpu_dev_num
        )
    )

    saver = tf.train.Saver(max_to_keep=10)

    # Start session
    with tf.Session(config=config) as sess:
        log(logflag, 'Pre-train : Test starts', 'info')

        sess.run(tf.global_variables_initializer())
        sess.run(global_iter.initializer)
        sess.run(scale_initialization(pre_gen_var, args))
        saver.restore(sess, tf.train.latest_checkpoint(args.pre_train_checkpoint_dir))

        writer = tf.summary.FileWriter(args.logdir, graph=sess.graph, filename_suffix='pre-train')

        #_datapathLR = np.sort(np.asarray(glob(os.path.join(args.data_dir + '/source/train_LR_aug/x4/', '*.png'))))
        #_datapathHR = np.sort(np.asarray(glob(os.path.join(args.data_dir + '/source/train_HR_aug/x4/', '*.png'))))
        _datapathLR = np.sort(np.asarray(glob(os.path.join(args.data_dir + '/validation/x/', '*.png'))))
        #_datapathHR = np.sort(np.asarray(glob(os.path.join(args.data_dir + '/validation/y/', '*.png'))))
        #_datapathLR = np.sort(np.asarray(glob(os.path.join('/home/super/PycharmProjects/NTIRE2020_tensorflow/sample/', 'A_*.jpg'))))
        _datapathHR = np.sort(np.asarray(glob(os.path.join(args.data_dir + '/validation/y/', '*.png'))))

        #idx = np.random.permutation(len(_datapathLR))
        #datapathLR = _datapathLR[idx]
        #datapathHR = _datapathHR[idx]
        datapathLR = _datapathLR
        datapathHR = _datapathHR

        for i in range(0,len(datapathLR),1) :
            log(logflag, 'Pre-train, info')
            start_time = time.time()
            dataLR, dataHR = generate_testset(datapathLR[i],
                                          datapathHR[i],
                                          args)
            dataLR = np.transpose(dataLR[:,:,:,np.newaxis], (3,0,1,2))
            dataHR = np.transpose(dataHR[:,:,:,np.newaxis], (3,0,1,2))
            feed_dict = {
                HR_data: dataHR,
                LR_data: dataLR
                }
            # update weights
            result = sess.run(fetches=fetches, feed_dict=feed_dict)
            current_iter = i
            save_image(args, result['gen_HR'], 'pre-test', current_iter, save_max_num=5)
            print("saved %d" %i)
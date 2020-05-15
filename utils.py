import logging
import os
import glob
import tensorflow as tf
import cv2
import numpy as np


def log(logflag, message, level='info'):
    """logging to stdout and logfile if flag is true"""
    print(message, flush=True)

    if logflag:
        if level == 'info':
            logging.info(message)
        elif level == 'warning':
            logging.warning(message)
        elif level == 'error':
            logging.error(message)
        elif level == 'critical':
            logging.critical(message)


def create_dirs(target_dirs):
    """create necessary directories to save output files"""
    for dir_path in target_dirs:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)


def normalize_images(*arrays):
    """normalize input image arrays"""
    return [arr / 127.5 - 1 for arr in arrays]


def de_normalize_image(image):
    """de-normalize input image array"""
    return (image + 1) * 127.5


def save_image(args, images, phase, global_iter, save_max_num=5):
    """save images in specified directory"""
    save_dir = ''
    if phase == 'pre-train':
        save_dir = args.pre_train_result_dir
    elif phase =='pre-raw':
        save_dir = args.pre_raw_result_dir
    elif phase == 'pre-valid' :
        save_dir = args.pre_valid_result_dir
    elif phase == 'pre-valid_LR' :
        save_dir = args.pre_valid_LR_result_dir
    elif phase == 'train' :
        save_dir = args.train_result_dir
    elif phase == 'valid' :
        save_dir = args.valid_result_dir
    elif phase == 'valid_LR' :
        save_dir = args.valid_LR_result_dir
    elif phase == 'test' :
        save_dir = args.test_result_dir
    elif phase == 'test_LR' :
        save_dir = args.test_LR_result_dir
    elif phase == 'pre-test' :
        save_dir = './pre_test_images/'
        if not os.path.isdir(save_dir) :
            os.makedirs(save_dir)
    else:
        print('specified phase is invalid')


    for i, img in enumerate(images):
        if i >= save_max_num:
            break
        #cv2.imwrite(save_dir + '/{0}.png'.format(global_iter+901), de_normalize_image(img))
        cv2.imwrite(save_dir +'/' +global_iter, de_normalize_image(img))


def crop(img, args):
    """crop patch from an image with specified size"""
    img_h, img_w, _ = img.shape

    rand_h = np.random.randint(img_h - args.crop_size)
    rand_w = np.random.randint(img_w - args.crop_size)

    return img[rand_h:rand_h + args.crop_size, rand_w:rand_w + args.crop_size, :]


def data_augmentation(LR_images, HR_images, aug_type='horizontal_flip'):
    """data augmentation. input arrays should be [N, H, W, C]"""

    if aug_type == 'horizontal_flip':
        return LR_images[:, :, ::-1, :], HR_images[:, :, ::-1, :]
    elif aug_type == 'rotation_90':
        return np.rot90(LR_images, k=1, axes=(1, 2)), np.rot90(HR_images, k=1, axes=(1, 2))


def generate_batch(NLR_list, NHR_list, CLR_list, CHR_list, args) :
    patch_h = args.crop_size
    patch_w = args.crop_size
    scale = args.scale_SR
    stride = args.stride

    patches_NLR = list()
    patches_NHR = list()
    patches_CLR = list()
    patches_CHR = list()
    for i in range(len(NLR_list)) :
        NLR = cv2.imread(NLR_list[i])
        NHR = cv2.imread(NHR_list[i])
        CLR = cv2.imread(CLR_list[i])
        CHR = cv2.imread(CHR_list[i])

        NLR = NLR / 127.5 - 1.
        NHR = NHR / 127.5 - 1.
        CLR = CLR / 127.5 - 1.
        CHR = CHR / 127.5 - 1
        NLR_x, NLR_y, depth = NLR.shape
        NHR_x, NHR_y, depth = NLR.shape
        CLR_x, CLR_y, depth = CLR.shape

        for h in range(0, NLR_x-patch_h, stride) :
            for w in range(0, NLR_y-patch_w, stride) :
                x = h
                y = w
                patch_NLR = NLR[x:x+patch_h,y:y+patch_w]
                patches_NLR.append(patch_NLR)

        for h in range(0, NHR_x-patch_h*4, stride*4) :
            for w in range(0, NHR_y-patch_w*4, stride*4) :
                x = h
                y = w
                patch_NHR = NHR[x:x+patch_h,y:y+patch_w]
                patches_NHR.append(patch_NHR)

        for h in range(0, CLR_x - patch_h, stride):
            for w in range(0, CLR_y - patch_w, stride):
                x = h
                y = w
                #x = np.random.randint(CLR_x - patch_h)
                #y = np.random.randint(CLR_y - patch_w)
                patch_CLR = CLR[x:x+patch_h,y:y+patch_w]
                patches_CLR.append(patch_CLR)

                x *= scale
                y *= scale

                patch_CHR = CHR[x:x+patch_h*scale,y:y+patch_w*scale]
                patches_CHR.append(patch_CHR)
                """
                s = cv2.cvtColor(patch_s, cv2.COLOR_BGR2RGB)
                t = cv2.cvtColor(patch_t, cv2.COLOR_BGR2RGB)
                import matplotlib.pyplot as plt
                plt.figure(0)
                plt.imshow(s)
    
                plt.figure(1)
                plt.imshow(t)
                plt.show()
                """
    if len(patches_NLR) > len(patches_CLR) :
        patches_NLR = patches_NLR[:len(patches_CLR)]
        patches_NHR = patches_NHR[:len(patches_CLR)]
    elif len(patches_NLR) < len(patches_CLR) :
        patches_CLR = patches_CLR[:len(patches_NLR)]
        patches_CHR = patches_CHR[:len(patches_CHR)]
    np.random.seed(36)
    np.random.shuffle(patches_NLR)
    np.random.seed(36)
    np.random.shuffle(patches_NHR)
    np.random.seed(36)
    np.random.shuffle(patches_CLR)
    np.random.seed(36)
    np.random.shuffle(patches_CHR)

    return np.array(patches_NLR), np.array(patches_NHR), np.array(patches_CLR), np.array(patches_CHR)

def generate_pretrain_batch(source_list, target_list, args) :
    patch_h = args.crop_size
    patch_w = args.crop_size
    scale = args.scale_SR
    stride = args.stride

    patch_source = list()
    patch_target = list()
    for i in range(len(source_list)) :
        source = cv2.imread(source_list[i])
        target = cv2.imread(target_list[i])

        source = source / 127.5 - 1.
        target = target / 127.5 - 1
        source_x, source_y, depth = source.shape
        target_x, target_y, depth = target.shape

        for h in range(0, source_x - patch_h, stride):
            for w in range(0, source_y - patch_w, stride):
                x = h
                y = w
                patch_s = source[x:x+patch_h,y:y+patch_w]
                patch_source.append(patch_s)

        for h in range(0, target_x - patch_h, stride):
            for w in range(0, target_y - patch_w, stride):
                x = h
                y = w
                patch_t = target[x:x+patch_h,y:y+patch_w]
                patch_target.append(patch_t)

    if len(patch_source) > len(patch_target) :
        patch_source = patch_source[:len(patch_target)]
    elif len(patch_target) < len(patch_source) :
        patch_target = patch_target[:len(patch_source)]

    np.random.seed(36)
    np.random.shuffle(patch_source)
    np.random.seed(36)
    np.random.shuffle(patch_target)

    return np.array(patch_source), np.array(patch_target)

def generate_pretrain_batch2(source_list, target_list, args) :
    patch_h = args.crop_size
    patch_w = args.crop_size
    scale = args.scale_SR
    stride = args.stride

    patch_source = list()
    patch_target = list()
    for i in range(len(source_list)) :
        source = cv2.imread(source_list[i])
        target = cv2.imread(target_list[i])

        source = source / 127.5 - 1.
        target = target / 127.5 - 1
        source_x, source_y, depth = source.shape
        target_x, target_y, depth = target.shape

        for h in range(0, source_x - patch_h, stride):
            for w in range(0, source_y - patch_w, stride):
                x = h
                y = w
                patch_s = source[x:x+patch_h,y:y+patch_w]
                patch_source.append(patch_s)

                x *= scale
                y *= scale

                patch_t = target[x:x+patch_h*scale, y:y+patch_w*scale]
                patch_target.append(patch_t)

    np.random.seed(36)
    np.random.shuffle(patch_source)
    np.random.seed(36)
    np.random.shuffle(patch_target)

    return np.array(patch_source), np.array(patch_target)

def generate_testset(source_list, target_list, args) :

    source = cv2.imread(source_list)
    image_source = source / 127.5 - 1

    target = cv2.imread(target_list)
    image_target = target / 127.5 - 1

    return np.array(image_source), np.array(image_target)

# TensorFlow Better Bicubic Downsample
# https://github.com/trevor-m/tensorflow-bicubic-downsample
def bicubic_kernel(x, a=-0.5):
  """https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic"""
  if abs(x) <= 1:
    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
  elif 1 < abs(x) and abs(x) < 2:
    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
  else:
    return 0

def build_filter(factor):
  size = factor*4
  k = np.zeros((size))
  for i in range(size):
    x = (1/factor)*(i- np.floor(size/2) +0.5)
    k[i] = bicubic_kernel(x)
  k = k / np.sum(k)
  # make 2d
  k = np.outer(k, k.T)
  k = tf.constant(k, dtype=tf.float32, shape=(size, size, 1, 1))
  return tf.concat([k, k, k], axis=2)

def apply_bicubic_downsample(x, filter, factor):
  """Downsample x by a factor of factor, using the filter built by build_filter()
  x: a rank 4 tensor with format NHWC
  filter: from build_filter(factor)
  factor: downsampling factor (ex: factor=2 means the output size is (h/2, w/2))
  """
  # using padding calculations from https://www.tensorflow.org/api_guides/python/nn#Convolution
  filter_height = factor*4
  filter_width = factor*4
  strides = factor
  pad_along_height = max(filter_height - strides, 0)
  pad_along_width = max(filter_width - strides, 0)
  # compute actual padding values for each side
  pad_top = pad_along_height // 2
  pad_bottom = pad_along_height - pad_top
  pad_left = pad_along_width // 2
  pad_right = pad_along_width - pad_left
  # apply mirror padding
  x = tf.pad(x, [[0,0], [pad_top,pad_bottom], [pad_left,pad_right], [0,0]], mode='REFLECT')
  # downsampling performed by strided conv
  x = tf.nn.depthwise_conv2d(x, filter=filter, strides=[1,strides,strides,1], padding='VALID')
  return x




#---------------------------- unused ----------------------
def load_and_save_data(args, logflag):
    """make HR and LR data. And save them as npz files"""
    assert os.path.isdir(args.data_dir) is True, 'Directory specified by data_dir does not exist or is not a directory'

    all_file_path = glob.glob(args.data_dir + '/*')
    assert len(all_file_path) > 0, 'No file in the directory'

    ret_HR_image = []
    ret_LR_image = []

    for file in all_file_path:
        img = cv2.imread(file)
        filename = file.rsplit('/', 1)[-1]

        # crop patches if flag is true. Otherwise just resize HR and LR images
        if args.crop:
            for _ in range(args.num_crop_per_image):
                img_h, img_w, _ = img.shape

                if (img_h < args.crop_size) or (img_w < args.crop_size):
                    print('Skip crop target image because of insufficient size')
                    continue

                HR_image = crop(img, args)
                LR_crop_size = np.int(np.floor(args.crop_size / args.scale_SR))
                LR_image = cv2.resize(HR_image, (LR_crop_size, LR_crop_size), interpolation=cv2.INTER_LANCZOS4)

                cv2.imwrite(args.HR_data_dir + '/' + filename, HR_image)
                cv2.imwrite(args.LR_data_dir + '/' + filename, LR_image)

                ret_HR_image.append(HR_image)
                ret_LR_image.append(LR_image)
        else:
            HR_image = cv2.resize(img, (args.HR_image_size, args.HR_image_size), interpolation=cv2.INTER_LANCZOS4)
            LR_image = cv2.resize(img, (args.LR_image_size, args.LR_image_size), interpolation=cv2.INTER_LANCZOS4)

            cv2.imwrite(args.HR_data_dir + '/' + filename, HR_image)
            cv2.imwrite(args.LR_data_dir + '/' + filename, LR_image)

            ret_HR_image.append(HR_image)
            ret_LR_image.append(LR_image)

    assert len(ret_HR_image) > 0 and len(ret_LR_image) > 0, 'No availale image is found in the directory'
    log(logflag, 'Data process : {} images are processed'.format(len(ret_HR_image)), 'info')

    ret_HR_image = np.array(ret_HR_image)
    ret_LR_image = np.array(ret_LR_image)

    if args.data_augmentation:
        LR_flip, HR_flip = data_augmentation(ret_LR_image, ret_HR_image, aug_type='horizontal_flip')
        LR_rot, HR_rot = data_augmentation(ret_LR_image, ret_HR_image, aug_type='rotation_90')

        ret_LR_image = np.append(ret_LR_image, LR_flip, axis=0)
        ret_HR_image = np.append(ret_HR_image, HR_flip, axis=0)
        ret_LR_image = np.append(ret_LR_image, LR_rot, axis=0)
        ret_HR_image = np.append(ret_HR_image, HR_rot, axis=0)

        del LR_flip, HR_flip, LR_rot, HR_rot

    np.savez(args.npz_data_dir + '/' + args.HR_npz_filename, images=ret_HR_image)
    np.savez(args.npz_data_dir + '/' + args.LR_npz_filename, images=ret_LR_image)

    return ret_HR_image, ret_LR_image


def load_npz_data(FLAGS):
    """load array data from data_path"""
    return np.load(FLAGS.npz_data_dir + '/' + FLAGS.HR_npz_filename)['images'], \
           np.load(FLAGS.npz_data_dir + '/' + FLAGS.LR_npz_filename)['images']


def load_inference_data(FLAGS):
    """load data from directory for inference"""
    assert os.path.isdir(FLAGS.data_dir) is True, 'Directory specified by data_dir does not exist or is not a directory'

    all_file_path = glob.glob(FLAGS.data_dir + '/*')
    assert len(all_file_path) > 0, 'No file in the directory'

    ret_LR_image = []
    ret_filename = []

    for file in all_file_path:
        img = cv2.imread(file)
        img = normalize_images(img)
        ret_LR_image.append(img[0][np.newaxis, ...])

        ret_filename.append(file.rsplit('/', 1)[-1])

    assert len(ret_LR_image) > 0, 'No available image is found in the directory'

    return ret_LR_image, ret_filename
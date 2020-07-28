import time
import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import sklearn.metrics as mt
from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument(
    '--flist', default='', type=str,
    help='The filenames of image to be processed: input, mask, output.')
parser.add_argument(
    '--image_height', default=-1, type=int,
    help='The height of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--image_width', default=-1, type=int,
    help='The width of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--checkpoint_dir', default='', type=str,
    help='The directory of tensorflow checkpoint.')

def cal_rmse(inp, out, mask):
    diff = ((inp - out) * (mask/255)) ** 2
    diff[diff == 0] = np.nan
    return np.nanmean(diff)**0.5

if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] =''
    args = parser.parse_args()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, args.image_height, args.image_width*2, 1))
    output = model.build_server_graph(FLAGS, input_image_ph)
    # output = (output + 1.) * 127.5
    #minV = FLAGS.min_dem
    #maxV = FLAGS.max_dem
    #output = (output + 1.)*(maxV - minV)/2 + minV
    output = tf.reverse(output, [-1])
    #output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    with open(args.flist, 'r') as f:
        lines = f.read().splitlines()[:100]
    t = time.time()
    rmses = []
    for line in lines:
        image, mask, out = line.split()
        base = os.path.basename(mask)
        image = cv2.imread(image, -1)
        raw_image = image
        im_min = image.min()
        im_max = image.max()
        image = cv2.normalize(image, None, 1, -1, cv2.NORM_MINMAX, cv2.CV_32F)
        if len(image.shape) < 3:
            image = image[..., np.newaxis]
        mask = cv2.imread(mask, -1)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        if len(mask.shape) < 3:
            mask = mask[..., np.newaxis]
        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 4
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        # load pretrained model
        result = sess.run(output, feed_dict={input_image_ph: input_image})
        print('Processed: {}'.format(out))
        result = result[0][:, :, ::-1]
        #result = (im_max-im_min)*(result - result.min())/(result.max()-result.min()) + im_min
        result = (im_max-im_min)*(result +1.)/2 + im_min
        # print(raw_image.shape, result.shape, mask.shape)
        rmse = cal_rmse(raw_image, result[:,:,0], mask[0,:,:,0])
        print('Rmse: {}'.format(rmse))
        rmses.append(rmse)
        cv2.imwrite(out, result)

    print('RMSE: ', np.mean(rmses))
    print('Time total: {}'.format(time.time() - t))

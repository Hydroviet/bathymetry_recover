import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()

    model = InpaintCAModel()
    image = cv2.imread(args.image, -1)
    original_shape = image.shape
    image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))
    im_min = image.min()
    im_max = image.max()
    image = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_32F)
    ho, wo = image.shape
    print('Shape of image: {}'.format(image.shape))
    image = np.pad(image, ((0, 7), (0, 7)), 'symmetric')
    if len(image.shape) < 3:
        image = image[..., np.newaxis]
    mask = cv2.imread(args.mask, -1)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask = cv2.resize(mask, (mask.shape[1]//4, mask.shape[0]//4))
    mask = np.pad(mask, ((0, 7), (0, 7)), 'symmetric')
    if len(mask.shape) < 3:
        mask = mask[..., np.newaxis]
        
    assert image.shape == mask.shape

    print('Shape of image1: {}'.format(image.shape))
    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image2: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
#         minV = FLAGS.min_dem
#         maxV = FLAGS.max_dem
#         output = (output + 1.)*(maxV - minV)/2 + minV
        output = tf.reverse(output, [-1])
#         output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        print('Checkpoint dir: ', args.checkpoint_dir)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        result = result[0][:, :, ::-1]
        result = (im_max-im_min)*(result - result.min())/(result.max()-result.min()) + im_min
        result = result[:ho, :wo, :]
        result = cv2.resize(result, original_shape[::-1])
        print('Result shape: ', result.shape)
        cv2.imwrite(args.output, result)

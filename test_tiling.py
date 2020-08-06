import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import math
from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--tile_height', default=256, type=int, help='Size of tile')
parser.add_argument('--tile_width', default=256, type=int, help='Size of tile')
parser.add_argument('--output', default='output.tif', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')


def tiling_image(image, tileSizeX=256, tileSizeY=256):
    numTilesX = 2
    numTilesY = 2
    tileSizeX = math.ceil(image.shape[1]/numTilesX)
    tileSizeY = math.ceil(image.shape[0]/numTilesY)
    makeLastPartFull = True; # in case you need even siez
    tiles = []

    for nTileX in range(numTilesX):
        for nTileY in range(numTilesY):
            startX = nTileX*tileSizeX
            endX = startX + tileSizeX
            startY = nTileY*tileSizeY
            endY = startY + tileSizeY;

            if(endY > image.shape[0]):
                endY = image.shape[0]

            if(endX > image.shape[1]):
                endX = image.shape[1]

            if( makeLastPartFull == True and (nTileX == numTilesX-1 or nTileY == numTilesY-1) ):
                startX = endX - tileSizeX
                startY = endY - tileSizeY

            currentTile = image[startY:endY, startX:endX]
            tiles.append((currentTile, (startY, endY), (startX, endX)))
    return tiles

def process_tile(image, mask):
    if np.sum(mask) == 0:
        return None
    if len(image.shape) < 3:
        image = image[..., np.newaxis]
    if len(mask.shape) < 3:
        mask = mask[..., np.newaxis]
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)
    return input_image

def predict_tile(image, mask, model, reuse=False):
    if np.sum(mask) == 0:
        return None
    raw_shape = image.shape
    image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    mask = cv2.resize(mask, (mask.shape[1]//2, mask.shape[0]//2))
    ho, wo = image.shape
    image = np.pad(image, ((0, 7), (0, 7)), 'symmetric')
    if len(image.shape) < 3:
        image = image[..., np.newaxis]
    mask = np.pad(mask, ((0, 7), (0, 7)), 'symmetric')
    if len(mask.shape) < 3:
        mask = mask[..., np.newaxis]
        
    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image,reuse=reuse)
        output = tf.reverse(output, [-1])
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # print('Checkpoint dir: ', args.checkpoint_dir)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        # print('Model loaded.')
        result = sess.run(output)
        result = result[0][:, :, ::-1]
        # result = (im_max-im_min)*(result +1)/2 + im_min
        result = result[:ho, :wo, :]
        result = cv2.resize(result, raw_shape[::-1])
    return result
            
if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()    
    image = cv2.imread(args.image, -1)
    mask = cv2.imread(args.mask, -1)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    im_min = image.min()
    im_max = image.max()
    image = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_32F)
    
    image_tiles = tiling_image(image)
    mask_tiles = tiling_image(mask)
    result = np.zeros(image.shape)
    ntiles = len(image_tiles)
    for i in range(ntiles):
        startY, endY = image_tiles[i][1]
        startX, endX = image_tiles[i][2]
        if i == 0:
            predicted_tile = predict_tile(image_tiles[i][0], mask_tiles[i][0], model, reuse=False)
        else:
            predicted_tile = predict_tile(image_tiles[i][0], mask_tiles[i][0], model, reuse=True)
        if predicted_tile is None:
            result[startY:endY, startX:endX] = image[startY:endY, startX:endX]
            continue
        result[startY:endY, startX:endX] = predicted_tile[:,:]
    
    result = (im_max-im_min)*(result - result.min())/(result.max()-result.min()) + im_min
    # result = (im_max-im_min)*(result +1.)/2 + im_min
    print('Writing result to {}'.format(args.output))
    cv2.imwrite(args.output, result)
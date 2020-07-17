import cv2
from concurrent.futures import ThreadPoolExecutor
from skimage import measure
import rasterio
import glob
import random
import numpy as np
import os
import argparse
import time
import neuralgym as ng
import tensorflow as tf
from tqdm import tqdm

def gen_mask(img, min_size=64*64, max_size=128*128):
    def get_mask(img, level):
        mask = (img <= level)*1
        all_labels = measure.label(mask, background=0)
        labels = np.unique(all_labels)
        connected = []
        for i in range(1, len(labels)):
            label = labels[i]
            connected.append((np.sum(all_labels == label), label))
        if len(connected) == 0:
            return 0, None
        mask = sorted(connected)[::-1][0]
        return mask[0], (all_labels == mask[1])*1
    
    minL = img.min()
    maxL = img.max()
    res = []
    while minL <= maxL:
        mid = (minL + maxL)//2
        mask = get_mask(img, mid)
        if mask[0] < min_size:
            minL = mid  + 1
        elif mask[0] > max_size:
            maxL = mid -1
        else:
            res = mask
            break
    if len(res) == 0:
        return img, None
    return (img, res[1])

def gen_mask_batch(batch_data, dest, start):
    indx = start
    for data in batch_data:
        img, mask = gen_mask(data)
        if mask is not None:
            mask = mask * 255
            filename_input = os.path.join(dest, '{}.tif'.format(str(indx).zfill(6)))
            filename_mask = os.path.join(dest, '{}mask.png'.format(str(indx).zfill(6)))
            if not cv2.imwrite(filename_input, img):
                print('Error writing file {}'.format(filename_input))
            if not cv2.imwrite(filename_mask, mask):
                print('Error writing mask file {}'.format(filename_mask))
        indx += 1

def preprocess(imgs):
    pimgs = list(map(lambda img: cv2.medianBlur(img, 5), imgs))
    pimgs = list(filter(lambda img: np.quantile(img, 0.90) - np.quantile(img, 0.05) >= 30 and img.min() >= 5, imgs))
    return pimgs

def read_batch(src, dest, batch_size=8, nthreads=8):
    print(src, dest)
    if not os.path.exists(dest):
        os.mkdir(dest)
        
    print('Read from: ', src)
    files = glob.glob(os.path.join(src, '*.tif'))
    num = len(files)
    data = ng.data.DataFromFNames(files, [256, 256, 1], nthreads=nthreads)
    batch_data = data.data_pipeline(batch_size)
    sess = tf.Session(config=tf.ConfigProto())
    tf.train.start_queue_runners(sess)
    stime = time.time()
    for i in tqdm(range(num//batch_size)):
        imgs = list(sess.run(batch_data))
        imgs = preprocess(imgs)
        gen_mask_batch(imgs, dest, batch_size*i)
    print('Total time {}'.format(time.time() - stime))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='', type=str, help='Path store source data')
    parser.add_argument('--dest', default='', type=str, help='Path store generated data')
    parser.add_argument('--batch_size', default='', type=str, help='Batch size to read data')
    args = parser.parse_args()
    read_batch(args.src, args.dest, int(args.batch_size))
    

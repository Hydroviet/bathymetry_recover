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

D_LEVEL = 50

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

def get_depth(img, mask):
    m = img*mask
    m[m==0] = np.nan
    return np.nanmax(m) - np.nanmin(m)
            
def adjust_level(img, level, size, dlevel, min_size=64*64, max_size=128*128, max_depth=50):
    masks = []
    avaiable_size = [size]
    for hi in range(int(level) - dlevel, int(level) + dlevel, 10):
        size_m, m = get_mask(img, hi)
        if m is not None:
            depth = get_depth(img, m)
            if size_m not in avaiable_size and size_m >= min_size and size_m <= max_size and depth <= max_depth:
                masks.append(m)
                avaiable_size.append(size_m)
    return masks
            
def gen_mask(img, min_size=64*64, max_size=128*128, max_depth=50):
    minL = img.min()
    maxL = img.max()
    res = None
    level = 0
    while minL <= maxL:
        mid = (minL + maxL)//2
        size_mask, mask = get_mask(img, mid)
        depth = get_depth(img, mask)
        if size_mask < min_size:
            minL = mid  + 1
        elif size_mask > max_size or depth > max_depth:
            maxL = mid -1
        else:
            res = mask
            level = mid
            break
    if res is None:
        return img, None
    masks = [res] + adjust_level(img, level, size_mask, D_LEVEL)
    return (img, masks)

def rotate(img, mask):
    mask = mask*255
    yield (img, mask)
    angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
    for ang in angles:
        rimg = cv2.rotate(img, ang)
        rmask = cv2.rotate(mask, ang)
        yield (rimg, rmask)

def flip(img, mask):
    mask = mask*255
    for fcode in range(-1, 1, 1):
        fimg = cv2.flip(img, fcode)
        fmask = cv2.flip(mask, fcode)
        yield (fimg, fmask)

def write_to_disk(img, mask, dest, indx):
    for rim, rmask in rotate(img, mask):
        filename_input = os.path.join(dest, '{}.tif'.format(str(indx).zfill(6)))
        filename_mask = os.path.join(dest, '{}mask.png'.format(str(indx).zfill(6)))
        if not cv2.imwrite(filename_input, rim):
            print('Error writing file {}'.format(filename_input))
        if not cv2.imwrite(filename_mask, rmask):
            print('Error writing mask file {}'.format(filename_mask))
        indx = indx + 1

    for fim, fmask in flip(img, mask):
        filename_input = os.path.join(dest, '{}.tif'.format(str(indx).zfill(6)))
        filename_mask = os.path.join(dest, '{}mask.png'.format(str(indx).zfill(6)))
        if not cv2.imwrite(filename_input, fim):
            print('Error writing file {}'.format(filename_input))
        if not cv2.imwrite(filename_mask, fmask):
            print('Error writing mask file {}'.format(filename_mask))
        indx = indx + 1
    return indx

def gen_mask_batch(batch_data, dest, start):
    indx = start
    for data in batch_data:
        img, masks = gen_mask(data)
        if masks is not None:
            for mask in masks:
                indx = write_to_disk(img, mask, dest, indx)
    return indx

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
        gen_mask_batch(imgs, dest, i*batch_size*10*7)
    print('Total time {}'.format(time.time() - stime))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='', type=str, help='Path store source data')
    parser.add_argument('--dest', default='', type=str, help='Path store generated data')
    parser.add_argument('--batch_size', default='', type=str, help='Batch size to read data')
    args = parser.parse_args()
    read_batch(args.src, args.dest, int(args.batch_size))
    

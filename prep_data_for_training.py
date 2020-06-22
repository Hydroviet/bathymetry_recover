import cv2
from concurrent.futures import ThreadPoolExecutor
from skimage import measure
import rasterio
import glob
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os
import argparse

def create_train_val(source_path):
    print('READING DATA FROM {}'.format(source_path))
    data = []
    for file in glob.glob(os.path.join(source_path, '*.tif')):
      img = cv2.imread(file, -1)
      data.append(img)
    print(len(data))
    data_resize = list(map(lambda x: cv2.resize(x, (256, 256)), data))
    n = len(data_resize)
    train, test = train_test_split(data_resize, train_size=int(n*0.8), test_size=int(0.2*n))
    print('TRAIN: {}, VAL: {}'.format(len(train), len(test)))
    return train, test

def random_free_mask(img, min_size=64*64, max_size=128*128):
    level = np.random.uniform(np.median(img) - 2*np.std(img), np.median(img) + 2*np.std(img))
    mask = (img <= level)*1
    all_labels = measure.label(mask, background=0)
    labels = np.unique(all_labels)
    connected = []
    for i in range(1, len(labels)):
        label = labels[i]
        connected.append((np.sum(all_labels == label), label))
    masks = list(filter(lambda x: x[0] >= min_size and x[0] <= max_size, connected))
    masks = list(map(lambda x: (img, (all_labels == x[1])*1), masks))
    return masks

def random_bbox(img_shape=(256, 256), mask_shape=(128, 128)):
    img_height, img_width = img_shape
    mask_height, mask_width = mask_shape
    max_top = img_height - mask_height
    max_left = img_width - mask_width
    top = int(random.uniform(0, max_top))
    left = int(random.uniform(0, max_left))
    
    mask = np.zeros((img_height, img_width), np.float32)
    mask[top:top+mask_height, left:left+mask_width] = 1
    return mask

def write_data(data, dir_name):
    MASK_PER_IMG = 10
    indx = 1
    for d in data:
        for time in range(MASK_PER_IMG):
            #num_masks = 0
            #loop = 0
        #while num_masks < MASK_PER_IMG and loop < 100:
            gen_res = random_free_mask(d)
            #loop += 1
            #num_masks += len(gen_res)
            for img, mask in gen_res:
                print('Process image {}{}'.format(dir_name, indx))
                mask = mask * 255
                filename_input = os.path.join(dir_name, '{}.tif'.format(str(indx).zfill(3)))
                filename_mask = os.path.join(dir_name, '{}mask.png'.format(str(indx).zfill(3)))
                if not cv2.imwrite(filename_input, img):
                    print('Error writing file {}'.format(filename_input))
                if not cv2.imwrite(filename_mask, mask):
                    print('Error writing mask file {}'.format(filename_mask))
                indx += 1

def gen(args):
    source_path = args.src
    dest_path = args.dest
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
        
    train, val = create_train_val(source_path)
    
    train_path = os.path.join(dest_path, 'train/')
    val_path = os.path.join(dest_path, 'validation/')
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    
    # print('Write train data to {}'.format(train_path))
    with ThreadPoolExecutor(max_workers=2) as e:
        e.submit(write_data, train, train_path)
        e.submit(write_data, val, val_path)
#    write_data(train, train_path)
#    write_data(val, val_path)
#     print('Write validation data to {}'.format(val_path))
#     write_data(val, val_path, True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='', type=str, help='Path store source data')
    parser.add_argument('--dest', default='', type=str, help='Path store generated data')
    args = parser.parse_args()
    gen(args)
    

#!/usr/bin/python

import argparse
import os
import glob
from random import shuffle
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='./training_data', type=str,
                    help='The folder source path')
parser.add_argument('--dest', default='./data_flist', type=str,
                    help='The folder destination path')

def write_flist(src, dest, is_val = True):
    files = []
    filename = ''
    print('Reading from ', os.path.join(src))
    files = glob.glob(os.path.join(src, '*.tif'))
    files = list(map(lambda f: os.path.abspath(f), files))
    
    train_fname = os.path.join(dest, 'train.flist')

    if is_val:
        val_fname = os.path.join(dest, 'validation.flist')
        ftrain, fval = train_test_split(files, test_size=0.1, shuffle=True)
    
        print('Writing to {}'.format(train_fname))
        fo = open(train_fname, "w")
        fo.write("\n".join(ftrain))
        fo.close()
        print('Writing to {}'.format(val_fname))
        fo = open(val_fname, "w")
        fo.write("\n".join(fval))
        fo.close()    
    else:
        print('Writing to {}'.format(train_fname))
        fo = open(train_fname, "w")
        fo.write("\n".join(files))
        fo.close()

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.dest):
        os.mkdir(args.dest)
    write_flist(args.src, args.dest)
    


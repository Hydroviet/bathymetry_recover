#!/usr/bin/python

import argparse
import os
import glob
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--src', default='./training_data', type=str,
                    help='The folder source path')
parser.add_argument('--dest', default='./data_flist', type=str,
                    help='The folder destination path')

def write_flist(src, dest, name, mask=True):
    files = []
    filename = ''
    print('Reading from ', os.path.join(src, name))
    if mask:
        files = glob.glob(os.path.join(src, name, '*.png'))
        filename = os.path.join(dest, '{}_mask.flist'.format(name))
    else:
        files = glob.glob(os.path.join(src, name, '*.tif'))
        filename = os.path.join(dest, '{}.flist'.format(name))
   
#     files = list(map(lambda f: os.path.abspath(f), files))
    print('Writing to ', filename)
    fo = open(filename, "w")
    fo.write("\n".join(sorted(files)))
    fo.close()
    

if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.dest):
        os.mkdir(args.dest)
    write_flist(args.src, args.dest, 'train', False)
    write_flist(args.src, args.dest, 'train', True)
    write_flist(args.src, args.dest, 'validation', False)
    write_flist(args.src, args.dest, 'validation', True)
    


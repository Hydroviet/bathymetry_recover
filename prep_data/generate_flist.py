#!/usr/bin/python

import argparse
import os
import glob
from random import shuffle
from sklearn.model_selection import train_test_split

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
parser = argparse.ArgumentParser()
parser.add_argument('--src', default='./training_data', type=str,
                    help='The folder source path')
parser.add_argument('--dest', default='./data_flist', type=str,
                    help='The folder destination path')
parser.add_argument('--val',type=str2bool, nargs='?',
                        const=True, default=False,
                    help='True if train/validation split')
parser.add_argument('--test',type=str2bool, nargs='?',
                        const=True, default=False,
                    help='True if generate flist for test')
parser.add_argument('--fname', default='', type=str,
                    help='File name in case no split train/val')


def cmask(files, fname):
    masks = list(map(lambda f: f[:-4]+'mask.png', files))
    f = open(fname, 'w')
    f.write('\n'.join(masks))
    f.close()

def write_flist(src, dest, is_val, fname=""):
    files = []
    filename = ''
    print('Reading from ', os.path.join(src))
    files = glob.glob(os.path.join(src, '*.tif'))
    files = list(map(lambda f: os.path.abspath(f), files))
    #files = files[:1000]
    if is_val:
        train_fname = os.path.join(dest, 'train.flist')
        val_fname = os.path.join(dest, 'validation.flist')
        ftrain, fval = train_test_split(files, test_size=0.1, shuffle=True)
        print('Train size: {}/ Val size {}'.format(len(ftrain), len(fval)))
    
        print('Writing to {}'.format(train_fname))
        fo = open(train_fname, "w")
        fo.write("\n".join(ftrain))
        fo.close()
        print('Create mask for train:' )
        cmask(ftrain, train_fname[:-6]+'_mask.flist')
        
        print('Writing to {}'.format(val_fname))
        fo = open(val_fname, "w")
        fo.write("\n".join(fval))
        fo.close()    
        print('Create maks for val:')
        cmask(fval, val_fname[:-6]+'_mask.flist') 
    else:
        fname = os.path.join(dest, fname)
        print('Writing to {}'.format(fname))
        fo = open(fname, "w")
        fo.write("\n".join(files))
        fo.close()
        cmask(files, fname[:-6]+'_mask.flist')
        
def write_flist_test(src, fout):
    
    files = sorted(glob.glob(os.path.join(src, 'inputs/*.tif')))
    masks = [file[:-4]+'mask.png' for file in files]
    outputs = list(map(lambda x: x.split('/')[-1], files))
    lines = [' '.join([fi, m, os.path.join(src, 'outputs/'+o)] )
                  for fi,m,o in zip(files, masks, outputs)]
    fo = open(os.path.join(fout, 'test.flist'), "w")
    fo.write("\n".join(lines))
    fo.close()


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.dest):
        os.mkdir(args.dest)
    print(args)
    if args.test:
        write_flist_test(args.src, args.dest)
    else:
        write_flist(args.src, args.dest, args.val, args.fname)
    


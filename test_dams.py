import os
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd

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
parser.add_argument('--dams_list', default='', type=str, help='The file of list of dams to be generated.')
parser.add_argument('--checkpoint_dir', type=str, help='The directory of tensorflow checkpoint.')
parser.add_argument('--input_dir', type=str, help='The directory of input files included dem and mask')
parser.add_argument('--output_dir', type=str, help='The directory of output files, if not provided, output will stored in input_dir')
parser.add_argument('--tiling', type=str2bool, nargs='?',
                        const=True, default=False, help='True if want test by generated image by tiles')

def run_gen(dams, checkpoint_dir, input_dir, output_dir, tiling=False):
    if tiling:
        cmd = "python test_tiling.py"
    else:
        cmd = "python test.py"
    if os.path.exists('tlogs'):
        os.remove('tlogs')
    for i, name in tqdm(enumerate(dams)):
        os.system("echo LOGS: {} >> tlogs".format(name))
        inp = os.path.join(input_dir, "{}.tif".format(name))
        mask = os.path.join(input_dir, "{}_mask.png".format(name))
        out = os.path.join(output_dir, "{}_out.tif".format(name))
        cmd = cmd + " --image {} --mask {} --checkpoint_dir {} --output {} 2>>tlogs".format(inp, mask, checkpoint_dir, out)
        os.system(cmd)
        
REF_DIR = 'data/aus_test/ref_bathymetry/'
def report_result(dams, input_dir, output_dir):
    rmses = dict()
    maes = dict()
    for x in dams:
        ref = os.path.join(REF_DIR, "{}.tif".format(x))
        mask = os.path.join(input_dir, "{}_mask.png".format(x))
        out = os.path.join(output_dir, "{}_out.tif".format(x))
        bathymetry = cv2.imread(ref, -1)
        img_out = cv2.imread(out, -1)
        img_out = cv2.resize(img_out, bathymetry.shape[::-1], interpolation = cv2.INTER_AREA)
        mask = cv2.imread(mask.format(x), -1)
        if len(mask.shape) >= 3:
            mask = mask[:,:,0]
        mask = cv2.resize(mask, bathymetry.shape[::-1], interpolation = cv2.INTER_AREA)
        mask =  mask/255
        mask[mask < 1] = 0
        bathymetry[bathymetry == 0] = np.nan
        diff = img_out*mask - bathymetry*mask
        diff[diff == 0] = np.nan   
        
        rmse = ((img_out*mask - bathymetry*mask)**2)
        rmse = np.nanmean(rmse)**0.5
        mae = np.abs(img_out*mask - bathymetry*mask)
        mae = np.nanmean(mae)

        rmses[x] = rmse
        maes[x] = mae
    return {
        'rmse': np.mean(list(rmses.values())), 
        'mae': np.mean(list(maes.values())),
        'report': pd.DataFrame([rmses, maes], columns=rmses.keys(), index=['rmse', 'mae'])
    }
    
    
if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    f = open(args.dams_list, 'r')
    dams  = list(map(lambda line: line.split(','), f.read().splitlines()))
    dams = [item for sublist in dams for item in sublist]
    dams = [item.strip() for item in dams]
    f.close()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    print('GENERATING TERRAIN')
    run_gen(dams, args.checkpoint_dir, args.input_dir, args.output_dir, args.tiling)
    print('RESULT')
    result = report_result(dams, args.input_dir, args.output_dir)
    print('Mean RMSE: {}\nMean MAE: {}'.format(result['rmse'], result['mae']))
    print(result['report'].to_string())
    
    
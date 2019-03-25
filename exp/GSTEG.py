#!/usr/bin/env python
import sys
import pdb
import traceback
#sys.path.insert(0, '..')
sys.path.insert(0, '.')
from main import main
from bdb import BdbQuit
import subprocess
subprocess.Popen('find ./exp/.. -iname "*.pyc" -delete'.split())

args = [
    '--name', __file__.split('/')[-1].split('.')[0],  # name is filename
    '--cache-dir', '/mnt/disk3/hubertt/cr_caches/',
    '--rgb-data', '/mnt/disk3/hubertt/Charades_v1_rgb/',
    '--rgb-pretrained-weights', '/mnt/disk3/hubertt/rgb_i3d_pretrained.pt',
    '--resume', '/mnt/disk3/hubertt/cr_caches/' + __file__.split('/')[-1].split('.')[0] + '/model.pth.tar',
    '--train-file', './exp/Charades_v1_train.csv',
    '--val-file', './exp/Charades_v1_test.csv',
    '--groundtruth-lookup', './utils/groundtruth.p'    
#'--evaluate',
]
sys.argv.extend(args)
try:
    main()
except BdbQuit:
    sys.exit(1)
except Exception:
    traceback.print_exc()
    print('')
    pdb.post_mortem()
    sys.exit(1)



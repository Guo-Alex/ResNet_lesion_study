'''
Filename: converted_to_spatial_representations.py
function: JPEG to spatial; And save as npy file
date: 2017-06-08
'''
import os
import sys
sys.path.append('/home/lgq/Workspace/Contribution_of_CIHW2018_paper/data_preparation/toolbox')
import numpy as np
from jpeg import jpeg

path1 = 'train/cover'
path2 = 'train/stego_j-uniward_40'
path3 = 'test/cover'
path4 = 'test/stego_j-uniward_40'
numdata = 250000

for path in [path1, path2, path3, path4]:
    flist = []
    for (dirpath,dirnames,filenames) in os.walk('../originals/'+path):
        flist = filenames

    for index in range(numdata):
        imc = jpeg('../originals/'+path+'/'+flist[index]).getSpatial()
        np.save('../spatial_representations/'+path+'/'+flist[index].split('.')[0] , imc)
        if (index+1)%100 == 0:
            print '{0}: {1} done'.format(path, str(index+1))

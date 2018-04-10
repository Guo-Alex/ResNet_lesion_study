#!/usr/bin/python
#The program generate the lmdb dataset for the Caffe input
#Implement in python-2.7

import caffe
import lmdb
import os
import numpy as np
from caffe.proto import caffe_pb2

#prepare for the image dir and names
#the images should be decompressed to spatial format first
path1 = '../spatial_representations/train/cover'
path2 = '../spatial_representations/train/stego_j-uniward_40'
path3 = '../spatial_representations/test/cover'
path4 = '../spatial_representations/test/stego_j-uniward_40'

#basic setting
train = '../lmdb/train'
test = '../lmdb/test'
numdata=250000
batch_size = 10000

for [cover_path, stego_path, lmdb_file] in [[path1, path2, train], \
                                            [path3, path4, test]]:
    #image name    
    image_names_string=os.popen("ls "+cover_path).read()
    image_names=image_names_string.split('\n')[0:-1]

    # create the lmdb file
    lmdb_env = lmdb.open(lmdb_file, map_size=int(1e13))
    lmdb_txn = lmdb_env.begin(write=True)
    datum = caffe_pb2.Datum()

    image_id= 0
    for item_id in range(2*numdata):
        #prepare the data and label
        #read in cover-stego pair
        if(item_id % 2) == 0:
            image_path=os.path.join(cover_path,image_names[image_id])
            data_temp=np.load(image_path)
            data=data_temp[np.newaxis,:,:]
            label=1
        else :
            image_path=os.path.join(stego_path,image_names[image_id])
            data_temp=np.load(image_path)
            data=data_temp[np.newaxis,:,:]
            label=0
            image_id+=1

        # save in datum
        datum = caffe.io.array_to_datum(data, label)
        keystr = '{:0>8d}'.format(item_id)
        lmdb_txn.put( keystr, datum.SerializeToString() )

        # write batch
        if(item_id + 1) % batch_size == 0:
            lmdb_txn.commit()
            lmdb_txn = lmdb_env.begin(write=True)
            print '{0}: {1}'.format(lmdb_file, str(item_id + 1))

    # write last batch
    if (item_id+1) % batch_size != 0:
        lmdb_txn.commit()
        print 'last batch'
        print (item_id + 1)

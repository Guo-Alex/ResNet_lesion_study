import caffe
from caffe.proto import caffe_pb2

import lmdb
# import cv2
import numpy as np

lmdb_env = lmdb.open('train', readonly=True)
lmdb_txn = lmdb_env.begin() 
lmdb_cursor = lmdb_txn.cursor() 
datum = caffe_pb2.Datum() 
num=0

for key, value in lmdb_cursor:
    #    datum.ParseFromString(value) 

    #    label = datum.label
    #    data = caffe.io.datum_to_array(datum)
    num = num + 1
    if num%50000 == 0:
        print num
    #    print data.shape
    #    print datum.channels
    #    image = data.transpose(1, 2, 0)
    #    cv2.imshow('cv2.png', image)
    #    cv2.waitKey(0)

print 'num is {}'.format(num)
# cv2.destroyAllWindows()
lmdb_env.close()

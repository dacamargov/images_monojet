from __future__ import print_function

import time
import warnings
warnings.filterwarnings("ignore")

import os
import theano
theano.config.openmp = True
theano.config.floatX = 'float32'
os.environ['THEANO_FLAGS'] = 'device=cpu,blas.ldflags=-lblas -lgfortran,mode=FAST_RUN'
os.environ['OMP_NUM_THREADS'] = '6'

from TrainClassifiers import main

params = {"input_path"              : "../data/",
          "output_path"             : "outputs/",
          "inputs"                  : "2d",
          "model_name"              : "2d_01",        
          "nb_epoch"                : 5,
          "batch_size"              : 512,
          "name_train"              : "train_imgs.h5" ,
          "name_test"               : "test_imgs.h5",
}

time_start = time.clock()
main(params)
time_elapsed = (time.clock() - time_start)
print("Running time =  "+str(time_elapsed/60.)+" minutes")

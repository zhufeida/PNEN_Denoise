import argparse
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import scipy.misc
import numpy as np
import tensorflow as tf
from model import mymodel
from tensorflow.contrib import slim
tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser(description='')

#### specify by train.sh
parser.add_argument('--checkpoint_path', default='checkpoints_path', help='models are saved here')
parser.add_argument('--results_dir', default='results_dir', help='results are saved here')
parser.add_argument('--noise_dir', default='noise_dir', help='noise images are saved here')
parser.add_argument('--model_def',  required=True, help='models are saved here')
parser.add_argument('--sigma', type=float,default=15, help='sigma')

#### default
### patch 80
parser.add_argument('--test_patch_size', type=int, default=96, help='then crop to this size')
### step 50
parser.add_argument('--test_step_size', type=int, default=40, help='# images in batch')

args = parser.parse_args()

def main(_):

    tfconfig = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        if args.model_def=='RDN':
            model = mymodel(sess, args)
            model.build_model_inference_2()
            t_vars = tf.trainable_variables()
            # slim.model_analyzer.analyze_vars(t_vars, print_info=True)
            
            model.inference_2('Set12') 
            model.inference_2('BSD68') 
            model.inference_2('Urban100')
        else:
            model = mymodel(sess, args)
            model.build_model_inference()
            t_vars = tf.trainable_variables()
            # slim.model_analyzer.analyze_vars(t_vars, print_info=True)
            
            model.inference('Set12') 
            model.inference('BSD68') 
            model.inference('Urban100')


if __name__ == '__main__':
    tf.app.run()









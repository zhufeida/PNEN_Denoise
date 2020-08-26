from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import skimage
import imageio
# from skimage import  color
# from module import model_try1
# from module import model_try2, model_try3, model_try4
from module import RDN,RDN_PNB, RDN_NLB
import sys
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)
import random 
import scipy.io as sio 

class mymodel(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args

        
        #self.loss_beta = 0.0001 
    def build_model(self):

        self.global_step = tf.Variable(0,trainable=False)
        self.global_step_add1_op= tf.assign_add(self.global_step,1)

        ### self.y, range [0,255]
        self.prepare_train_patch()
        self.y = self.y / 255
 
        
        ################# add noise to the ground truth image
        noise = tf.random.normal(self.y.get_shape(), mean=0, stddev=self.args.sigma /255.0)
        self.x = self.y + noise

        #################  build model
        TempModel=str_to_class(self.args.model_def)
        self.y_ , self.loss = TempModel(self.x, self.y, is_train=True,reuse=False)
        
        self.lr_input = tf.placeholder(tf.float32, shape=[], name="LearningRate")
        self.train_step = tf.train.AdamOptimizer(self.lr_input).minimize(self.loss)
        self.saver=tf.train.Saver()

        ########## test
        self.test_input = tf.placeholder(tf.float32,[1, self.args.test_patch_size, self.args.test_patch_size ,1], name='test') # range [0 255]
        self.test_output, _ = TempModel(self.test_input, self.test_input, is_train=False,reuse=True)

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        start_time = time.time()

        print(" [*] Reading checkpoint...")
        ckpt = tf.train.get_checkpoint_state(self.args.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print(ckpt.model_checkpoint_path)
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        try:
            while not coord.should_stop():
                _ = self.sess.run([self.train_step], feed_dict={self.lr_input: self.args.lr })    
                  
                self.iter_count = self.sess.run(self.global_step)

                #### converges at around 630002
                if self.iter_count > self.args.iteration_num:
                    break

                if np.mod(self.iter_count,100) == 1:
                    print("iter: [%.7d/%.7d] time: %4.4f"%(self.iter_count,self.args.iteration_num,time.time()-start_time))
                    start_time=time.time()

                if np.mod(self.iter_count, 5000) == 2:
                    self.saver.save(self.sess,os.path.join(self.args.checkpoint_dir, 'model'),global_step=self.iter_count)
                    self.test(testset='Set12')
                    # self.test(testset='BSD68') ### Set12, BSD68, Urban100
                    # self.test(testset='Urban100')
                
                self.sess.run(self.global_step_add1_op)

        except tf.errors.OutOfRangeError:
            print('Done Training.')
        finally:
            coord.request_stop()
            coord.join(threads)


    def prepare_train_patch(self):
        # tf_list=['TrainData/TrainData.tfrecords']
        tf_list=['TrainData/TrainData.tfrecords','TrainData/TrainData2.tfrecords']
        filename_queue = tf.train.string_input_producer(tf_list,num_epochs=None)
        ReadOptions = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        reader = tf.TFRecordReader(options=ReadOptions)
        _, single_example = reader.read(filename_queue)

        features = tf.parse_single_example(single_example,features={
                    'img': tf.FixedLenFeature([], tf.string),
                    'h': tf.FixedLenFeature([], tf.float32),
                    'w': tf.FixedLenFeature([], tf.float32),
                       })

        h = features['h']
        w = features['w']
        im_y = tf.cast(tf.reshape(tf.decode_raw(features['img'], tf.uint8),[h,w,1]),tf.float32)
        ###### random crop, flip, rotate
        im_y = tf.random_crop(im_y,[self.args.patch_size,self.args.patch_size,1])
        im_y = tf.image.random_flip_left_right(im_y)
        temp_k = tf.random.uniform([],minval=0,maxval=4,dtype=tf.int32) ##[minval,maxval)
        im_y = tf.image.rot90(im_y,k=temp_k)

        min_after_dequeue = 1000
        num_threads = 4 ######## threads
        batch_size = self.args.batch_size  ########## batchsize
        capacity = min_after_dequeue + (num_threads + 2) * batch_size

        self.y = tf.train.shuffle_batch([im_y],batch_size=batch_size,capacity=capacity,num_threads=num_threads,min_after_dequeue=min_after_dequeue)

    def test(self, testset):
        np.random.seed(seed=0) #### for reproduce

        total_psnr=0
        total_ssim=0 
        test_total_count=0
        start=time.time()
        im_list = glob('TestData/%s/*.png' % (testset))
        im_list = sorted(im_list)
        for i in range(len(im_list)):
            ###### convert to float [0, 1]
            im_path = im_list[i]
            im_gt = imageio.imread(im_path) / 255.0 ###### range[0,1]
            im_noise = im_gt + np.random.normal(0, self.args.sigma/255.0, im_gt.shape)
           
            ############################ test patch by patch
            def get_patch_start(length,patch_size,step_size):
                start_list=[x for x in range(0,length-patch_size,step_size)]
                start_list.append(length-patch_size)
                return start_list
            h,w=im_noise.shape
            patch_size=self.args.test_patch_size ### 144
            step_size=self.args.test_step_size  ### 100
            temp_out = np.zeros((h,w),dtype=np.float64)
            temp_count = np.zeros((h,w),dtype=np.float64)
            for h0 in get_patch_start(h,patch_size,step_size):
                for w0 in get_patch_start(w,patch_size,step_size):
                    batch_images = im_noise[np.newaxis, h0:h0+patch_size, w0:w0+patch_size, np.newaxis]
                    test_output_eval = self.sess.run(self.test_output,feed_dict={self.test_input: batch_images}) ## range [0,1]
                    test_output_eval = test_output_eval[0,:,:,0]
                    test_output_eval = np.clip(test_output_eval,0,1)
                    temp_out[h0:h0+patch_size,w0:w0+patch_size]  =temp_out[h0:h0+patch_size,w0:w0+patch_size] + test_output_eval
                    temp_count[h0:h0+patch_size,w0:w0+patch_size]=temp_count[h0:h0+patch_size,w0:w0+patch_size] + 1
            im_out = temp_out/temp_count
                      
            ###### convert back to uint8 [0 255]
            im_gt = np.uint8(im_gt*255)
            im_out = np.uint8(np.clip(im_out*255,0,255))
            #### save results
            #save_path = '%s/%s' % (result_dir,os.path.basename(im_path))
            #imageio.imsave(save_path,im_out)
            
            total_psnr=total_psnr + psnr(im_gt,im_out)
            total_ssim=total_ssim + ssim(im_gt,im_out)
        
        print ("average run time: ",(time.time()-start)/len(im_list))
        print ('%s, %d ,psnr: %.2f, ssim: %.4f' % (testset,self.args.sigma,total_psnr/len(im_list),total_ssim/len(im_list)))




    def build_model_inference(self):     
        ########## test
        TempModel=str_to_class(self.args.model_def)
        self.test_input = tf.placeholder(tf.float32,[1, self.args.test_patch_size, self.args.test_patch_size, 1], name='test') # range [0 255]
        self.test_output, _ = TempModel(self.test_input, self.test_input, is_train=False,reuse=False)
        self.saver=tf.train.Saver()
        self.saver.restore(self.sess, self.args.checkpoint_path)
        #print("test_output shape",self.test_output.get_shape())
    def inference(self,testset):

        np.random.seed(seed=0) #### for reproduce

        total_psnr=0
        total_ssim=0 
        test_total_count=0
        start=time.time()
        im_list = glob('TestData/%s/*.png' % (testset))
        im_list = sorted(im_list)
        for i in range(len(im_list)):
            ###### convert to float [0, 1]
            im_path = im_list[i]
            im_gt = imageio.imread(im_path) / 255.0 ###### range[0,1]
            im_noise = im_gt + np.random.normal(0, self.args.sigma/255.0, im_gt.shape)
           
            ############################ test patch by patch
            def get_patch_start(length,patch_size,step_size):
                start_list=[x for x in range(0,length-patch_size,step_size)]
                start_list.append(length-patch_size)
                return start_list
            h,w=im_noise.shape
            patch_size=self.args.test_patch_size ### 144
            step_size=self.args.test_step_size  ### 100
            temp_out = np.zeros((h,w),dtype=np.float64)
            temp_count = np.zeros((h,w),dtype=np.float64)
            for h0 in get_patch_start(h,patch_size,step_size):
                for w0 in get_patch_start(w,patch_size,step_size):
                    batch_images = im_noise[np.newaxis, h0:h0+patch_size, w0:w0+patch_size, np.newaxis]
                    test_output_eval = self.sess.run(self.test_output,feed_dict={self.test_input: batch_images}) ## range [0,1]
                    test_output_eval = test_output_eval[0,:,:,0]
                    test_output_eval = np.clip(test_output_eval,0,1)
                    temp_out[h0:h0+patch_size,w0:w0+patch_size]  =temp_out[h0:h0+patch_size,w0:w0+patch_size] + test_output_eval
                    temp_count[h0:h0+patch_size,w0:w0+patch_size]=temp_count[h0:h0+patch_size,w0:w0+patch_size] + 1
            im_out = temp_out/temp_count
                      
            ###### convert back to uint8 [0 255]
            im_noise = np.uint8(np.clip(im_noise,0,1)*255)
            im_gt = np.uint8(im_gt*255)
            im_out = np.uint8(im_out*255)
            
            ### save noise
            temp_dir = '%s/%s' % (self.args.noise_dir,testset)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir) 
            save_path = '%s/%s' % (temp_dir,os.path.basename(im_path))
            imageio.imsave(save_path,im_noise)
            #### save output
            temp_dir = '%s/%s' % (self.args.results_dir,testset)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir) 
            save_path = '%s/%s' % (temp_dir,os.path.basename(im_path))
            imageio.imsave(save_path,im_out)

            total_psnr=total_psnr + psnr(im_gt,im_out)
            total_ssim=total_ssim + ssim(im_gt,im_out)
        
        print ("average run time: ",(time.time()-start)/len(im_list))
        print ('%s, %d ,psnr: %.2f, ssim: %.4f' % (testset,self.args.sigma,total_psnr/len(im_list),total_ssim/len(im_list)))

       


    def build_model_inference_2(self):     
        ########## test
        TempModel=str_to_class(self.args.model_def)
        self.test_input = tf.placeholder(tf.float32,[1, None, None, 1], name='test') # range [0 255]
        self.test_output, _ = TempModel(self.test_input, self.test_input, is_train=False,reuse=False)
        self.saver=tf.train.Saver()
        self.saver.restore(self.sess, self.args.checkpoint_path)
        #print("test_output shape",self.test_output.get_shape())
    def inference_2(self,testset):

        np.random.seed(seed=0) #### for reproduce

        total_psnr=0
        total_ssim=0 
        test_total_count=0
        start=time.time()
        im_list = glob('TestData/%s/*.png' % (testset))
        im_list = sorted(im_list)
        for i in range(len(im_list)):
            ###### convert to float [0, 1]
            im_path = im_list[i]
            im_gt = imageio.imread(im_path) / 255.0 ###### range[0,1]
            im_noise = im_gt + np.random.normal(0, self.args.sigma/255.0, im_gt.shape)
           
            batch_images = im_noise[np.newaxis, :,:, np.newaxis]
            test_output_eval = self.sess.run(self.test_output,feed_dict={self.test_input: batch_images}) ## range [0,1]
            test_output_eval = test_output_eval[0,:,:,0]
            im_out = np.clip(test_output_eval,0,1)
                               
            ###### convert back to uint8 [0 255]
            im_noise = np.uint8(np.clip(im_noise,0,1)*255)
            im_gt = np.uint8(im_gt*255)
            im_out = np.uint8(im_out*255)
            
            ### save noise
            temp_dir = '%s/%s' % (self.args.noise_dir,testset)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir) 
            save_path = '%s/%s' % (temp_dir,os.path.basename(im_path))
            imageio.imsave(save_path,im_noise)
            #### save output
            temp_dir = '%s/%s' % (self.args.results_dir,testset)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir) 
            save_path = '%s/%s' % (temp_dir,os.path.basename(im_path))
            imageio.imsave(save_path,im_out)

            total_psnr=total_psnr + psnr(im_gt,im_out)
            total_ssim=total_ssim + ssim(im_gt,im_out)
        
        print ("average run time: ",(time.time()-start)/len(im_list))
        print ('%s, %d ,psnr: %.2f, ssim: %.4f' % (testset,self.args.sigma,total_psnr/len(im_list),total_ssim/len(im_list)))















import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def RDB(input_layer, is_train=False, reuse = False, RDB_name='RDB'):
    ## C=8
    w_init = tf.random_normal_initializer(stddev=0.02)
    # b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope(RDB_name, reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n0 = input_layer
        n1 = Conv2d(n0, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='%s_c1'%RDB_name)
        n1 = ConcatLayer([n0,n1],concat_dim=3, name='%s_concat1'%RDB_name)
        n2 = Conv2d(n1, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='%s_c2'%RDB_name)
        n2 = ConcatLayer([n1,n2],concat_dim=3, name='%s_concat2'%RDB_name)
        n3 = Conv2d(n2, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='%s_c3'%RDB_name)
        n3 = ConcatLayer([n2,n3],concat_dim=3, name='%s_concat3'%RDB_name)
        n4 = Conv2d(n3, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='%s_c4'%RDB_name)
        n4 = ConcatLayer([n3,n4],concat_dim=3, name='%s_concat4'%RDB_name)
        n5 = Conv2d(n4, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='%s_c5'%RDB_name)
        n5 = ConcatLayer([n4,n5],concat_dim=3, name='%s_concat5'%RDB_name)
        n6 = Conv2d(n5, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='%s_c6'%RDB_name)
        n6 = ConcatLayer([n5,n6],concat_dim=3, name='%s_concat6'%RDB_name)
        n7 = Conv2d(n6, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='%s_c7'%RDB_name)
        n7 = ConcatLayer([n6,n7],concat_dim=3, name='%s_concat7'%RDB_name)
        n8 = Conv2d(n7, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='%s_c8'%RDB_name)
        n8 = ConcatLayer([n7,n8],concat_dim=3, name='%s_concat8'%RDB_name)

        n_LF = Conv2d(n8, 64, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, name='%s_fusion'%RDB_name)
        n_out = ElementwiseLayer([n0, n_LF], tf.add, '%s_residual' % RDB_name)
        return n_out

def RDN(input_image,gt_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    # b_init =  tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("RDN", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n_input = InputLayer(input_image, name='in')
        f_1 = Conv2d(n_input, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='conv1')
        f0 = Conv2d(f_1, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='conv2')
        f1 = RDB(f0,is_train=is_train,reuse=reuse, RDB_name='RDB1')
        f2 = RDB(f1,is_train=is_train,reuse=reuse, RDB_name='RDB2')
        f3 = RDB(f2,is_train=is_train,reuse=reuse, RDB_name='RDB3')
        f4 = RDB(f3,is_train=is_train,reuse=reuse, RDB_name='RDB4')
        f5 = RDB(f4,is_train=is_train,reuse=reuse, RDB_name='RDB5')
        f6 = RDB(f5,is_train=is_train,reuse=reuse, RDB_name='RDB6')
        f7 = RDB(f6,is_train=is_train,reuse=reuse, RDB_name='RDB7')
        f8 = RDB(f7,is_train=is_train,reuse=reuse, RDB_name='RDB8')
        f9 = RDB(f8,is_train=is_train,reuse=reuse, RDB_name='RDB9')
        f10 = RDB(f9,is_train=is_train,reuse=reuse, RDB_name='RDB10')
        f11 = RDB(f10,is_train=is_train,reuse=reuse, RDB_name='RDB11')
        f12 = RDB(f11,is_train=is_train,reuse=reuse, RDB_name='RDB12')
        f13 = RDB(f12,is_train=is_train,reuse=reuse, RDB_name='RDB13')
        f14 = RDB(f13,is_train=is_train,reuse=reuse, RDB_name='RDB14')
        f15 = RDB(f14,is_train=is_train,reuse=reuse, RDB_name='RDB15')
        f16 = RDB(f15,is_train=is_train,reuse=reuse, RDB_name='RDB16')
        f_d = ConcatLayer([f1,f2,f4,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16],concat_dim=3,name='Fd_concat')
        f_d = Conv2d(f_d, 64, (1, 1), (1, 1), padding='SAME', W_init=w_init, name='conv3')
        f_gf = Conv2d(f_d, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='conv4')
        f_df = ElementwiseLayer([f_1, f_gf], tf.add, 'Fdf_residual' )

        output = Conv2d(f_df, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='out_temp1')
        output = Conv2d(output, 1, (1, 1), (1, 1), act= None, padding='SAME', W_init=w_init, name='out')
        output = ElementwiseLayer([n_input,output],tf.add,'out_residual' )
        output = output.outputs

        loss = tf.reduce_mean(tf.squared_difference(output, gt_image))
        return output, loss


def RDN_NLB(input_image,gt_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    # b_init =  tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("RDN", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n_input = InputLayer(input_image, name='in')
        f_1 = Conv2d(n_input, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='conv1')
        f0 = Conv2d(f_1, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='conv2')
        f1 = RDB(f0,is_train=is_train,reuse=reuse, RDB_name='RDB1')
        f2 = RDB(f1,is_train=is_train,reuse=reuse, RDB_name='RDB2')
        f3 = RDB(f2,is_train=is_train,reuse=reuse, RDB_name='RDB3')
        f4 = RDB(f3,is_train=is_train,reuse=reuse, RDB_name='RDB4')
        f5 = RDB(f4,is_train=is_train,reuse=reuse, RDB_name='RDB5')
        f5 = NonLocalLayer_Concat(f5,filter_num=32,output_num=64,strides=[1],name='nl1')
        f6 = RDB(f5,is_train=is_train,reuse=reuse, RDB_name='RDB6')
        f7 = RDB(f6,is_train=is_train,reuse=reuse, RDB_name='RDB7')
        f8 = RDB(f7,is_train=is_train,reuse=reuse, RDB_name='RDB8')
        f9 = RDB(f8,is_train=is_train,reuse=reuse, RDB_name='RDB9')
        f10 = RDB(f9,is_train=is_train,reuse=reuse, RDB_name='RDB10')
        f10 = NonLocalLayer_Concat(f10,filter_num=32,output_num=64,strides=[1],name='nl2')
        f11 = RDB(f10,is_train=is_train,reuse=reuse, RDB_name='RDB11')
        f12 = RDB(f11,is_train=is_train,reuse=reuse, RDB_name='RDB12')
        f13 = RDB(f12,is_train=is_train,reuse=reuse, RDB_name='RDB13')
        f14 = RDB(f13,is_train=is_train,reuse=reuse, RDB_name='RDB14')
        f15 = RDB(f14,is_train=is_train,reuse=reuse, RDB_name='RDB15')
        f16 = RDB(f15,is_train=is_train,reuse=reuse, RDB_name='RDB16')
        f_d = ConcatLayer([f1,f2,f4,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16],concat_dim=3,name='Fd_concat')
        f_d = Conv2d(f_d, 64, (1, 1), (1, 1), padding='SAME', W_init=w_init, name='conv3')
        f_gf = Conv2d(f_d, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='conv4')
        f_df = ElementwiseLayer([f_1, f_gf], tf.add, 'Fdf_residual' )

        output = Conv2d(f_df, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='out_temp1')
        output = Conv2d(output, 1, (1, 1), (1, 1), act= None, padding='SAME', W_init=w_init, name='out')
        output = ElementwiseLayer([n_input,output],tf.add,'out_residual' )
        output = output.outputs

        loss = tf.reduce_mean(tf.squared_difference(output, gt_image))
        return output, loss

def RDN_PNB(input_image,gt_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    # b_init =  tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("RDN", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n_input = InputLayer(input_image, name='in')
        f_1 = Conv2d(n_input, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='conv1')
        f0 = Conv2d(f_1, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='conv2')
        f1 = RDB(f0,is_train=is_train,reuse=reuse, RDB_name='RDB1')
        f2 = RDB(f1,is_train=is_train,reuse=reuse, RDB_name='RDB2')
        f3 = RDB(f2,is_train=is_train,reuse=reuse, RDB_name='RDB3')
        f4 = RDB(f3,is_train=is_train,reuse=reuse, RDB_name='RDB4')
        f5 = RDB(f4,is_train=is_train,reuse=reuse, RDB_name='RDB5')
        f5 = NonLocalLayer_Concat(f5,filter_num=32,output_num=64,strides=[2,4,8],name='nl1')
        f6 = RDB(f5,is_train=is_train,reuse=reuse, RDB_name='RDB6')
        f7 = RDB(f6,is_train=is_train,reuse=reuse, RDB_name='RDB7')
        f8 = RDB(f7,is_train=is_train,reuse=reuse, RDB_name='RDB8')
        f9 = RDB(f8,is_train=is_train,reuse=reuse, RDB_name='RDB9')
        f10 = RDB(f9,is_train=is_train,reuse=reuse, RDB_name='RDB10')
        f10 = NonLocalLayer_Concat(f10,filter_num=32,output_num=64,strides=[2,4,8],name='nl2')
        f11 = RDB(f10,is_train=is_train,reuse=reuse, RDB_name='RDB11')
        f12 = RDB(f11,is_train=is_train,reuse=reuse, RDB_name='RDB12')
        f13 = RDB(f12,is_train=is_train,reuse=reuse, RDB_name='RDB13')
        f14 = RDB(f13,is_train=is_train,reuse=reuse, RDB_name='RDB14')
        f15 = RDB(f14,is_train=is_train,reuse=reuse, RDB_name='RDB15')
        f16 = RDB(f15,is_train=is_train,reuse=reuse, RDB_name='RDB16')
        f_d = ConcatLayer([f1,f2,f4,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16],concat_dim=3,name='Fd_concat')
        f_d = Conv2d(f_d, 64, (1, 1), (1, 1), padding='SAME', W_init=w_init, name='conv3')
        f_gf = Conv2d(f_d, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='conv4')
        f_df = ElementwiseLayer([f_1, f_gf], tf.add, 'Fdf_residual' )

        output = Conv2d(f_df, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='out_temp1')
        output = Conv2d(output, 1, (1, 1), (1, 1), act= None, padding='SAME', W_init=w_init, name='out')
        output = ElementwiseLayer([n_input,output],tf.add,'out_residual' )
        output = output.outputs

        loss = tf.reduce_mean(tf.squared_difference(output, gt_image))
        return output, loss

def NonLocalLayer_Concat(input_layer, filter_num, output_num, strides=[2,4,8], name='nl'):

    input_tensor =  input_layer.outputs
    ###  pyramid 
    NonLocal_tensor_list = [] 
    for stride in strides:
        
        x_theta = tf.layers.conv2d(input_tensor, filter_num, 1, padding='same', activation=None, name=name +'_'+str(stride)+ '_theta')

        #x_downscale = downscale(input_tensor,r=stride)
        #x_phi = tf.layers.conv2d(x_downscale, filter_num, 1, padding='same', activation=None, name=name +'_'+str(stride)+ '_phi')
        # x_phi = tf.layers.conv2d(input_tensor, filter_num, kernel_size=(2*stride-1,2*stride-1), strides=(stride,stride),padding='same', activation=None, name=name +'_'+str(stride)+ '_phi')
        x_phi = tf.layers.conv2d(input_tensor, filter_num, kernel_size=(3,3), strides=(stride,stride), padding='same', activation=None, name=name +'_'+str(stride)+ '_phi')

        # x_g = tf.layers.conv2d(input_tensor, output_num, kernel_size=(2*stride-1,2*stride-1), strides=(stride,stride), padding='same', activation=None, name=name+'_' +str(stride)+ '_g')
        x_g = tf.layers.conv2d(input_tensor, output_num, kernel_size=(3,3), strides=(stride,stride), padding='same', activation=None, name=name+'_' +str(stride)+ '_g')

        x_theta_reshaped = tf.reshape(x_theta, [tf.shape(x_theta)[0], tf.shape(x_theta)[1] * tf.shape(x_theta)[2],tf.shape(x_theta)[3]])
        x_phi_reshaped = tf.reshape(x_phi,  [tf.shape(x_phi)[0], tf.shape(x_phi)[1] * tf.shape(x_phi)[2], tf.shape(x_phi)[3]])
        x_phi_permuted = tf.transpose(x_phi_reshaped, perm=[0, 2, 1])
        x_mul1 = tf.matmul(x_theta_reshaped, x_phi_permuted)
        x_mul1_softmax = tf.nn.softmax(x_mul1, axis=-1)  # normalization for embedded Gaussian

        x_g_reshaped = tf.reshape(x_g, [tf.shape(x_g)[0], tf.shape(x_g)[1] * tf.shape(x_g)[2], tf.shape(x_g)[3]])
        x_mul2 = tf.matmul(x_mul1_softmax, x_g_reshaped)
        x_mul2_reshaped = tf.reshape(x_mul2, [tf.shape(x_g)[0], tf.shape(x_theta)[1], tf.shape(x_theta)[2], tf.shape(x_g)[3]])
    
        NonLocal_tensor_list.append(x_mul2_reshaped)

    NonLocal_tensor = tf.concat(NonLocal_tensor_list,axis=3)
    NonLocal_tensor = tf.layers.conv2d(NonLocal_tensor, output_num, 1, padding='same', activation=None, name=name+ '_out')
    output_tensor = tf.add(input_tensor,NonLocal_tensor)
    output_layer = InputLayer(output_tensor, name=name + 'output')
    #output_layer = InputLayer(x_out, name=name + 'output')
    return output_layer





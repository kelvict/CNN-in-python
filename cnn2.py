#!/usr/bin/python
# encoding: utf-8

import numpy as np
import scipy.signal as signal
import scipy
import gzip
import cPickle

def load_data(path):
    f=gzip.open(path,'rb')
    train_set,valid_set,test_set=cPickle.load(f)
    return [train_set,valid_set,test_set]

class layer:
    def sigmoid(input):
        return 1.0 / (1.0 + np.exp(-input))

    def derive_sigmoid(input):
        """
        this is the derive function of the sigmoid
        """
        return sigmoid(input) * (1.0 - sigmoid(input))

    def tanh(input):
        """
        this is the output activation function f(.)
        """
        return 1.7159 * np.tanh(2.0 / 3.0 * input)
    def derive_tanh(input):
        """
        this is the derive function of the tanh
        """
        return 2.0 / 3.0 / 1.7159 * (1.7159 + input) * (1.7159 - input)

    def __init__(self,name = "please set me"):
        self.name = name
        self.activation = sigmoid
        self.dActivation = derive_sigmoid


class conv_layer(layer):
    """
    subsample use average_pooling
    """
    # TODO understand why do this
    def init_weight(self,ct,kernel_shape):
        kernel_num=ct[0].shape[0]
        kernel_height=kernel_shape[0]
        kernel_width=kernel_shape[1]
        weights=np.zeros((kernel_num,kernel_height,kernel_width))
        for i in range(kernel_num):
            connected=np.array((ct[1]==ct[0,i]),dtype='int')
            connected=np.array(np.nonzero(connected))
            connected=connected.shape[0]
            fanin=connected*kernel_height*kernel_width
            sd=1.0/np.sqrt(fanin)
            weights[i]=-1.0*sd+2*sd*np.random.random_sample((kernel_height,kernel_width))
        print("weight")
        print(weights.shape)
        return weights


    def __init__(self,connect,shape_image,shape_pooling,shape_kernel,stride):
        #  layer.__init__("conv")

        connect = np.transpose(connect)
        num_out_maps = connect.shape[0]
        connect = np.nonzero(connect)
        connect = np.array((connect[1],connect[0]))
        # TODO delete
        print(connect)
        self.connect = connect

        self.shape_image = shape_image
        self.shape_pooling = shape_pooling
        self.shape_kernel = shape_kernel
        self.stride = stride

        image_height = shape_image[0]
        image_width = shape_image[1]

        kernel_height = shape_kernel[0]
        kernel_width = shape_kernel[1]

        pool_height = shape_pooling[0]
        pool_width = shape_pooling[1]

        self.conv_w = self.init_weight(connect,shape_kernel)
        self.conv_b = np.zeros(num_out_maps)
        self.sub_w = np.ones(shape_pooling) / pool_height / pool_width
        # becuase the numpy's + is elementary wise
        self.sub_b = np.zeros(num_out_maps)

        conv_out_height = image_height - kernel_height + 1
        conv_out_width = image_width - kernel_width + 1
        sub_out_height = (conv_out_height - pool_height) / stride + 1
        sub_out_width = (conv_out_width - pool_width) / stride + 1
        self.shape_conv_out = [conv_out_height,conv_out_width]
        self.shape_sub_out = [sub_out_height,sub_out_width]

    def convolutional(in_image,connect,kernel,bias,stride = 1,out_fun = 'sigmoid'):
        """
        convolutional function
        in_image is 3d
        kernel is 3d
        bias is 1d
        """
        shape_image = np.shape(in_image)
        shape_kernel = np.shape(kernel)
        out_dim1 = np.max(connect[1]) + 1
        shape_out_feature = (out_dim1,(shape_image[1] - shape_kernel[1]) / stride + 1,
                (shape_image[2] - shape_kernel[2]) / stride + 1)
        out_feature = np.zeros(shape_out_feature)
        for index in range(0,out_dim1):
            out_feature[index] += bias[index]
        for index in range(0,shape_kernel[0]):
            this_image = in_image[connect[0,index]]
            this_kernel = kernel[index]
            # FIXME should this to be rot180
            this_out = signal.convolve2d(this_image,this_kernel,mode = 'valid')
            out_feature_index = connect[1,index]
            out_feature[out_feature_index] += this_conv
        return sigmoid(out_feature)

    def pooling(in_image,weight,bias,shape_pooling,stride):
        """
        subsampling the out_feature
        use average_pooling
        """

        def copy_out(in_image,stride):
            """
            in_image is 2d
            """
            out_image = []
            shape_image = np.shape(in_image)
            for y in range(0,shape_image[0],stride):
                y_result = []
                y_data = in_image[y]
                for x in range(0,shape_image[1],stride):
                    y_result.append(y_data[x])
                out_image.append(y_result)
            return np.array(y_result)

        shape_image = np.shape(in_image)
        shape_out_feature = (np.shape(in_image[0]),(shape_image[1] - shape_pooling[0]) / stride + 1,
                (shape_image[2] - shape_pooling[1]) / stride + 1)
        out_feature = np.zeros(shape_out_feature)
        kernel = np.ones(shape_pooling)
        for index in range(0,shape_out_feature[0]):
            this_image = in_image[index]
            this_kernel = kernel * weight[index]
            this_out = signal.convolve2d(this_image,this_kernel,mode = 'valid')
            out_feature[index] = copy_out(this_out,stride)
            out_feature[index] += bias[index]
            out_feature = sigmoid(out_feature)
            return out_feature

    def ff(self,in_images):
        connect = self.connect
        conv_w = self.conv_w
        conv_b = self.conv_b
        self.conv_out = convolutional(in_images,connect,conv_w,conv_b)

        sub_w = self.sub_w
        sub_b = self.sub_b
        shape_pooling = self.shape_pooling
        stride = self.stride
        self.sub_out = pooling(conv_out,sub_w,sub_b,shape_pooling,stride)
        return self.sub_out

#      def bp(self,d_last):
        #  # TODO must use index
        #  kernel = self.conv_w
        #  self.d_conv_out = np.zeros(self.shape_image)
        #  for index in range(0,d_last.shape[0]):
            #  self.d_conv_out = np.kron(self.conv_w,d_last[index]) * self.conv_out * (1 - self.conv_out)

        #  connect = self.connect
        #  self.d_prev = np.zeros((self.shape_image))
        #  shape_kernel = self.shape_kernel
        #  kernel = self.conv_w

        #  for index in range(0,shape_kernel[0]):
            #  prev_index = connect[0,index]
            #  this_kernel = kernel[index]
            #  #  NOTE should this to be rot180
            #  this_out = signal.convolve2d(this_image,np.rot(this_kernel,2),mode = 'full')
            #  #  out_feature_index = connect[1,index]
            #  #  out_feature[out_feature_index] += this_conv
        #  #  return sigmoid(out_feature)



class out_layer:
    def __init__(self,n_input,n_output):
        #  layer.__init__("out")

        sd = 1.0 / np.sqrt(n_input)
        self.w = -sd + 2 * sd * np.random.random_sample((n_input,n_output))
        self.b = np.zeros(n_output)
        self.n_input = n_input
        self.n_output = n_output

    def ff(self,in_images):
        w = self.w
        b = self.b
        self.out = np.dot(in_images,w)
        self.out += b
        self.out = self.activation(self.out)
        return self.out


class cnn:
    def __init__(self,shape_image,num_tags):
        self.layers = []

        connect1=np.array([[1,1,1,1]])
        connect2=np.array([[1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
                      [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                      [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
                      [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0]])

        shape_kernel1 = [5,5]
        shape_kernel2 = [5,5]
        shape_pooling1 = [2,2]
        shape_pooling2 = [2,2]
        stride1 = 2
        stride2 = 2
        num_out_maps1 = connect1.shape[1]
        num_out_maps2 = connect2.shape[1]
        self.num_out_maps2 = num_out_maps2

        self.layers.append(conv_layer(connect1,shape_image,shape_pooling1,shape_kernel1,stride1))
        self.layers.append(conv_layer(connect2,self.layers[0].shape_sub_out,shape_pooling2,shape_kernel2,stride2))

        out_layer_input = num_out_maps2 * np.prod(self.layers[1].shape_sub_out)
        self.layers.append(out_layer(out_layer_input,num_tags))

        self.error_count = 0

    def max_with_index(input):
        """
        for the out result
        """
        max = np.max(input)
        index = np.argmax(input)
        return [max,index]


    def ff(self,input):
        layer1_out = self.layers[0].ff(input)
        layer2_out = self.layers[1].ff(layer1_out)

        layer3_input = layer2_out.reshape(self.layers[2].n_input)
        layer3_out = self.layers[2].ff(layer2_out)

        self.out = max_with_index(layer3_out)
        return self.out

    def apply_gradient(self):
        pass


    def train(self,data,target):
        out = ff(data)
        if out[1] != target:
            self.error_count += 1

        # compute the delta of the out layer
        self.error = self.layers[2].out
        self.error[target] -= 1
        self.error_fun = 0
        for x in self.error:
            self.error_fun += x ** 2
        self.d_out = self.error * self.out * (1 - self.out)

        # compute the delta of the second layer
        self.layers[1].d_sub = np.dot(self.layers[2].w,self.d_out)
        # because it has reshape the output of the sub layer,we must reshape it back
        self.layers[1].d_sub = self.layers[1].d_sub.reshape([self.num_out_maps2,self.layers[2].shape_sub_out])
        num_input_images = self.layers[1].shape_image[0]
        d_conv1 = np.array(self.layers[1].shape_image)
        conv_out1 = self.layers[1].conv_out
        weight1 = self.layers[1].sub_w
        for i in range(0,num_input_images):
            d_conv1[i] = np.kron(weight1,self.layers[1].d_sub[i]) * conv_out[i] * (1 - conv_out[i])
        self.layers[1].d_conv = d_conv1

        # TODO compute the delta of the first layer
        weights = self.layers[1].conv_w
        d_last = d_conv1
        # FIXME shape_conv_out is 2D
        self.layers[0].d_sub = np.zeros(self.layers[0].shape_conv_out)

        apply_gradient()

    def check(self):
        for index in range(0,2):
            print(self.layers[index].conv_w)
            print(self.layers[index].conv_b)
            print(self.layers[index].sub_w)
            print(self.layers[index].sub_b)
            print()
        print(self.layers[2].w)
        print(self.layers[2].b)
        print()



if __name__ == '__main__':
    train_set,valid_set,test_set = load_data('./mnist.pkl.gz')
    conv_net = cnn([28,28],10)
    train_count =  train_set[0].shape[0]

    conv_net.check()
    #  for epoch in range(0,10):
        #  conv_net.error_count = 0
        #  print("the %dth train" % epoch)
        #  for i in range(0,train_count):
            #  image = train_set[0][i]
            #  data = image.reshape([1,28,28])
            #  target = train_set[1][i]
            #  conv_net.train(data,target)


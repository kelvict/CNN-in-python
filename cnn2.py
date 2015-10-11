#!/usr/bin/python
# encoding: utf-8

import numpy as np
import scipy.signal as signal
import scipy
import gzip
import cPickle
import copy

def load_data(path):
    f=gzip.open(path,'rb')
    train_set,valid_set,test_set=cPickle.load(f)
    return [train_set,valid_set,test_set]

class layer:
    def sigmoid(self,input):
        return 1.0 / (1.0 + np.exp(-input))

    def derive_sigmoid(self,input):
        """
        this is the derive function of the sigmoid
        """
        return sigmoid(input) * (1.0 - sigmoid(input))

    def tanh(self,input):
        """
        this is the output activation function f(.)
        """
        return 1.7159 * np.tanh(2.0 / 3.0 * input)
    def derive_tanh(self,input):
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
        #  print("weight")
        #  print(weights.shape)
        return weights


    def __init__(self,connect,shape_image,shape_pooling,shape_kernel,stride):
        """
        connect is 2D
        shape_image is 2D
        shape_pooling is 2D
        shape_kernel is 2D
        shape_conv_out is 3D
        shape_sub_out is 3D
        """
        #  layer.__init__("conv")

        connect = np.transpose(connect)
        num_out_maps = connect.shape[0]
        connect = np.nonzero(connect)
        connect = np.array((connect[1],connect[0]))
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
        self.conv_b = np.zeros([num_out_maps,1])
        self.sub_w = np.ones(shape_pooling) / pool_height / pool_width
        # because the numpy's + is elementary wise
        self.sub_b = np.zeros([num_out_maps,1])

        conv_out_height = image_height - kernel_height + 1
        conv_out_width = image_width - kernel_width + 1
        sub_out_height = (conv_out_height - pool_height) / stride + 1
        sub_out_width = (conv_out_width - pool_width) / stride + 1
        self.shape_conv_out = [num_out_maps,conv_out_height,conv_out_width]
        self.shape_sub_out = [num_out_maps,sub_out_height,sub_out_width]


    def convolutional(self,in_image,connect,kernel,bias,stride = 1,out_fun = 'sigmoid'):
        """
        convolutional function
        in_image is 3d
        kernel is 3d
        bias is 1d
        """
        shape_image = np.shape(in_image)
        shape_kernel = np.shape(kernel)
        #  print(connect.shape)
        out_dim1 = np.max(connect[1]) + 1
        shape_out_feature = (out_dim1,(shape_image[1] - shape_kernel[1]) / stride + 1,
                (shape_image[2] - shape_kernel[2]) / stride + 1)
        out_feature = np.zeros(shape_out_feature)
        for index in range(0,out_dim1):
            out_feature[index] += bias[index][0]
        for index in range(0,shape_kernel[0]):
            this_image = in_image[connect[0,index]]
            this_kernel = kernel[index]
            # this should be rot180 because this is a relative operation
            this_out = signal.convolve2d(this_image,np.rot90(this_kernel,2),mode = 'valid')
            out_feature_index = connect[1,index]
            out_feature[out_feature_index] += this_out

        self.conv_z = copy.deepcopy(out_feature)
        out_feature = self.sigmoid(out_feature)
        #  print(out_feature)
        return out_feature

    def pooling(self,in_image,weight,bias,shape_pooling,stride):
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

            return np.array(out_image)

        shape_image = np.shape(in_image)
        shape_out_feature = (in_image.shape[0],(shape_image[1] - shape_pooling[0]) / stride + 1,
                (shape_image[2] - shape_pooling[1]) / stride + 1)
        #  print("shape_out_feature")
        #  print(shape_out_feature)
        out_feature = np.zeros(shape_out_feature)
        kernel = np.ones(shape_pooling)
        for index in range(0,shape_out_feature[0]):
            this_image = in_image[index]
            this_kernel = kernel * weight
            # this should be rot180 because this is a relative operation
            this_out = signal.convolve2d(this_image,np.rot90(this_kernel,2),mode = 'valid')
            out_feature[index] = copy_out(this_out,stride)
            out_feature[index] += bias[index][0]

        self.sub_z = copy.deepcopy(out_feature)
        # FIXME use sigmoid?
        #  out_feature = self.sigmoid(out_feature)
        return out_feature

    def ff_conv(self,in_images):
        connect = self.connect
        conv_w = self.conv_w
        conv_b = self.conv_b
        self.conv_out = self.convolutional(in_images,connect,conv_w,conv_b)

    def ff_sub(self,in_images):
        sub_w = self.sub_w
        sub_b = self.sub_b
        shape_pooling = self.shape_pooling
        stride = self.stride
        self.sub_out = self.pooling(self.conv_out,sub_w,sub_b,shape_pooling,stride)

    def ff(self,in_images):
        self.ff_conv(in_images)
        self.ff_sub(in_images)
        return self.sub_out

    def gradient(self,inputs):
        """
        inputs is 3D
        """
        # calculate the gradient of the bias
        num_b = self.shape_conv_out[0]
        self.db = np.zeros([num_b,1])
        for i in range(0,num_b):
            self.db[i][0] += np.sum(self.d_conv[i])

        # calculate the gradient of the weight
        connect = self.connect
        kernels = self.conv_w
        num_kernels = kernels.shape[0]
        self.dw = np.zeros(kernels.shape)
        for i in range(0,num_kernels):
            input_index = connect[0,i]
            output_index = connect[1,i]
            this_input = inputs[input_index]
            this_d = self.d_conv[output_index]
            # this should rot180 because this is the a relative operation
            self.dw[i] += signal.convolve2d(this_input,np.rot90(this_d,2),mode = 'valid')

    def apply_gradient(self,alpha):
        num_w = self.conv_w.shape[0]
        # apply_gradient for weights
        for i in range(0,num_w):
            self.conv_w[i] -= alpha * self.dw[i]
        # apply_gradient for bias
        self.conv_b -= alpha * self.db



class out_layer(layer):
    def __init__(self,n_input,n_output):
        #  layer.__init__("out")

        sd = 1.0 / np.sqrt(n_input)
        self.w = -sd + 2 * sd * np.random.random_sample((n_output,n_input))
        self.b = np.zeros([n_output,1])
        self.n_input = n_input
        self.n_output = n_output

    def ff(self,in_images):
        """
        in_images is 1D
        w is 10 * 256
        """
        inputs = in_images.reshape([self.n_input,1])
        #  print(inputs[0][0])
        w = self.w
        b = self.b
        self.out = np.dot(w,inputs)
        self.out += b

        #FIXME for debug
        self.z = self.out
        self.out = self.sigmoid(self.out)
        return self.out

    def gradient(self,inputs):
        # FIXME to delete
        self.dw = np.zeros([self.n_output,self.n_input])
        # when the array has just one dimension it has no transpose,must to be (1,x)
        self.dw = np.dot(self.d_out,inputs.reshape([1,self.n_input]))
        self.db = self.d_out

    def apply_gradient(self,alpha):
        self.w -= alpha * self.dw
        self.b -= alpha * self.db

class cnn:
    def __init__(self,shape_image,num_tags):
        self.alpha = 0.02

        # set the layer
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
        self.layers.append(conv_layer(connect2,self.layers[0].shape_sub_out[1:],shape_pooling2,shape_kernel2,stride2))

        out_layer_input = np.prod(self.layers[1].shape_sub_out)
        self.layers.append(out_layer(out_layer_input,num_tags))

        self.error_count = 0

    def max_with_index(self,input):
        """
        for the out result
        """
        max = np.max(input)
        # NOTE np.argmax return a number
        index = np.argmax(input)
        return [max,index]


    def ff(self,input):
        self.input = input
        layer1_out = self.layers[0].ff(input)
        layer2_out = self.layers[1].ff(layer1_out)
        layer3_out = self.layers[2].ff(layer2_out)

        self.out = self.max_with_index(layer3_out)
        return self.out

    def compute_gradient(self):
        self.layers[2].gradient(self.layers[1].sub_out)
        self.layers[1].gradient(self.layers[0].sub_out)
        self.layers[0].gradient(self.input)

    def apply_gradient(self):
        for i in range(0,2):
            self.layers[i].apply_gradient(self.alpha)


    def train(self,data,target):
        self.target = target
        out = self.ff(data)
        if out[1] != target:
            self.error_count += 1

        self.compute_error()

        self.bp()
        self.compute_gradient()
        self.apply_gradient()

    def check_gradient(self,image,target):
        delta = 0.000001
        self.target = target

        #FIXME for debug
        out = self.ff(image)
        if out[1] != target:
            self.error_count += 1

        self.compute_error()

        self.bp()



        #  print("check layer 2")
        #  print("change w")

        #  for i in range(3,5):
            #  for j in range(5,6):
                #  out0 = self.ff(image)
                #  self.compute_error()
                #  self.bp()
                #  self.compute_gradient()

                #  print("bp's result is %f" % self.layers[2].dw[i][j])
                #  error_fun0 = self.error_fun

                #  self.layers[2].w[i][j] += delta

                #  out1 = self.ff(image)
                #  self.compute_error()
                #  error_fun1 = self.error_fun

                #  delta_fun = error_fun1 - error_fun0
                #  derive = delta_fun / delta
                #  print("real result is %f" % derive)
                #  print(" ")

        #  print("change b")
        #  for i in range(1,3):
            #  out0 = self.ff(image)
            #  self.compute_error()
            #  self.bp()
            #  self.compute_gradient()

            #  print("bp's result is %f" % self.layers[2].db[i][0])
            #  error_fun0 = self.error_fun

            #  self.layers[2].b[i][0] += delta

            #  out1 = self.ff(image)
            #  self.compute_error()
            #  error_fun1 = self.error_fun

            #  delta_fun = error_fun1 - error_fun0
            #  derive = delta_fun / delta
            #  print("real result is %f" % derive)
            #  print(" ")

        # check_gradient layer0 layer[1]
        #  for i in range(0,2):
            #  print("check layer %d" % (1 - i))
            #  self.check_gradient_w(1 - i,delta)
            #  self.check_gradient_b(1 - i,delta)
            #  self.check_gradient_s(1 - i,delta)


    def check_gradient_w(self,layer,delta):
        print("change w")

        for i in range(1,3):
            for j in range(1,2):
                out0 = self.ff(image)
                self.compute_error()
                self.bp()
                self.compute_gradient()

                print("bp's result is %f" % self.layers[layer].dw[0][i][j])
                error_fun0 = self.error_fun

                self.layers[layer].conv_w[0][i][j] += delta

                out1 = self.ff(image)
                self.compute_error()
                error_fun1 = self.error_fun

                delta_fun = error_fun1 - error_fun0
                derive = delta_fun / delta
                print("real result is %f" % derive)
                print(" ")

    def check_gradient_b(self,layer,delta):
        print("change b")
        for i in range(1,3):
            out0 = self.ff(image)
            self.compute_error()
            self.bp()
            self.compute_gradient()

            print("bp's result is %f" % self.layers[layer].db[i][0])
            error_fun0 = self.error_fun

            self.layers[layer].conv_b[i][0] += delta

            out1 = self.ff(image)
            self.compute_error()
            error_fun1 = self.error_fun

            delta_fun = error_fun1 - error_fun0
            derive = delta_fun / delta
            print("real result is %f" % derive)
            print(" ")

    def check_gradient_s(self,layer,delta):
        """
        check the gradient of the sensitive
        """
        print("change d_sub")
        out = self.ff(image)

        for i in range(1,3):
            for j in range(1,2):
                for k in range(layer + 1,3):
                    inputs = self.layers[k - 1].sub_out
                    out0 = self.layers[k].ff(inputs)
                self.compute_error()
                self.bp()
                self.compute_gradient()

                print("bp's result is %f" % self.layers[layer].d_sub[0][i][j])
                error_fun0 = self.error_fun

                self.layers[layer].sub_out[0][i][j] += delta

                for k in range(layer + 1,3):
                    inputs = self.layers[k - 1].sub_out
                    out1 = self.layers[k].ff(inputs)
                self.compute_error()
                error_fun1 = self.error_fun

                delta_fun = error_fun1 - error_fun0
                derive = delta_fun / delta
                print("real result is %f" % derive)
                print(" ")

    def check(self):
        image = np.ones([1,28,28])
        out = self.ff(image)
        for i in range(0,2):
            print(self.layers[i].conv_out[0])
            print(self.layers[i].conv_out[1])
            print(self.layers[i].sub_out[0])
            print(self.layers[i].sub_out[1])


    def compute_error(self):
        # compute the error of the net
        target = self.target
        self.error = self.layers[2].out
        self.error[target][0] -= 1
        self.error_fun = 0
        for x in self.error:
            self.error_fun += x ** 2
        self.error_fun /= 2

        #FIXME for debug
        #  error_fun0 = self.error_fun
        #  self.layers[2].d_out = self.error * self.layers[2].out * (1 - self.layers[2].out)
        #  print("bp %f" % self.layers[2].d_out[0][0])
        #  delta = 0.0000001
        #  self.layers[2].z[0][0] += delta
        #  self.layers[2].out = self.layers[2].sigmoid(self.layers[2].z)

        #  self.error = self.layers[2].out
        #  self.error[target][0] -= 1
        #  self.error_fun = 0
        #  for x in self.error:
            #  self.error_fun += x ** 2
        #  self.error_fun /= 2
        #  error_fun1 = self.error_fun
        #  derive = (error_fun1 - error_fun0) / delta
        #  print("real %f" % derive)




    def bp(self):
        # compute the delta of the out layer
        self.layers[2].d_out = self.error * self.layers[2].out * (1 - self.layers[2].out)
        #  print(self.layers[2].d_out)

        # compute the delta of the second layer
        # compute the d_sub
        #  print(self.layers[2].d_out.shape)
        self.layers[1].d_sub = np.dot(self.layers[2].w.transpose(),self.layers[2].d_out)

        # FIXME for debug
        # --------------------------------------------------------------
        delta = 0.000001

        z0 = copy.deepcopy(self.layers[2].z)
        i,j = (0,0)
        error_fun0 = self.error_fun
        print("bp's result is %f" % self.layers[1].d_sub[i][j])

        derive_t = 0
        for i in range(0,10):
            derive_t += self.layers[2].d_out[i][0] * self.layers[2].w[i,0]
            print(self.layers[2].w[i,0])
        print("")
        print(derive_t)

        self.layers[1].sub_z[0][0][0] += delta
        self.layers[2].ff(self.layers[1].sub_z)
        z1 = copy.deepcopy(self.layers[2].z)
        #  print(self.layers[2].out)
        self.compute_error()
        error_fun1 = self.error_fun

        delta_fun = error_fun1 - error_fun0
        derive = delta_fun / delta
        derive_z = (z1 - z0) / delta
        print("real result is %f" % derive)
        print(derive_z)

        #  because it has reshape the output of the sub layer,we must reshape it back
        shape_in = [self.num_out_maps2,self.layers[1].shape_sub_out[1],self.layers[1].shape_sub_out[2]]
        self.layers[1].d_sub = self.layers[1].d_sub.reshape(shape_in)
        # compute the d_conv
        num_input_images = self.layers[1].shape_conv_out[0]
        d_conv1 = np.zeros(self.layers[1].shape_conv_out)
        conv_out1 = self.layers[1].conv_out
        weight1 = self.layers[1].sub_w
        for i in range(0,num_input_images):
            d_conv1[i] = np.kron(weight1,self.layers[1].d_sub[i]) * conv_out1[i] * (1 - conv_out1[i])
        self.layers[1].d_conv = d_conv1


        # compute the delta of the first layer
        # compute the d_sub
        kernels = self.layers[1].conv_w
        num_kernels = kernels.shape[0]
        d_last = d_conv1
        connect = self.layers[1].connect
        d_sub0 = np.zeros(self.layers[0].shape_sub_out)
        for i in range(0,num_kernels):
            prev_index = connect[0,i]
            last_index = connect[1,i]
            this_kernel = kernels[i]
            this_d = d_last[last_index]
            # this is the true convolutional operation
            d_sub0[prev_index] += signal.convolve2d(this_d,this_kernel,mode = 'full')
        self.layers[0].d_sub = d_sub0
        # compute the d_conv
        num_input_images = self.layers[0].shape_conv_out[0]
        d_conv0 = np.zeros(self.layers[0].shape_conv_out)
        conv_out0 = self.layers[0].conv_out
        weight0 = self.layers[0].sub_w
        for i in range(0,num_input_images):
            d_conv0[i] = np.kron(weight0,self.layers[0].d_sub[i]) * conv_out0[i] * (1 - conv_out0[i])
        self.layers[0].d_conv = d_conv0


    def init_check(self):
        for index in range(0,2):
            print(self.layers[index].conv_w[0])
            print(self.layers[index].conv_w[1])
            print(self.layers[index].conv_b.shape)
            print(self.layers[index].shape_conv_out)
            print(self.layers[index].sub_w[0])
            print(self.layers[index].sub_w[1])
            print(self.layers[index].sub_b.shape)
            print(self.layers[index].shape_sub_out)
            #  print("connect")
            #  print(self.layers[index].connect)
        print(self.layers[2].w[0].shape)
        print(self.layers[2].b)
        #  print()

    def test(self,data,target):
        self.target = target
        out = self.ff(data)
        if out[1] != target:
            self.error_count += 1



if __name__ == '__main__':
    train_set,valid_set,test_set = load_data('./mnist.pkl.gz')
    train_count =  train_set[0].shape[0]
    test_count = test_set[0].shape[0]

    # set the conv_net
    conv_net = cnn([28,28],10)


    image = train_set[0][0]
    image = image.reshape([1,28,28])
    target = train_set[1][0]
    conv_net.check_gradient(image,target)


    #  for epoch in range(0,10):
        #  conv_net.error_count = 0
        #  print("the %dth train" % epoch)
        #  for i in range(0,train_count):
            #  image = train_set[0][i]
            #  data = image.reshape([1,28,28])
            #  target = train_set[1][i]
            #  conv_net.train(data,target)
            #  if (i + 1) % 1000 == 0:
                #  print("error_rate %f" % (conv_net.error_count / 1000.0))
                #  conv_net.error_count = 0

    #  for i in range(0,test_count):
        #  conv_net.error_count = 0
        #  image = test_set[0][i]
        #  data = image.reshape([1,28,28])
        #  target = test_set[1][i]
        #  conv_net.test(data,target)
    #  print("precision is %f" % (conv_net.error_count / floa(test_count)))


import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import tensor_shape
import tensorflow as tf
class GaussianBlur():
	def __init__(self,img,sigma,kSize):
		self.sigma=sigma
		self.kSize=kSize
		self.gaussianKernelNumpy=(np.zeros([kSize,kSize]))
		self.findKernel()
		self.gaussianKernelNumpy=np.expand_dims(self.gaussianKernelNumpy,axis=2)
		
		
		img=np.expand_dims(img,axis=2)
		
		img=tf.convert_to_tensor(img)

		self.gaussianKernelTensor=tf.convert_to_tensor(self.gaussianKernelNumpy)

		gaussian_filter_shape=self.gaussianKernelTensor.get_shape()


		self.gaussianKernelTensor=tf.reshape(self.gaussianKernelTensor,[self.kSize,self.kSize,1])
		img=tf.reshape(img,[img.shape[0],img.shape[1],1])
		
		self.conv_ops=nn_ops.Convolution(input_shape=img.get_shape(),filter_shape=tensor_shape.as_shape(self.gaussianKernelTensor.shape),padding='same')
		
		
	def findKernel(self):
		for i in range(-self.kSize//2,self.kSize//2+1):
			for j in range(-self.kSize//2,self.kSize//2+1):
				self.gaussianKernelNumpy[i+self.kSize//2][j+self.kSize//2]=1/(2*np.pi*(self.sigma)**2)*np.exp(-(i**2+j**2)/(2*self.sigma**2))
		return

		
		
	def convolve(self):
		out=self.conv_ops(img,self.gaussianKernelTensor)
		return out
		
		
	

		


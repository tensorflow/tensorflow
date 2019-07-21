import tensorflow as tf
tf.compat.v1.logging.set_verbosity(50)

import numpy as np
#from tensorflow.python.ops import gen_math_ops
#from tensorflow.contrib import t2t
from tensorflow.contrib.t2t.python.ops import t2t_ops
from tensorflow.contrib.t2t.python.ops import gen_t2t_ops
from tensorflow.python.framework import ops

def toss():
	return np.random.random()>0.5

def ref_layer_norm_compute(x, epsilon, scale, bias):
  if not x.dtype == tf.dtypes.float32:
  	epsilon = tf.cast(epsilon, x.dtype)
  	scale = tf.cast(scale, x.dtype)
  	bias = tf.cast(bias, x.dtype)
  mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
  sqds = tf.math.square(x-mean)
  variance = tf.reduce_mean(sqds, axis=[-1], keepdims=True) 
  return (x - mean) * scale * tf.rsqrt(variance + epsilon) + bias


def testInner(v):
	eps=np.random.random()
	scale=np.random.rand(v.shape[-1])
	bias=np.random.rand(v.shape[-1])
	#print('scale', scale)
	#print('bias', bias)
	intl=tf.convert_to_tensor(tf.random.normal(v.shape, seed=0))
	result = gen_t2t_ops.custom_l2_norm(v, eps, scale, bias)
	intl2=tf.convert_to_tensor(tf.random.normal(v.shape, seed=0))
	ref = ref_layer_norm_compute(v, eps, scale, bias)
	grad_ref=None
	grad_test=None
	grad_ref = tf.gradients(ref, v, grad_ys=tf.cast(intl,v.dtype))
	#print("Eps is", eps)
	grad_test = tf.gradients(result, v, grad_ys=tf.cast(intl2,v.dtype))
	return result, ref, grad_test, grad_ref

def ref_broadcast_dropout(v, rng, thresh):
	rng=tf.broadcast_to(tf.cast(tf.greater_equal(rng,tf.broadcast_to(thresh, rng.shape)), tf.float32), v.shape)
	return v * rng / ( 1. - thresh)

def testInnerDropout(sh):
	thresh=np.random.random()*0.979+0.001 #don't let 'thresh' get too high or we'll start failing mismatch tests due to fp precision issues
	shd = [(1 if toss() else x) for x in sh]
	v1=np.random.random_sample(sh)
	v2=np.random.random_sample(shd)
	v1=tf.cast(v1,tf.float32)
	v2=tf.cast(v2,tf.float32)
	intl=tf.convert_to_tensor(tf.random.normal(v1.shape, seed=0))
	intl2=tf.convert_to_tensor(tf.random.normal(v1.shape, seed=0))	
	ref = ref_broadcast_dropout(v1, v2, thresh)
	test = gen_t2t_ops.custom_dropout(v1, v2, thresh)
	grad_ref=None
	grad_test=None
	grad_ref = tf.gradients(ref, v1, grad_ys=tf.cast(intl,v1.dtype))
	#print("Eps is", eps)
	grad_test = tf.gradients(test, v1, grad_ys=tf.cast(intl2,v1.dtype))
	return test, ref, grad_test, grad_ref


v1 = [5.0, 4.0, 3.0, 2.0, -1.0]
v2 = [5.0, 5.1, 7.1, 8.0]
v3 = [[-1., 0., 9.], [1., 2., 3.], [5., -5., 5.], [0., 0., 0.]]
v4 = np.random.rand(18,21)
v5 = np.random.rand(13,24,18)
v6 = np.random.rand(218,121)
v7 = np.random.rand(19,55,600)

"""
class CustomL2Test(tf.test.TestCase):
	def doTest(self, v):
		for dev in range(2):
			for ntest in range(2):
				with self.session(force_gpu=(dev==1)):
						v = tf.to_float(v) if ntest==0 else tf.cast(v, tf.dtypes.float16)
						result, ref, grad_test, grad_ref = testInner(v)
						#print(result.eval())
						#print(ref.eval())
						self.assertAllCloseAccordingToType(result.eval(), ref.eval(), float_atol=1e-5, half_rtol=0.005, half_atol=0.005)
						#print(grad_test[0].eval())
						#print(grad_ref[0].eval())
						self.assertAllCloseAccordingToType(grad_test[0], grad_ref[0], float_atol=1e-5, half_rtol=0.005, half_atol=0.005)
	def test1(self):
		self.doTest(v1)
	def test2(self):
		self.doTest(v2)
	def test3(self):
		self.doTest(v3)
	def test4(self):
		self.doTest(v4)
	def test5(self):
		self.doTest(v5)
	def test6(self):
		self.doTest(v6)
	def test7(self):
		self.doTest(v7)
"""

class CustomDropoutTest(tf.test.TestCase):
	def test1(self):
		#config = tf.ConfigProto(allow_soft_placement=False)
		for dev in [1]:#range(2):
			with self.session(force_gpu=(dev==1)):
				for series in range(5):
					for n in range(50):
						#shape=[np.random.randint(1,2000),np.random.randint(1,100),np.random.randint(1,100)] if toss() else [np.random.randint(1,2000),np.random.randint(1,100)]
						if series==0: # test Functor3_v2
							shape=[np.random.randint(1,200),np.random.randint(1,10),np.random.randint(1,10)] 
							np.random.shuffle(shape)
						elif series==1: #test Functor2
							shape=[np.random.randint(1,1000),np.random.randint(1,10)]
							np.random.shuffle(shape)
						elif series==2: # test Functor3->Functor2 fallback
							shape=[1, np.random.randint(1,200),np.random.randint(1,200)]
						elif series==3: #test Functor3
							shape=[np.random.randint(1,200),np.random.randint(1,10),np.random.randint(1024,3072)] 
							if n<2:
								shape=[70,49,2048]
						else:
							shape=[np.random.randint(1,2000),np.random.randint(1,10),np.random.randint(1,10), np.random.randint(1,10)] 
							np.random.shuffle(shape)
						ref, test, grad_ref, grad_test=testInnerDropout(shape)
						self.assertAllCloseAccordingToType(test.eval(), ref.eval(), float_atol=1e-4)
						self.assertAllCloseAccordingToType(grad_test[0].eval(), grad_ref[0].eval(), float_atol=1e-4)


if __name__ == "__main__":
  tf.test.main()

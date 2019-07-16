import tensorflow as tf
import numpy as np
#from tensorflow.python.ops import gen_math_ops
#from tensorflow.contrib import t2t
from tensorflow.contrib.t2t.python.ops import t2t_ops
from tensorflow.contrib.t2t.python.ops import gen_t2t_ops


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
	grad_ref = tf.gradients(ref, v, grad_ys=tf.cast(intl,v.dtype))
	#print("Eps is", eps)
	grad_test = tf.gradients(result, v, grad_ys=tf.cast(intl2,v.dtype))
	return result, ref, grad_test, grad_ref

v1 = [5.0, 4.0, 3.0, 2.0, 1.0]
v2 = [5.0, 6.0, 7.0, 8.0]
v3 = [[-1., 0., 1], [1., 2., 3.], [5., -5., 5.], [0., 0., 0.]]
v4 = np.random.rand(18,21)
v5 = np.random.rand(13,24,18)
v6 = np.random.rand(218,121)
v7 = np.random.rand(19,55,600)


class CustomL2Test(tf.test.TestCase):
	def doTest(self, v):
		for ntest in range(2):
			with self.test_session():
				v = tf.to_float(v) if ntest==0 else tf.cast(v, tf.dtypes.float16)
				result, ref, grad_test, grad_ref = testInner(v)
				#print(result.eval())
				#print(ref.eval())
				self.assertAllCloseAccordingToType(result.eval(), ref.eval(), half_rtol=0.005, half_atol=0.005)
				#print(grad_test[0].eval())
				#print(grad_ref[0].eval())
				self.assertAllCloseAccordingToType(grad_test[0], grad_ref[0], half_rtol=0.005, half_atol=0.005)

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

if __name__ == "__main__":
  tf.test.main()

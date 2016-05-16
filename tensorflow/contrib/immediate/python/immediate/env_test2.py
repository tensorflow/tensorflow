# Tests for immediate.Env

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate

# TODO(yaroslavvb): overload identity in array_ops

class EnvTest(tf.test.TestCase):

  def atestConcatGenop(self):
    env = immediate.Env(tf)
    gen_array_ops = env.wrapped_namespaces['gen_array_ops']
    values = [env.tf.ones((2, 2)), env.tf.ones((2, 2))]
    concat_dim = 0
    name = "concat"
    print gen_array_ops._concat(concat_dim=concat_dim,
                               values=values,
                               name=name)

# OpFactory __call__: gen_array_ops._concat((), {'concat_dim': 0, 'values': [iTensor([[ 1.  1.]
#  [ 1.  1.]]), iTensor([[ 1.  1.]
#  [ 1.  1.]])], 'name': 'concat'})
    
# OpFactor redirect: gen_array_ops._concat((), {'concat_dim': 0, 'values': [iTensor([[ 1.  1.]
#  [ 1.  1.]]), iTensor([[ 1.  1.]
#  [ 1.  1.]])], 'name': 'concat'})



  def atestConcat(self):
    env = immediate.Env(tf)
    val0 = env.numpy_to_tensor(0)
    # test special degenerate case
    self.assertEqual(env.tf.concat(0, 5), 5)

    val1 = env.numpy_to_tensor([1,2])
    val2 = env.numpy_to_tensor([3,4])
    val3 = env.tf.concat(0, [val1, val2]) 
    print val3
   


#  def testRandomUniform(self):
#    env = immediate.Env(tf)
#    val = env.tf.random_uniform([3, 3], -2, 2)
#    print val

if __name__ == "__main__":
  tf.test.main()

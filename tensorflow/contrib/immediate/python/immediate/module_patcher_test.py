# Tests for module rewriter
# pylint: disable=missing-docstring,invalid-name

import tensorflow as tf
import tensorflow.contrib.immediate as immediate
import tensorflow.contrib.immediate.python.immediate.module_patcher as module_patcher
import sys, traceback, pdb


from tensorflow.python.platform import googletest

class ModulePatcherTest(googletest.TestCase):

  def atestPatch0(self):
    import __builtin__
    result = __builtin__.__import__("tensorflow.python.ops", fromlist=["gen_math_ops", "gen_array_ops"])
    print result.gen_math_ops
    print result.gen_string_ops

#    result = __builtin__

#    result = __builtin__.__import__("tensorflow.python.ops", fromlist=["gen_math_ops", "gen_array_ops"])
    #    print result.gen_math_ops
#    print result.gen_string_ops
#    print result.gen_string_ops

  def atestPatch(self):
    try:
      import math
      import random

      def patched_log(x):
          print('Computing log({:g})'.format(x))
          return math.log(x)

      def patched_add(x):
          print('Computing add({:g})'.format(x))
          return None

      from tensorflow.python.ops import op_def_library
      # need to patch 
      patches = {"op_def_library"}

      import sys
      modules_before = set(sys.modules)
      patches = {'math.log': patched_log, "tensorflow.add": patched_add}
      cloned_modules = module_patcher.clone_modules(patches, ['random'])
      new_math = cloned_modules['math']
      new_random = cloned_modules['random']
      print('Original log:         ', math.log(2.0))
      print('Patched log:          ', new_math.log(2.0))
      print('Original expovariate: ', random.expovariate(2.0))
      print('Patched expovariate:  ', new_random.expovariate(2.0))
      #new_tf = cloned_modules["tensorflow"]
      #print('Patched tf:  ', new_tf.add(2))
      
    except:
      import sys, traceback, pdb
      exc_type, value, tb = sys.exc_info()
      traceback.print_exc()
      pdb.post_mortem(tb)

  def atestPatch15(self):
    holder1, tensor1 = tf.get_session_tensor(tf.int32)
    holder2, tensor2 = tf.get_session_tensor(tf.int32)
    result = tf.add(tensor1, tensor2)

# problem with module patcher is that it creates objects of wrong class
# (immediate.Tensor instead of original Tensor)

#    Called apply op with <tensorflow.python.ops.op_def_library.OpDefLibrary object at 0x10ab7eb50> Add {'y': <tf.Tensor 'GetSessionTensor_1:0' shape=<unknown> dtype=int32>, 'x': <tf.Tensor 'GetSessionTensor:0' shape=<unknown> dtype=int32>}

#    Called apply op with <tensorflow.python.ops.op_def_library.OpDefLibrary object at 0x10e2ee5d0> Add {'y': <tf.Tensor 'GetSessionTensor_3:0' shape=<unknown> dtype=int32>, 'x': <tf.Tensor 'GetSessionTensor_2:0' shape=<unknown> dtype=int32>}

    # ops.apply_op is called with a Tensor that's different type than tensor
    # in ops.py
    # also, importing tensorflow as tf, gets a different Tensor type than in
    # ops.py. 
  def testPatch2(self):
    env = immediate.Env(tf)
    cloned_modules = module_patcher.patch_immediate(env, ["math", "random"])
    new_tf = cloned_modules["tensorflow"]
    new_gen_math_ops = cloned_modules["gen_math_ops"]
    print("Adding things")
    val1 = env.numpy_to_tensor(1)
      #      val2 = env.numpy_to_tensor(2)
    print('Patched tf:  ', new_gen_math_ops.cast(val1, tf.float32))



  def atestImmediatePatcher(self):
    import pdb, traceback, sys


    try:
      env = immediate.Env(tf)
      patcher = module_patcher.ImmediatePatcher(env)
      new_tf = patcher(tf)

      val1 = env.numpy_to_tensor(2)
      val2 = env.numpy_to_tensor(3)
      val3 = env.numpy_to_tensor([1,2,3, 4])
      print new_tf.add(val1, val2)
      return

      print new_tf.concat(0, [val3, val3])

      print new_tf.add(val1, val2)
      print new_tf.reduce_sum(val3)
      print new_tf.ones((3, 3))
      print new_tf.constant([1,2,3])
      print new_tf.random_uniform([2,2], 0, 10)
      val4 = new_tf.reshape(val3, [2, 2])
      print val4
      print new_tf.nn.relu(val1)
      print new_tf.matmul(val4, val4)
      print new_tf.equal(val1, val2)
      
      print new_tf.image.random_brightness(val4, 1.)
      tensor1 = env.numpy_to_tensor([0, 0, 0, 0])
      tensor2 = env.numpy_to_tensor([1, 1, 1, 1])
      bool_tensor = env.numpy_to_tensor([True, False, True, False])
      print new_tf.select(bool_tensor, tensor1, tensor2)
      print new_tf.mod(5, 2)
      print new_tf.cast(5, tf.float32)
      print new_tf.transpose(new_tf.ones((2,2)))

      # TODO(yaroslavvb) remove numpy string restriction

      print new_tf.string_to_number("123")
      print new_tf.nn.top_k(val3)
      print new_tf.range(10)
      print new_tf.nn.softmax([[1.,2.],[3., 4.]])
      print new_tf.div(5,2)
      print new_tf.squeeze([[1,2,3]])
      print new_tf.split(0, 2, tensor2)

      # decode_csv missing
      # try Print
      
    except:
      exc_type, value, tb = sys.exc_info()
      traceback.print_exc()
      pdb.post_mortem(tb)

if __name__ == "__main__":
  googletest.main()

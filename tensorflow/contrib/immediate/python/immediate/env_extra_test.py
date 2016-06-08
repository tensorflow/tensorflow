# Small set of broken tests for env (to be moved into env_test.py after they
# work )

import tensorflow as tf
import numpy as np
import tensorflow.contrib.immediate as immediate

from tensorflow.contrib.immediate.python.immediate import test_util
from tensorflow.python.framework import ops as ops

import threading


def print_gdef_diff(gdef1, gdef2):
  print("GraphDef difference")
  print("-"*80)
  dict1 = {node.name: node for node in gdef1.node}
  dict2 = {node.name: node for node in gdef2.node}
  names1 = set(dict1.keys())
  names2 = set(dict2.keys())
  if names1 == names2:
    return
  for name in sorted(names2.difference(names1)):
    print dict2[name]

_cached_graph_version = 0
def _is_graph_changed(env):
  global _cached_graph_version
  is_changed = (env._graph_version != _cached_graph_version)
  _cached_graph_version = env._graph_version
  return is_changed

class ExtraEnvTest(test_util.ImmediateTestCase):


  def disabled_testAdd(self):
    types = [np.int32, np.int64, np.float32, np.float64]
    with self.test_env(tf) as env:
      val0 = np.ones(())
      #      vals = [env.numpy_to_itensor(val0, dtype=dt) for dt in types]
      #      vals = [env.tf.ones((), dtype=dt) for dt in types]
      vals = [env.tf.ones(())]
      for val in vals:
        for d in ["cpu:0", "gpu:0", "cpu:0", "gpu:0"]:
          with env.device(d):
            placed_val = env.tf.identity(val)
            for i in range(5):
              result = placed_val+placed_val
            print("Result for %s %s %s %s"%(result, d, 
                                            placed_val.tf_handle,
                                            result.tf_handle))

  def disabled_testOnes(self):
    try:
      with self.test_env(tf) as env:
        val1 = env.tf.ones(shape=(3, 3))
        self.assertAllEqual(val1.as_numpy(), np.ones((3, 3)))
    except:
      import sys, pdb, traceback
      type, value, tb = sys.exc_info()
      traceback.print_exc()
      pdb.post_mortem(tb)



    
  
if __name__ == "__main__":
  tf.test.main()

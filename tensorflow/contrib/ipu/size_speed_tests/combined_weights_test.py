from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.contrib.ipu import utils
from tensorflow.contrib.ipu import ops as ipu_ops
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest

# This verifies that in a replicated graph, the weights are copied to IPU 0 on
# the device as a merged set, then copied to the other IPUs as a merged set.

datatype = np.float16


def _get_variable(name, shape, init):
  return vs.get_variable(name, shape, initializer=init, dtype=datatype)


def inference(x):

  with vs.variable_scope('all', use_resource=True):
    x = fc("fc0", x, 256)
    x = fc("fc1", x, 256)
    x = fc("fc2", x, 256)
    x = fc("fc3", x, 256)
    x = fc("fc4", x, 256)
    x = fc("fc5", x, 256)
    x = fc("fc6", x, 256)
    x = fc("fc7", x, 256)
    x = fc("fc8", x, 256)
    x = fc("fc9", x, 256)

  return x


def fc(name, x, num_units_out):
  num_units_in = x.shape[1]
  weights_initializer = init_ops.truncated_normal_initializer(stddev=0.01)

  with vs.variable_scope(name):
    weights = _get_variable(
        'weights',
        shape=[num_units_in, num_units_out],
        init=weights_initializer)
    biases = _get_variable(
        'biases',
        shape=[num_units_out],
        init=init_ops.constant_initializer(0.0))

    x = nn_ops.xw_plus_b(x, weights, biases)

  return x


class CombinedWightsTest(test_util.TensorFlowTestCase):
  def testMergedWeightDownload(self):
    x = array_ops.placeholder(datatype, shape=[16, 4])
    y_ = array_ops.placeholder(datatype, shape=[16, 256])

    with ipu_ops.ipu_scope("/device:IPU:0"):
      logits = inference(x)

      loss = math_ops.reduce_mean(
          nn_ops.softmax_cross_entropy_with_logits_v2(
              logits=logits, labels=array_ops.stop_gradient(y_)))

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    opts = utils.create_ipu_config(profiling=True)
    opts = utils.set_ipu_model_options(opts, True)
    opts = utils.auto_select_ipus(opts, 1, sharded=False, number_of_replicas=2)
    utils.configure_ipu_system(opts)
    sess = sl.Session()

    sess.run(variables.global_variables_initializer())
    sess.run(report)

    data = np.zeros([16, 4])
    labels = np.zeros([16, 256])

    sess.run(loss, feed_dict={x: data, y_: labels})
    out = sess.run(report)

    sess.close()

    evts = utils.extract_all_events(out)
    r = utils.extract_compile_reports(out)
    self.assertEqual(len(r), 1)
    j = json.loads(r[0][1])

    # Find the switch
    switch_index = 0
    for p in j['programs']:
      if p['type'] == 'Switch':
        break
      switch_index = switch_index + 1

    # Find the first case - the download weights sequence
    download_weights_index = j['programs'][switch_index]['children'][0]

    # The download weights sequence should not have lots of entries (because the
    # copies will have been merged)
    self.assertTrue(len(j['programs'][download_weights_index]['children']) < 6)

    # Also check the overall size
    size = utils.get_memory_size_from_events(evts)
    self.assertTrue(size < 17600000)


if __name__ == "__main__":
  googletest.main()

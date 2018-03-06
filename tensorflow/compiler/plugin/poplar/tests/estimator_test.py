# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import random
import shutil
import glob

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.core.protobuf import config_pb2

import test_utils

def model_fn(features, labels, mode):
  # Set variables to resource variables
  vscope = tf.get_variable_scope()
  vscope.set_use_resource(True)

  with tf.device("/device:IPU:0"):
    x = tf.reshape(features, [-1, 4])
    x = tf.layers.dense(inputs=x, units=10)
    x = tf.layers.dense(inputs=x, units=3)

    if (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL):
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=labels))
    else:
      loss = None

    if mode == tf.estimator.ModeKeys.TRAIN:
      opt = tf.train.GradientDescentOptimizer(0.01)
      train = opt.minimize(loss, tf.train.get_global_step())
    else:
      train = None

  test_utils.ipu_compile_summary("compile_summary", [train, loss])

  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=x,
    loss=loss,
    train_op=train)

def gen_input():
  type = random.randint(0, 2)
  t = [random.random(), random.random(), random.random(), random.random()]
  t[type] += random.uniform(1.0, 3.0)
  v = [0,0,0]
  v[type] = 1.0
  yield (t, v)

def input_fn():
  dataset = tf.data.Dataset.from_generator(
    gen_input,
    (tf.float32, tf.float32),
    (tf.TensorShape([4]), tf.TensorShape([3])))
  dataset = dataset.batch(4)
  return dataset.make_one_shot_iterator().get_next()



class IpuEstimatorTest(test_util.TensorFlowTestCase):

  def testTrain(self):

    shutil.rmtree("testlogs", True)

    opts = config_pb2.IPUOptions()
    dev = opts.device_config.add()
    dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
    dev.enable_profile = True

    sess_cfg = tf.ConfigProto(ipu_options=opts)
    run_cfg = tf.estimator.RunConfig(session_config=sess_cfg)

    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        config=run_cfg,
                                        model_dir="testlogs")

    classifier.train(input_fn=input_fn, steps=16)

    event_file = glob.glob("testlogs/event*")

    self.assertTrue(len(event_file) == 1)

    compile_count = 0
    for summary in tf.train.summary_iterator(event_file[0]):
      for val in summary.summary.value:
        if val.tag == "compile_summary":
          compile_count += len(val.tensor.string_val)

    self.assertTrue(compile_count == 2)


if __name__ == "__main__":
  googletest.main()

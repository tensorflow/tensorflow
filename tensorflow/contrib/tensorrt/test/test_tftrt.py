# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Script to test TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np


def getSimpleGraphDef():
  """Create a simple graph and return its graph_def"""
  g = tf.Graph()
  with g.as_default():
    A = tf.placeholder(dtype=tf.float32, shape=(None, 24, 24, 2), name="input")
    e = tf.constant(
        [[[[1., 0.5, 4., 6., 0.5, 1.], [1., 0.5, 1., 1., 0.5, 1.]]]],
        name="weights",
        dtype=tf.float32)
    conv = tf.nn.conv2d(
        input=A, filter=e, strides=[1, 2, 2, 1], padding="SAME", name="conv")
    b = tf.constant([4., 1.5, 2., 3., 5., 7.], name="bias", dtype=tf.float32)
    t = tf.nn.bias_add(conv, b, name="biasAdd")
    relu = tf.nn.relu(t, "relu")
    idty = tf.identity(relu, "ID")
    v = tf.nn.max_pool(
        idty, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
    out = tf.squeeze(v, name="output")
  return g.as_graph_def()


def runGraph(gdef, dumm_inp):
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
  tf.reset_default_graph()
  g = tf.Graph()
  with g.as_default():
    inp, out = tf.import_graph_def(
        graph_def=gdef, return_elements=["input", "output"])
    inp = inp.outputs[0]
    out = out.outputs[0]
  with tf.Session(
      config=tf.ConfigProto(gpu_options=gpu_options), graph=g) as sess:
    val = sess.run(out, {inp: dumm_inp})
  return val


if "__main__" in __name__:
  inpDims = (100, 24, 24, 2)
  dummy_input = np.random.random_sample(inpDims)
  gdef = getSimpleGraphDef()
  trt_graph = trt.CreateInferenceGraph(gdef, ["output"],
                                       inpDims[0])  # Get optimized graph
  o1 = runGraph(gdef, dummy_input)
  o2 = runGraph(trt_graph, dummy_input)
  assert (np.array_equal(o1, o2))
  print("Pass")

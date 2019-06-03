# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow as tf
from tensorflow.lite.toco import model_flags_pb2
from tensorflow.lite.toco import toco_flags_pb2
from tensorflow.lite.toco import types_pb2
from tensorflow.python.platform import googletest
from tensorflow.python.platform import resource_loader


def TensorName(x):
  """Get the canonical (non foo:0 name)."""
  return x.name.split(":")[0]


class TocoFromProtosTest(googletest.TestCase):

  def _run(self, sess, in_tensor, out_tensor, should_succeed):
    """Use toco binary to check conversion from graphdef to tflite.

    Args:
      sess: Active TensorFlow session containing graph.
      in_tensor: TensorFlow tensor to use as input.
      out_tensor: TensorFlow tensor to use as output.
      should_succeed: Whether this is a valid conversion.
    """
    # Build all protos and extract graphdef
    graph_def = sess.graph_def
    toco_flags = toco_flags_pb2.TocoFlags()
    toco_flags.input_format = toco_flags_pb2.TENSORFLOW_GRAPHDEF
    toco_flags.output_format = toco_flags_pb2.TFLITE
    toco_flags.inference_input_type = types_pb2.FLOAT
    toco_flags.inference_type = types_pb2.FLOAT
    toco_flags.allow_custom_ops = True
    model_flags = model_flags_pb2.ModelFlags()
    input_array = model_flags.input_arrays.add()
    input_array.name = TensorName(in_tensor)
    input_array.shape.dims.extend(map(int, in_tensor.shape))
    model_flags.output_arrays.append(TensorName(out_tensor))
    # Shell out to run toco (in case it crashes)
    with tempfile.NamedTemporaryFile() as fp_toco, \
           tempfile.NamedTemporaryFile() as fp_model, \
           tempfile.NamedTemporaryFile() as fp_input, \
           tempfile.NamedTemporaryFile() as fp_output:
      fp_model.write(model_flags.SerializeToString())
      fp_toco.write(toco_flags.SerializeToString())
      fp_input.write(graph_def.SerializeToString())
      fp_model.flush()
      fp_toco.flush()
      fp_input.flush()
      tflite_bin = resource_loader.get_path_to_datafile("toco_from_protos.par")
      cmdline = " ".join([
          tflite_bin, fp_model.name, fp_toco.name, fp_input.name, fp_output.name
      ])
      exitcode = os.system(cmdline)
      if exitcode == 0:
        stuff = fp_output.read()
        self.assertEqual(stuff is not None, should_succeed)
      else:
        self.assertFalse(should_succeed)

  def test_toco(self):
    """Run a couple of TensorFlow graphs against TOCO through the python bin."""
    with tf.Session() as sess:
      img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
      val = img + tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
      out = tf.identity(val, name="out")
      out2 = tf.sin(val, name="out2")
      # This is a valid mdoel
      self._run(sess, img, out, True)
      # This uses an invalid function.
      # TODO(aselle): Check to make sure a warning is included.
      self._run(sess, img, out2, True)
      # This is an identity graph, which doesn't work
      self._run(sess, img, img, False)


if __name__ == "__main__":
  googletest.main()

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for signature_def_util.py.

   - Tests adding a SignatureDef to TFLite metadata.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.lite.tools.signature import signature_def_utils


class SignatureDefUtilsTest(tf.test.TestCase):

  def testAddSignatureDefToFlatbufferMetadata(self):
    """Test a SavedModel conversion has correct Metadata."""
    filename = tf.compat.v1.resource_loader.get_path_to_datafile(
        '../../testdata/add.bin')
    if not tf.io.gfile.exists(filename):
      raise IOError('File "{0}" does not exist in {1}.'.format(
          filename,
          tf.compat.v1.resource_loader.get_root_dir_with_all_resources()))

    with tf.io.gfile.GFile(filename, 'rb') as fp:
      tflite_model = bytearray(fp.read())

    self.assertIsNotNone(tflite_model, 'TFLite model is none')
    sig_input_tensor = meta_graph_pb2.TensorInfo(
        dtype=tf.as_dtype(tf.float32).as_datatype_enum,
        tensor_shape=tf.TensorShape([1, 8, 8, 3]).as_proto())
    sig_input_tensor_signature = {'x': sig_input_tensor}
    sig_output_tensor = meta_graph_pb2.TensorInfo(
        dtype=tf.as_dtype(tf.float32).as_datatype_enum,
        tensor_shape=tf.TensorShape([1, 8, 8, 3]).as_proto())
    sig_output_tensor_signature = {'y': sig_output_tensor}
    predict_signature_def = (
        tf.compat.v1.saved_model.build_signature_def(
            sig_input_tensor_signature, sig_output_tensor_signature,
            tf.saved_model.PREDICT_METHOD_NAME))
    serving_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    signature_def_map = {serving_key: predict_signature_def}
    tflite_model = signature_def_utils.set_signature_defs(
        tflite_model, signature_def_map)
    saved_signature_def_map = signature_def_utils.get_signature_defs(
        tflite_model)
    signature_def = saved_signature_def_map.get(serving_key)
    self.assertIsNotNone(signature_def, 'SignatureDef not found')
    self.assertEqual(signature_def.SerializeToString(),
                     predict_signature_def.SerializeToString())
    remove_tflite_model = (
        signature_def_utils.clear_signature_defs(tflite_model))
    signature_def_map = signature_def_utils.get_signature_defs(
        remove_tflite_model)
    self.assertIsNone(signature_def_map.get(serving_key),
                      'SignatureDef found, but should be missing')


if __name__ == '__main__':
  tf.test.main()

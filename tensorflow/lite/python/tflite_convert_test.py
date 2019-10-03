# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tflite_convert.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model
from tensorflow.python.saved_model.save import save
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.training_util import write_graph


class TestModels(test_util.TensorFlowTestCase):

  def _getFilepath(self, filename):
    return os.path.join(self.get_temp_dir(), filename)

  def _run(self, flags_str, should_succeed):
    output_file = os.path.join(self.get_temp_dir(), 'model.tflite')
    tflite_bin = resource_loader.get_path_to_datafile('tflite_convert.par')
    cmdline = '{0} --output_file={1} {2}'.format(tflite_bin, output_file,
                                                 flags_str)

    exitcode = os.system(cmdline)
    if exitcode == 0:
      with gfile.Open(output_file, 'rb') as model_file:
        content = model_file.read()
      self.assertEqual(content is not None, should_succeed)
      os.remove(output_file)
    else:
      self.assertFalse(should_succeed)

  def _getKerasModelFile(self):
    x = np.array([[1.], [2.]])
    y = np.array([[2.], [4.]])

    model = keras.models.Sequential([
        keras.layers.Dropout(0.2, input_shape=(1,)),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=1)

    keras_file = self._getFilepath('model.h5')
    keras.models.save_model(model, keras_file)
    return keras_file


class TfLiteConvertV1Test(TestModels):

  def _run(self, flags_str, should_succeed):
    if tf2.enabled():
      flags_str += ' --enable_v1_converter'
    super(TfLiteConvertV1Test, self)._run(flags_str, should_succeed)

  def testFrozenGraphDef(self):
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32)
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = self._getFilepath('model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    flags_str = ('--graph_def_file={0} --input_arrays={1} '
                 '--output_arrays={2}'.format(graph_def_file,
                                              'Placeholder', 'add'))
    self._run(flags_str, should_succeed=True)
    os.remove(graph_def_file)

  def testSavedModel(self):
    saved_model_dir = self._getFilepath('model')
    with ops.Graph().as_default():
      with session.Session() as sess:
        in_tensor = array_ops.placeholder(
            shape=[1, 16, 16, 3], dtype=dtypes.float32, name='inputB')
        out_tensor = in_tensor + in_tensor
        inputs = {'x': in_tensor}
        outputs = {'z': out_tensor}
        saved_model.simple_save(sess, saved_model_dir, inputs, outputs)

    flags_str = '--saved_model_dir={}'.format(saved_model_dir)
    self._run(flags_str, should_succeed=True)

  def testKerasFile(self):
    keras_file = self._getKerasModelFile()

    flags_str = '--keras_model_file={}'.format(keras_file)
    self._run(flags_str, should_succeed=True)
    os.remove(keras_file)

  def testKerasFileMLIR(self):
    keras_file = self._getKerasModelFile()

    flags_str = ('--keras_model_file={} --experimental_enable_mlir_converter'
                 .format(keras_file))
    self._run(flags_str, should_succeed=True)
    os.remove(keras_file)


class TfLiteConvertV2Test(TestModels):

  @test_util.run_v2_only
  def testSavedModel(self):
    input_data = constant_op.constant(1., shape=[1])
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: 2. * x)
    to_save = root.f.get_concrete_function(input_data)

    saved_model_dir = self._getFilepath('model')
    save(root, saved_model_dir, to_save)

    flags_str = '--saved_model_dir={}'.format(saved_model_dir)
    self._run(flags_str, should_succeed=True)

  @test_util.run_v2_only
  def testKerasFile(self):
    keras_file = self._getKerasModelFile()

    flags_str = '--keras_model_file={}'.format(keras_file)
    self._run(flags_str, should_succeed=True)
    os.remove(keras_file)

  @test_util.run_v2_only
  def testKerasFileMLIR(self):
    keras_file = self._getKerasModelFile()

    flags_str = ('--keras_model_file={} --experimental_enable_mlir_converter'
                 .format(keras_file))
    self._run(flags_str, should_succeed=True)
    os.remove(keras_file)

  def testMissingRequired(self):
    self._run('--invalid_args', should_succeed=False)

  def testMutuallyExclusive(self):
    self._run(
        '--keras_model_file=model.h5 --saved_model_dir=/tmp/',
        should_succeed=False)


if __name__ == '__main__':
  test.main()

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
from tensorflow.python.ops import random_ops
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
    tflite_bin = resource_loader.get_path_to_datafile('tflite_convert')
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

  # Run `tflite_convert` explicitly with the legacy converter.
  # Before the new converter is enabled by default, this flag has no real
  # effects.
  def testFrozenGraphDefWithLegacyConverter(self):
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
                 '--output_arrays={2} --experimental_legacy_converter'.format(
                     graph_def_file, 'Placeholder', 'add'))
    self._run(flags_str, should_succeed=True)
    os.remove(graph_def_file)

  def testFrozenGraphDefNonPlaceholder(self):
    with ops.Graph().as_default():
      in_tensor = random_ops.random_normal(shape=[1, 16, 16, 3], name='random')
      _ = in_tensor + in_tensor
      sess = session.Session()

    # Write graph to file.
    graph_def_file = self._getFilepath('model.pb')
    write_graph(sess.graph_def, '', graph_def_file, False)
    sess.close()

    flags_str = ('--graph_def_file={0} --input_arrays={1} '
                 '--output_arrays={2}'.format(graph_def_file,
                                              'random', 'add'))
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

    flags_str = ('--keras_model_file={} --experimental_new_converter'
                 .format(keras_file))
    self._run(flags_str, should_succeed=True)
    os.remove(keras_file)

  def testConversionSummary(self):
    keras_file = self._getKerasModelFile()
    log_dir = self.get_temp_dir()

    flags_str = ('--keras_model_file={} --experimental_new_converter  '
                 '--conversion_summary_dir={}'.format(keras_file, log_dir))
    self._run(flags_str, should_succeed=True)
    os.remove(keras_file)

    num_items_conversion_summary = len(os.listdir(log_dir))
    self.assertTrue(num_items_conversion_summary)

  def testConversionSummaryWithOldConverter(self):
    keras_file = self._getKerasModelFile()
    log_dir = self.get_temp_dir()

    flags_str = ('--keras_model_file={} '
                 '--conversion_summary_dir={}'.format(keras_file, log_dir))
    self._run(flags_str, should_succeed=True)
    os.remove(keras_file)

    num_items_conversion_summary = len(os.listdir(log_dir))
    self.assertEqual(num_items_conversion_summary, 0)

  def _initObjectDetectionArgs(self):
    # Initializes the arguments required for the object detection model.
    # Looks for the model file which is saved in a different location internally
    # and externally.
    filename = resource_loader.get_path_to_datafile('testdata/tflite_graph.pb')
    if not os.path.exists(filename):
      filename = os.path.join(
          resource_loader.get_root_dir_with_all_resources(),
          '../tflite_mobilenet_ssd_quant_protobuf/tflite_graph.pb')
      if not os.path.exists(filename):
        raise IOError("File '{0}' does not exist.".format(filename))

    self._graph_def_file = filename
    self._input_arrays = 'normalized_input_image_tensor'
    self._output_arrays = (
        'TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,'
        'TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3')
    self._input_shapes = '1,300,300,3'

  def testObjectDetection(self):
    """Tests object detection model through TOCO."""
    self._initObjectDetectionArgs()
    flags_str = ('--graph_def_file={0} --input_arrays={1} '
                 '--output_arrays={2} --input_shapes={3} '
                 '--allow_custom_ops'.format(self._graph_def_file,
                                             self._input_arrays,
                                             self._output_arrays,
                                             self._input_shapes))
    self._run(flags_str, should_succeed=True)

  def testObjectDetectionMLIR(self):
    """Tests object detection model through MLIR converter."""
    self._initObjectDetectionArgs()
    custom_opdefs_str = (
        'name: \'TFLite_Detection_PostProcess\' '
        'input_arg: { name: \'raw_outputs/box_encodings\' type: DT_FLOAT } '
        'input_arg: { name: \'raw_outputs/class_predictions\' type: DT_FLOAT } '
        'input_arg: { name: \'anchors\' type: DT_FLOAT } '
        'output_arg: { name: \'TFLite_Detection_PostProcess\' type: DT_FLOAT } '
        'output_arg: { name: \'TFLite_Detection_PostProcess:1\' '
        'type: DT_FLOAT } '
        'output_arg: { name: \'TFLite_Detection_PostProcess:2\' '
        'type: DT_FLOAT } '
        'output_arg: { name: \'TFLite_Detection_PostProcess:3\' '
        'type: DT_FLOAT } '
        'attr : { name: \'h_scale\' type: \'float\'} '
        'attr : { name: \'max_classes_per_detection\' type: \'int\'} '
        'attr : { name: \'max_detections\' type: \'int\'} '
        'attr : { name: \'nms_iou_threshold\' type: \'float\'} '
        'attr : { name: \'nms_score_threshold\' type: \'float\'} '
        'attr : { name: \'num_classes\' type: \'int\'} '
        'attr : { name: \'w_scale\' type: \'int\'} '
        'attr : { name: \'x_scale\' type: \'int\'} '
        'attr : { name: \'y_scale\' type: \'int\'}')

    flags_str = ('--graph_def_file={0} --input_arrays={1} '
                 '--output_arrays={2} --input_shapes={3} '
                 '--custom_opdefs="{4}"'.format(self._graph_def_file,
                                                self._input_arrays,
                                                self._output_arrays,
                                                self._input_shapes,
                                                custom_opdefs_str))

    # Ensure --experimental_new_converter.
    flags_str_final = ('{} --allow_custom_ops').format(flags_str)
    self._run(flags_str_final, should_succeed=False)

    # Ensure --allow_custom_ops.
    flags_str_final = ('{} --experimental_new_converter').format(flags_str)
    self._run(flags_str_final, should_succeed=False)

    # Valid conversion.
    flags_str_final = ('{} --allow_custom_ops '
                       '--experimental_new_converter').format(flags_str)
    self._run(flags_str_final, should_succeed=True)


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

    flags_str = ('--keras_model_file={} --experimental_new_converter'
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

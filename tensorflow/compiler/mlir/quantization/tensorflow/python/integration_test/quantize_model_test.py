# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for quantize_model."""

from absl.testing import parameterized
import numpy as np
import tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.quantization.tensorflow.python import quantize_model
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader_impl as saved_model_loader
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training.tracking import tracking


def _contains_quantized_function_call(meta_graphdef):
  """Returns true if the graph def has quantized function call."""
  for func in meta_graphdef.graph_def.library.function:
    if func.signature.name.startswith('quantized_'):
      return True
  return False


class QuantizationTest(test.TestCase, parameterized.TestCase):

  def test_qat_model(self):

    class QATModelWithAdd(tracking.AutoTrackable):
      """Basic model with Fake quant + add."""

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[10], dtype=dtypes.float32, name='x'),
          tensor_spec.TensorSpec(shape=[10], dtype=dtypes.float32, name='y')
      ])
      def add(self, x, y):
        float_res = math_ops.add(x, y)
        x = array_ops.fake_quant_with_min_max_args(
            x, min=-0.1, max=0.2, num_bits=8, narrow_range=False)
        y = array_ops.fake_quant_with_min_max_args(
            y, min=-0.3, max=0.4, num_bits=8, narrow_range=False)
        res = math_ops.add(x, y)
        res = array_ops.fake_quant_with_min_max_args(
            res, min=-0.4, max=0.6, num_bits=8, narrow_range=False)
        return {'output': res, 'float_output': float_res}

    root = QATModelWithAdd()

    temp_path = self.create_tempdir().full_path
    saved_model_save.save(
        root, temp_path, signatures=root.add.get_concrete_function())

    output_directory = self.create_tempdir().full_path
    tags = [tag_constants.SERVING]
    model = quantize_model.quantize(temp_path, ['serving_default'],
                                    [tag_constants.SERVING], output_directory)
    self.assertIsNotNone(model)
    self.assertEqual(
        list(model.signatures._signatures.keys()), ['serving_default'])
    func = model.signatures['serving_default']
    func_res = func(
        x=array_ops.constant(0.1, shape=[10]),
        y=array_ops.constant(0.1, shape=[10]))
    self.assertAllClose(
        func_res['output'], array_ops.constant(0.2, shape=[10]), atol=0.01)
    self.assertAllClose(
        func_res['float_output'],
        array_ops.constant(0.2, shape=[10]),
        atol=1e-3)

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(_contains_quantized_function_call(output_meta_graphdef))

  def test_ptq_model(self):

    class PTQModelWithAdd(tracking.AutoTrackable):
      """Basic model with addition."""

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[10], dtype=dtypes.float32, name='x'),
          tensor_spec.TensorSpec(shape=[10], dtype=dtypes.float32, name='y')
      ])
      def add(self, x, y):
        res = math_ops.add(x, y)
        return {'output': res, 'x': x, 'y': y}

    def data_gen():
      for _ in range(255):
        yield {
            'x':
                ops.convert_to_tensor(
                    np.random.uniform(size=(10)).astype('f4')),
            'y':
                ops.convert_to_tensor(
                    np.random.uniform(size=(10)).astype('f4'))
        }

    root = PTQModelWithAdd()

    temp_path = self.create_tempdir().full_path
    saved_model_save.save(
        root, temp_path, signatures=root.add.get_concrete_function())

    output_directory = self.create_tempdir().full_path
    tags = [tag_constants.SERVING]
    model = quantize_model.quantize(
        temp_path, ['serving_default'],
        tags,
        output_directory,
        representative_dataset=data_gen)
    self.assertIsNotNone(model)
    self.assertEqual(
        list(model.signatures._signatures.keys()), ['serving_default'])
    func = model.signatures['serving_default']
    func_res = func(
        x=array_ops.constant(0.1, shape=[10]),
        y=array_ops.constant(0.1, shape=[10]))
    self.assertAllClose(
        func_res['output'], array_ops.constant(0.2, shape=[10]), atol=0.01)
    xy_atol = 1e-6
    self.assertAllClose(
        func_res['x'], array_ops.constant(0.1, shape=[10]), atol=xy_atol)
    self.assertAllClose(
        func_res['y'], array_ops.constant(0.1, shape=[10]), atol=xy_atol)

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(_contains_quantized_function_call(output_meta_graphdef))

  @parameterized.named_parameters(
      ('none', None),
      ('relu', nn_ops.relu),
      ('relu6', nn_ops.relu6),
  )
  def test_qat_conv_model(self, activation_fn):

    class ConvModel(module.Module):

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(
              name='input', shape=[1, 3, 4, 3], dtype=dtypes.float32),
          tensor_spec.TensorSpec(
              name='filter', shape=[2, 3, 3, 2], dtype=dtypes.float32),
      ])
      def conv(self, input_tensor, filter_tensor):
        q_input = array_ops.fake_quant_with_min_max_args(
            input_tensor, min=-0.1, max=0.2, num_bits=8, narrow_range=False)
        q_filters = array_ops.fake_quant_with_min_max_args(
            filter_tensor, min=-1.0, max=2.0, num_bits=8, narrow_range=False)
        bias = array_ops.constant([0, 0], dtype=dtypes.float32)
        out = nn_ops.conv2d(
            q_input,
            q_filters,
            strides=[1, 1, 2, 1],
            dilations=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC')
        out = nn_ops.bias_add(out, bias, data_format='NHWC')
        if activation_fn is not None:
          out = activation_fn(out)
        q_out = array_ops.fake_quant_with_min_max_args(
            out, min=-0.3, max=0.4, num_bits=8, narrow_range=False)
        return {'output': q_out}

    model = ConvModel()
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = [tag_constants.SERVING]
    output_directory = self.create_tempdir().full_path
    converted_model = quantize_model.quantize(
        input_saved_model_path, [signature_key],
        tags,
        output_directory=output_directory)
    self.assertIsNotNone(converted_model)
    self.assertEqual(
        list(converted_model.signatures._signatures.keys()), [signature_key])

    input_data = np.random.uniform(
        low=-0.1, high=0.2, size=(1, 3, 4, 3)).astype('f4')
    filter_data = np.random.uniform(
        low=-0.5, high=0.5, size=(2, 3, 3, 2)).astype('f4')

    expected_outputs = model.conv(input_data, filter_data)
    got_outputs = converted_model.signatures[signature_key](
        input=ops.convert_to_tensor(input_data),
        filter=ops.convert_to_tensor(filter_data))
    # TODO(b/215633216): Check if the accuracy is acceptable.
    self.assertAllClose(expected_outputs, got_outputs, atol=0.01)

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(_contains_quantized_function_call(output_meta_graphdef))

  def test_conv_ptq_model(self):

    class ConvModel(module.Module):

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[1, 3, 4, 3], dtype=dtypes.float32)
      ])
      def conv(self, input_tensor):
        filters = np.random.uniform(
            low=-10, high=10, size=(2, 3, 3, 2)).astype('f4')
        bias = np.random.uniform(low=0, high=10, size=(2)).astype('f4')
        out = nn_ops.conv2d(
            input_tensor,
            filters,
            strides=[1, 1, 2, 1],
            dilations=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC')
        out = nn_ops.bias_add(out, bias, data_format='NHWC')
        out = nn_ops.relu6(out)
        return {'output': out}

    model = ConvModel()
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    def data_gen():
      for _ in range(255):
        yield {
            'input_tensor':
                ops.convert_to_tensor(
                    np.random.uniform(low=0, high=150,
                                      size=(1, 3, 4, 3)).astype('f4')),
        }

    tags = [tag_constants.SERVING]
    output_directory = self.create_tempdir().full_path
    converted_model = quantize_model.quantize(
        input_saved_model_path, ['serving_default'],
        tags,
        output_directory,
        representative_dataset=data_gen)
    self.assertIsNotNone(converted_model)
    self.assertEqual(
        list(converted_model.signatures._signatures.keys()),
        ['serving_default'])

    output_loader = saved_model_loader.SavedModelLoader(output_directory)
    output_meta_graphdef = output_loader.get_meta_graph_def_from_tags(tags)
    self.assertTrue(_contains_quantized_function_call(output_meta_graphdef))


if __name__ == '__main__':
  test.main()

# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
import os
import re
from typing import Mapping, Optional, Sequence

from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from tensorflow.compiler.mlir.quantization.common.python import testing
from tensorflow.compiler.mlir.quantization.stablehlo import quantization_config_pb2 as qc
from tensorflow.compiler.mlir.quantization.stablehlo.python import quantization
from tensorflow.compiler.mlir.quantization.stablehlo.python.integration_test import quantize_model_test_base
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.types import core

_CalibrationMethod = qc.CalibrationOptions.CalibrationMethod


# Test cases for Static Range Quantization.
# Tries to run all tests cases in both the graph mode (default in TF1) and the
# eager mode (default in TF2) to ensure support for when TF2 is disabled.
class StaticRangeQuantizationTest(quantize_model_test_base.QuantizedModelTest):

  @parameterized.parameters(
      testing.parameter_combinations([{
          'bias_fn': (
              None,
              nn_ops.bias_add,
          ),
          'activation_fn': (
              None,
              nn_ops.relu,
              nn_ops.relu6,
          ),
          'dim_sizes': (
              # tf.MatMul cases.
              ([None, 1024], [1024, 3]),  # dynamic batch dim.
              ([1, 1024], [1024, 3]),
              # tf.BatchMatMul cases.
              ([10, 1, 1024], [10, 1024, 3]),
              ([2, 3, 1, 1024], [2, 3, 1024, 3]),
          ),
          'merge_fusion_with_dequantize': (False, True),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_ptq_model(
      self,
      bias_fn: Optional[ops.Operation],
      activation_fn: Optional[ops.Operation],
      dim_sizes: Sequence[int],
      merge_fusion_with_dequantize: bool,
  ):
    lhs_dim_size, rhs_dim_size = dim_sizes
    input_shape = (*lhs_dim_size,)
    filter_shape = (*rhs_dim_size,)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]
    model = self._create_matmul_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        bias_fn,
        activation_fn,
    )

    rng = np.random.default_rng(seed=42)
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        pipeline_config=qc.PipelineConfig(
            merge_fusion_with_dequantize=merge_fusion_with_dequantize
        ),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.matmul(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    module_str = self._extract_first_xla_call_module_op(
        self._output_saved_model_path
    )
    self.assertTrue(re.search('stablehlo.dot_general.*xi8>', module_str))
    if bias_fn:
      self.assertTrue(re.search('stablehlo.add.*xi32>', module_str))
    # Consider if there is a way to check if activation fusion is properly
    # done in MLIR level.
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.3, atol=0.2)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.65,
    )

    if merge_fusion_with_dequantize:
      # Check activation functions are explicitly present.
      # If present the last op before return should be stablehlo.clamp for relu6
      # and stablehlo.maximum for relu.
      if activation_fn is nn_ops.relu6:
        self.assertRegex(module_str, r'stablehlo.clamp.*\n.*return')
      elif activation_fn is nn_ops.relu:
        self.assertRegex(module_str, r'stablehlo.maximum.*\n.*return')
    else:
      # Check activation functions are implicit.
      self.assertNotRegex(module_str, r'stablehlo.clamp.*\n.*return')
      self.assertNotRegex(module_str, r'stablehlo.maximum.*\n.*return')

  @parameterized.parameters(
      testing.parameter_combinations([{
          'same_scale_op': (
              'concatenate',
              'gather',
              'max_pool',
              'pad',
              'reshape',
              'select',
              'slice',
              'transpose',
          ),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_and_same_scale_ptq_model(
      self,
      same_scale_op: str,
  ):
    input_shape = (2, 3, 1, 1024)
    filter_shape = (2, 3, 1024, 3)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]

    model = self._create_matmul_and_same_scale_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        same_scale_op,
    )

    rng = np.random.default_rng(seed=42)
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.matmul_and_same_scale(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.03, atol=0.2)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.65,
    )

  @parameterized.parameters(
      testing.parameter_combinations([{
          'same_scale_op': (
              'reshape',  # This corresponds to stablehlo.dynamic_reshape
              'slice',  # This corresponds to stablehlo.dynamic_slice.
              # TODO: b/326242075 - Support other same-scale ops.
          ),
          'dim_sizes': (([None, 1024], [1024, 3]),),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_and_same_scale_ptq_model_dynamic(
      self,
      same_scale_op: str,
      dim_sizes: Sequence[int],
  ):
    input_dim_size, filter_dim_size = dim_sizes
    input_shape = (*input_dim_size,)
    filter_shape = (*filter_dim_size,)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]

    model = self._create_matmul_and_same_scale_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        same_scale_op,
    )

    rng = np.random.default_rng(seed=42)
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.matmul_and_same_scale(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.03, atol=0.2)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.6,
    )

  @parameterized.parameters(
      testing.parameter_combinations([{
          'bias_fn': (
              None,
              nn_ops.bias_add,
          ),
          'activation_fn': (
              None,
              nn_ops.relu,
              nn_ops.relu6,
          ),
          'has_batch_norm': (False, True),
          'input_shape_dynamic': (
              False,
              True,
          ),
          'enable_per_channel_quantized_weight': (
              False,
              True,
          ),
          'merge_fusion_with_dequantize': (False, True),
          'has_func_alias': (False, True),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_ptq_model(
      self,
      bias_fn: Optional[ops.Operation],
      activation_fn: Optional[ops.Operation],
      has_batch_norm: bool,
      input_shape_dynamic: bool,
      enable_per_channel_quantized_weight: bool,
      merge_fusion_with_dequantize: bool,
      dilations: Sequence[int] = None,
      has_func_alias: bool = False,
  ):
    input_shape = (None, 3, 4, 3) if input_shape_dynamic else (1, 3, 4, 3)
    filter_shape = (2, 3, 3, 2)
    strides = (1, 1, 1, 1)
    model = self._create_conv2d_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        bias_fn,
        activation_fn,
        has_batch_norm,
        strides,
        dilations,
        'SAME',
        has_func_alias,
    )
    # TODO: b/331809306 - Investigate why these test fail then re-enable.
    if has_batch_norm and (bias_fn or not input_shape_dynamic):
      return

    # TODO: b/331120943 - Re-enable this after correctly handling quantization
    # granularity per quantizable scope.
    if has_batch_norm and (not bias_fn and input_shape_dynamic):
      return

    # Generate model input data.
    rng = np.random.default_rng(seed=42)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ],
            enable_per_channel_quantized_weight=enable_per_channel_quantized_weight,
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        pipeline_config=qc.PipelineConfig(
            merge_fusion_with_dequantize=merge_fusion_with_dequantize
        ),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.conv2d(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    module_str = self._extract_first_xla_call_module_op(
        self._output_saved_model_path
    )
    self.assertTrue(re.search('stablehlo.convolution.*xi8>', module_str))
    if bias_fn:
      self.assertTrue(re.search('stablehlo.add.*xi32>', module_str))
    # Consider if there is a way to check if activation fusion is properly
    # done in MLIR level.
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.02, atol=0.05)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.61,
    )

    if merge_fusion_with_dequantize:
      # Check activation functions are explicitly present.
      # If present the last op before return should be stablehlo.clamp for relu6
      # and stablehlo.maximum for relu.
      if activation_fn is nn_ops.relu6:
        self.assertRegex(module_str, r'stablehlo.clamp.*\n.*return')
      elif activation_fn is nn_ops.relu:
        self.assertRegex(module_str, r'stablehlo.maximum.*\n.*return')
    else:
      # Check activation functions are implicit.
      self.assertNotRegex(module_str, r'stablehlo.clamp.*\n.*return')
      self.assertNotRegex(module_str, r'stablehlo.maximum.*\n.*return')

    if has_func_alias:
      func_aliases = self._get_function_aliases(
          self._output_saved_model_path, [tag_constants.SERVING]
      )
      self.assertCountEqual(
          func_aliases.values(), [quantize_model_test_base.FUNC_ALIAS]
      )

  @parameterized.parameters(
      testing.parameter_combinations([{
          'equation': (
              'abc,cde->abde',
              'abc,dce->abde',
          ),
      }])
  )
  def test_einsum_ptq_model(
      self,
      equation: str,
  ):
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes(equation, use_bias=True)
    )

    model = self._create_einsum_model(
        self._input_saved_model_path,
        equation,
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
    )

    # Generate model input data.
    rng = np.random.default_rng(seed=42)
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=x_signature).astype('f4')
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(100):
        yield {
            'x': ops.convert_to_tensor(
                np.random.uniform(low=0.0, high=1.0, size=x_signature).astype(
                    'f4'
                )
            ),
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.einsum_with_kernel(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        x=ops.convert_to_tensor(input_data)
    )
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.02, atol=0.04)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.65,
    )

  def test_reuse_calibration_data(self):
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes('abc,cde->abde', use_bias=True)
    )

    self._create_einsum_model(
        self._input_saved_model_path,
        'abc,cde->abde',
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
    )

    # Generate model input data.
    rng = np.random.default_rng(seed=42)
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=x_signature).astype('f4')
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(100):
        yield {
            'x': ops.convert_to_tensor(
                np.random.uniform(low=0.0, high=1.0, size=x_signature).astype(
                    'f4'
                )
            ),
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    calibration_data_dir = self.create_tempdir('calibration_data').full_path
    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        calibration_options=qc.CalibrationOptions(
            calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX,
            calibration_data_dir=calibration_data_dir,
        ),
    )

    # Run quantization the first time, calibration is expected to be run.
    with self.assertLogs(level='INFO') as info_logs:
      quantization.quantize_saved_model(
          self._input_saved_model_path,
          self._output_saved_model_path,
          config,
      )
      self.assertTrue(
          self._any_log_contains(
              'Calibration step is executed in graph mode.',
              info_logs.records,
          )
      )
      module_str = self._extract_first_xla_call_module_op(
          self._output_saved_model_path
      )
      self.assertTrue(
          re.search('stablehlo.dot_general.*xi8>.*xi8>.*xi32>', module_str)
      )

    # Run quantization the first time, calibration is expected to be skipped.
    output_saved_model_path_2 = self.create_tempdir('output2').full_path
    with self.assertLogs(level='INFO') as info_logs:
      quantization.quantize_saved_model(
          self._input_saved_model_path,
          output_saved_model_path_2,
          config,
      )
      self.assertFalse(
          self._any_log_contains(
              'Calibration step is executed in graph mode.',
              info_logs.records,
          )
      )
      module_str = self._extract_first_xla_call_module_op(
          output_saved_model_path_2
      )
      self.assertTrue(
          re.search('stablehlo.dot_general.*xi8>.*xi8>.*xi32>', module_str)
      )

    # Expect both quantized model to produce the same results.
    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})
    new_outputs_1 = root.signatures['serving_default'](
        x=ops.convert_to_tensor(input_data)
    )

    root = load.load(output_saved_model_path_2)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})
    new_outputs_2 = root.signatures['serving_default'](
        x=ops.convert_to_tensor(input_data)
    )

    self.assertAllClose(new_outputs_1, new_outputs_2)

  @parameterized.named_parameters(
      ('use_constant_with_int32_input', np.int32, False),
      ('use_variable_with_int32_input', np.int32, True),
      ('use_constant_with_int64_input', np.int64, False),
      ('use_variable_with_int64_input', np.int64, True),
  )
  @test_util.run_v2_only
  def test_gather_model(self, input_type, use_variable):
    model = self._create_gather_model(input_type, use_variable)

    save.save(model, self._input_saved_model_path)

    rng = np.random.default_rng(seed=42)
    static_input_shape = [6]

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=10, size=static_input_shape
            ).astype(input_type)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})
    module_str = self._extract_first_xla_call_module_op(
        self._output_saved_model_path
    )
    self.assertTrue(re.search('stablehlo.gather.*xi8>', module_str))

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        1 / 3,
    )

  def test_when_preset_not_srq_raises_error(self):
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    config = qc.QuantizationConfig()
    with self.assertRaisesRegex(ValueError, 'only supports static-range PTQ'):
      quantization.quantize_saved_model(
          self._input_saved_model_path,
          self._output_saved_model_path,
          config,
      )

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_denylist_basic(self):
    """Tests that the op is not quantized when no quantization is enabled."""
    input_shape = (1, 2)
    model = self._create_matmul_model(
        input_shape,
        weight_shape=(2, 3),
        saved_model_path=self._input_saved_model_path,
    )

    rng = np.random.default_rng(1230)
    random_tensor_gen_fn = lambda: rng.uniform(
        low=0.0, high=1.0, size=input_shape
    ).astype(np.float32)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(50):
        yield {'input_tensor': random_tensor_gen_fn()}

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        # Disable quantization for the quantizable unit (lifted function) whose
        # function name starts with "composite_dot_general".
        specs=qc.QuantizationSpecs(
            specs=[
                qc.QuantizationSpec(
                    matcher=qc.MatcherSpec(
                        function_name=qc.FunctionNameMatcherSpec(
                            regex='composite_dot_general.*'
                        )
                    ),
                    method=qc.Method(no_quantization={}),
                )
            ]
        ),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    input_data = ops.convert_to_tensor(random_tensor_gen_fn())
    expected_outputs = model.matmul(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )

    # Indirectly tests that the model is not quantized by asserting that there
    # are negligible numeric difference.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.000001)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.4,
    )

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_selective_denylist(self):
    """Tests that the op is not quantized when no quantization is enabled."""

    rng = np.random.default_rng(1230)
    random_tensor_gen_fn = lambda shape: rng.uniform(
        low=-1.0, high=1.0, size=shape
    ).astype(np.float32)

    class TwoMatmulModel(module.Module):
      """A model with two matmul ops."""

      @def_function.function
      def matmul(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        """Performs a matrix multiplication.

        Args:
          input_tensor: Input tensor to matmul with the filter.

        Returns:
          A 'output' -> output tensor mapping
        """
        out = math_ops.matmul(input_tensor, random_tensor_gen_fn((2, 3)))
        out = math_ops.matmul(out, random_tensor_gen_fn((3, 4)))
        return {'output': out}

    model = TwoMatmulModel()
    input_shape = (1, 2)

    save.save(
        model,
        self._input_saved_model_path,
        signatures=model.matmul.get_concrete_function(
            tensor_spec.TensorSpec(
                shape=input_shape, dtype=dtypes.float32, name='input_tensor'
            )
        ),
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(50):
        yield {'input_tensor': random_tensor_gen_fn(input_shape)}

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                ),
            ],
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        # Disable quantization for the quantizable unit (lifted function) whose
        # function name matches "composite_dot_general_fn_1".
        # "composite_dot_general_fn_2" will be quantized.
        specs=qc.QuantizationSpecs(
            specs=[
                qc.QuantizationSpec(
                    matcher=qc.MatcherSpec(
                        function_name=qc.FunctionNameMatcherSpec(
                            regex='composite_dot_general_fn_1'
                        )
                    ),
                    method=qc.Method(no_quantization={}),
                )
            ]
        ),
    )

    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    input_data = ops.convert_to_tensor(random_tensor_gen_fn(input_shape))
    expected_outputs = model.matmul(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )

    # Indirectly tests that the model is only partially quantized.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.011)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.55,
    )

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_quantization_method_not_applied_when_matcher_mismatch(self):
    """Tests that quantization method is not applied to unmatched units."""
    input_shape = (1, 2)
    model = self._create_matmul_model(
        input_shape,
        weight_shape=(2, 3),
        saved_model_path=self._input_saved_model_path,
    )

    rng = np.random.default_rng(1230)
    random_tensor_gen_fn = lambda: rng.uniform(
        low=0.0, high=1.0, size=input_shape
    ).astype(np.float32)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(50):
        yield {'input_tensor': random_tensor_gen_fn()}

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        specs=qc.QuantizationSpecs(
            specs=[
                qc.QuantizationSpec(
                    # Provide a regex that wouldn't match any quantizable units.
                    matcher=qc.MatcherSpec(
                        function_name=qc.FunctionNameMatcherSpec(
                            regex='.*invalid_function_name.*'
                        ),
                    ),
                    method=qc.Method(no_quantization={}),
                ),
            ],
        ),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    input_data = ops.convert_to_tensor(random_tensor_gen_fn())
    expected_outputs = model.matmul(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )

    # Tests that the quantized graph outputs similar values. They also shouldn't
    # be exactly the same. Indirectly proves that the `FunctionNameMatcherSpec`
    # with regex '.*invalid_function_name.*' did not match the quantizable unit.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.04)
    self.assertNotAllClose(new_outputs, expected_outputs, 1e-7)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.4,
    )

  def test_save_quantization_report_file(self):
    """Tests that the quantization report file is created.

    Also test that it is populated with textproto of `QuantizationResults`.
    """
    input_shape = (1, 16)
    filter_shape = (16, 3)
    self._create_matmul_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
    )

    rng = np.random.default_rng(seed=42)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=input_shape
            ).astype(np.float32)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    report_file_path = self.create_tempfile('report.txtpb').full_path
    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        report_file_path=report_file_path,
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    # Test the contents of the report file, which is a textproto of
    # `QuantizationResults`.
    self.assertTrue(os.path.exists(report_file_path))
    with open(report_file_path, 'r') as f:
      quantization_results_textpb = f.read()

    results = qc.QuantizationResults()
    text_format.Parse(quantization_results_textpb, results)

    self.assertProtoEquals(
        expected_message_maybe_ascii=r"""
        results {
          quantizable_unit { name: "composite_dot_general_fn_1" }
          method { static_range_ptq {} }
        }
        """,
        message=results,
    )


@test_util.run_all_in_graph_and_eager_modes
class CalibrationOptionsTest(quantize_model_test_base.QuantizedModelTest):
  """Test cases regarding the use of CalibrationOptions proto.

  Run all tests cases in both the graph mode (default in TF1) and the eager mode
  (default in TF2) to ensure support for when TF2 is disabled.
  """

  @parameterized.parameters(
      {
          'calibration_options': qc.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX
          )
      },
      {
          'calibration_options': qc.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX
          ),
      },
      {
          'calibration_options': qc.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE,
              calibration_parameters=qc.CalibrationOptions.CalibrationParameters(
                  num_bins=10,
              ),
          ),
      },
      {
          'calibration_options': qc.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
              calibration_parameters=qc.CalibrationOptions.CalibrationParameters(
                  num_bins=10,
              ),
          ),
      },
      {
          'calibration_options': qc.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY,
              calibration_parameters=qc.CalibrationOptions.CalibrationParameters(
                  num_bins=10,
              ),
          ),
      },
      {
          'calibration_options': qc.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC,
              calibration_parameters=qc.CalibrationOptions.CalibrationParameters(
                  num_bins=10,
              ),
          ),
      },
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_ptq_model_by_calibration_options(
      self,
      calibration_options: qc.CalibrationOptions,
  ):
    bias_fn = nn_ops.bias_add
    activation_fn = nn_ops.relu6
    enable_per_channel_quantized_weight = False
    has_batch_norm = True
    dilations = None
    input_shape = (1, 3, 4, 3)
    filter_shape = (2, 3, 3, 2)
    strides = (1, 1, 1, 1)
    model = self._create_conv2d_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        bias_fn,
        activation_fn,
        has_batch_norm,
        strides,
        dilations,
    )

    # Generate model input data.
    input_data = ops.convert_to_tensor(
        np.random.uniform(low=0.0, high=10, size=input_shape).astype('f4')
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(100):
        yield {
            'input_tensor': ops.convert_to_tensor(
                np.random.uniform(low=0, high=10, size=input_shape).astype('f4')
            ),
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ],
            enable_per_channel_quantized_weight=enable_per_channel_quantized_weight,
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        calibration_options=calibration_options,
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.conv2d(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.02, atol=0.5)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.46,
    )


class WeightOnlyQuantizationTest(quantize_model_test_base.QuantizedModelTest):

  @parameterized.parameters(
      testing.parameter_combinations([{
          'bias_fn': (
              None,
              nn_ops.bias_add,
          ),
          'activation_fn': (
              None,
              nn_ops.relu,
              nn_ops.relu6,
          ),
          'dim_sizes': (
              # tf.MatMul cases.
              ([None, 1024], [1024, 3]),  # dynamic batch dim.
              ([1, 1024], [1024, 3]),
              # tf.BatchMatMul cases.
              ([10, 1, 1024], [10, 1024, 3]),
              ([2, 3, 1, 1024], [2, 3, 1024, 3]),
          ),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_weight_only_model(
      self,
      bias_fn: Optional[ops.Operation],
      activation_fn: Optional[ops.Operation],
      dim_sizes: Sequence[int],
  ):
    lhs_dim_size, rhs_dim_size = dim_sizes
    input_shape = (*lhs_dim_size,)
    filter_shape = (*rhs_dim_size,)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]
    model = self._create_matmul_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        bias_fn,
        activation_fn,
    )

    rng = np.random.default_rng(1234)
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )

    config = qc.QuantizationConfig(
        weight_only_ptq_preset=qc.WeightOnlyPtqPreset(),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.matmul(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.03, atol=0.2)

    module_str = self._extract_first_xla_call_module_op(
        self._output_saved_model_path
    )

    # Tests that the output graph contains multiply for symmetric
    # dequantization.
    self.assertTrue(re.search('stablehlo.multiply', module_str))
    # Tests that the output graph contains float dot_general.
    self.assertTrue(
        re.search('stablehlo.dot_general.*xf32>.*xf32>.*xf32>', module_str)
    )

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.3,
    )

  @parameterized.parameters(
      testing.parameter_combinations([{
          'bias_fn': (
              None,
              nn_ops.bias_add,
          ),
          'activation_fn': (
              None,
              nn_ops.relu,
              nn_ops.relu6,
          ),
          'has_batch_norm': (False,),
          'input_shape_dynamic': (
              False,
              True,
          ),
          'has_func_alias': (False, True),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_weight_only_model(
      self,
      bias_fn: Optional[ops.Operation],
      activation_fn: Optional[ops.Operation],
      has_batch_norm: bool,
      input_shape_dynamic: bool,
      dilations: Sequence[int] = None,
      has_func_alias: bool = False,
  ):
    input_shape = (None, 3, 4, 3) if input_shape_dynamic else (1, 3, 4, 3)
    filter_shape = (2, 3, 3, 2)
    strides = (1, 1, 1, 1)
    model = self._create_conv2d_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        bias_fn,
        activation_fn,
        has_batch_norm,
        strides,
        dilations,
        'SAME',
        has_func_alias,
    )

    rng = np.random.default_rng(1234)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )

    config = qc.QuantizationConfig(
        weight_only_ptq_preset=qc.WeightOnlyPtqPreset(),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.conv2d(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.03, atol=0.2)

    module_str = self._extract_first_xla_call_module_op(
        self._output_saved_model_path
    )

    # Tests that the output graph contains multiply op for symmetric
    # dequantization.
    self.assertTrue(re.search('stablehlo.multiply', module_str))
    # Tests that the output graph contains float dot_general.
    self.assertTrue(
        re.search('stablehlo.convolution.*xf32>.*xf32>.*xf32>', module_str)
    )

    if has_func_alias:
      func_aliases = self._get_function_aliases(
          self._output_saved_model_path, [tag_constants.SERVING]
      )
      self.assertCountEqual(
          func_aliases.values(), [quantize_model_test_base.FUNC_ALIAS]
      )

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.4,
    )

  @parameterized.parameters(
      testing.parameter_combinations([{
          'shape_dynamic': (
              False,
              True,
          ),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_add_ptq_model(
      self,
      shape_dynamic: bool,
  ):
    input_shape = (None, 3, 4, 3) if shape_dynamic else (2, 3, 4, 3)
    self._create_add_model(
        input_shape,
        self._input_saved_model_path,
    )

    # Generate model input data.
    rng = np.random.default_rng(seed=42)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ],
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    self.assertEqual(
        self._get_num_xla_call_module_op(self._output_saved_model_path), 1
    )
    module_str = self._extract_first_xla_call_module_op(
        self._output_saved_model_path
    )

    # Check add is not quantized.
    self.assertTrue(re.search(r'stablehlo.add.*f32>', module_str))

  @parameterized.parameters(
      testing.parameter_combinations([{
          'shape_dynamic': (
              False,
              True,
          ),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_add_weight_only_model(
      self,
      shape_dynamic: bool,
  ):
    input_shape = (None, 3, 4, 3) if shape_dynamic else (2, 3, 4, 3)
    self._create_add_model(
        input_shape,
        self._input_saved_model_path,
    )

    # Generate model input data.
    rng = np.random.default_rng(seed=42)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        weight_only_ptq_preset=qc.WeightOnlyPtqPreset(),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    self.assertEqual(
        self._get_num_xla_call_module_op(self._output_saved_model_path), 1
    )
    module_str = self._extract_first_xla_call_module_op(
        self._output_saved_model_path
    )

    # Check add is not quantized.
    self.assertTrue(re.search(r'stablehlo.add.*f32>', module_str), module_str)

  def test_save_quantization_report_file(self):
    """Tests that the quantization report file is created.

    Also test that it is populated with textproto of `QuantizationResults`.
    """

    input_shape = (1, 3, 4, 3)
    filter_shape = (2, 3, 3, 2)
    self._create_conv2d_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
    )

    report_file_path = self.create_tempfile('report.txtpb').full_path
    config = qc.QuantizationConfig(
        weight_only_ptq_preset=qc.WeightOnlyPtqPreset(),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        report_file_path=report_file_path,
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    # Test the contents of the report file, which is a textproto of
    # `QuantizationResults`.
    self.assertTrue(os.path.exists(report_file_path))
    with open(report_file_path, 'r') as f:
      quantization_results_textpb = f.read()

    results = qc.QuantizationResults()
    text_format.Parse(quantization_results_textpb, results)

    self.assertProtoEquals(
        expected_message_maybe_ascii=r"""
        results {
          quantizable_unit { name: "composite_conv_fn_1" }
          method {
            weight_only_ptq {
              input_quantized_types {
                key: 1
                value { dimension_specs {} }
              }
            }
          }
        }
        """,
        message=results,
    )


if __name__ == '__main__':
  test.main()

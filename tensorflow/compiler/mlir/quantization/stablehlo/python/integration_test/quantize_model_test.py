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
import itertools
from typing import Optional, Sequence

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.mlir.quantization.stablehlo.python import quantization
from tensorflow.compiler.mlir.quantization.stablehlo.python.integration_test import quantize_model_test_base
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import tag_constants

# Type aliases for quantization method protobuf enums.
_PresetMethod = quant_opts_pb2.QuantizationMethod.PresetMethod


def parameter_combinations(test_parameters):
  """Generate all combinations of test parameters."""
  real_parameters = []
  for parameters in test_parameters:
    keys = parameters.keys()
    for curr in itertools.product(*parameters.values()):
      real_parameters.append(dict(zip(keys, curr)))
  return real_parameters


# Test cases for Static Range Quantization.
# Tries to run all tests cases in both the graph mode (default in TF1) and the
# eager mode (default in TF2) to ensure support for when TF2 is disabled.
class StaticRangeQuantizationTest(quantize_model_test_base.QuantizedModelTest):

  @parameterized.parameters(
      parameter_combinations([{
          'activation_fn': [None],
          'has_bias': [True, False],
          'dim_sizes': [
              # tf.MatMul cases.
              ([None, 1024], [1024, 3]),  # dynamic batch dim.
              ([1, 1024], [1024, 3]),
              # tf.BatchMatMul cases.
              ([10, 1, 1024], [10, 1024, 3]),
              ([2, 3, 1, 1024], [2, 3, 1024, 3]),
          ],
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_ptq_model(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      dim_sizes: Sequence[int],
  ):
    target_opset = quant_opts_pb2.STABLEHLO

    lhs_dim_size, rhs_dim_size = dim_sizes
    input_shape = (*lhs_dim_size,)
    filter_shape = (*rhs_dim_size,)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]
    model = self._create_matmul_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        has_bias,
        activation_fn,
    )

    rng = np.random.default_rng(seed=1235)
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

    config = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags={tag_constants.SERVING},
        signature_keys=['serving_default'],
        op_set=target_opset,
        representative_datasets={
            'serving_default': quant_opts_pb2.RepresentativeDatasetFile(
                tfrecord_file_path=dataset_path
            )
        },
        calibration_options=quant_opts_pb2.CalibrationOptions(
            calibration_method=quant_opts_pb2.CalibrationOptions.CALIBRATION_METHOD_MIN_MAX
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
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.03, atol=0.2)

  @parameterized.parameters(
      parameter_combinations([{
          'same_scale_op': [
              'concatenate',
              'gather',
              'pad',
              'reshape',
              'select',
              'slice',
              'transpose',
          ],
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_and_same_scale_ptq_model(
      self,
      same_scale_op: str,
  ):
    target_opset = quant_opts_pb2.STABLEHLO

    input_shape = (2, 3, 1, 1024)
    filter_shape = (2, 3, 1024, 3)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]

    model = self._create_matmul_and_same_scale_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        same_scale_op,
    )

    rng = np.random.default_rng(seed=1235)
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

    config = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags={tag_constants.SERVING},
        signature_keys=['serving_default'],
        op_set=target_opset,
        representative_datasets={
            'serving_default': quant_opts_pb2.RepresentativeDatasetFile(
                tfrecord_file_path=dataset_path
            )
        },
        calibration_options=quant_opts_pb2.CalibrationOptions(
            calibration_method=quant_opts_pb2.CalibrationOptions.CALIBRATION_METHOD_MIN_MAX
        ),
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

  @parameterized.named_parameters(
      {
          'testcase_name': 'none',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.STABLEHLO,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_ptq_model(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      has_batch_norm: bool,
      target_opset: quant_opts_pb2.OpSet,
      input_shape_dynamic: bool,
      enable_per_channel_quantization: bool,
      dilations: Sequence[int] = None,
  ):
    input_shape = (None, None, None, 3) if input_shape_dynamic else (1, 3, 4, 3)
    filter_shape = (2, 3, 3, 2)
    strides = (1, 1, 1, 1)
    model = self._create_conv2d_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        has_bias,
        has_batch_norm,
        activation_fn,
        strides,
        dilations,
    )

    # Generate model input data.
    rng = np.random.default_rng(seed=1224)
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

    config = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags={tag_constants.SERVING},
        signature_keys=['serving_default'],
        op_set=target_opset,
        representative_datasets={
            'serving_default': quant_opts_pb2.RepresentativeDatasetFile(
                tfrecord_file_path=dataset_path
            )
        },
        enable_per_channel_quantization=enable_per_channel_quantization,
        calibration_options=quant_opts_pb2.CalibrationOptions(
            calibration_method=quant_opts_pb2.CalibrationOptions.CALIBRATION_METHOD_MIN_MAX
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
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.02, atol=0.05)

  @parameterized.parameters(
      ('abc,cde->abde', quant_opts_pb2.STABLEHLO),
      ('abc,dce->abde', quant_opts_pb2.STABLEHLO),
  )
  def test_einsum_ptq_model(
      self,
      equation: str,
      target_opset: quant_opts_pb2.OpSet,
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
    rng = np.random.default_rng(seed=1231)
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

    config = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags={tag_constants.SERVING},
        signature_keys=['serving_default'],
        op_set=target_opset,
        representative_datasets={
            'serving_default': quant_opts_pb2.RepresentativeDatasetFile(
                tfrecord_file_path=dataset_path
            )
        },
        calibration_options=quant_opts_pb2.CalibrationOptions(
            calibration_method=quant_opts_pb2.CalibrationOptions.CALIBRATION_METHOD_MIN_MAX
        ),
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

  def test_when_preset_not_srq_raise_error(self):
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    config = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_NO_QUANTIZE
        ),
        tags={tag_constants.SERVING},
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.STABLEHLO,
    )

    with self.assertRaisesRegex(ValueError, 'only supports static-range PTQ'):
      quantization.quantize_saved_model(
          self._input_saved_model_path,
          self._output_saved_model_path,
          config,
      )


if __name__ == '__main__':
  test.main()

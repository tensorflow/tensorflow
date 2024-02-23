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
from typing import Mapping, Optional, Sequence

from absl.testing import parameterized
import numpy as np

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
          'rng_seed': (1230, 1231, 1232, 1233),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_ptq_model(
      self,
      bias_fn: Optional[ops.Operation],
      activation_fn: Optional[ops.Operation],
      dim_sizes: Sequence[int],
      rng_seed: int,
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

    rng = np.random.default_rng(rng_seed)
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
          'rng_seed': (0, 11, 222, 3333),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_and_same_scale_ptq_model(
      self,
      same_scale_op: str,
      rng_seed: int,
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

    rng = np.random.default_rng(rng_seed)
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
          'enable_per_channel_quantized_weight': (
              False,
              True,
          ),
          'rng_seed': (10, 11, 12, 13),
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
      rng_seed: int,
      dilations: Sequence[int] = None,
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
    )

    # Generate model input data.
    rng = np.random.default_rng(rng_seed)
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
      testing.parameter_combinations([{
          'equation': (
              'abc,cde->abde',
              'abc,dce->abde',
          ),
          'rng_seed': (82, 82732, 4444, 14),
      }])
  )
  def test_einsum_ptq_model(
      self,
      equation: str,
      rng_seed: int,
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
    rng = np.random.default_rng(rng_seed)
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
    self.assertNotAllClose(new_outputs, expected_outputs, rtol=0.00001)


if __name__ == '__main__':
  test.main()

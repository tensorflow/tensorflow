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

from tensorflow.compiler.mlir.quantization.stablehlo.python.integration_test import quantize_model_test_base
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import quantize_model
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
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
          'has_bias': [False],
          'batch_sizes': [([], []), ([10], [10]), ([2, 3], [2, 3])],
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_ptq_model(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      batch_sizes: Sequence[int],
  ):
    target_opset = quant_opts_pb2.XLA

    lhs_batch_size, rhs_batch_size = batch_sizes
    input_shape = (*lhs_batch_size, 1, 1024)
    filter_shape = (*rhs_batch_size, 1024, 3)
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
    expected_outputs = model.matmul(input_data)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags={tag_constants.SERVING},
        signature_keys=['serving_default'],
        op_set=target_opset,
    )
    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    new_outputs = converted_model.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # Tests that the quantized graph outputs similar values. The rtol value is
    # arbitrary.
    # TODO(b/309674337): Fix the large numerical errors.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.3)


if __name__ == '__main__':
  test.main()

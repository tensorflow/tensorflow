/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/kernels/gru_cell.h"

#include <vector>

#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"

namespace tflite {
namespace ops {
namespace custom {
namespace gru_cell {

using optimized_ops::ArrayMap;
using optimized_ops::FullyConnected;
using optimized_ops::MapAsArrayWithLastDimAsRows;
using reference_ops::Concatenation;

void GruCell(const RuntimeShape& input_shape, const float* input,
             const RuntimeShape& state_shape, const float* input_state,
             const RuntimeShape& gate_weight_shape, const float* gate_weight,
             const RuntimeShape& gate_bias_shape, const float* gate_bias,
             const RuntimeShape& candidate_weight_shape,
             const float* candidate_weight,
             const RuntimeShape& candidate_bias_shape,
             const float* candidate_bias, const RuntimeShape& output_shape,
             float* output, float* output_state,
             const RuntimeShape& activation_shape, float* activation,
             const RuntimeShape& concat_shape, float* concat,
             const tflite::FullyConnectedParams& fc_params,
             tflite::CpuBackendContext* cpu_backend_context) {
  const int n_batch = input_shape.Dims(0);
  const int n_input = input_shape.Dims(1);
  const int n_output = state_shape.Dims(1);

  // [x h] = concat(input, state)
  std::vector<float const*> concat_arrays_data;
  std::vector<RuntimeShape const*> concat_arrays_shapes;
  concat_arrays_data.push_back(input);
  concat_arrays_data.push_back(input_state);
  concat_arrays_shapes.push_back(&input_shape);
  concat_arrays_shapes.push_back(&state_shape);
  tflite::ConcatenationParams concat_params;
  concat_params.axis = 1;
  concat_params.inputs_count = concat_arrays_data.size();
  Concatenation(concat_params, &(concat_arrays_shapes[0]),
                &(concat_arrays_data[0]), concat_shape, concat);

  // [r u] = [x h] * gate_weight + gate_bias
  FullyConnected(fc_params, concat_shape, concat, gate_weight_shape,
                 gate_weight, gate_bias_shape, gate_bias, activation_shape,
                 activation, cpu_backend_context);

  // [r u] = sigmoid([r u])
  auto ru = MapAsArrayWithLastDimAsRows(activation, activation_shape);
  ru = ru.unaryExpr(Eigen::internal::scalar_logistic_op<float>());
  auto r = ru.block(0 * n_output, 0, n_output, n_batch);
  auto u = ru.block(1 * n_output, 0, n_output, n_batch);

  // hr = h .* r
  auto h = MapAsArrayWithLastDimAsRows(input_state, state_shape);
  auto xh = MapAsArrayWithLastDimAsRows(concat, concat_shape);
  auto hr = xh.block(n_input, 0, n_output, n_batch);
  hr = h * r;

  // c = [x hr] * candidate_weight + candidate_bias
  FullyConnected(fc_params, concat_shape, concat, candidate_weight_shape,
                 candidate_weight, candidate_bias_shape, candidate_bias,
                 output_shape, output, cpu_backend_context);

  auto c = MapAsArrayWithLastDimAsRows(output, output_shape);
  // output = (1 - u) .* tanh(c) + u .* h
  c = (1.0 - u) * c.tanh() + u * h;

  memcpy(output_state, output, n_batch * n_output * sizeof(float));
}

}  // namespace gru_cell
}  // namespace custom
}  // namespace ops
}  // namespace tflite

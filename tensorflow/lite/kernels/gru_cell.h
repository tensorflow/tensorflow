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

#ifndef TENSORFLOW_LITE_KERNELS_GRU_CELL_H_
#define TENSORFLOW_LITE_KERNELS_GRU_CELL_H_

#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/tensor.h"

namespace tflite {
namespace ops {
namespace custom {
namespace gru_cell {

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
             tflite::CpuBackendContext* cpu_backend_context);

}  // namespace gru_cell
}  // namespace custom
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_GRU_CELL_H_

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
#ifndef TENSORFLOW_LITE_TOOLS_VERSIONING_OP_VERSION_H_
#define TENSORFLOW_LITE_TOOLS_VERSIONING_OP_VERSION_H_

#include <vector>

#include "tensorflow/lite/schema/mutable/schema_generated.h"

namespace tflite {

// OpSignature contains operator parameters for version functions.
typedef struct {
  BuiltinOperator op;
  std::vector<TensorType> input_types;
  std::vector<TensorType> output_types;
  union {
    struct {
      int32_t dilation_w_factor;
      int32_t dilation_h_factor;
      bool is_per_channel_quantized;
    } depthwise_conv_2d;
    struct {
      bool narrow_range;
    } fakequant;
    struct {
      bool keep_num_dims;
      FullyConnectedOptionsWeightsFormat weights_format;
      // TODO(b/156530611): Make this global when more ops support sparse
      // computation.
      bool sparse_weight;
      bool asymmetric_quantize_inputs;
    } fully_connected;
    struct {
      float input1_scale;
      float input2_scale;
      float output_scale;
    } mul;
    struct {
      LSTMKernelType kernel_type;
      bool asymmetric_quantize_inputs;
    } lstm;
    struct {
      bool half_pixel_centers;
      bool align_corners;
    } resize;
    struct {
      int32_t num_dims;
    } single_input_op;
    struct {
      int32_t num_dims;
      bool need_broadcast;
    } broadcast;
    struct {
      bool is_per_channel_quantized;
    } conv_2d;
    struct {
      bool asymmetric_quantize_inputs;
    } input_quantization;
  } options;
} OpSignature;

// Returns version of builtin ops by the given signature.
int GetBuiltinOperatorVersion(const OpSignature& op_sig);

// Update operator's version of the given TFL flatbuffer model.
void UpdateOpVersion(uint8_t* model_buffer_pointer);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_VERSIONING_OP_VERSION_H_

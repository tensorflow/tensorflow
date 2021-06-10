/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_VERSIONING_OP_SIGNATURE_H_
#define TENSORFLOW_LITE_TOOLS_VERSIONING_OP_SIGNATURE_H_

#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

const auto kTensorTypeNone = static_cast<::tflite::TensorType>(-1);

// OpSignature contains operator parameters for version functions.
typedef struct {
  TensorType type;
  std::vector<int32_t> dims;
} OpSignatureTensorSpec;

typedef struct {
  BuiltinOperator op;
  std::vector<OpSignatureTensorSpec> inputs;
  std::vector<OpSignatureTensorSpec> outputs;
  void* builtin_data;
  union {
    struct {
      bool is_per_channel_quantized;
    } conv_2d;
    struct {
      bool is_per_channel_quantized;
    } depthwise_conv_2d;
    struct {
      // TODO(b/156530611): Make this global when more ops support sparse
      // computation.
      bool sparse_weight;
    } fully_connected;
    struct {
      float input1_scale;
      float input2_scale;
      float output_scale;
    } mul;
    struct {
      int32_t num_dims;
    } strided_slice;
    struct {
      bool input_quantized;
    } abs;
  } ext_options;
} OpSignature;

// Generate OpSignature with the given OperatorCode, Operator and Tensors (from
// SubGraph). The OpSignature will be used by GetBuiltinOperatorVersion() and
// mostly input and output tensor types are enough to figure out op version.
// But some ops (DEPTHWISE_CONV_2D,  FULLY_CONNECTED, ...) require to pass their
// options to decide op version.
//
// WARNING: The caller is responsible to free the allocated
// OpSignature.builtin_data memory.
OpSignature GetOpSignature(const OperatorCode* op_code, const Operator* op,
                           const SubGraph* subgraph);

}  // namespace tflite
#endif  // TENSORFLOW_LITE_TOOLS_VERSIONING_OP_SIGNATURE_H_

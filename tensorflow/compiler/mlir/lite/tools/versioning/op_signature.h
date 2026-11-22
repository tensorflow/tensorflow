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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TOOLS_VERSIONING_OP_SIGNATURE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TOOLS_VERSIONING_OP_SIGNATURE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/compiler/mlir/lite/core/c/tflite_types.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace tflite {

// OpSignature contains operator parameters for version functions.
typedef struct {
  TfLiteType type;
  std::vector<int32_t> dims;
  bool is_const;
  bool is_shape_dynamic;
} OpSignatureTensorSpec;

struct OpSignature {
  BuiltinOperator op = BuiltinOperator_ADD;
  std::vector<OpSignatureTensorSpec> inputs;
  std::vector<OpSignatureTensorSpec> outputs;
  void* builtin_data = nullptr;
  int version = 0;
  const void* custom_initial_data = nullptr;
  std::string custom_name;
  struct {
    struct {
      bool is_per_channel_quantized = false;
      bool is_grouped_convolution = false;
    } conv_2d;
    struct {
      bool is_per_channel_quantized = false;
    } depthwise_conv_2d;
    struct {
      // TODO(b/156530611): Make this global when more ops support sparse
      // computation.
      bool sparse_weight = false;
      bool is_per_channel_quantized = false;
      // The weight (input 1) is quantized with `BlockwiseQuantization` with a
      // non-empty `block_shape`.
      bool is_weight_blockwise_quantized = false;
    } fully_connected;
    struct {
      float input1_scale = 0.0f;
      float input2_scale = 0.0f;
      float output_scale = 0.0f;
      bool input_quantized = false;
    } mul;
    struct {
      int32_t num_dims = 0;
    } strided_slice;
    struct {
      bool input_quantized = false;
    } abs;
    struct {
      bool is_per_channel_quantized = false;
      // The input is quantized with `BlockwiseQuantization` with a non-empty
      // `block_shape`.
      bool is_blockwise_quantized = false;
    } dequantize;
    struct {
      bool is_per_channel_quantized = false;
    } quantize;
    struct {
      bool input_quantized = false;
    } add;
    struct {
      bool is_per_channel_quantized = false;
      // The table (input 1) is quantized with `BlockwiseQuantization` with a
      // non-empty `block_shape`.
      bool is_blockwise_quantized = false;
    } embedding_lookup;
    struct {
      // RHS (input 1) is quantized with `BlockwiseQuantization` with a
      // non-empty `block_shape`.
      bool is_rhs_blockwise_quantized = false;
    } batch_matmul;
  } ext_options = {};
};

// Generate OpSignature with the given OperatorCode, Operator and Tensors (from
// SubGraph). The OpSignature will be used by GetBuiltinOperatorVersion() and
// mostly input and output tensor types are enough to figure out op version.
// But some ops (DEPTHWISE_CONV_2D,  FULLY_CONNECTED, ...) require to pass their
// options to decide op version.
//
// WARNING: The caller is responsible to free the allocated
// OpSignature.builtin_data memory.
OpSignature GetOpSignature(const OperatorCode* op_code, const Operator* op,
                           const SubGraph* subgraph, const Model* model);

}  // namespace tflite
#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TOOLS_VERSIONING_OP_SIGNATURE_H_

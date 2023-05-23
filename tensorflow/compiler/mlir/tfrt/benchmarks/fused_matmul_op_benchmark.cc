/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/fused_matmul_op_benchmark.h"

#include <string>

namespace tensorflow {

// Use type aliases compatible with MLIR type names.
using f32 = float;

static const char* const activations[] = {
    "Undefined",  // FusedComputationType::kUndefined
    "",           // FusedComputationType::kBiasAdd
    "Relu",       // FusedComputationType::kBiasAddWithRelu
    "Relu6",      // FusedComputationType::kBiasAddWithRelu6
    "Tanh",       // FusedComputationType::kBiasAddWithTanh
    "Sigmoid",    // FusedComputationType::kBiasAddWithSigmoid
    "Elu"         // FusedComputationType::kBiasAddWithElu
};

static const char* fused_matmul_ir_skeleton = R"(
func.func @fused_matmul(%arg0: {0}, %arg1: {1}, %arg2: {2}) -> {3} {
    %0 = "tf._FusedMatMul"(%arg0, %arg1, %arg2) {{
           epsilon = {4} : f32,
           leakyrelu_alpha = {5} : f32,
           fused_ops = ["BiasAdd"{6}],
           transpose_a = false,
           transpose_b = false
         } : ({0}, {1}, {2}) -> {3}
    func.return %0 : {3}
  }
)";

std::string GetFusedMatmulIR(
    llvm::ArrayRef<int32_t> arg0_shape, llvm::ArrayRef<bool> arg0_dyn_dims,
    llvm::ArrayRef<int32_t> arg1_shape, llvm::ArrayRef<bool> arg1_dyn_dims,
    llvm::ArrayRef<int32_t> arg2_shape, llvm::ArrayRef<bool> arg2_dyn_dims,
    llvm::ArrayRef<int32_t> out_shape, llvm::ArrayRef<bool> out_dyn_dims,
    llvm::StringRef element_type, unsigned activation, llvm::StringRef epsilon,
    llvm::StringRef leakyrelu_alpha) {
  llvm::SmallVector<int64_t, 2> mlir_arg0_shape, mlir_arg1_shape,
      mlir_arg2_shape, mlir_out_shape;
  for (int i = 0; i < arg0_shape.size(); ++i) {
    mlir_arg0_shape.push_back(arg0_dyn_dims[i] ? kDynSize : arg0_shape[i]);
  }
  for (int i = 0; i < arg1_shape.size(); ++i) {
    mlir_arg1_shape.push_back(arg1_dyn_dims[i] ? kDynSize : arg1_shape[i]);
  }
  for (int i = 0; i < arg2_shape.size(); ++i) {
    mlir_arg2_shape.push_back(arg2_dyn_dims[i] ? kDynSize : arg2_shape[i]);
  }
  for (int i = 0; i < out_shape.size(); ++i) {
    mlir_out_shape.push_back(out_dyn_dims[i] ? kDynSize : out_shape[i]);
  }
  return llvm::formatv(
      fused_matmul_ir_skeleton,
      PrintTensorType(mlir_arg0_shape, element_type),  // arg0 type {0}
      PrintTensorType(mlir_arg1_shape, element_type),  // arg1 type {1}
      PrintTensorType(mlir_arg2_shape, element_type),  // arg2 type {2}
      PrintTensorType(mlir_out_shape, element_type),   // Out type {3}
      epsilon, leakyrelu_alpha,
      activations[activation][0] == '\0'
          ? ""
          : ", \"" + std::string(activations[activation]) + '"');
}

// With BiasAdd
BM_TFMlir_DYNAMIC_ALL(1, 10, 32, 10, 1, "fused_matmul", f32);
BM_TFMlir_STATIC_ALL(1, 10, 32, 10, 1, "fused_matmul", f32);
BM_Eigen_WRAPPER(1, 10, 32, 10, 1, f32);

BM_TFMlir_DYNAMIC_ALL(1, 32, 268, 32, 1, "fused_matmul", f32);
BM_TFMlir_STATIC_ALL(1, 32, 268, 32, 1, "fused_matmul", f32);
BM_Eigen_WRAPPER(1, 32, 268, 1024, 1, f32);

BM_TFMlir_DYNAMIC_ALL(1, 1024, 32, 1024, 1, "fused_matmul", f32);
BM_TFMlir_STATIC_ALL(1, 1024, 32, 1024, 1, "fused_matmul", f32);
BM_Eigen_WRAPPER(1, 1024, 32, 1024, 1, f32);

BM_TFMlir_DYNAMIC_ALL(1, 32, 1024, 32, 1, "fused_matmul", f32);
BM_TFMlir_STATIC_ALL(1, 32, 1024, 32, 1, "fused_matmul", f32);
BM_Eigen_WRAPPER(1, 32, 1024, 32, 1, f32);

BM_TFMlir_DYNAMIC_ALL(1, 1024, 16, 1024, 1, "fused_matmul", f32);
BM_TFMlir_STATIC_ALL(1, 1024, 16, 1024, 1, "fused_matmul", f32);
BM_Eigen_WRAPPER(1, 1024, 16, 1024, 1, f32);

// With BiasAdd + Relu
BM_TFMlir_DYNAMIC_ALL(1, 1024, 32, 1024, 2, "fused_matmul", f32);
BM_TFMlir_STATIC_ALL(1, 1024, 32, 1024, 2, "fused_matmul", f32);
BM_Eigen_WRAPPER(1, 1024, 32, 1024, 2, f32);

}  // namespace tensorflow

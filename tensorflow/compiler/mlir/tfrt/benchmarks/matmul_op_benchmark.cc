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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/matmul_op_benchmark.h"

#include <string>

namespace tensorflow {

// Use type aliases compatible with MLIR type names.
using f32 = float;

static const char* matmul_ir_skeleton = R"(
func.func @matmul(%arg0: {0}, %arg1: {1}) -> {2} {
    %0 = "tf.MatMul"(%arg0, %arg1) {{
           transpose_a = false,
           transpose_b = false
         } : ({0}, {1}) -> {2}
    func.return %0 : {2}
  }
)";

std::string GetMatmulIR(llvm::ArrayRef<int32_t> lhs_shape,
                        llvm::ArrayRef<bool> lhs_dyn_dims,
                        llvm::ArrayRef<int32_t> rhs_shape,
                        llvm::ArrayRef<bool> rhs_dyn_dims,
                        llvm::ArrayRef<int32_t> out_shape,
                        llvm::ArrayRef<bool> out_dyn_dims,
                        llvm::StringRef element_type) {
  llvm::SmallVector<int64_t, 2> mlir_lhs_shape, mlir_rhs_shape, mlir_out_shape;
  for (int i = 0; i < lhs_shape.size(); ++i) {
    mlir_lhs_shape.push_back(lhs_dyn_dims[i] ? kDynSize : lhs_shape[i]);
  }
  for (int i = 0; i < rhs_shape.size(); ++i) {
    mlir_rhs_shape.push_back(rhs_dyn_dims[i] ? kDynSize : rhs_shape[i]);
  }
  for (int i = 0; i < out_shape.size(); ++i) {
    mlir_out_shape.push_back(out_dyn_dims[i] ? kDynSize : out_shape[i]);
  }
  return llvm::formatv(
      matmul_ir_skeleton,
      PrintTensorType(mlir_lhs_shape, element_type),  // LHS type {0}
      PrintTensorType(mlir_rhs_shape, element_type),  // RHS type {1}
      PrintTensorType(mlir_out_shape, element_type)   // Out type {2}
  );
}

constexpr bool kPack = true;

BM_TFMlir_DYNAMIC_ALL(16, 16, 16, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(16, 16, 16, 8, 8, 8, kPack, "matmul", f32);
BM_Eigen_WRAPPER(16, 16, 16, f32);

BM_TFMlir_DYNAMIC_ALL(64, 64, 64, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(64, 64, 64, 8, 8, 8, kPack, "matmul", f32);
BM_Eigen_WRAPPER(64, 64, 64, f32);

BM_TFMlir_DYNAMIC_ALL(128, 128, 128, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(128, 128, 128, 8, 8, 8, kPack, "matmul", f32);
BM_Eigen_WRAPPER(128, 128, 128, f32);

BM_TFMlir_DYNAMIC_ALL(256, 256, 256, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(256, 256, 256, 8, 8, 8, kPack, "matmul", f32);
BM_Eigen_WRAPPER(256, 256, 256, f32);

BM_TFMlir_DYNAMIC_ALL(512, 512, 512, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(512, 512, 512, 8, 8, 8, kPack, "matmul", f32);
BM_Eigen_WRAPPER(512, 512, 512, f32);

BM_TFMlir_DYNAMIC_ALL(1024, 1024, 1024, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(1024, 1024, 1024, 8, 8, 8, kPack, "matmul", f32);
BM_Eigen_WRAPPER(1024, 1024, 1024, f32);

BM_TFMlir_DYNAMIC_ALL(2048, 2048, 2048, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(2048, 2048, 2048, 8, 8, 8, kPack, "matmul", f32);
BM_Eigen_WRAPPER(2048, 2048, 2048, f32);

BM_TFMlir_DYNAMIC_ALL(100, 100, 100, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(100, 100, 100, 8, 8, 8, kPack, "matmul", f32);
BM_Eigen_WRAPPER(100, 100, 100, f32);

BM_TFMlir_DYNAMIC_ALL(1, 18, 300, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(1, 18, 300, 8, 8, 8, kPack, "matmul", f32);
BM_Eigen_WRAPPER(1, 18, 300, f32);

BM_TFMlir_DYNAMIC_ALL(1, 1, 300, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(1, 1, 300, 8, 8, 8, kPack, "matmul", f32);
BM_Eigen_WRAPPER(1, 1, 300, f32);

BM_TFMlir_DYNAMIC_ALL(18, 1, 300, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(18, 1, 300, 8, 8, 8, kPack, "matmul", f32);
BM_Eigen_WRAPPER(18, 1, 300, f32);

BM_TFMlir_DYNAMIC_ALL(18, 300, 1, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(18, 300, 1, 8, 8, 8, kPack, "matmul", f32);
BM_Eigen_WRAPPER(18, 300, 1, f32);

BM_TFMlir_DYNAMIC_ALL(1, 300, 300, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(1, 300, 300, 8, 8, 8, kPack, "matmul", f32);
BM_Eigen_WRAPPER(1, 300, 300, f32);

BM_TFMlir_DYNAMIC_ALL(1, 300, 1, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(1, 300, 1, 8, 8, 8, kPack, "matmul", f32);
BM_Eigen_WRAPPER(1, 300, 1, f32);

BM_TFMlir_DYNAMIC_ALL(10, 10, 10, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(10, 10, 10, 8, 8, 8, kPack, "matmul", f32);
BM_TFMlir_DYNAMIC_ALL(10, 10, 10, 4, 4, 4, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(10, 10, 10, 4, 4, 4, kPack, "matmul", f32);
BM_TFMlir_DYNAMIC_ALL(10, 10, 10, 2, 2, 2, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(10, 10, 10, 2, 2, 2, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(10, 10, 10, 2, 2, 8, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(10, 10, 10, 2, 8, 2, kPack, "matmul", f32);
BM_TFMlir_STATIC_ALL(10, 10, 10, 8, 2, 2, kPack, "matmul", f32);
BM_Eigen_WRAPPER(10, 10, 10, f32);

}  // namespace tensorflow

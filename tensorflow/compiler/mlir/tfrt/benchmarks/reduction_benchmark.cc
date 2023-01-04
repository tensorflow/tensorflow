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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/reduction_benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"

namespace tensorflow {
namespace {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::llvm::StringRef;

const char* kReductionIR = R"(
  func.func @main(%input: {1}) -> {2} {
    %dim_to_reduce = "tf.Const"() {{
      value = {3} : {4},
      device = "/job:localhost/replica:0/task:0/device:CPU:0"
    } : () -> {4}
    %result = "{0}"(%input, %dim_to_reduce) {{
      keep_dims = false,
      device = "/job:localhost/replica:0/task:0/device:CPU:0"
    } : ({1}, {4}) -> {2}
    func.return %result : {2}
  }
)";

}  // namespace

std::string GetReductionIR(StringRef op_name, ArrayRef<int32_t> input_shape,
                           ArrayRef<bool> dynamic_dims,
                           ArrayRef<int32_t> dims_to_reduce,
                           StringRef element_type) {
  SmallVector<int64_t, 2> mlir_input_shape, mlir_output_shape;
  for (int i = 0; i < input_shape.size(); ++i) {
    mlir_input_shape.push_back(dynamic_dims[i] ? kDynSize : input_shape[i]);
    if (llvm::find(dims_to_reduce, i) == dims_to_reduce.end())
      mlir_output_shape.push_back(mlir_input_shape[i]);
  }
  return llvm::formatv(
      kReductionIR, op_name,                             // TF op to use {0},
      PrintTensorType(mlir_input_shape, element_type),   // Input type {1}
      PrintTensorType(mlir_output_shape, element_type),  // Output type {2}
      PrintDenseArray(dims_to_reduce),  // Dims to reduce attr {3}
      PrintTensorType(static_cast<int64_t>(dims_to_reduce.size()),
                      "i32")  // Dims to reduce type {4}
  );
}

std::string GetSumF32IR(llvm::ArrayRef<int32_t> input_shape,
                        llvm::ArrayRef<bool> dynamic_dims,
                        llvm::ArrayRef<int32_t> dims_to_reduce) {
  return GetReductionIR("tf.Sum", input_shape, dynamic_dims, dims_to_reduce,
                        "f32");
}

std::string GetMeanF32IR(llvm::ArrayRef<int32_t> input_shape,
                         llvm::ArrayRef<bool> dynamic_dims,
                         llvm::ArrayRef<int32_t> dims_to_reduce) {
  return GetReductionIR("tf.Mean", input_shape, dynamic_dims, dims_to_reduce,
                        "f32");
}

llvm::SmallVector<InputTensorSpec> GetInputSpec(
    llvm::ArrayRef<ssize_t> input_shape) {
  return {InputTensorSpec(DT_FLOAT, input_shape)};
}

}  // namespace tensorflow

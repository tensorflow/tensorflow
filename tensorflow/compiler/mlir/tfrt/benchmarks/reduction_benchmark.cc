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

const bool kStatic = false;
const bool kDynamic = true;

static const char* kReductionIR = R"(
  func @main(%input: {1}) -> {2} {
    %dim_to_reduce = "tf.Const"() {{value = {3} : {4}} : () -> {4}
    %result = "{0}"(%input, %dim_to_reduce) {{keep_dims = false}
      : ({1}, {4}) -> {2}
    return %result : {2}
  }
)";

std::string GetIR(StringRef op_name, ArrayRef<int64_t> input_shape,
                  ArrayRef<int64_t> output_shape,
                  ArrayRef<int32_t> dims_to_reduce, StringRef element_type) {
  return llvm::formatv(
      kReductionIR, op_name,                        // TF op to use {0},
      PrintTensorType(input_shape, element_type),   // Input type {1}
      PrintTensorType(output_shape, element_type),  // Output type {2}
      PrintDenseArray(dims_to_reduce),              // Dims to reduce attr {3}
      PrintTensorType(static_cast<int64_t>(dims_to_reduce.size()),
                      "i32")  // Dims to reduce type {4}
  );
}

}  // namespace tensorflow

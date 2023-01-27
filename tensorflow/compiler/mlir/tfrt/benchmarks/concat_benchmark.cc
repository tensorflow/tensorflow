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

#include <string>
#include <vector>

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {
namespace {

static const char* kUnaryConcatIR = R"(
func.func @main(%arg: tensor<8x16xf32>) -> tensor<8x32xf32> {
  %axis = "tf.Const"() { value = dense<1> : tensor<i64> } : () -> tensor<i64>
  %0 = "tf.ConcatV2"(%arg, %arg, %axis)
      : (tensor<8x16xf32>, tensor<8x16xf32>, tensor<i64>) -> tensor<8x32xf32>
  func.return %0 : tensor<8x32xf32>
}
)";

std::string GetUnaryConcatIR() { return kUnaryConcatIR; }

template <int64_t R, int64_t D>
auto GetEigenUnaryConcatFn() {
  return [](llvm::ArrayRef<Tensor> inputs,
            llvm::Optional<Eigen::ThreadPoolDevice> device) {
    assert(inputs.size() == 1);

    // Determine result shape.
    auto inShape = inputs[0].shape();
    std::vector<int64_t> outShape;
    outShape.reserve(inShape.dims());
    for (int64_t i = 0; i < inShape.dims(); ++i)
      outShape.push_back(inShape.dim_size(i));
    outShape[D] *= 2;

    Tensor output(DT_FLOAT, TensorShape(outShape));

    auto in = inputs[0].tensor<float, R>();
    auto out = output.tensor<float, R>();
    if (device.has_value()) {
      out.device(*device) = in.concatenate(in, D);
    } else {
      out = in.concatenate(in, D);
    }
  };
}

BM_Jitrt(Concat, GetUnaryConcatIR(), "main",
         {InputTensorSpec(DT_FLOAT, {8, 16})})
    ->Arg(0);

BM_Eigen(Concat, (GetEigenUnaryConcatFn</*rank=*/2, /*axis=*/1>()),
         {InputTensorSpec(DT_FLOAT, {8, 16})})
    ->Arg(0);

}  // namespace
}  // namespace tensorflow

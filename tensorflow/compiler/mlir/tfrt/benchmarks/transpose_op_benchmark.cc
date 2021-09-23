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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {

static const char* mlir_input = R"(
func @compute(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %0 = "tf.Const"() {value = dense<[0, 2, 1]> : tensor<3xi64>}
         : () -> tensor<3xi64>
    %1 = "tf.Transpose"(%arg0, %0)
         : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
    return %1 : tensor<?x?x?xf32>
  }
)";

static void Shuffle(llvm::ArrayRef<Tensor> inputs,
                    llvm::Optional<Eigen::ThreadPoolDevice> device) {
  std::array<int64_t, 3> perm = {0, 2, 1};

  std::array<int64_t, 3> shuffled;
  for (unsigned d = 0; d < 3; d++) shuffled[d] = inputs[0].dim_size(perm[d]);

  Tensor output(DT_FLOAT, TensorShape(shuffled));

  auto in0 = inputs[0].tensor<float, 3>();
  auto out0 = output.tensor<float, 3>();

  if (device.hasValue()) {
    out0.device(*device) = in0.shuffle(perm);
  } else {
    out0 = in0.shuffle(perm);
  }
}

static llvm::SmallVector<InputTensorSpec> Inputs(ssize_t dim) {
  return {InputTensorSpec(DT_FLOAT, {dim, dim, dim})};
}

BM_Mlir(Transpose, mlir_input, "compute", Inputs(256))
    ->Arg(0)
    ->Arg(4)
    ->Arg(16)
    ->Arg(32);

BM_Eigen(Transpose, Shuffle, Inputs(256))->Arg(0)->Arg(4)->Arg(16)->Arg(32);

}  // namespace tensorflow

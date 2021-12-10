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

#include <array>
#include <string>

#include "llvm/Support/FormatVariadic.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {

static const char* mlir_input = R"(
func @compute(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {{
    %0 = "tf.Const"()
         {{value = dense<[{0}, {1}, {2}]> : tensor<3xi64>,
          device = "/job:localhost/replica:0/task:0/device:CPU:0"}
         : () -> tensor<3xi64>
    %1 = "tf.Transpose"(%arg0, %0)
         {{device = "/job:localhost/replica:0/task:0/device:CPU:0"}
         : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
    return %1 : tensor<?x?x?xf32>
  }
)";

static std::string Transpose(std::array<int32_t, 3> perm) {
  return llvm::formatv(mlir_input, perm[0], perm[1], perm[2]);
}

static auto Shuffle(std::array<int32_t, 3> perm) {
  return [perm](llvm::ArrayRef<Tensor> inputs,
                llvm::Optional<Eigen::ThreadPoolDevice> device) {
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
  };
}

static llvm::SmallVector<InputTensorSpec> Inputs(ssize_t dim) {
  return {InputTensorSpec(DT_FLOAT, {dim, dim, dim})};
}

#define BM(FN) BM_##FN->Arg(0)->Arg(4)->Arg(8);

// Transpose: [0, 2, 1]
BM(Cpurt(Transpose_0x2x1, Transpose({0, 2, 1}), "compute", Inputs(256)));
BM(CpurtV(Transpose_0x2x1, Transpose({0, 2, 1}), "compute", Inputs(256)));
BM(Tfrt(Transpose_0x2x1, Transpose({0, 2, 1}), "compute", Inputs(256)));
BM(Eigen(Transpose_0x2x1, Shuffle({0, 2, 1}), Inputs(256)));

// Transpose: [2, 0, 1]
BM(Cpurt(Transpose_2x0x1, Transpose({2, 0, 1}), "compute", Inputs(256)));
BM(CpurtV(Transpose_2x0x1, Transpose({2, 0, 1}), "compute", Inputs(256)));
BM(Tfrt(Transpose_2x0x1, Transpose({2, 0, 1}), "compute", Inputs(256)));
BM(Eigen(Transpose_2x0x1, Shuffle({2, 0, 1}), Inputs(256)));

// Transpose: [2, 1, 0]
BM(Cpurt(Transpose_2x1x0, Transpose({2, 1, 0}), "compute", Inputs(256)));
BM(CpurtV(Transpose_2x1x0, Transpose({2, 1, 0}), "compute", Inputs(256)));
BM(Tfrt(Transpose_2x1x0, Transpose({2, 1, 0}), "compute", Inputs(256)));
BM(Eigen(Transpose_2x1x0, Shuffle({2, 1, 0}), Inputs(256)));

// Transpose: [1, 2, 0]
BM(Cpurt(Transpose_1x2x0, Transpose({1, 2, 0}), "compute", Inputs(256)));
BM(CpurtV(Transpose_1x2x0, Transpose({1, 2, 0}), "compute", Inputs(256)));
BM(Tfrt(Transpose_1x2x0, Transpose({1, 2, 0}), "compute", Inputs(256)));
BM(Eigen(Transpose_1x2x0, Shuffle({1, 2, 0}), Inputs(256)));

}  // namespace tensorflow

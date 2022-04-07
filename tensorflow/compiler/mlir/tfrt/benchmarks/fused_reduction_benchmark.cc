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
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {
namespace {

const char* kReductionIR = R"(
  func.func @main(%lhs: {0}, %rhs: {0}) -> tensor<f32> {
    %lhs_abs = "tf.Abs"(%lhs) {{
      device = "/job:localhost/replica:0/task:0/device:CPU:0"
    } : ({0}) -> {0}
    %rhs_exp = "tf.Exp"(%rhs) {{
      device = "/job:localhost/replica:0/task:0/device:CPU:0"
    } : ({0}) -> {0}

    %add = "tf.Add"(%lhs_abs, %rhs_exp) {{
      device = "/job:localhost/replica:0/task:0/device:CPU:0"
    } : ({0}, {0}) -> {0}

    %dim_to_reduce = "tf.Const"() {{
      value = dense<[0]> : tensor<1xi32>,
      device = "/job:localhost/replica:0/task:0/device:CPU:0"
    } : () -> tensor<1xi32>
    %result = "tf.Prod"(%add, %dim_to_reduce) {{
      keep_dims = false,
      device = "/job:localhost/replica:0/task:0/device:CPU:0"
    } : ({0}, tensor<1xi32>) -> tensor<f32>
    func.return %result : tensor<f32>
  }
)";

std::string FusedReduction1D(bool dynamic, int64_t size) {
  return llvm::formatv(kReductionIR,
                       PrintTensorType({dynamic ? kDynSize : size}, "f32"));
}

auto EigenFusedReduction1D() {
  return [](llvm::ArrayRef<Tensor> inputs,
            llvm::Optional<Eigen::ThreadPoolDevice> device) {
    std::array<int64_t, 1> dims_to_reduce{0};
    Tensor output(DT_FLOAT, {});

    auto lhs = inputs[0].tensor<float, 1>();
    auto rhs = inputs[1].tensor<float, 1>();
    auto out = output.tensor<float, 0>();
    out.setZero();

    if (device.hasValue()) {
      out.device(*device) = (lhs.abs() + rhs.exp()).sum(dims_to_reduce);
    } else {
      out = (lhs.abs() + rhs.exp()).prod(dims_to_reduce);
    }
  };
}

llvm::SmallVector<InputTensorSpec> Inputs(ssize_t dim) {
  return {InputTensorSpec(DT_FLOAT, {dim}), InputTensorSpec(DT_FLOAT, {dim})};
}

#define BM(FN) BM_##FN->Arg(0);

#define BM_SUITE(NAME, DYNAMIC, SIZE)                                      \
  BM(JitrtV(NAME, FusedReduction1D(DYNAMIC, SIZE), "main", Inputs(SIZE))); \
  BM(Eigen(NAME, EigenFusedReduction1D(), Inputs(SIZE)));                  \
  BM(Tfrt(NAME, FusedReduction1D(DYNAMIC, SIZE), "main", Inputs(SIZE)))

#define BM_DYNAMIC(SIZE) \
  BM_SUITE(FusedReductionDynamic_##SIZE, kDynamicDim, SIZE)
BM_DYNAMIC(3);
BM_DYNAMIC(8);
BM_DYNAMIC(80);
BM_DYNAMIC(800);
BM_DYNAMIC(8000);
BM_DYNAMIC(8131);
BM_DYNAMIC(1000000);
BM_DYNAMIC(1010131);

#define BM_STATIC(SIZE) BM_SUITE(FusedReductionStatic_##SIZE, kStaticDim, SIZE)
BM_STATIC(3);
BM_STATIC(8);
BM_STATIC(80);
BM_STATIC(800);
BM_STATIC(8000);
BM_STATIC(8131);
BM_STATIC(1000000);
BM_STATIC(1010131);

}  // namespace
}  // namespace tensorflow

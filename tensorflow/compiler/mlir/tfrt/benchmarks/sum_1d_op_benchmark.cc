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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/reduction_benchmark.h"

namespace tensorflow {
namespace {

std::string Sum1D(bool dynamic, int32_t size) {
  return GetReductionIR("tf.Sum", {size}, {dynamic}, {0}, "f32");
}

auto EigenSum1D() {
  return [](llvm::ArrayRef<Tensor> inputs,
            llvm::Optional<Eigen::ThreadPoolDevice> device) {
    std::array<int64_t, 1> dims_to_reduce{0};
    Tensor output(DT_FLOAT, {});

    auto in = inputs[0].tensor<float, 1>();
    auto out = output.tensor<float, 0>();
    out.setZero();

    if (device.hasValue()) {
      out.device(*device) = in.sum(dims_to_reduce);
    } else {
      out = in.sum(dims_to_reduce);
    }
  };
}

llvm::SmallVector<InputTensorSpec> Inputs(ssize_t dim) {
  return {InputTensorSpec(DT_FLOAT, {dim})};
}

#define BM(FN) BM_##FN->Arg(0);

#define BM_SUITE(NAME, DYNAMIC, SIZE)                           \
  BM(CpurtV(NAME, Sum1D(DYNAMIC, SIZE), "main", Inputs(SIZE))); \
  BM(Eigen(NAME, EigenSum1D(), Inputs(SIZE)));                  \
  BM(Tfrt(NAME, Sum1D(DYNAMIC, SIZE), "main", Inputs(SIZE)))

#define BM_DYNAMIC(SIZE) BM_SUITE(SumDynamic_##SIZE, kDynamicDim, SIZE)
BM_DYNAMIC(3);
BM_DYNAMIC(8);
BM_DYNAMIC(80);
BM_DYNAMIC(800);
BM_DYNAMIC(8000);
BM_DYNAMIC(8131);
BM_DYNAMIC(1000000);
BM_DYNAMIC(1010131);

#define BM_STATIC(SIZE) BM_SUITE(SumStatic_##SIZE, kStaticDim, SIZE)
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

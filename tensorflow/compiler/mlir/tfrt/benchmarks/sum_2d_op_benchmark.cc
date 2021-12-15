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

std::string Sum2D(bool dynamic_row, bool dynamic_col, int32_t rows,
                  int32_t cols) {
  return GetReductionIR("tf.Sum", {rows, cols}, {dynamic_row, dynamic_col},
                        {0, 1}, "f32");
}

auto EigenSum2D() {
  return [](llvm::ArrayRef<Tensor> inputs,
            llvm::Optional<Eigen::ThreadPoolDevice> device) {
    Tensor output(DT_FLOAT, {});

    auto in = inputs[0].tensor<float, 2>();
    auto out = output.tensor<float, 0>();
    out.setZero();

    std::array<int64_t, 2> dims_to_reduce{0, 1};
    if (device.hasValue()) {
      out.device(*device) = in.sum(dims_to_reduce);
    } else {
      out = in.sum(dims_to_reduce);
    }
  };
}

llvm::SmallVector<InputTensorSpec> Inputs(ssize_t rows, ssize_t cols) {
  return {InputTensorSpec(DT_FLOAT, {rows, cols})};
}

#define BM(FN) BM_##FN->Arg(0);

#define BM_SUITE(NAME, DYNAMIC_ROW, DYNAMIC_COL, ROWS, COLS)           \
  BM(CpurtV(NAME, Sum2D(DYNAMIC_ROW, DYNAMIC_COL, ROWS, COLS), "main", \
            Inputs(ROWS, COLS)));                                      \
  BM(Eigen(NAME, EigenSum2D(), Inputs(ROWS, COLS)));                   \
  BM(Tfrt(NAME, Sum2D(DYNAMIC_ROW, DYNAMIC_COL, ROWS, COLS), "main",   \
          Inputs(ROWS, COLS)))

// TODO(b/207822945): Enable after reduction grouper pass is implemented.
#define BM_DYNAMIC_ALL(ROWS, COLS)                                          \
  BM_SUITE(Sum2DDynamicAll_##ROWS##_##COLS, kDynamicDim, kDynamicDim, ROWS, \
           COLS)
BM_DYNAMIC_ALL(2, 80);
// BM_DYNAMIC_ALL(8, 6);
// BM_DYNAMIC_ALL(80, 1);
// BM_DYNAMIC_ALL(80, 60);
// BM_DYNAMIC_ALL(81, 61);
// BM_DYNAMIC_ALL(800, 600);
// BM_DYNAMIC_ALL(802, 602);

#define BM_STATIC_ROW(ROWS, COLS) \
  BM_SUITE(Sum2DStaticRow##ROWS##_##COLS, kStaticDim, kDynamicDim, ROWS, COLS)
// BM_STATIC_ROW(2, 80);
// BM_STATIC_ROW(8, 6);
// BM_STATIC_ROW(80, 1);
// BM_STATIC_ROW(80, 60);
// BM_STATIC_ROW(81, 61);
// BM_STATIC_ROW(800, 600);
// BM_STATIC_ROW(802, 602);

#define BM_STATIC_COL(ROWS, COLS) \
  BM_SUITE(Sum2DStaticCol_##ROWS##_##COLS, kDynamicDim, kStaticDim, ROWS, COLS)
// BM_STATIC_COL(2, 80);
// BM_STATIC_COL(8, 6);
// BM_STATIC_COL(80, 1);
// BM_STATIC_COL(80, 60);
// BM_STATIC_COL(81, 61);
// BM_STATIC_COL(800, 600);
// BM_STATIC_COL(802, 602);

#define BM_STATIC_ALL(ROWS, COLS) \
  BM_SUITE(Sum2DStaticAll_##ROWS##_##COLS, kStaticDim, kStaticDim, ROWS, COLS)
// BM_STATIC_ALL(2, 80);
// BM_STATIC_ALL(8, 6);
// BM_STATIC_ALL(80, 1);
// BM_STATIC_ALL(80, 60);
// BM_STATIC_ALL(81, 61);
// BM_STATIC_ALL(800, 600);
// BM_STATIC_ALL(802, 602);

}  // namespace
}  // namespace tensorflow

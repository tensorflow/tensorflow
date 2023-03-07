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

#include <optional>
#include <string>

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {
namespace {

const char* kMapIR = R"(
  func.func @main(%arg0: {0}) -> {0} {
    %abs = "tf.Abs"(%arg0)
      {{device = "/job:localhost/replica:0/task:0/device:CPU:0"}
      : ({0}) -> {0}
    %exp = "tf.Exp"(%abs)
      {{device = "/job:localhost/replica:0/task:0/device:CPU:0"}
      : ({0}) -> {0}
    %tanh = "tf.Tanh"(%exp)
      {{device = "/job:localhost/replica:0/task:0/device:CPU:0"}
      : ({0}) -> {0}
    func.return %tanh : {0}
  }
)";

std::string Map(llvm::ArrayRef<bool> dynamic_dims,
                llvm::ArrayRef<ssize_t> input_shape) {
  llvm::SmallVector<int64_t, 2> mlir_input_shape;
  for (int i = 0; i < input_shape.size(); ++i) {
    mlir_input_shape.push_back(dynamic_dims[i] ? kDynSize : input_shape[i]);
  }
  return llvm::formatv(kMapIR, PrintTensorType(mlir_input_shape, "f32"));
}

auto EigenMap() {
  return [](llvm::ArrayRef<Tensor> inputs,
            std::optional<Eigen::ThreadPoolDevice>) {
    Tensor output(DT_FLOAT, {inputs[0].dim_size(0), inputs[0].dim_size(1)});

    auto in = inputs[0].tensor<float, 2>();
    auto out = output.tensor<float, 2>();
    out.setZero();
    Eigen::DefaultDevice d;
    out.device(d) = in.abs().exp().tanh();
  };
}

llvm::SmallVector<InputTensorSpec> Inputs(ssize_t rows, ssize_t cols) {
  return {InputTensorSpec(DT_FLOAT, {rows, cols})};
}

#define BM(FN) BM_##FN->Arg(0);

#define BM_SUITE(NAME, DYNAMIC_ROW, DYNAMIC_COL, ROWS, COLS)             \
  BM(JitrtV(NAME, Map({DYNAMIC_ROW, DYNAMIC_COL}, {ROWS, COLS}), "main", \
            Inputs(ROWS, COLS)));                                        \
  BM(Eigen(NAME, EigenMap(), Inputs(ROWS, COLS)));                       \
  BM(Tfrt(NAME, Map({DYNAMIC_ROW, DYNAMIC_COL}, {ROWS, COLS}), "main",   \
          Inputs(ROWS, COLS)))

#define BM_DYNAMIC_ALL(ROWS, COLS) \
  BM_SUITE(MapDynamicAll_##ROWS##_##COLS, kDynamicDim, kDynamicDim, ROWS, COLS)
BM_DYNAMIC_ALL(2, 80);
BM_DYNAMIC_ALL(8, 6);
BM_DYNAMIC_ALL(80, 1);
BM_DYNAMIC_ALL(80, 60);
BM_DYNAMIC_ALL(81, 61);
BM_DYNAMIC_ALL(800, 600);
BM_DYNAMIC_ALL(802, 602);

#define BM_STATIC_ROW(ROWS, COLS) \
  BM_SUITE(MapStaticRow_##ROWS##_##COLS, kStaticDim, kDynamicDim, ROWS, COLS)
BM_STATIC_ROW(2, 80);
BM_STATIC_ROW(8, 6);
BM_STATIC_ROW(80, 1);
BM_STATIC_ROW(80, 60);
BM_STATIC_ROW(81, 61);
BM_STATIC_ROW(800, 600);
BM_STATIC_ROW(802, 602);

#define BM_STATIC_COL(ROWS, COLS) \
  BM_SUITE(MapStaticCol_##ROWS##_##COLS, kDynamicDim, kStaticDim, ROWS, COLS)
BM_STATIC_COL(2, 80);
BM_STATIC_COL(8, 6);
BM_STATIC_COL(80, 1);
BM_STATIC_COL(80, 60);
BM_STATIC_COL(81, 61);
BM_STATIC_COL(800, 600);
BM_STATIC_COL(802, 602);

#define BM_STATIC_ALL(ROWS, COLS) \
  BM_SUITE(MapStaticAll_##ROWS##_##COLS, kStaticDim, kStaticDim, ROWS, COLS)
BM_STATIC_ALL(2, 80);
BM_STATIC_ALL(8, 6);
BM_STATIC_ALL(80, 1);
BM_STATIC_ALL(80, 60);
BM_STATIC_ALL(81, 61);
BM_STATIC_ALL(800, 600);
BM_STATIC_ALL(802, 602);

}  // namespace
}  // namespace tensorflow

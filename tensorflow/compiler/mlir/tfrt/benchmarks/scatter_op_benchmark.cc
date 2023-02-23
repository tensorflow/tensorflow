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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {
namespace {

// {0} -- updates_shape
// {1} -- output_shape
// {2} -- indices_value
// {3} -- indices_value
const char* kMapScatterIR = R"(
  func.func @main(%updates: {0}, %out: {1}) -> {1} {
    %indices = "tf.Const"()
      {{value = {2} : {3},
      device = "/job:localhost/replica:0/task:0/device:CPU:0"}
      : () -> {3}
    %updates_exp = "tf.Exp"(%updates)
      {{device = "/job:localhost/replica:0/task:0/device:CPU:0"}
      : ({0}) -> {0}
    %scattered = "tf.TensorScatterAdd"(%out, %indices, %updates)
      {{device = "/job:localhost/replica:0/task:0/device:CPU:0"}
      : ({1}, {3}, {0}) -> {1}
    func.return %scattered : {1}
  }
)";

std::string GetScatterIndices(llvm::ArrayRef<ssize_t> updates_shape,
                              ssize_t rows) {
  std::string result{"dense<["};
  llvm::raw_string_ostream ss(result);
  for (size_t i = 0; i < updates_shape[0]; ++i) {
    if (i > 0) ss << ',';
    ss << "[0, " << (i * 5) % rows << "]";
  }
  ss << "]>";
  return result;
}

std::string MapScatter(llvm::ArrayRef<bool> dynamic_dims,
                       llvm::ArrayRef<ssize_t> updates_shape,
                       llvm::ArrayRef<ssize_t> output_shape) {
  llvm::SmallVector<int64_t, 2> mlir_output_shape;
  for (int i = 0; i < output_shape.size(); ++i) {
    mlir_output_shape.push_back(dynamic_dims[i] ? kDynSize : output_shape[i]);
  }
  llvm::SmallVector<int64_t, 2> indeces_shape = {updates_shape[0], 2};
  return llvm::formatv(kMapScatterIR, PrintTensorType(updates_shape, "f32"),
                       PrintTensorType(mlir_output_shape, "f32"),
                       GetScatterIndices(updates_shape, output_shape[1]),
                       PrintTensorType(indeces_shape, "i32"));
}

llvm::SmallVector<InputTensorSpec> Inputs(ssize_t rows, ssize_t cols,
                                          ssize_t rows_upd) {
  return {InputTensorSpec(DT_FLOAT, {rows_upd, cols}),  // updates_shape
          InputTensorSpec(DT_FLOAT, {1, rows, cols})};  // output_shapes
}

// This benchmark checks the insertion of full rows (Tfrt requirement).
#define BM(FN) BM_##FN->Arg(0);

#define BM_SUITE(NAME, DYNAMIC_ROW, DYNAMIC_COL, ROWS, COLS, ROWS_UPD) \
  BM(JitrtV(NAME,                                                      \
            MapScatter({DYNAMIC_ROW, DYNAMIC_COL}, {ROWS_UPD, COLS},   \
                       {1, ROWS, COLS}),                               \
            "main", Inputs(ROWS, COLS, ROWS_UPD)));                    \
  BM(Tfrt(NAME,                                                        \
          MapScatter({DYNAMIC_ROW, DYNAMIC_COL}, {ROWS_UPD, COLS},     \
                     {1, ROWS, COLS}),                                 \
          "main", Inputs(ROWS, COLS, ROWS_UPD)))

#define BM_STATIC_ALL(ROWS, COLS, ROWS_UPD)                              \
  BM_SUITE(MapScatterStaticAll_##ROWS##_##COLS##_##ROWS_UPD, kStaticDim, \
           kStaticDim, ROWS, COLS, ROWS_UPD)
BM_STATIC_ALL(11, 1, 5);
BM_STATIC_ALL(20, 11, 5);
BM_STATIC_ALL(1, 80, 100);
BM_STATIC_ALL(80, 1, 5);
BM_STATIC_ALL(800, 600, 10);
BM_STATIC_ALL(802, 602, 100);

#define BM_DYNAMIC_ALL(ROWS, COLS, ROWS_UPD)                               \
  BM_SUITE(MapScatterDynamicAll_##ROWS##_##COLS##_##ROWS_UPD, kDynamicDim, \
           kDynamicDim, ROWS, COLS, ROWS_UPD)
BM_DYNAMIC_ALL(11, 1, 5);
BM_DYNAMIC_ALL(20, 11, 5);
BM_DYNAMIC_ALL(1, 80, 100);
BM_DYNAMIC_ALL(80, 1, 5);
BM_DYNAMIC_ALL(800, 600, 10);
BM_DYNAMIC_ALL(802, 602, 100);

#define BM_STATIC_ROW(ROWS, COLS, ROWS_UPD)                              \
  BM_SUITE(MapScatterStaticRow_##ROWS##_##COLS##_##ROWS_UPD, kStaticDim, \
           kDynamicDim, ROWS, COLS, ROWS_UPD)
BM_STATIC_ROW(11, 1, 5);
BM_STATIC_ROW(20, 11, 5);
BM_STATIC_ROW(1, 80, 100);
BM_STATIC_ROW(80, 1, 5);
BM_STATIC_ROW(800, 600, 10);
BM_STATIC_ROW(802, 602, 100);

#define BM_STATIC_COL(ROWS, COLS, ROWS_UPD)                               \
  BM_SUITE(MapScatterStaticCol_##ROWS##_##COLS##_##ROWS_UPD, kDynamicDim, \
           kStaticDim, ROWS, COLS, ROWS_UPD)
BM_STATIC_COL(11, 1, 5);
BM_STATIC_COL(20, 11, 5);
BM_STATIC_COL(1, 80, 100);
BM_STATIC_COL(80, 1, 5);
BM_STATIC_COL(800, 600, 10);
BM_STATIC_COL(802, 602, 100);

}  // namespace
}  // namespace tensorflow

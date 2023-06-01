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

#include <optional>
#include <string>

#include "llvm/Support/FormatVariadic.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {
namespace {

const char* kReverseIR = R"(
  func.func @main(%input: {0}) -> {0} {
    %reverse_dims = "tf.Const"() {{
      value = {1} : {2},
      device = "/job:localhost/replica:0/task:0/device:CPU:0"
    } : () -> {2}
    %result = "tf.ReverseV2"(%input, %reverse_dims)
      {{device = "/job:localhost/replica:0/task:0/device:CPU:0"}
      : ({0}, {2}) -> {0}
    func.return %result : {0}
  }
)";

std::string Reverse(llvm::ArrayRef<int64_t> input_shape,
                    llvm::ArrayRef<bool> dynamic_dims,
                    llvm::ArrayRef<int32_t> reverse_dims,
                    llvm::StringRef element_type) {
  llvm::SmallVector<int64_t, 4> mlir_input_shape =
      GetTensorTypeShape(input_shape, dynamic_dims);
  return llvm::formatv(
      kReverseIR,
      PrintTensorType(mlir_input_shape, element_type),  // Input type {0}
      PrintDenseArray(reverse_dims),  // Dims to reverse attr {1}
      PrintTensorType(static_cast<int64_t>(reverse_dims.size()),
                      "i32")  // Dims to reverse type {2}
  );
}

template <int64_t INPUT_RANK, size_t N_REVERSE_DIMS>
auto EigenReverse(std::array<int64_t, N_REVERSE_DIMS> reverse_dims) {
  return [reverse_dims](llvm::ArrayRef<Tensor> inputs,
                        std::optional<Eigen::ThreadPoolDevice> device) {
    std::array<bool, INPUT_RANK> bool_reverse_dims;
    bool_reverse_dims.fill(false);
    for (auto i : reverse_dims) {
      bool_reverse_dims[i] = true;
    }
    Tensor output(DT_FLOAT, inputs[0].shape());
    auto in = inputs[0].tensor<float, INPUT_RANK>();
    auto out = output.tensor<float, INPUT_RANK>();
    if (device.has_value()) {
      out.device(*device) = in.reverse(bool_reverse_dims);
    } else {
      out = in.reverse(bool_reverse_dims);
    }
  };
}

llvm::SmallVector<InputTensorSpec> GetInputSpec(
    llvm::ArrayRef<ssize_t> input_shape) {
  return {InputTensorSpec(DT_FLOAT, input_shape)};
}

#define INTS(...) __VA_ARGS__
#define BOOLS(...) __VA_ARGS__

#define BM(KIND, ...) BM_##KIND(__VA_ARGS__)->Arg(0);

#define BM_SUITE(NAME, INPUT_RANK, INPUT_SHAPE, DYNAMIC_DIMS, N_REVERSE_DIMS, \
                 REVERSE_DIMS)                                                \
  BM(JitrtV, NAME,                                                            \
     Reverse({INPUT_SHAPE}, {DYNAMIC_DIMS}, {REVERSE_DIMS}, "f32"), "main",   \
     GetInputSpec({INPUT_SHAPE}));                                            \
  BM(Eigen, NAME,                                                             \
     (EigenReverse<INPUT_RANK>(                                               \
         std::array<int64_t, N_REVERSE_DIMS>{REVERSE_DIMS})),                 \
     GetInputSpec({INPUT_SHAPE}));                                            \
  BM(Tfrt, NAME,                                                              \
     Reverse({INPUT_SHAPE}, {DYNAMIC_DIMS}, {REVERSE_DIMS}, "f32"), "main",   \
     GetInputSpec({INPUT_SHAPE}))

////////////////////////////////////////////////////////////////////////////////
// Reverse 1D tensors.
////////////////////////////////////////////////////////////////////////////////

#define BM_STATIC_1D(SIZE)                                               \
  BM_SUITE(ReverseStatic_1D_##SIZE, 1, INTS(SIZE), BOOLS(kStaticDim), 1, \
           INTS(0))
BM_STATIC_1D(3);
BM_STATIC_1D(8);
BM_STATIC_1D(80);
BM_STATIC_1D(800);
BM_STATIC_1D(8000);
BM_STATIC_1D(8131);
BM_STATIC_1D(1000000);
BM_STATIC_1D(1010131);

#define BM_DYNAMIC_1D(SIZE)                                                \
  BM_SUITE(ReverseDynamic_1D_##SIZE, 1, INTS(SIZE), BOOLS(kDynamicDim), 1, \
           INTS(0))
BM_DYNAMIC_1D(3);
BM_DYNAMIC_1D(8);
BM_DYNAMIC_1D(80);
BM_DYNAMIC_1D(800);
BM_DYNAMIC_1D(8000);
BM_DYNAMIC_1D(8131);
BM_DYNAMIC_1D(1000000);
BM_DYNAMIC_1D(1010131);

////////////////////////////////////////////////////////////////////////////////
// Reverse 2D tensors.
////////////////////////////////////////////////////////////////////////////////

#define BM_STATIC_2D_ROW(ROWS, COLS)                                  \
  BM_SUITE(ReverseStatic_2D_ROW_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
           BOOLS(kStaticDim, kStaticDim), 1, INTS(0))
BM_STATIC_2D_ROW(2, 80);
BM_STATIC_2D_ROW(8, 6);
BM_STATIC_2D_ROW(80, 1);
BM_STATIC_2D_ROW(80, 3);
BM_STATIC_2D_ROW(80, 7);
BM_STATIC_2D_ROW(80, 60);
BM_STATIC_2D_ROW(81, 61);
BM_STATIC_2D_ROW(800, 600);
BM_STATIC_2D_ROW(802, 602);

#define BM_STATIC_2D_COL(ROWS, COLS)                                  \
  BM_SUITE(ReverseStatic_2D_COL_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
           BOOLS(kStaticDim, kStaticDim), 1, INTS(1))
BM_STATIC_2D_COL(2, 80);
BM_STATIC_2D_COL(8, 6);
BM_STATIC_2D_COL(80, 1);
BM_STATIC_2D_COL(80, 3);
BM_STATIC_2D_COL(80, 7);
BM_STATIC_2D_COL(80, 60);
BM_STATIC_2D_COL(81, 61);
BM_STATIC_2D_COL(800, 600);
BM_STATIC_2D_COL(802, 602);

#define BM_STATIC_2D_ALL(ROWS, COLS)                                  \
  BM_SUITE(ReverseStatic_2D_ALL_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
           BOOLS(kStaticDim, kStaticDim), 2, INTS(0, 1))
BM_STATIC_2D_ALL(2, 80);
BM_STATIC_2D_ALL(8, 6);
BM_STATIC_2D_ALL(80, 1);
BM_STATIC_2D_ALL(80, 3);
BM_STATIC_2D_ALL(80, 7);
BM_STATIC_2D_ALL(80, 60);
BM_STATIC_2D_ALL(81, 61);
BM_STATIC_2D_ALL(800, 600);
BM_STATIC_2D_ALL(802, 602);

#define BM_DYNAMIC_2D_ROW(ROWS, COLS)                                  \
  BM_SUITE(ReverseDynamic_2D_ROW_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
           BOOLS(kDynamicDim, kStaticDim), 1, INTS(0))
BM_DYNAMIC_2D_ROW(2, 80);
BM_DYNAMIC_2D_ROW(8, 6);
BM_DYNAMIC_2D_ROW(80, 1);
BM_DYNAMIC_2D_ROW(80, 3);
BM_DYNAMIC_2D_ROW(80, 7);
BM_DYNAMIC_2D_ROW(80, 60);
BM_DYNAMIC_2D_ROW(81, 61);
BM_DYNAMIC_2D_ROW(800, 600);
BM_DYNAMIC_2D_ROW(802, 602);

#define BM_DYNAMIC_2D_COL(ROWS, COLS)                                  \
  BM_SUITE(ReverseDynamic_2D_COL_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
           BOOLS(kStaticDim, kDynamicDim), 1, INTS(1))
BM_DYNAMIC_2D_COL(2, 80);
BM_DYNAMIC_2D_COL(8, 6);
BM_DYNAMIC_2D_COL(80, 1);
BM_DYNAMIC_2D_COL(80, 3);
BM_DYNAMIC_2D_COL(80, 7);
BM_DYNAMIC_2D_COL(80, 60);
BM_DYNAMIC_2D_COL(81, 61);
BM_DYNAMIC_2D_COL(800, 600);
BM_DYNAMIC_2D_COL(802, 602);

#define BM_DYNAMIC_2D_ALL(ROWS, COLS)                                  \
  BM_SUITE(ReverseDynamic_2D_ALL_##ROWS##_##COLS, 2, INTS(ROWS, COLS), \
           BOOLS(kDynamicDim, kDynamicDim), 2, INTS(0, 1))
BM_DYNAMIC_2D_ALL(2, 80);
BM_DYNAMIC_2D_ALL(8, 6);
BM_DYNAMIC_2D_ALL(80, 1);
BM_DYNAMIC_2D_ALL(80, 3);
BM_DYNAMIC_2D_ALL(80, 7);
BM_DYNAMIC_2D_ALL(80, 60);
BM_DYNAMIC_2D_ALL(81, 61);
BM_DYNAMIC_2D_ALL(800, 600);
BM_DYNAMIC_2D_ALL(802, 602);

}  // namespace
}  // namespace tensorflow

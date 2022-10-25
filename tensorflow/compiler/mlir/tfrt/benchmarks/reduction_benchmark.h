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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_REDUCTION_BENCHMARK_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_REDUCTION_BENCHMARK_H_

#include <string>

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {

std::string GetSumF32IR(llvm::ArrayRef<int32_t> input_shape,
                        llvm::ArrayRef<bool> dynamic_dims,
                        llvm::ArrayRef<int32_t> dims_to_reduce);

std::string GetMeanF32IR(llvm::ArrayRef<int32_t> input_shape,
                         llvm::ArrayRef<bool> dynamic_dims,
                         llvm::ArrayRef<int32_t> dims_to_reduce);

template <size_t N>
TensorShape ReducedTensorShape(TensorShape input_shape,
                               std::array<int32_t, N> dims_to_reduce) {
  std::vector<int64_t> result_shape;
  int j = 0;
  for (int i = 0; i < input_shape.dims(); ++i) {
    if (j < dims_to_reduce.size() && i == dims_to_reduce[j]) {
      j++;
      continue;
    }
    result_shape.push_back(input_shape.dim_size(i));
  }
  return TensorShape(result_shape);
}

template <int32_t INPUT_RANK, size_t N_DIMS_TO_REDUCE>
auto GetEigenSumF32Function(
    std::array<int32_t, N_DIMS_TO_REDUCE> dims_to_reduce) {
  return [dims_to_reduce](llvm::ArrayRef<Tensor> inputs,
                          llvm::Optional<Eigen::ThreadPoolDevice> device) {
    Tensor output(DT_FLOAT,
                  ReducedTensorShape(inputs[0].shape(), dims_to_reduce));
    auto in = inputs[0].tensor<float, INPUT_RANK>();
    auto out = output.tensor<float, INPUT_RANK - N_DIMS_TO_REDUCE>();
    out.setZero();
    if (device.has_value()) {
      out.device(*device) = in.sum(dims_to_reduce);
    } else {
      out = in.sum(dims_to_reduce);
    }
  };
}

template <int32_t INPUT_RANK, size_t N_DIMS_TO_REDUCE>
auto GetEigenMeanF32Function(
    std::array<int32_t, N_DIMS_TO_REDUCE> dims_to_reduce) {
  return [dims_to_reduce](llvm::ArrayRef<Tensor> inputs,
                          llvm::Optional<Eigen::ThreadPoolDevice> device) {
    Tensor output(DT_FLOAT,
                  ReducedTensorShape(inputs[0].shape(), dims_to_reduce));
    auto in = inputs[0].tensor<float, INPUT_RANK>();
    auto out = output.tensor<float, INPUT_RANK - N_DIMS_TO_REDUCE>();
    out.setZero();
    if (device.has_value()) {
      out.device(*device) = in.mean(dims_to_reduce);
    } else {
      out = in.mean(dims_to_reduce);
    }
  };
}

llvm::SmallVector<InputTensorSpec> GetInputSpec(
    llvm::ArrayRef<ssize_t> input_shape);

}  // namespace tensorflow

#define INTS(...) __VA_ARGS__
#define BOOLS(...) __VA_ARGS__

#define BM(KIND, ...) BM_##KIND(__VA_ARGS__)->Arg(0);

#define BM_SUITE_SUM_F32(NAME, INPUT_RANK, INPUT_SHAPE, DYNAMIC_DIMS,          \
                         N_DIMS_TO_REDUCE, DIMS_TO_REDUCE)                     \
  BM(JitrtV, NAME,                                                             \
     GetSumF32IR({INPUT_SHAPE}, {DYNAMIC_DIMS}, {DIMS_TO_REDUCE}), "main",     \
     GetInputSpec({INPUT_SHAPE}));                                             \
  BM(Eigen, NAME,                                                              \
     (GetEigenSumF32Function<INPUT_RANK>(                                      \
         std::array<int32_t, N_DIMS_TO_REDUCE>{DIMS_TO_REDUCE})),              \
     GetInputSpec({INPUT_SHAPE}));                                             \
  BM(Tfrt, NAME, GetSumF32IR({INPUT_SHAPE}, {DYNAMIC_DIMS}, {DIMS_TO_REDUCE}), \
     "main", GetInputSpec({INPUT_SHAPE}))

#define BM_SUITE_MEAN_F32(NAME, INPUT_RANK, INPUT_SHAPE, DYNAMIC_DIMS,      \
                          N_DIMS_TO_REDUCE, DIMS_TO_REDUCE)                 \
  BM(JitrtV, NAME,                                                          \
     GetMeanF32IR({INPUT_SHAPE}, {DYNAMIC_DIMS}, {DIMS_TO_REDUCE}), "main", \
     GetInputSpec({INPUT_SHAPE}));                                          \
  BM(Eigen, NAME,                                                           \
     (GetEigenMeanF32Function<INPUT_RANK>(                                  \
         std::array<int32_t, N_DIMS_TO_REDUCE>{DIMS_TO_REDUCE})),           \
     GetInputSpec({INPUT_SHAPE}));                                          \
  BM(Tfrt, NAME,                                                            \
     GetMeanF32IR({INPUT_SHAPE}, {DYNAMIC_DIMS}, {DIMS_TO_REDUCE}), "main", \
     GetInputSpec({INPUT_SHAPE}))

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_REDUCTION_BENCHMARK_H_

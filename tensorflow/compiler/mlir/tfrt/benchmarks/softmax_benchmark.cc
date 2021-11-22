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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/softmax_benchmark.h"

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"

namespace tensorflow {

static const char* kReductionIR = R"(
  func @main(%input: {0}) -> {0} {
    %result = "tf.Softmax"(%input) : ({0}) -> {0}
    return %result : {0}
  }
)";

std::string GetSoftMaxIR(ArrayRef<int64_t> shape, StringRef element_type) {
  return llvm::formatv(kReductionIR, PrintTensorType(shape, element_type));
}

namespace {

#define ARGS_2D          \
  Args({2, 80})          \
      ->Args({8, 6})     \
      ->Args({80, 1})    \
      ->Args({80, 60})   \
      ->Args({81, 61})   \
      ->Args({800, 600}) \
      ->Args({802, 602})

BM_TFMlir2_SingleThread(Softmax2DDynamicAll, f32,
                        MlirSpec("f32", {kDynamicDim, kDynamicDim}))
    ->ARGS_2D;
BM_TFMlir2_SingleThread(Softmax2DRowStatic, f32,
                        MlirSpec("f32", {kStaticDim, kDynamicDim}))
    ->ARGS_2D;
BM_TFMlir2_SingleThread(Softmax2DColStatic, f32,
                        MlirSpec("f32", {kDynamicDim, kStaticDim}))
    ->ARGS_2D;
BM_TFMlir2_SingleThread(Softmax2DStaticAll, f32,
                        MlirSpec("f32", {kStaticDim, kStaticDim}))
    ->ARGS_2D;
BM_Eigen2_SingleThread(Softmax2D, f32)->ARGS_2D;

}  // namespace
}  // namespace tensorflow

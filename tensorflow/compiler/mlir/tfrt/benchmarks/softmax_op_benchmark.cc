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

#include "llvm/Support/FormatVariadic.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

namespace tensorflow {
namespace {

const char* kSoftmaxIR = R"(
  func.func @main(%input: {0}) -> {0} {
    %result = "tf.Softmax"(%input)
      {{device = "/job:localhost/replica:0/task:0/device:CPU:0"}
      : ({0}) -> {0}
    func.return %result : {0}
  }
)";

std::string Softmax(llvm::ArrayRef<bool> dynamic_dims,
                    llvm::ArrayRef<ssize_t> input_shape) {
  llvm::SmallVector<int64_t, 2> mlir_input_shape;
  for (int i = 0; i < input_shape.size(); ++i) {
    mlir_input_shape.push_back(dynamic_dims[i] ? kDynSize : input_shape[i]);
  }
  return llvm::formatv(kSoftmaxIR, PrintTensorType(mlir_input_shape, "f32"));
}

// Eigen code implementing SoftmaxFunctor::operator() carefully taken from
// tensorflow/core/kernels/softmax_op_functor.h
template <typename InT, typename OutT>
static void ComputeSoftmax(const Eigen::DefaultDevice& d, InT logits,
                           OutT softmax) {
  const int kBatchDim = 0;
  const int kClassDim = 1;

  const int batch_size = logits.dimension(kBatchDim);
  const int num_classes = logits.dimension(kClassDim);

  // These arrays are used to reduce along the class dimension, and broadcast
  // the resulting value to all classes.
  Eigen::IndexList<Eigen::type2index<kClassDim> > along_class;
  Eigen::IndexList<int, Eigen::type2index<1> > batch_by_one;
  batch_by_one.set(0, batch_size);
  Eigen::IndexList<Eigen::type2index<1>, int> one_by_class;
  one_by_class.set(1, num_classes);
  // shifted_logits = logits - max(logits along classes);
  auto shifted_logits = (logits - logits.maximum(along_class)
                                      .eval()
                                      .reshape(batch_by_one)
                                      .broadcast(one_by_class));
  softmax.device(d) = shifted_logits.exp();
  softmax.device(d) = (softmax * softmax.sum(along_class)
                                     .inverse()
                                     .eval()
                                     .reshape(batch_by_one)
                                     .broadcast(one_by_class));
}

auto EigenSoftmax() {
  return [](llvm::ArrayRef<Tensor> inputs,
            std::optional<Eigen::ThreadPoolDevice>) {
    Tensor output(DT_FLOAT, {inputs[0].dim_size(0), inputs[0].dim_size(1)});

    auto in = inputs[0].tensor<float, 2>();
    auto out = output.tensor<float, 2>();
    out.setZero();

    Eigen::DefaultDevice default_device;
    ComputeSoftmax<decltype(in), decltype(out)>(default_device, in, out);
  };
}

llvm::SmallVector<InputTensorSpec> Inputs(ssize_t rows, ssize_t cols) {
  return {InputTensorSpec(DT_FLOAT, {rows, cols})};
}

#define BM(FN) BM_##FN->Arg(0);

#define BM_SUITE(NAME, DYNAMIC_ROW, DYNAMIC_COL, ROWS, COLS)                 \
  BM(JitrtV(NAME, Softmax({DYNAMIC_ROW, DYNAMIC_COL}, {ROWS, COLS}), "main", \
            Inputs(ROWS, COLS)));                                            \
  BM(Eigen(NAME, EigenSoftmax(), Inputs(ROWS, COLS)));                       \
  BM(Tfrt(NAME, Softmax({DYNAMIC_ROW, DYNAMIC_COL}, {ROWS, COLS}), "main",   \
          Inputs(ROWS, COLS)))

#define BM_DYNAMIC_ALL(ROWS, COLS)                                            \
  BM_SUITE(SoftmaxDynamicAll_##ROWS##_##COLS, kDynamicDim, kDynamicDim, ROWS, \
           COLS)
BM_DYNAMIC_ALL(2, 80);
BM_DYNAMIC_ALL(8, 6);
BM_DYNAMIC_ALL(80, 1);
BM_DYNAMIC_ALL(80, 60);
BM_DYNAMIC_ALL(81, 61);
BM_DYNAMIC_ALL(800, 600);
BM_DYNAMIC_ALL(802, 602);

#define BM_STATIC_ROW(ROWS, COLS)                                           \
  BM_SUITE(SoftmaxStaticRow_##ROWS##_##COLS, kStaticDim, kDynamicDim, ROWS, \
           COLS)
BM_STATIC_ROW(2, 80);
BM_STATIC_ROW(8, 6);
BM_STATIC_ROW(80, 1);
BM_STATIC_ROW(80, 60);
BM_STATIC_ROW(81, 61);
BM_STATIC_ROW(800, 600);
BM_STATIC_ROW(802, 602);

#define BM_STATIC_COL(ROWS, COLS)                                           \
  BM_SUITE(SoftmaxStaticCol_##ROWS##_##COLS, kDynamicDim, kStaticDim, ROWS, \
           COLS)
BM_STATIC_COL(2, 80);
BM_STATIC_COL(8, 6);
BM_STATIC_COL(80, 1);
BM_STATIC_COL(80, 60);
BM_STATIC_COL(81, 61);
BM_STATIC_COL(800, 600);
BM_STATIC_COL(802, 602);

#define BM_STATIC_ALL(ROWS, COLS) \
  BM_SUITE(SoftmaxStaticAll_##ROWS##_##COLS, kStaticDim, kStaticDim, ROWS, COLS)
BM_STATIC_ALL(2, 80);
BM_STATIC_ALL(8, 6);
BM_STATIC_ALL(80, 1);
BM_STATIC_ALL(80, 60);
BM_STATIC_ALL(81, 61);
BM_STATIC_ALL(800, 600);
BM_STATIC_ALL(802, 602);

}  // namespace
}  // namespace tensorflow

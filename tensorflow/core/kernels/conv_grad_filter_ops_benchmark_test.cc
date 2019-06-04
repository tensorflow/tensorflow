/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

template <typename T>
static Tensor MakeRandomTensor(const TensorShape& shape) {
  Tensor tensor(DataTypeToEnum<T>::value, TensorShape(shape));
  tensor.flat<T>() = tensor.flat<T>().setRandom();
  return tensor;
}

template <typename T>
static Graph* Conv2DBackpropFilter(int batch, int height, int width,
                                   int in_depth, int filter_w, int filter_h,
                                   int out_depth, TensorFormat data_format) {
  auto* graph = new Graph(OpRegistry::Global());

  Tensor input_t = data_format == FORMAT_NHWC
                       ? MakeRandomTensor<T>({batch, height, width, in_depth})
                       : MakeRandomTensor<T>({batch, in_depth, height, width});
  Tensor filter_t =
      MakeRandomTensor<T>({filter_w, filter_h, in_depth, out_depth});

  // Compute dimensions for the `out_backprop` tensor.
  Conv2DParameters params;
  params.dilations = {1, 1, 1, 1};
  params.strides = {1, 1, 1, 1};
  params.padding = Padding::SAME;
  params.data_format = data_format;

  Conv2DDimensions conv2d_dims;
  TF_CHECK_OK(ComputeConv2DDimension(params, input_t, filter_t, &conv2d_dims));

  Tensor out_backprop_t =
      data_format == FORMAT_NHWC
          ? MakeRandomTensor<T>(
                {batch, conv2d_dims.out_rows, conv2d_dims.out_cols, out_depth})
          : MakeRandomTensor<T>(
                {batch, out_depth, conv2d_dims.out_rows, conv2d_dims.out_cols});

  Tensor filter_dims_t(DT_INT32, TensorShape({4}));
  filter_dims_t.flat<int32>()(0) = filter_h;
  filter_dims_t.flat<int32>()(1) = filter_w;
  filter_dims_t.flat<int32>()(2) = in_depth;
  filter_dims_t.flat<int32>()(3) = out_depth;

  Node* input = test::graph::Constant(graph, input_t, "input");
  Node* filter_dims =
      test::graph::HostConstant(graph, filter_dims_t, "filter_dims");
  Node* backprop = test::graph::Constant(graph, out_backprop_t, "backprop");

  Node* conv2d;
  TF_CHECK_OK(
      NodeBuilder(graph->NewName("conv_bwd_filter"), "Conv2DBackpropFilter")
          .Input(input)
          .Input(filter_dims)
          .Input(backprop)
          .Attr("T", DataTypeToEnum<T>::value)
          .Attr("strides", {1, 1, 1, 1})
          .Attr("padding", "SAME")
          .Attr("data_format", ToString(data_format))
          .Finalize(graph, &conv2d));

  return graph;
}

// -------------------------------------------------------------------------- //
// The following benchmarks are used to compare different data format
// performance for different data types. They make sense only when CUDA enabled,
// because on CPU we only support data in NHWC.
// -------------------------------------------------------------------------- //

// Macro arguments names: --------------------------------------------------- //
//      T: data type
// FORMAT: data format (NHWC or NCHW)
//      N: batch size
//      H: height
//      W: width
//      C: channels
//     FC: filter count
//     FH: filter height
//     FW: filter width

#define BM_NAME(name, type, T, FORMAT, N, H, W, C, FW, FH, FC) \
  name##_##T##_##FORMAT##_##type##_##N##_##H##_##W##_##C##_##FW##_##FH##_##FC

#define BM_Conv2DBwdFilterFmt(T, FORMAT, N, H, W, C, FW, FH, FC, type)        \
  static void BM_NAME(BM_Conv2DBackpropFilter, type, T, FORMAT, N, H, W, C,   \
                      FW, FH, FC)(int iters) {                                \
    testing::ItemsProcessed(static_cast<int64>(iters) * (N) * (H) * (W) *     \
                            (C));                                             \
    test::Benchmark(#type, Conv2DBackpropFilter<T>(N, H, W, C, FW, FH, FC,    \
                                                   FORMAT_##FORMAT))          \
        .Run(iters);                                                          \
  }                                                                           \
  BENCHMARK(BM_NAME(BM_Conv2DBackpropFilter, type, T, FORMAT, N, H, W, C, FW, \
                    FH, FC));

#if GOOGLE_CUDA
using fp32 = float;
using fp16 = Eigen::half;

// ResNet50-ish convolutions.
#define BENCHMARK_DTYPE(FORMAT, BATCH, T)                                \
  BM_Conv2DBwdFilterFmt(T, FORMAT, BATCH, 56, 56, 64, 1, 1, 64, gpu);    \
  BM_Conv2DBwdFilterFmt(T, FORMAT, BATCH, 56, 56, 64, 1, 1, 256, gpu);   \
  BM_Conv2DBwdFilterFmt(T, FORMAT, BATCH, 56, 56, 256, 1, 1, 64, gpu);   \
  BM_Conv2DBwdFilterFmt(T, FORMAT, BATCH, 56, 56, 64, 3, 3, 64, gpu);    \
                                                                         \
  BM_Conv2DBwdFilterFmt(T, FORMAT, BATCH, 28, 28, 128, 1, 1, 128, gpu);  \
  BM_Conv2DBwdFilterFmt(T, FORMAT, BATCH, 28, 28, 128, 1, 1, 512, gpu);  \
  BM_Conv2DBwdFilterFmt(T, FORMAT, BATCH, 28, 28, 512, 1, 1, 128, gpu);  \
  BM_Conv2DBwdFilterFmt(T, FORMAT, BATCH, 28, 28, 512, 3, 3, 128, gpu);  \
                                                                         \
  BM_Conv2DBwdFilterFmt(T, FORMAT, BATCH, 14, 14, 256, 1, 1, 256, gpu);  \
  BM_Conv2DBwdFilterFmt(T, FORMAT, BATCH, 14, 14, 256, 1, 1, 1024, gpu); \
  BM_Conv2DBwdFilterFmt(T, FORMAT, BATCH, 14, 14, 1024, 1, 1, 256, gpu); \
  BM_Conv2DBwdFilterFmt(T, FORMAT, BATCH, 14, 14, 256, 3, 3, 256, gpu);

BENCHMARK_DTYPE(NHWC, 32, fp32);
BENCHMARK_DTYPE(NCHW, 32, fp32);

BENCHMARK_DTYPE(NHWC, 32, fp16);
BENCHMARK_DTYPE(NCHW, 32, fp16);

BENCHMARK_DTYPE(NHWC, 64, fp32);
BENCHMARK_DTYPE(NCHW, 64, fp32);

BENCHMARK_DTYPE(NHWC, 64, fp16);
BENCHMARK_DTYPE(NCHW, 64, fp16);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
static Graph* Conv2DBackpropInput(int batch, int height, int width,
                                  int in_depth, int filter_h, int filter_w,
                                  int out_depth, int stride_h, int stride_w,
                                  Padding padding, TensorFormat data_format) {
  auto* graph = new Graph(OpRegistry::Global());

  Tensor input_t = data_format == FORMAT_NHWC
                       ? MakeRandomTensor<T>({batch, height, width, in_depth})
                       : MakeRandomTensor<T>({batch, in_depth, height, width});
  Tensor filter_t =
      MakeRandomTensor<T>({filter_w, filter_h, in_depth, out_depth});

  // Compute dimensions for the `out_backprop` tensor.
  Conv2DParameters params;
  params.dilations = {1, 1, 1, 1};
  params.strides = {1, stride_h, stride_w, 1};
  params.padding = padding;
  params.data_format = data_format;

  Conv2DDimensions conv2d_dims;
  TF_CHECK_OK(ComputeConv2DDimension(params, input_t, filter_t, &conv2d_dims));

  Tensor out_backprop_t =
      data_format == FORMAT_NHWC
          ? MakeRandomTensor<T>(
                {batch, conv2d_dims.out_rows, conv2d_dims.out_cols, out_depth})
          : MakeRandomTensor<T>(
                {batch, out_depth, conv2d_dims.out_rows, conv2d_dims.out_cols});

  Tensor input_dims_t(DT_INT32, TensorShape({4}));
  input_dims_t.flat<int32>()(0) = input_t.dim_size(0);
  input_dims_t.flat<int32>()(1) = input_t.dim_size(1);
  input_dims_t.flat<int32>()(2) = input_t.dim_size(2);
  input_dims_t.flat<int32>()(3) = input_t.dim_size(3);

  Node* input_dims =
      test::graph::HostConstant(graph, input_dims_t, "input_dims");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");
  Node* backprop = test::graph::Constant(graph, out_backprop_t, "backprop");

  Node* conv2d;
  TF_CHECK_OK(
      NodeBuilder(graph->NewName("conv_bwd_filter"), "Conv2DBackpropInput")
          .Input(input_dims)
          .Input(filter)
          .Input(backprop)
          .Attr("T", DataTypeToEnum<T>::value)
          .Attr("strides", {1, stride_h, stride_w, 1})
          .Attr("padding", padding == Padding::SAME    ? "SAME"
                           : padding == Padding::VALID ? "VALID"
                                                       : "N/A")
          .Attr("data_format", ToString(data_format))
          .Finalize(graph, &conv2d));

  return graph;
}

// Macro arguments names: --------------------------------------------------- //
//      T: data type
// FMT: data format (NHWC or NCHW)
//      N: batch size
//      H: height
//      W: width
//      C: channels
//     FC: filter count
//     FH: filter height
//     FW: filter width
//     SH: stride height
//     SW: stride width

#define BM_CONCAT(a, b) a##_##b

#define BM_NAME(name, type, T, FMT, N, H, W, C, FH, FW, FC, SH, SW, PADDING) \
  BM_CONCAT(name##_##T##_##FMT##_##type##_in##N##x##H##x##W##x##C,           \
            f##FH##x##FW##x##FC##_##s##SH##x##SW##_##PADDING)

#define BM_Conv2DBwdInput(T, FMT, N, H, W, C, FW, FH, FC, SH, SW, PADDING,    \
                          type)                                               \
  static void BM_NAME(BM_Conv2DBackpropInput, type, T, FMT, N, H, W, C, FH,   \
                      FW, FC, SH, SW,                                         \
                      PADDING)(::testing::benchmark::State & state) {         \
    test::Benchmark(#type,                                                    \
                    Conv2DBackpropInput<T>(N, H, W, C, FH, FW, FC, SH, SW,    \
                                           PADDING, FORMAT_##FMT),            \
                    /*old_benchmark_api*/ false)                              \
        .Run(state);                                                          \
    state.SetItemsProcessed(state.iterations() * (N) * (H) * (W) * (C));      \
  }                                                                           \
  BENCHMARK(BM_NAME(BM_Conv2DBackpropInput, type, T, FMT, N, H, W, C, FH, FW, \
                    FC, SH, SW, PADDING));

using fp32 = float;
using fp16 = Eigen::half;

// ResNet50-ish convolutions.
#define BENCHMARK_DTYPE(FMT, BATCH, T, D)                                   \
  BM_Conv2DBwdInput(T, FMT, BATCH, 112, 112, 64, 2, 2, 64, 2, 2, SAME, D);  \
  BM_Conv2DBwdInput(T, FMT, BATCH, 56, 56, 128, 2, 2, 128, 2, 2, SAME, D);  \
  BM_Conv2DBwdInput(T, FMT, BATCH, 28, 28, 256, 2, 2, 256, 2, 2, SAME, D);  \
                                                                            \
  BM_Conv2DBwdInput(T, FMT, BATCH, 112, 112, 64, 2, 2, 64, 2, 2, VALID, D); \
  BM_Conv2DBwdInput(T, FMT, BATCH, 56, 56, 128, 2, 2, 128, 2, 2, VALID, D); \
  BM_Conv2DBwdInput(T, FMT, BATCH, 28, 28, 256, 2, 2, 256, 2, 2, VALID, D); \
                                                                            \
  BM_Conv2DBwdInput(T, FMT, BATCH, 56, 56, 64, 1, 1, 64, 1, 1, SAME, D);    \
  BM_Conv2DBwdInput(T, FMT, BATCH, 56, 56, 64, 1, 1, 256, 1, 1, SAME, D);   \
  BM_Conv2DBwdInput(T, FMT, BATCH, 56, 56, 256, 1, 1, 64, 1, 1, SAME, D);   \
  BM_Conv2DBwdInput(T, FMT, BATCH, 56, 56, 64, 3, 3, 64, 1, 1, SAME, D);    \
                                                                            \
  BM_Conv2DBwdInput(T, FMT, BATCH, 28, 28, 128, 1, 1, 128, 1, 1, SAME, D);  \
  BM_Conv2DBwdInput(T, FMT, BATCH, 28, 28, 128, 1, 1, 512, 1, 1, SAME, D);  \
  BM_Conv2DBwdInput(T, FMT, BATCH, 28, 28, 512, 1, 1, 128, 1, 1, SAME, D);  \
  BM_Conv2DBwdInput(T, FMT, BATCH, 28, 28, 512, 3, 3, 128, 1, 1, SAME, D);  \
                                                                            \
  BM_Conv2DBwdInput(T, FMT, BATCH, 14, 14, 256, 1, 1, 256, 1, 1, SAME, D);  \
  BM_Conv2DBwdInput(T, FMT, BATCH, 14, 14, 256, 1, 1, 1024, 1, 1, SAME, D); \
  BM_Conv2DBwdInput(T, FMT, BATCH, 14, 14, 1024, 1, 1, 256, 1, 1, SAME, D); \
  BM_Conv2DBwdInput(T, FMT, BATCH, 14, 14, 256, 3, 3, 256, 1, 1, SAME, D);

BENCHMARK_DTYPE(NHWC, 8, fp32, cpu);
BENCHMARK_DTYPE(NHWC, 16, fp32, cpu);
BENCHMARK_DTYPE(NHWC, 32, fp32, cpu);

#if GOOGLE_CUDA
// -------------------------------------------------------------------------- //
// The following benchmarks are used to compare different data format
// performance for different data types. They make sense only when CUDA enabled,
// because on CPU we only support data in NHWC.
// -------------------------------------------------------------------------- //

BENCHMARK_DTYPE(NHWC, 32, fp32, gpu);
BENCHMARK_DTYPE(NCHW, 32, fp32, gpu);

BENCHMARK_DTYPE(NHWC, 32, fp16, gpu);
BENCHMARK_DTYPE(NCHW, 32, fp16, gpu);

BENCHMARK_DTYPE(NHWC, 64, fp32, gpu);
BENCHMARK_DTYPE(NCHW, 64, fp32, gpu);

BENCHMARK_DTYPE(NHWC, 64, fp16, gpu);
BENCHMARK_DTYPE(NCHW, 64, fp16, gpu);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow

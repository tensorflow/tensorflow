/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/mkl_util.h"

// TODO(intel-tf): Add numerical tests that will compare results of default
// (aka Eigen) convolutions with MKL convolutions.

// -------------------------------------------------------------------------- //
// Performance Benchmarks.                                                    //
// -------------------------------------------------------------------------- //

// Compare performance of default Tensorflow convolution kernels (Eigen) with
// MKL kernels on CPU.

// Before running these benchmarks configure OpenMP environment variables:
//   export KMP_BLOCKTIME=0
//   export OMP_NUM_THREADS=${num_threads}

namespace tensorflow {

struct Conv2DDimensions {
  Conv2DDimensions(int n, int h, int w, int c, int fc, int fh, int fw)
      : input_batches(n),
        input_height(h),
        input_width(w),
        input_depth(c),
        filter_count(fc),
        filter_height(fh),
        filter_width(fw) {}

  int input_batches;
  int input_height;
  int input_width;
  int input_depth;
  int filter_count;
  int filter_height;
  int filter_width;
};

static Tensor GetRandomTensor(const TensorShape& shape) {
  Tensor tensor(DT_FLOAT, TensorShape(shape));
  tensor.flat<float>() = tensor.flat<float>().setRandom();
  return tensor;
}

// Get a random Tensor for the Conv2D input.
static Tensor GetRandomInputTensor(const Conv2DDimensions& dims) {
  return GetRandomTensor({dims.input_batches, dims.input_height,
                          dims.input_width, dims.input_depth});
}

// Get a random Tensor for the Conv2D filter.
static Tensor GetRandomFilterTensor(const Conv2DDimensions& dims) {
  return GetRandomTensor({dims.filter_height, dims.filter_width,
                          dims.input_depth, dims.filter_count});
}

// Get a random Tensor for the Conv2D output (assuming SAME padding).
static Tensor GetRandomOutputTensor(const Conv2DDimensions& dims) {
  return GetRandomTensor({dims.input_batches, dims.input_height,
                          dims.input_width, dims.filter_count});
}

// Get a Tensor encoding Conv2D input shape.
static Tensor GetInputSizesTensor(const Conv2DDimensions& dims) {
  return test::AsTensor<int32>({dims.input_batches, dims.input_height,
                                dims.input_width, dims.input_depth});
}

// Get a Tensor encoding Conv2D filter shape.
static Tensor GetFilterSizesTensor(const Conv2DDimensions& dims) {
  return test::AsTensor<int32>({dims.filter_height, dims.filter_width,
                                dims.input_depth, dims.filter_count});
}

static Graph* DefaultConv2D(const Conv2DDimensions& dims) {
  auto* graph = new Graph(OpRegistry::Global());

  Tensor input_t = GetRandomInputTensor(dims);
  Tensor filter_t = GetRandomFilterTensor(dims);

  Node* input = test::graph::Constant(graph, input_t, "input");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");

  Node* conv2d;
  TF_CHECK_OK(NodeBuilder(graph->NewName("conv_2d"), "Conv2D")
                  .Input(input)
                  .Input(filter)
                  .Attr("T", DT_FLOAT)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Finalize(graph, &conv2d));

  return graph;
}

static Graph* MklConv2D(const Conv2DDimensions& dims) {
  auto* graph = new Graph(OpRegistry::Global());

  Tensor input_t = GetRandomInputTensor(dims);
  Tensor filter_t = GetRandomFilterTensor(dims);

  Node* input = test::graph::Constant(graph, input_t, "input");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");

  Node* not_mkl_shape =
      test::graph::Constant(graph, GetMklMetaTensor(), "not_mkl");

  Node* conv2d;
  TF_CHECK_OK(NodeBuilder(graph->NewName("mkl_conv_2d"), "_MklConv2D")
                  .Input(input)
                  .Input(filter)
                  .Input(not_mkl_shape)
                  .Input(not_mkl_shape)
                  .Attr("T", DT_FLOAT)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Attr("_kernel", "MklOp")
                  .Finalize(graph, &conv2d));

  return graph;
}

static Graph* DefaultConv2DBwdInput(const Conv2DDimensions& dims) {
  auto* graph = new Graph(OpRegistry::Global());

  Tensor input_sizes_t = GetInputSizesTensor(dims);
  Tensor filter_t = GetRandomFilterTensor(dims);
  Tensor out_backprop_t = GetRandomOutputTensor(dims);  // assuming SAME padding

  Node* input_sizes =
      test::graph::Constant(graph, input_sizes_t, "input_sizes");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");
  Node* out_backprop =
      test::graph::Constant(graph, out_backprop_t, "out_backprop");

  Node* conv2d_bwd_input;
  TF_CHECK_OK(
      NodeBuilder(graph->NewName("conv_2d_bwd_input"), "Conv2DBackpropInput")
          .Input(input_sizes)
          .Input(filter)
          .Input(out_backprop)
          .Attr("T", DT_FLOAT)
          .Attr("strides", {1, 1, 1, 1})
          .Attr("padding", "SAME")
          .Finalize(graph, &conv2d_bwd_input));

  return graph;
}

static Graph* MklConv2DBwdInput(const Conv2DDimensions& dims) {
  auto* graph = new Graph(OpRegistry::Global());

  Tensor input_sizes_t = GetInputSizesTensor(dims);
  Tensor filter_t = GetRandomFilterTensor(dims);
  Tensor out_backprop_t = GetRandomOutputTensor(dims);  // assuming SAME padding

  Node* input_sizes =
      test::graph::Constant(graph, input_sizes_t, "input_sizes");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");
  Node* out_backprop =
      test::graph::Constant(graph, out_backprop_t, "out_backprop");

  Node* not_mkl_shape =
      test::graph::Constant(graph, GetMklMetaTensor(), "not_mkl");

  Node* conv2d_bwd_input;
  TF_CHECK_OK(NodeBuilder(graph->NewName("conv_2d_bwd_input"),
                          "_MklConv2DBackpropInput")
                  .Input(input_sizes)
                  .Input(filter)
                  .Input(out_backprop)
                  .Input(not_mkl_shape)
                  .Input(not_mkl_shape)
                  .Input(not_mkl_shape)
                  .Attr("T", DT_FLOAT)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Attr("_kernel", "MklOp")
                  .Finalize(graph, &conv2d_bwd_input));

  return graph;
}

static Graph* DefaultConv2DBwdFilter(const Conv2DDimensions& dims) {
  auto* graph = new Graph(OpRegistry::Global());

  Tensor input_t = GetRandomInputTensor(dims);
  Tensor filter_sizes_t = GetFilterSizesTensor(dims);
  Tensor filter_t = GetRandomFilterTensor(dims);
  Tensor out_backprop_t = GetRandomOutputTensor(dims);  // assuming SAME padding

  Node* input = test::graph::Constant(graph, input_t, "input");
  Node* filter_sizes =
      test::graph::Constant(graph, filter_sizes_t, "filter_sizes");
  Node* out_backprop =
      test::graph::Constant(graph, out_backprop_t, "out_backprop");

  Node* conv2d_bwd_filter;
  TF_CHECK_OK(
      NodeBuilder(graph->NewName("conv_2d_bwd_filter"), "Conv2DBackpropFilter")
          .Input(input)
          .Input(filter_sizes)
          .Input(out_backprop)
          .Attr("T", DT_FLOAT)
          .Attr("strides", {1, 1, 1, 1})
          .Attr("padding", "SAME")
          .Finalize(graph, &conv2d_bwd_filter));

  return graph;
}

static Graph* MklConv2DBwdFilter(const Conv2DDimensions& dims) {
  Graph* graph = new Graph(OpRegistry::Global());

  Tensor input_t = GetRandomInputTensor(dims);
  Tensor filter_sizes_t = GetFilterSizesTensor(dims);
  Tensor filter_t = GetRandomFilterTensor(dims);
  Tensor out_backprop_t = GetRandomOutputTensor(dims);  // assuming SAME padding

  Node* input = test::graph::Constant(graph, input_t, "input");
  Node* filter_sizes =
      test::graph::Constant(graph, filter_sizes_t, "filter_sizes");
  Node* out_backprop =
      test::graph::Constant(graph, out_backprop_t, "out_backprop");

  Node* not_mkl_shape =
      test::graph::Constant(graph, GetMklMetaTensor(), "not_mkl");

  Node* conv2d_bwd_filter;
  TF_CHECK_OK(NodeBuilder(graph->NewName("conv_2d_bwd_filter"),
                          "_MklConv2DBackpropFilter")
                  .Input(input)
                  .Input(filter_sizes)
                  .Input(out_backprop)
                  .Input(not_mkl_shape)
                  .Input(not_mkl_shape)
                  .Input(not_mkl_shape)
                  .Attr("T", DT_FLOAT)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Attr("_kernel", "MklOp")
                  .Finalize(graph, &conv2d_bwd_filter));

  return graph;
}

// Macro arguments names: --------------------------------------------------- //
//    N: batch size
//    H: height
//    W: width
//    C: channels
//   FC: filter count
//   FH: filter height
//   FW: filter width

#define BM_CONCAT(a, b) a##b

#define BM_NAME(p, type, N, H, W, C, FC, FH, FW) \
  BM_CONCAT(BM_##p##_##type##_in_##N##_##H##_##W##_##C, _f_##FC##_##FH##_##FW)

// Flops computation in these benchmarks are the same as in
// eigen_benchmark_cpu_test.cc.

#define BM_Conv2DT(kind, N, H, W, C, FC, FH, FW, type, LABEL)           \
  static void BM_NAME(Conv2D_##kind, type, N, H, W, C, FC, FH,          \
                      FW)(::testing::benchmark::State & state) {        \
    state.SetLabel(LABEL);                                              \
                                                                        \
    int64 num_computed_elements = (N) * (H) * (W) * (FC);               \
    int64 flops_per_iter = num_computed_elements * ((C) * (FH) * (FW)); \
                                                                        \
    Conv2DDimensions dims(N, H, W, C, FC, FW, FH);                      \
    test::Benchmark(#type, BM_CONCAT(kind, Conv2D)(dims),               \
                    /*old_benchmark_api*/ false)                        \
        .Run(state);                                                    \
    state.SetItemsProcessed(state.iterations() * flops_per_iter);       \
  }                                                                     \
  BENCHMARK(BM_NAME(Conv2D_##kind, type, N, H, W, C, FC, FH, FW))

#define BM_Conv2D(N, H, W, C, FC, FH, FW, type, LABEL)      \
  BM_Conv2DT(Default, N, H, W, C, FC, FH, FW, type, LABEL); \
  BM_Conv2DT(Mkl, N, H, W, C, FC, FH, FW, type, LABEL);

#define BM_Conv2DBwdInputT(kind, N, H, W, C, FC, FH, FW, type, LABEL)   \
  static void BM_NAME(Conv2DBwdInput_##kind, type, N, H, W, C, FC, FH,  \
                      FW)(::testing::benchmark::State & state) {        \
    state.SetLabel(LABEL);                                              \
                                                                        \
    int64 num_computed_elements = (N) * (H) * (W) * (C);                \
    int64 flops_per_iter = num_computed_elements * ((C) * (FH) * (FW)); \
                                                                        \
    Conv2DDimensions dims(N, H, W, C, FC, FW, FH);                      \
    test::Benchmark(#type, BM_CONCAT(kind, Conv2DBwdInput)(dims),       \
                    /*old_benchmark_api*/ false)                        \
        .Run(state);                                                    \
    state.SetItemsProcessed(state.iterations() * flops_per_iter);       \
  }                                                                     \
  BENCHMARK(BM_NAME(Conv2DBwdInput_##kind, type, N, H, W, C, FC, FH, FW))

#define BM_Conv2DBwdInput(N, H, W, C, FC, FH, FW, type, LABEL)      \
  BM_Conv2DBwdInputT(Default, N, H, W, C, FC, FH, FW, type, LABEL); \
  BM_Conv2DBwdInputT(Mkl, N, H, W, C, FC, FH, FW, type, LABEL);

#define BM_Conv2DBwdFilterT(kind, N, H, W, C, FC, FH, FW, type, LABEL)  \
  static void BM_NAME(Conv2DBwdFilter_##kind, type, N, H, W, C, FC, FH, \
                      FW)(::testing::benchmark::State & state) {        \
    state.SetLabel(LABEL);                                              \
                                                                        \
    int64 num_computed_elements = (FH) * (FW) * (C) * (FC);             \
    int64 flops_per_iter = num_computed_elements * ((N) * (H) * (W));   \
                                                                        \
    Conv2DDimensions dims(N, H, W, C, FC, FW, FH);                      \
    test::Benchmark(#type, BM_CONCAT(kind, Conv2DBwdFilter)(dims),      \
                    /*old_benchmark_api*/ false)                        \
        .Run(state);                                                    \
    state.SetItemsProcessed(state.iterations() * flops_per_iter);       \
  }                                                                     \
  BENCHMARK(BM_NAME(Conv2DBwdFilter_##kind, type, N, H, W, C, FC, FH, FW))

#define BM_Conv2DBwdFilter(N, H, W, C, FC, FH, FW, type, LABEL)      \
  BM_Conv2DBwdFilterT(Default, N, H, W, C, FC, FH, FW, type, LABEL); \
  BM_Conv2DBwdFilterT(Mkl, N, H, W, C, FC, FH, FW, type, LABEL);

// ImageNet Convolutions ---------------------------------------------------- //

BM_Conv2D(32, 28, 28, 96, 128, 3, 3, cpu, "conv3a_00_3x3");
BM_Conv2D(32, 28, 28, 16, 32, 5, 5, cpu, "conv3a_00_5x5");
BM_Conv2D(32, 28, 28, 128, 192, 3, 3, cpu, "conv3_00_3x3");
BM_Conv2D(32, 28, 28, 32, 96, 5, 5, cpu, "conv3_00_5x5");
BM_Conv2D(32, 14, 14, 96, 204, 3, 3, cpu, "conv4a_00_3x3");
BM_Conv2D(32, 14, 14, 16, 48, 5, 5, cpu, "conv4a_00_5x5");
BM_Conv2D(32, 14, 14, 112, 224, 3, 3, cpu, "conv4b_00_3x3");

BM_Conv2DBwdInput(32, 28, 28, 96, 128, 3, 3, cpu, "conv3a_00_3x3");
BM_Conv2DBwdInput(32, 28, 28, 16, 32, 5, 5, cpu, "conv3a_00_5x5");
BM_Conv2DBwdInput(32, 28, 28, 128, 192, 3, 3, cpu, "conv3_00_3x3");
BM_Conv2DBwdInput(32, 28, 28, 32, 96, 5, 5, cpu, "conv3_00_5x5");
BM_Conv2DBwdInput(32, 14, 14, 96, 204, 3, 3, cpu, "conv4a_00_3x3");
BM_Conv2DBwdInput(32, 14, 14, 16, 48, 5, 5, cpu, "conv4a_00_5x5");
BM_Conv2DBwdInput(32, 14, 14, 112, 224, 3, 3, cpu, "conv4b_00_3x3");

BM_Conv2DBwdFilter(32, 28, 28, 96, 128, 3, 3, cpu, "conv3a_00_3x3");
BM_Conv2DBwdFilter(32, 28, 28, 16, 32, 5, 5, cpu, "conv3a_00_5x5");
BM_Conv2DBwdFilter(32, 28, 28, 128, 192, 3, 3, cpu, "conv3_00_3x3");
BM_Conv2DBwdFilter(32, 28, 28, 32, 96, 5, 5, cpu, "conv3_00_5x5");
BM_Conv2DBwdFilter(32, 14, 14, 96, 204, 3, 3, cpu, "conv4a_00_3x3");
BM_Conv2DBwdFilter(32, 14, 14, 16, 48, 5, 5, cpu, "conv4a_00_5x5");
BM_Conv2DBwdFilter(32, 14, 14, 112, 224, 3, 3, cpu, "conv4b_00_3x3");

}  // namespace tensorflow

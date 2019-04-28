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
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

////////////////////////////////////////////////////////////////////////////////
// Performance benchmarks for the FusedConv2Op.                               //
////////////////////////////////////////////////////////////////////////////////

struct Conv2DGraph {
  Graph* graph;
  Node* conv2d;
};

struct Conv2DWithBiasGraph {
  Graph* graph;
  Node* conv2d;
  Node* bias;
};

struct Conv2DWithBiasAndActivationGraph {
  Graph* graph;
  Node* conv2d;
  Node* bias;
  Node* activation;
};

struct Conv2DWithBatchNormGraph {
  Graph* graph;
  Node* conv2d;
  Node* batch_norm;
};

struct Conv2DWithBatchNormAndActivationGraph {
  Graph* graph;
  Node* conv2d;
  Node* batch_norm;
  Node* activation;
};

static Tensor MakeRandomTensor(const TensorShape& shape) {
  Tensor tensor(DT_FLOAT, TensorShape(shape));
  tensor.flat<float>() = tensor.flat<float>().setRandom();
  return tensor;
}

// Creates a simple Tensorflow graph with single Conv2D node.
static Conv2DGraph Conv2D(int batch, int height, int width, int in_depth,
                          int filter_w, int filter_h, int out_depth) {
  Graph* graph = new Graph(OpRegistry::Global());

  Tensor images_t = MakeRandomTensor({batch, height, width, in_depth});
  Tensor filter_t = MakeRandomTensor({filter_w, filter_h, in_depth, out_depth});

  Node* images = test::graph::Constant(graph, images_t, "images");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");

  Node* conv2d;
  TF_CHECK_OK(NodeBuilder(graph->NewName("conv"), "Conv2D")
                  .Input(images)
                  .Input(filter)
                  .Attr("T", DT_FLOAT)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Finalize(graph, &conv2d));

  return {graph, conv2d};
}

// Creates a Tensorflow graph with a Conv2D node followed by BiasAdd.
static Conv2DWithBiasGraph Conv2DWithBias(int batch, int height, int width,
                                          int in_depth, int filter_w,
                                          int filter_h, int out_depth) {
  Conv2DGraph conv_graph =
      Conv2D(batch, height, width, in_depth, filter_w, filter_h, out_depth);

  Graph* graph = conv_graph.graph;
  Node* conv2d = conv_graph.conv2d;

  Tensor bias_t = MakeRandomTensor({out_depth});
  Node* bias = test::graph::Constant(graph, bias_t, "bias");

  Node* out;
  TF_CHECK_OK(NodeBuilder(graph->NewName("bias"), "BiasAdd")
                  .Input(conv2d)
                  .Input(bias)
                  .Attr("T", DT_FLOAT)
                  .Attr("data_format", "NHWC")
                  .Finalize(graph, &out));

  return {graph, conv2d, out};
}

// Creates a Tensorflow graph with a Conv2D node followed by BiasAdd and
// activation (Relu, Relu6, etc...).
static Conv2DWithBiasAndActivationGraph Conv2DWithBiasAndActivation(
    int batch, int height, int width, int in_depth, int filter_w, int filter_h,
    int out_depth, const string& activation_type) {
  Conv2DWithBiasGraph conv_graph = Conv2DWithBias(
      batch, height, width, in_depth, filter_w, filter_h, out_depth);

  Graph* graph = conv_graph.graph;
  Node* conv2d = conv_graph.conv2d;
  Node* bias = conv_graph.bias;

  Node* activation;
  TF_CHECK_OK(NodeBuilder(graph->NewName("activation"), activation_type)
                  .Input(bias)
                  .Attr("T", DT_FLOAT)
                  .Finalize(graph, &activation));

  return {graph, conv2d, bias, activation};
}

// Creates a Tensorflow graph with a Conv2D node followed by FusedBatchNorm.
static Conv2DWithBatchNormGraph Conv2DWithBatchNorm(int batch, int height,
                                                    int width, int in_depth,
                                                    int filter_w, int filter_h,
                                                    int out_depth) {
  Conv2DGraph conv_graph =
      Conv2D(batch, height, width, in_depth, filter_w, filter_h, out_depth);

  Graph* graph = conv_graph.graph;
  Node* conv2d = conv_graph.conv2d;

  Tensor scale_t = MakeRandomTensor({out_depth});
  Tensor offset_t = MakeRandomTensor({out_depth});
  Tensor mean_t = MakeRandomTensor({out_depth});
  Tensor variance_t = MakeRandomTensor({out_depth});

  Node* scale = test::graph::Constant(graph, scale_t, "scale");
  Node* offset = test::graph::Constant(graph, offset_t, "offset");
  Node* mean = test::graph::Constant(graph, mean_t, "mean");
  Node* variance = test::graph::Constant(graph, variance_t, "variance");

  Node* out;
  TF_CHECK_OK(NodeBuilder(graph->NewName("batch_norm"), "FusedBatchNorm")
                  .Input(conv2d)
                  .Input(scale)
                  .Input(offset)
                  .Input(mean)
                  .Input(variance)
                  .Attr("T", DT_FLOAT)
                  .Attr("is_training", false)
                  .Finalize(graph, &out));

  return {graph, conv2d, out};
}

// Creates a Tensorflow graph with a Conv2D node followed by FusedBatchNorm and
// activation (Relu, Relu6, etc...).
static Conv2DWithBatchNormAndActivationGraph Conv2DWithBatchNormAndActivation(
    int batch, int height, int width, int in_depth, int filter_w, int filter_h,
    int out_depth, const string& activation_type) {
  Conv2DWithBatchNormGraph conv_graph = Conv2DWithBatchNorm(
      batch, height, width, in_depth, filter_w, filter_h, out_depth);

  Graph* graph = conv_graph.graph;
  Node* conv2d = conv_graph.conv2d;
  Node* batch_norm = conv_graph.batch_norm;

  Node* activation;
  TF_CHECK_OK(NodeBuilder(graph->NewName("activation"), activation_type)
                  .Input(batch_norm)
                  .Attr("T", DT_FLOAT)
                  .Finalize(graph, &activation));

  return {graph, conv2d, batch_norm, activation};
}

// Creates a tensorflow graph with a single FusedConv2D (with BiasAdd) node and
// fuses into it additional computations (e.g. Relu).
static Graph* FusedConv2DWithBias(int batch, int height, int width,
                                  int in_depth, int filter_w, int filter_h,
                                  int out_depth,
                                  const std::vector<string>& fused_ops = {}) {
  Graph* graph = new Graph(OpRegistry::Global());

  Tensor images_t = MakeRandomTensor({batch, height, width, in_depth});
  Tensor filter_t = MakeRandomTensor({filter_w, filter_h, in_depth, out_depth});
  Tensor bias_t = MakeRandomTensor({out_depth});

  Node* images = test::graph::Constant(graph, images_t, "images");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");
  Node* bias = test::graph::Constant(graph, bias_t, "bias");

  std::vector<NodeBuilder::NodeOut> args = {bias};

  Node* conv;
  TF_CHECK_OK(NodeBuilder(graph->NewName("conv"), "_FusedConv2D")
                  .Input(images)
                  .Input(filter)
                  .Attr("num_args", 1)
                  .Input(args)
                  .Attr("T", DT_FLOAT)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Attr("fused_ops", fused_ops)
                  .Finalize(graph, &conv));

  return graph;
}

// Creates a tensorflow graph with a single FusedConv2D (with FusedBatchNorm)
// node and fuses into it additional computations (e.g. Relu).
static Graph* FusedConv2DWithBatchNorm(
    int batch, int height, int width, int in_depth, int filter_w, int filter_h,
    int out_depth, const std::vector<string>& fused_ops = {}) {
  Graph* graph = new Graph(OpRegistry::Global());

  Tensor images_t = MakeRandomTensor({batch, height, width, in_depth});
  Tensor filter_t = MakeRandomTensor({filter_w, filter_h, in_depth, out_depth});
  Tensor scale_t = MakeRandomTensor({out_depth});
  Tensor offset_t = MakeRandomTensor({out_depth});
  Tensor mean_t = MakeRandomTensor({out_depth});
  Tensor variance_t = MakeRandomTensor({out_depth});

  Node* images = test::graph::Constant(graph, images_t, "images");
  Node* filter = test::graph::Constant(graph, filter_t, "filter");
  Node* scale = test::graph::Constant(graph, scale_t, "scale");
  Node* offset = test::graph::Constant(graph, offset_t, "offset");
  Node* mean = test::graph::Constant(graph, mean_t, "mean");
  Node* variance = test::graph::Constant(graph, variance_t, "variance");

  std::vector<NodeBuilder::NodeOut> args = {scale, offset, mean, variance};

  Node* conv;
  TF_CHECK_OK(NodeBuilder(graph->NewName("conv"), "_FusedConv2D")
                  .Input(images)
                  .Input(filter)
                  .Attr("num_args", 4)
                  .Input(args)
                  .Attr("T", DT_FLOAT)
                  .Attr("strides", {1, 1, 1, 1})
                  .Attr("padding", "SAME")
                  .Attr("fused_ops", fused_ops)
                  .Finalize(graph, &conv));

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

#define BM_SETUP(N, H, W, C, type, LABEL, NAME)                               \
  testing::ItemsProcessed(static_cast<int64>(iters) * (N) * (H) * (W) * (C)); \
  testing::SetLabel(LABEL);

#define BM_NAME(name, type, N, H, W, C, FW, FH, FC) \
  name##_##type##_##N##_##H##_##W##_##C##_##FW##_##FH##_##FC

#define BM_Conv2D(N, H, W, C, FW, FH, FC, type, LABEL)                       \
  static void BM_NAME(BM_Conv2D, type, N, H, W, C, FW, FH, FC)(int iters) {  \
    BM_SETUP(N, H, W, C, type, LABEL, Conv2D);                               \
    test::Benchmark(#type, Conv2D(N, H, W, C, FW, FH, FC).graph).Run(iters); \
  }                                                                          \
  BENCHMARK(BM_NAME(BM_Conv2D, type, N, H, W, C, FW, FH, FC));

#define BM_Conv2DWithBias(N, H, W, C, FW, FH, FC, type, LABEL)           \
  static void BM_NAME(BM_Conv2DWithBias, type, N, H, W, C, FW, FH,       \
                      FC)(int iters) {                                   \
    BM_SETUP(N, H, W, C, type, LABEL, Conv2D);                           \
    test::Benchmark(#type, Conv2DWithBias(N, H, W, C, FW, FH, FC).graph) \
        .Run(iters);                                                     \
  }                                                                      \
  BENCHMARK(BM_NAME(BM_Conv2DWithBias, type, N, H, W, C, FW, FH, FC));

#define BM_Conv2DWithBiasAndRelu(N, H, W, C, FW, FH, FC, type, LABEL)      \
  static void BM_NAME(BM_Conv2DWithBiasAndRelu, type, N, H, W, C, FW, FH,  \
                      FC)(int iters) {                                     \
    BM_SETUP(N, H, W, C, type, LABEL, Conv2D);                             \
    test::Benchmark(                                                       \
        #type,                                                             \
        Conv2DWithBiasAndActivation(N, H, W, C, FW, FH, FC, "Relu").graph) \
        .Run(iters);                                                       \
  }                                                                        \
  BENCHMARK(BM_NAME(BM_Conv2DWithBiasAndRelu, type, N, H, W, C, FW, FH, FC));

#define BM_FusedConv2DWithBias(N, H, W, C, FW, FH, FC, type, LABEL)           \
  static void BM_NAME(BM_FusedConv2DWithBias, type, N, H, W, C, FW, FH,       \
                      FC)(int iters) {                                        \
    BM_SETUP(N, H, W, C, type, LABEL, Conv2D);                                \
    test::Benchmark(#type,                                                    \
                    FusedConv2DWithBias(N, H, W, C, FW, FH, FC, {"BiasAdd"})) \
        .Run(iters);                                                          \
  }                                                                           \
  BENCHMARK(BM_NAME(BM_FusedConv2DWithBias, type, N, H, W, C, FW, FH, FC));

#define BM_FusedConv2DWithBiasAndRelu(N, H, W, C, FW, FH, FC, type, LABEL)     \
  static void BM_NAME(BM_FusedConv2DWithBiasAndRelu, type, N, H, W, C, FW, FH, \
                      FC)(int iters) {                                         \
    BM_SETUP(N, H, W, C, type, LABEL, Conv2D);                                 \
    test::Benchmark(#type, FusedConv2DWithBias(N, H, W, C, FW, FH, FC,         \
                                               {"BiasAdd", "Relu"}))           \
        .Run(iters);                                                           \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_NAME(BM_FusedConv2DWithBiasAndRelu, type, N, H, W, C, FW, FH, FC));

#define BM_Conv2DWithBatchNorm(N, H, W, C, FW, FH, FC, type, LABEL)           \
  static void BM_NAME(BM_Conv2DWithBatchNorm, type, N, H, W, C, FW, FH,       \
                      FC)(int iters) {                                        \
    BM_SETUP(N, H, W, C, type, LABEL, Conv2D);                                \
    test::Benchmark(#type, Conv2DWithBatchNorm(N, H, W, C, FW, FH, FC).graph) \
        .Run(iters);                                                          \
  }                                                                           \
  BENCHMARK(BM_NAME(BM_Conv2DWithBatchNorm, type, N, H, W, C, FW, FH, FC));

#define BM_Conv2DWithBatchNormAndRelu(N, H, W, C, FW, FH, FC, type, LABEL)     \
  static void BM_NAME(BM_Conv2DWithBatchNormAndRelu, type, N, H, W, C, FW, FH, \
                      FC)(int iters) {                                         \
    BM_SETUP(N, H, W, C, type, LABEL, Conv2D);                                 \
    test::Benchmark(#type, Conv2DWithBatchNormAndActivation(N, H, W, C, FW,    \
                                                            FH, FC, "Relu")    \
                               .graph)                                         \
        .Run(iters);                                                           \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_NAME(BM_Conv2DWithBatchNormAndRelu, type, N, H, W, C, FW, FH, FC));

#define BM_FusedConv2DWithBatchNorm(N, H, W, C, FW, FH, FC, type, LABEL)     \
  static void BM_NAME(BM_FusedConv2DWithBatchNorm, type, N, H, W, C, FW, FH, \
                      FC)(int iters) {                                       \
    BM_SETUP(N, H, W, C, type, LABEL, Conv2D);                               \
    test::Benchmark(#type, FusedConv2DWithBatchNorm(N, H, W, C, FW, FH, FC,  \
                                                    {"FusedBatchNorm"}))     \
        .Run(iters);                                                         \
  }                                                                          \
  BENCHMARK(BM_NAME(BM_FusedConv2DWithBatchNorm, type, N, H, W, C, FW, FH, FC));

#define BM_FusedConv2DWithBatchNormAndRelu(N, H, W, C, FW, FH, FC, type,      \
                                           LABEL)                             \
  static void BM_NAME(BM_FusedConv2DWithBatchNormAndRelu, type, N, H, W, C,   \
                      FW, FH, FC)(int iters) {                                \
    BM_SETUP(N, H, W, C, type, LABEL, Conv2D);                                \
    test::Benchmark(#type,                                                    \
                    FusedConv2DWithBatchNorm(N, H, W, C, FW, FH, FC,          \
                                             {"FusedBatchNorm", "Relu"}))     \
        .Run(iters);                                                          \
  }                                                                           \
  BENCHMARK(BM_NAME(BM_FusedConv2DWithBatchNormAndRelu, type, N, H, W, C, FW, \
                    FH, FC));

// -------------------------------------------------------------------------- //
// Pixel CNN convolutions.
// -------------------------------------------------------------------------- //

// 1x1 Convolution: MatMulFunctor

BM_Conv2D(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_Conv2D(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_Conv2D(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

// 1) BiasAdd {+ Relu}

BM_Conv2DWithBias(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_Conv2DWithBias(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_Conv2DWithBias(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

BM_Conv2DWithBiasAndRelu(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_Conv2DWithBiasAndRelu(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_Conv2DWithBiasAndRelu(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

BM_FusedConv2DWithBias(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_FusedConv2DWithBias(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_FusedConv2DWithBias(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

BM_FusedConv2DWithBiasAndRelu(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_FusedConv2DWithBiasAndRelu(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_FusedConv2DWithBiasAndRelu(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

// 2) FusedBatchNorm {+ Relu}

BM_Conv2DWithBatchNorm(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_Conv2DWithBatchNorm(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_Conv2DWithBatchNorm(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

BM_Conv2DWithBatchNormAndRelu(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_Conv2DWithBatchNormAndRelu(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_Conv2DWithBatchNormAndRelu(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

BM_FusedConv2DWithBatchNorm(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_FusedConv2DWithBatchNorm(16, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 16");
BM_FusedConv2DWithBatchNorm(32, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 32");

BM_FusedConv2DWithBatchNormAndRelu(8, 32, 32, 128, 1, 1, 1024, cpu, "1x1 /b 8");
BM_FusedConv2DWithBatchNormAndRelu(16, 32, 32, 128, 1, 1, 1024, cpu,
                                   "1x1 /b 16");
BM_FusedConv2DWithBatchNormAndRelu(32, 32, 32, 128, 1, 1, 1024, cpu,
                                   "1x1 /b 32");

// -------------------------------------------------------------------------- //
// 3x3 Convolution: SpatialConvolution
// -------------------------------------------------------------------------- //

BM_Conv2D(8, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 8");
BM_Conv2D(16, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 16");
BM_Conv2D(32, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 32");

// 1) BiasAdd {+ Relu}

BM_Conv2DWithBias(8, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 8");
BM_Conv2DWithBias(16, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 16");
BM_Conv2DWithBias(32, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 32");

BM_Conv2DWithBiasAndRelu(8, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 8");
BM_Conv2DWithBiasAndRelu(16, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 16");
BM_Conv2DWithBiasAndRelu(32, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 32");

BM_FusedConv2DWithBias(8, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 8");
BM_FusedConv2DWithBias(16, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 16");
BM_FusedConv2DWithBias(32, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 32");

BM_FusedConv2DWithBiasAndRelu(8, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 8");
BM_FusedConv2DWithBiasAndRelu(16, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 16");
BM_FusedConv2DWithBiasAndRelu(32, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 32");

// 2) FusedBatchNorm {+ Relu}

BM_Conv2DWithBatchNorm(8, 32, 32, 128, 3, 3, 1024, cpu, "1x1 /b 8");
BM_Conv2DWithBatchNorm(16, 32, 32, 128, 3, 3, 1024, cpu, "1x1 /b 16");
BM_Conv2DWithBatchNorm(32, 32, 32, 128, 3, 3, 1024, cpu, "1x1 /b 32");

BM_Conv2DWithBatchNormAndRelu(8, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 8");
BM_Conv2DWithBatchNormAndRelu(16, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 16");
BM_Conv2DWithBatchNormAndRelu(32, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 32");

BM_FusedConv2DWithBatchNorm(8, 32, 32, 128, 3, 3, 1024, cpu, "1x1 /b 8");
BM_FusedConv2DWithBatchNorm(16, 32, 32, 128, 3, 3, 1024, cpu, "1x1 /b 16");
BM_FusedConv2DWithBatchNorm(32, 32, 32, 128, 3, 3, 1024, cpu, "1x1 /b 32");

BM_FusedConv2DWithBatchNormAndRelu(8, 32, 32, 128, 3, 3, 1024, cpu, "3x3 /b 8");
BM_FusedConv2DWithBatchNormAndRelu(16, 32, 32, 128, 3, 3, 1024, cpu,
                                   "3x3 /b 16");
BM_FusedConv2DWithBatchNormAndRelu(32, 32, 32, 128, 3, 3, 1024, cpu,
                                   "3x3 /b 32");

#if GOOGLE_CUDA
// -------------------------------------------------------------------------- //
// 1x1 Convolution
// -------------------------------------------------------------------------- //

BM_Conv2D(8, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 8");
BM_Conv2D(16, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 16");
BM_Conv2D(32, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 32");

BM_Conv2DWithBiasAndRelu(8, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 8");
BM_Conv2DWithBiasAndRelu(16, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 16");
BM_Conv2DWithBiasAndRelu(32, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 32");

BM_FusedConv2DWithBiasAndRelu(8, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 8");
BM_FusedConv2DWithBiasAndRelu(16, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 16");
BM_FusedConv2DWithBiasAndRelu(32, 32, 32, 128, 1, 1, 1024, gpu, "1x1 /b 32");

// -------------------------------------------------------------------------- //
// 3x3 Convolution
// -------------------------------------------------------------------------- //

BM_Conv2D(8, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 8");
BM_Conv2D(16, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 16");
BM_Conv2D(32, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 32");

BM_Conv2DWithBiasAndRelu(8, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 8");
BM_Conv2DWithBiasAndRelu(16, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 16");
BM_Conv2DWithBiasAndRelu(32, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 32");

BM_FusedConv2DWithBiasAndRelu(8, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 8");
BM_FusedConv2DWithBiasAndRelu(16, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 16");
BM_FusedConv2DWithBiasAndRelu(32, 32, 32, 128, 3, 3, 1024, gpu, "3x3 /b 32");
#endif

}  // namespace tensorflow

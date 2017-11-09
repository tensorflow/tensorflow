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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/eigen_thread_pool.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

static void SetConstOp(const string& name, std::initializer_list<int64> dims,
                       DataType data_type, NodeDef* node) {
  Tensor tensor(data_type, TensorShape(dims));
  for (int64 i = 0; i < tensor.NumElements(); ++i) {
    switch (data_type) {
      case DT_FLOAT:
        tensor.flat<float>()(i) = i / 10.0f;
        break;
      case DT_HALF:
        tensor.flat<Eigen::half>()(i) = Eigen::half(i / 10.0f);
        break;
      default:
        LOG(FATAL) << "Unknown data type " << data_type;
    }
  }
  TF_CHECK_OK(NodeDefBuilder(name, "Const")
                  .Attr("dtype", data_type)
                  .Attr("value", tensor)
                  .Finalize(node));
}

static void SetConstSizesOp(const string& name, const std::vector<int32>& sizes,
                            NodeDef* node) {
  TensorShape shape;
  shape.AddDim(sizes.size());
  Tensor tensor(DT_INT32, shape);
  for (int64 i = 0; i < tensor.NumElements(); ++i) {
    tensor.flat<int32>()(i) = sizes[i];
  }
  TF_CHECK_OK(NodeDefBuilder(name, "Const")
                  .Attr("dtype", DT_INT32)
                  .Attr("value", tensor)
                  .Finalize(node));
}

namespace {

enum CONV_OP {
  CONV_OP_FORWARD = 0,
  CONV_OP_BACKPROP_INPUT = 1,
  CONV_OP_BACKPROP_FILTER = 2,
  CONV_OP_FUSED = 3,
};

}  // namespace

static void BM_ConvFloat(int iters, int batch, int rows, int cols, int in_depth,
                         int out_depth, int filter_rows, int filter_cols,
                         CONV_OP op, int num_threads, int stride,
                         Padding padding, bool use_gpu, DataType data_type,
                         const string& label) {
  if (!IsGoogleCudaEnabled() && use_gpu) {
    testing::SetLabel(
        strings::StrCat("Skipping GPU test (no --config=cuda): ", label));
    return;
  }
  testing::SetLabel(label);

  // Set the number of threads
  SessionOptions options;
  options.config.set_intra_op_parallelism_threads(num_threads);

  // We set up a graph for computing convolution.
  GraphDef graph;

  // For this, we need an input tensor and a filter tensor.
  // Compute the output size.
  int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
  TF_CHECK_OK(GetWindowedOutputSize(rows, filter_rows, stride, padding,
                                    &out_rows, &pad_rows));
  TF_CHECK_OK(GetWindowedOutputSize(cols, filter_cols, stride, padding,
                                    &out_cols, &pad_cols));
  // Counting the number of floating point operations (both MUL and ADD)
  int64 num_ops = 0;
  if (op == CONV_OP_FORWARD) {
    // Forward computation:
    // BATCH x OUT_ROW X OUT_COL X IN_DEPTH X PATCH_ROW X PATH_COL X OUT_DEPTH
    // We multiply by two since there are multiplications and additions.
    num_ops = static_cast<int64>(batch * in_depth * out_depth) *
              static_cast<int64>(filter_rows * filter_cols) *
              static_cast<int64>(out_rows * out_cols) * 2;
  } else {
    // Backward computation:
    // BATCH x IN_ROW X IN_COL X IN_DEPTH X PATCH_ROW X PATCH_COL X OUT_DEPTH
    // We multiply by two since there are multiplications and additions.
    num_ops = static_cast<int64>(batch * in_depth * out_depth) *
              static_cast<int64>(filter_rows * filter_cols) *
              static_cast<int64>(rows * cols) * 2;
  }

  SetConstOp("input", {batch, rows, cols, in_depth}, data_type,
             graph.add_node());
  SetConstOp("filter", {filter_rows, filter_cols, in_depth, out_depth},
             data_type, graph.add_node());
  SetConstOp("output_backprop", {batch, out_rows, out_cols, out_depth},
             data_type, graph.add_node());
  SetConstSizesOp("input_sizes",
                  std::vector<int32>({batch, rows, cols, in_depth}),
                  graph.add_node());
  SetConstSizesOp(
      "filter_sizes",
      std::vector<int32>({filter_rows, filter_cols, in_depth, out_depth}),
      graph.add_node());
  SetConstSizesOp("resize_size", std::vector<int32>({rows, cols}),
                  graph.add_node());

  TensorShape paddings_shape({4, 2});
  Tensor paddings_tensor(DT_INT32, paddings_shape);
  for (int64 i = 0; i < paddings_tensor.NumElements(); ++i) {
    paddings_tensor.flat<int32>()(i) = 0;
  }
  TF_CHECK_OK(NodeDefBuilder("paddings", "Const")
                  .Attr("dtype", DT_INT32)
                  .Attr("value", paddings_tensor)
                  .Finalize(graph.add_node()));

  // Now add the convolution op
  NodeDef* conv = graph.add_node();
  switch (op) {
    case CONV_OP_FORWARD:
      TF_CHECK_OK(NodeDefBuilder("conv2d", "Conv2D")
                      .Input("input", 0, data_type)
                      .Input("filter", 0, data_type)
                      .Attr("strides", {1, stride, stride, 1})
                      .Attr("padding", padding == VALID ? "VALID" : "SAME")
                      .Finalize(conv));
      break;
    case CONV_OP_BACKPROP_INPUT:
      TF_CHECK_OK(NodeDefBuilder("conv2d", "Conv2DBackpropInput")
                      .Input("input_sizes", 0, DT_INT32)
                      .Input("filter", 0, data_type)
                      .Input("output_backprop", 0, data_type)
                      .Attr("strides", {1, stride, stride, 1})
                      .Attr("padding", padding == VALID ? "VALID" : "SAME")
                      .Finalize(conv));
      break;
    case CONV_OP_BACKPROP_FILTER:
      TF_CHECK_OK(NodeDefBuilder("conv2d", "Conv2DBackpropFilter")
                      .Input("input", 0, data_type)
                      .Input("filter_sizes", 0, DT_INT32)
                      .Input("output_backprop", 0, data_type)
                      .Attr("strides", {1, stride, stride, 1})
                      .Attr("padding", padding == VALID ? "VALID" : "SAME")
                      .Finalize(conv));
      break;
    case CONV_OP_FUSED:
      TF_CHECK_OK(NodeDefBuilder("conv2d", "FusedResizeAndPadConv2D")
                      .Input("input", 0, data_type)
                      .Input("resize_size", 0, DT_INT32)
                      .Input("paddings", 0, DT_INT32)
                      .Input("filter", 0, data_type)
                      .Attr("mode", "REFLECT")
                      .Attr("strides", {1, stride, stride, 1})
                      .Attr("padding", padding == VALID ? "VALID" : "SAME")
                      .Attr("resize_align_corners", false)
                      .Finalize(conv));
      break;
  }
  Graph* g = new Graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph, g));

  string device = use_gpu ? "gpu" : "cpu";
  testing::UseRealTime();
  test::Benchmark(device, g, &options).Run(iters);
  testing::ItemsProcessed(num_ops * iters);
}

// BS: batch_size
// R: tensor_in_rows
// C: tensor_in_cols
// ID: input_depth
// OD: output_depth
// KR: kernel_rows
// KC: kernel_cols
#define BM_ConvFloatFwd(BS, R, C, ID, OD, KR, KC, STR, PAD, LABEL)             \
  static void BM_ConvFloatFwdCPU1_##LABEL(int iters) {                         \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_FORWARD, 1, STR,     \
                 PAD, false, DT_FLOAT,                                         \
                 strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", OD, "_",    \
                                 KR, "_", KC, "_", STR, "_", PAD, "_f_cpu1")); \
  }                                                                            \
  static void BM_ConvFloatFwdCPU4_##LABEL(int iters) {                         \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_FORWARD, 4, STR,     \
                 PAD, false, DT_FLOAT,                                         \
                 strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", OD, "_",    \
                                 KR, "_", KC, "_", STR, "_", PAD, "_f_cpu4")); \
  }                                                                            \
  static void BM_ConvFloatFusedCPU1_##LABEL(int iters) {                       \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_FUSED, 1, STR, PAD,  \
                 false, DT_FLOAT,                                              \
                 strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", OD, "_",    \
                                 KR, "_", KC, "_", STR, "_", PAD, "_f_cpu1")); \
  }                                                                            \
  static void BM_ConvFloatFusedCPU4_##LABEL(int iters) {                       \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_FUSED, 4, STR, PAD,  \
                 false, DT_FLOAT,                                              \
                 strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", OD, "_",    \
                                 KR, "_", KC, "_", STR, "_", PAD, "_f_cpu4")); \
  }                                                                            \
  static void BM_ConvFloatFwdGPU_##LABEL(int iters) {                          \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_FORWARD, 1, STR,     \
                 PAD, true, DT_FLOAT,                                          \
                 strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", OD, "_",    \
                                 KR, "_", KC, "_", STR, "_", PAD, "_f_gpu"));  \
  }                                                                            \
  static void BM_ConvHalfFwdGPU_##LABEL(int iters) {                           \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_FORWARD, 1, STR,     \
                 PAD, true, DT_HALF,                                           \
                 strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", OD, "_",    \
                                 KR, "_", KC, "_", STR, "_", PAD, "_h_gpu"));  \
  }                                                                            \
  BENCHMARK(BM_ConvFloatFwdCPU1_##LABEL);                                      \
  BENCHMARK(BM_ConvFloatFwdCPU4_##LABEL);                                      \
  BENCHMARK(BM_ConvFloatFusedCPU1_##LABEL);                                    \
  BENCHMARK(BM_ConvFloatFusedCPU4_##LABEL);                                    \
  BENCHMARK(BM_ConvFloatFwdGPU_##LABEL);                                       \
  BENCHMARK(BM_ConvHalfFwdGPU_##LABEL)

BM_ConvFloatFwd(32, 5, 5, 1248, 128, 1, 1, 1, SAME, conv0);
BM_ConvFloatFwd(32, 8, 8, 384, 384, 1, 3, 1, SAME, conv1);
BM_ConvFloatFwd(32, 8, 8, 384, 384, 3, 1, 1, SAME, conv2);
BM_ConvFloatFwd(32, 8, 8, 2048, 192, 1, 1, 1, SAME, conv3);
BM_ConvFloatFwd(32, 8, 8, 448, 384, 3, 3, 1, SAME, conv4);
BM_ConvFloatFwd(32, 8, 8, 2048, 320, 1, 1, 1, SAME, conv5);
BM_ConvFloatFwd(32, 8, 8, 2048, 448, 1, 1, 1, SAME, conv6);
BM_ConvFloatFwd(32, 8, 8, 2048, 384, 1, 1, 1, SAME, conv7);
BM_ConvFloatFwd(32, 8, 8, 1760, 384, 1, 1, 1, SAME, conv8);
BM_ConvFloatFwd(32, 8, 8, 1760, 192, 1, 1, 1, SAME, conv9);
BM_ConvFloatFwd(32, 8, 8, 1760, 448, 1, 1, 1, SAME, conv10);
BM_ConvFloatFwd(32, 8, 8, 1760, 320, 1, 1, 1, SAME, conv11);
BM_ConvFloatFwd(32, 17, 17, 192, 192, 3, 3, 2, VALID, conv12);
BM_ConvFloatFwd(32, 17, 17, 192, 192, 3, 3, 1, SAME, conv13);
BM_ConvFloatFwd(32, 17, 17, 1248, 192, 1, 1, 1, SAME, conv14);
BM_ConvFloatFwd(32, 17, 17, 128, 320, 3, 3, 2, VALID, conv15);
BM_ConvFloatFwd(32, 17, 17, 1248, 128, 1, 1, 1, SAME, conv16);
BM_ConvFloatFwd(32, 17, 17, 224, 224, 1, 3, 1, SAME, conv17);
BM_ConvFloatFwd(32, 17, 17, 192, 256, 3, 1, 1, SAME, conv18);
BM_ConvFloatFwd(32, 17, 17, 192, 256, 1, 3, 1, SAME, conv19);
BM_ConvFloatFwd(32, 17, 17, 1216, 192, 1, 1, 1, SAME, conv20);
BM_ConvFloatFwd(32, 17, 17, 1216, 96, 1, 1, 1, SAME, conv21);
BM_ConvFloatFwd(32, 17, 17, 224, 224, 3, 1, 1, SAME, conv22);
BM_ConvFloatFwd(32, 17, 17, 192, 224, 3, 3, 1, SAME, conv23);
BM_ConvFloatFwd(32, 17, 17, 192, 192, 1, 3, 1, SAME, conv24);
BM_ConvFloatFwd(32, 17, 17, 1152, 192, 1, 1, 1, SAME, conv25);
BM_ConvFloatFwd(32, 17, 17, 1152, 128, 1, 1, 1, SAME, conv26);
BM_ConvFloatFwd(32, 17, 17, 192, 192, 3, 1, 1, SAME, conv27);
BM_ConvFloatFwd(32, 17, 17, 160, 192, 3, 3, 1, SAME, conv28);
BM_ConvFloatFwd(32, 17, 17, 1152, 160, 1, 1, 1, SAME, conv29);
BM_ConvFloatFwd(32, 17, 17, 1024, 128, 1, 1, 1, SAME, conv30);
BM_ConvFloatFwd(32, 17, 17, 128, 192, 1, 3, 1, SAME, conv31);
BM_ConvFloatFwd(32, 17, 17, 1024, 160, 1, 1, 1, SAME, conv32);
BM_ConvFloatFwd(32, 17, 17, 128, 192, 3, 1, 1, SAME, conv33);
BM_ConvFloatFwd(32, 17, 17, 1024, 256, 1, 1, 1, SAME, conv34);
BM_ConvFloatFwd(32, 17, 17, 128, 128, 3, 1, 1, SAME, conv35);
BM_ConvFloatFwd(32, 17, 17, 768, 192, 1, 1, 1, SAME, conv36);
BM_ConvFloatFwd(32, 17, 17, 128, 128, 1, 3, 1, SAME, conv37);
BM_ConvFloatFwd(32, 17, 17, 128, 128, 3, 3, 1, SAME, conv38);
BM_ConvFloatFwd(32, 17, 17, 768, 128, 1, 1, 1, SAME, conv39);
BM_ConvFloatFwd(32, 17, 17, 768, 320, 1, 1, 1, SAME, conv40);
BM_ConvFloatFwd(32, 35, 35, 96, 96, 3, 3, 2, VALID, conv41);
BM_ConvFloatFwd(32, 35, 35, 288, 384, 3, 3, 2, VALID, conv42);
BM_ConvFloatFwd(32, 35, 35, 64, 96, 3, 3, 1, SAME, conv43);
BM_ConvFloatFwd(32, 35, 35, 288, 64, 1, 1, 1, SAME, conv44);
BM_ConvFloatFwd(32, 35, 35, 256, 64, 1, 1, 1, SAME, conv45);
BM_ConvFloatFwd(32, 35, 35, 48, 64, 5, 5, 1, SAME, conv46);
BM_ConvFloatFwd(32, 35, 35, 256, 48, 1, 1, 1, SAME, conv47);
BM_ConvFloatFwd(32, 35, 35, 96, 96, 3, 3, 1, SAME, conv48);
BM_ConvFloatFwd(32, 35, 35, 192, 32, 1, 1, 1, SAME, conv49);
BM_ConvFloatFwd(32, 35, 35, 192, 64, 1, 1, 1, SAME, conv50);
BM_ConvFloatFwd(32, 35, 35, 192, 48, 1, 1, 1, SAME, conv51);
BM_ConvFloatFwd(32, 73, 73, 64, 192, 3, 3, 1, VALID, conv52);
BM_ConvFloatFwd(32, 73, 73, 64, 64, 1, 1, 1, VALID, conv53);
BM_ConvFloatFwd(32, 147, 147, 24, 64, 1, 1, 1, VALID, conv54);

#define BM_ConvFloatBkInAndFilter(BS, R, C, ID, OD, KR, KC, STR, PAD, LABEL)  \
  static void BM_ConvFloatBkInCPU1_##LABEL(int iters) {                       \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_BACKPROP_INPUT, 1,  \
                 STR, PAD, false, DT_FLOAT,                                   \
                 strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", OD, "_",   \
                                 KR, "_", KC, "_", STR, "_", PAD, "_cpu1"));  \
  }                                                                           \
  static void BM_ConvFloatBkInCPU4_##LABEL(int iters) {                       \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_BACKPROP_INPUT, 4,  \
                 STR, PAD, false, DT_FLOAT,                                   \
                 strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", OD, "_",   \
                                 KR, "_", KC, "_", STR, "_", PAD, "_cpu4"));  \
  }                                                                           \
  static void BM_ConvFloatBkInGPU_##LABEL(int iters) {                        \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_BACKPROP_INPUT, 1,  \
                 STR, PAD, true, DT_FLOAT,                                    \
                 strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", OD, "_",   \
                                 KR, "_", KC, "_", STR, "_", PAD, "_gpu"));   \
  }                                                                           \
  static void BM_ConvFloatBkFilterCPU1_##LABEL(int iters) {                   \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_BACKPROP_FILTER, 1, \
                 STR, PAD, false, DT_FLOAT,                                   \
                 strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", OD, "_",   \
                                 KR, "_", KC, "_", STR, "_", PAD, "_cpu1"));  \
  }                                                                           \
  static void BM_ConvFloatBkFilterCPU4_##LABEL(int iters) {                   \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_BACKPROP_FILTER, 4, \
                 STR, PAD, false, DT_FLOAT,                                   \
                 strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", OD, "_",   \
                                 KR, "_", KC, "_", STR, "_", PAD, "_cpu4"));  \
  }                                                                           \
  static void BM_ConvFloatBkFilterGPU_##LABEL(int iters) {                    \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_BACKPROP_FILTER, 1, \
                 STR, PAD, true, DT_FLOAT,                                    \
                 strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", OD, "_",   \
                                 KR, "_", KC, "_", STR, "_", PAD, "_gpu"));   \
  }                                                                           \
  static void BM_ConvHalfBkInGPU_##LABEL(int iters) {                         \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_BACKPROP_INPUT, 1,  \
                 STR, PAD, true, DT_HALF,                                     \
                 strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", OD, "_",   \
                                 KR, "_", KC, "_", STR, "_", PAD, "_gpu"));   \
  }                                                                           \
  static void BM_ConvHalfBkFilterGPU_##LABEL(int iters) {                     \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_BACKPROP_FILTER, 1, \
                 STR, PAD, true, DT_HALF,                                     \
                 strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", OD, "_",   \
                                 KR, "_", KC, "_", STR, "_", PAD, "_gpu"));   \
  }                                                                           \
  BENCHMARK(BM_ConvFloatBkInCPU1_##LABEL);                                    \
  BENCHMARK(BM_ConvFloatBkInCPU4_##LABEL);                                    \
  BENCHMARK(BM_ConvFloatBkInGPU_##LABEL);                                     \
  BENCHMARK(BM_ConvFloatBkFilterCPU1_##LABEL);                                \
  BENCHMARK(BM_ConvFloatBkFilterCPU4_##LABEL);                                \
  BENCHMARK(BM_ConvFloatBkFilterGPU_##LABEL);                                 \
  BENCHMARK(BM_ConvHalfBkInGPU_##LABEL);                                      \
  BENCHMARK(BM_ConvHalfBkFilterGPU_##LABEL)

// Benchmarks from the inception model

BM_ConvFloatBkInAndFilter(32, 5, 5, 1248, 128, 1, 1, 1, SAME, conv0);
BM_ConvFloatBkInAndFilter(32, 8, 8, 384, 384, 1, 3, 1, SAME, conv1);
BM_ConvFloatBkInAndFilter(32, 8, 8, 384, 384, 3, 1, 1, SAME, conv2);
BM_ConvFloatBkInAndFilter(32, 8, 8, 2048, 192, 1, 1, 1, SAME, conv3);
BM_ConvFloatBkInAndFilter(32, 8, 8, 448, 384, 3, 3, 1, SAME, conv4);
BM_ConvFloatBkInAndFilter(32, 8, 8, 2048, 320, 1, 1, 1, SAME, conv5);
BM_ConvFloatBkInAndFilter(32, 8, 8, 2048, 448, 1, 1, 1, SAME, conv6);
BM_ConvFloatBkInAndFilter(32, 8, 8, 2048, 384, 1, 1, 1, SAME, conv7);
BM_ConvFloatBkInAndFilter(32, 8, 8, 1760, 384, 1, 1, 1, SAME, conv8);
BM_ConvFloatBkInAndFilter(32, 8, 8, 1760, 192, 1, 1, 1, SAME, conv9);
BM_ConvFloatBkInAndFilter(32, 8, 8, 1760, 448, 1, 1, 1, SAME, conv10);
BM_ConvFloatBkInAndFilter(32, 8, 8, 1760, 320, 1, 1, 1, SAME, conv11);
BM_ConvFloatBkInAndFilter(32, 17, 17, 192, 192, 3, 3, 2, VALID, conv12);
BM_ConvFloatBkInAndFilter(32, 17, 17, 192, 192, 3, 3, 1, SAME, conv13);
BM_ConvFloatBkInAndFilter(32, 17, 17, 1248, 192, 1, 1, 1, SAME, conv14);
BM_ConvFloatBkInAndFilter(32, 17, 17, 128, 320, 3, 3, 2, VALID, conv15);
BM_ConvFloatBkInAndFilter(32, 17, 17, 1248, 128, 1, 1, 1, SAME, conv16);
BM_ConvFloatBkInAndFilter(32, 17, 17, 224, 224, 1, 3, 1, SAME, conv17);
BM_ConvFloatBkInAndFilter(32, 17, 17, 192, 256, 3, 1, 1, SAME, conv18);
BM_ConvFloatBkInAndFilter(32, 17, 17, 192, 256, 1, 3, 1, SAME, conv19);
BM_ConvFloatBkInAndFilter(32, 17, 17, 1216, 192, 1, 1, 1, SAME, conv20);
BM_ConvFloatBkInAndFilter(32, 17, 17, 1216, 96, 1, 1, 1, SAME, conv21);
BM_ConvFloatBkInAndFilter(32, 17, 17, 224, 224, 3, 1, 1, SAME, conv22);
BM_ConvFloatBkInAndFilter(32, 17, 17, 192, 224, 3, 3, 1, SAME, conv23);
BM_ConvFloatBkInAndFilter(32, 17, 17, 192, 192, 1, 3, 1, SAME, conv24);
BM_ConvFloatBkInAndFilter(32, 17, 17, 1152, 192, 1, 1, 1, SAME, conv25);
BM_ConvFloatBkInAndFilter(32, 17, 17, 1152, 128, 1, 1, 1, SAME, conv26);
BM_ConvFloatBkInAndFilter(32, 17, 17, 192, 192, 3, 1, 1, SAME, conv27);
BM_ConvFloatBkInAndFilter(32, 17, 17, 160, 192, 3, 3, 1, SAME, conv28);
BM_ConvFloatBkInAndFilter(32, 17, 17, 1152, 160, 1, 1, 1, SAME, conv29);
BM_ConvFloatBkInAndFilter(32, 17, 17, 1024, 128, 1, 1, 1, SAME, conv30);
BM_ConvFloatBkInAndFilter(32, 17, 17, 128, 192, 1, 3, 1, SAME, conv31);
BM_ConvFloatBkInAndFilter(32, 17, 17, 1024, 160, 1, 1, 1, SAME, conv32);
BM_ConvFloatBkInAndFilter(32, 17, 17, 128, 192, 3, 1, 1, SAME, conv33);
BM_ConvFloatBkInAndFilter(32, 17, 17, 1024, 256, 1, 1, 1, SAME, conv34);
BM_ConvFloatBkInAndFilter(32, 17, 17, 128, 128, 3, 1, 1, SAME, conv35);
BM_ConvFloatBkInAndFilter(32, 17, 17, 768, 192, 1, 1, 1, SAME, conv36);
BM_ConvFloatBkInAndFilter(32, 17, 17, 128, 128, 1, 3, 1, SAME, conv37);
BM_ConvFloatBkInAndFilter(32, 17, 17, 128, 128, 3, 3, 1, SAME, conv38);
BM_ConvFloatBkInAndFilter(32, 17, 17, 768, 128, 1, 1, 1, SAME, conv39);
BM_ConvFloatBkInAndFilter(32, 17, 17, 768, 320, 1, 1, 1, SAME, conv40);
BM_ConvFloatBkInAndFilter(32, 35, 35, 96, 96, 3, 3, 2, VALID, conv41);
BM_ConvFloatBkInAndFilter(32, 35, 35, 288, 384, 3, 3, 2, VALID, conv42);
BM_ConvFloatBkInAndFilter(32, 35, 35, 64, 96, 3, 3, 1, SAME, conv43);
BM_ConvFloatBkInAndFilter(32, 35, 35, 288, 64, 1, 1, 1, SAME, conv44);
BM_ConvFloatBkInAndFilter(32, 35, 35, 256, 64, 1, 1, 1, SAME, conv45);
BM_ConvFloatBkInAndFilter(32, 35, 35, 48, 64, 5, 5, 1, SAME, conv46);
BM_ConvFloatBkInAndFilter(32, 35, 35, 256, 48, 1, 1, 1, SAME, conv47);
BM_ConvFloatBkInAndFilter(32, 35, 35, 96, 96, 3, 3, 1, SAME, conv48);
BM_ConvFloatBkInAndFilter(32, 35, 35, 192, 32, 1, 1, 1, SAME, conv49);
BM_ConvFloatBkInAndFilter(32, 35, 35, 192, 64, 1, 1, 1, SAME, conv50);
BM_ConvFloatBkInAndFilter(32, 35, 35, 192, 48, 1, 1, 1, SAME, conv51);
BM_ConvFloatBkInAndFilter(32, 73, 73, 64, 192, 3, 3, 1, VALID, conv52);
BM_ConvFloatBkInAndFilter(32, 73, 73, 64, 64, 1, 1, 1, VALID, conv53);
BM_ConvFloatBkInAndFilter(32, 147, 147, 24, 64, 1, 1, 1, VALID, conv54);

#define BM_ConvFloatBkFCPU(BS, R, C, ID, OD, KR, KC, TH, LABEL)                \
  static void                                                                  \
      BM_ConvFloatBkFCPU_##BS##_##R##_##C##_##ID##_##OD##_##KR##_##KC##_##TH(  \
          int iters) {                                                         \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_BACKPROP_FILTER, TH, \
                 1, VALID, false, DT_FLOAT, LABEL);                            \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_ConvFloatBkFCPU_##BS##_##R##_##C##_##ID##_##OD##_##KR##_##KC##_##TH);

// Benchmarks from https://github.com/soumith/convnet-benchmarks
BM_ConvFloatBkFCPU(128, 128, 128, 3, 96, 11, 11, 4, "convnet-layer1");
BM_ConvFloatBkFCPU(128, 64, 64, 64, 128, 9, 9, 4, "convnet-layer2");
BM_ConvFloatBkFCPU(128, 32, 32, 128, 128, 9, 9, 4, "convnet-layer3");
BM_ConvFloatBkFCPU(128, 16, 16, 128, 128, 7, 7, 4, "convnet-layer4");
BM_ConvFloatBkFCPU(128, 13, 13, 384, 384, 3, 3, 4, "convnet-layer5");

#define BM_ConvFloatBkFGPU(BS, R, C, ID, OD, KR, KC, LABEL)                    \
  static void BM_ConvFloatBkFGPU_##BS##_##R##_##C##_##ID##_##OD##_##KR##_##KC( \
      int iters) {                                                             \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_BACKPROP_FILTER, 1,  \
                 1, VALID, true, DT_FLOAT, LABEL);                             \
  }                                                                            \
  static void BM_ConvHalfBkFGPU_##BS##_##R##_##C##_##ID##_##OD##_##KR##_##KC(  \
      int iters) {                                                             \
    BM_ConvFloat(iters, BS, R, C, ID, OD, KR, KC, CONV_OP_BACKPROP_FILTER, 1,  \
                 1, VALID, true, DT_HALF, LABEL);                              \
  }                                                                            \
  BENCHMARK(BM_ConvFloatBkFGPU_##BS##_##R##_##C##_##ID##_##OD##_##KR##_##KC);  \
  BENCHMARK(BM_ConvHalfBkFGPU_##BS##_##R##_##C##_##ID##_##OD##_##KR##_##KC)

// Benchmarks from https://github.com/soumith/convnet-benchmarks
BM_ConvFloatBkFGPU(128, 128, 128, 3, 96, 11, 11, "convnet-layer1");
BM_ConvFloatBkFGPU(128, 64, 64, 64, 128, 9, 9, "convnet-layer2");
BM_ConvFloatBkFGPU(128, 32, 32, 128, 128, 9, 9, "convnet-layer3");
BM_ConvFloatBkFGPU(128, 16, 16, 128, 128, 7, 7, "convnet-layer4");
BM_ConvFloatBkFGPU(128, 13, 13, 384, 384, 3, 3, "convnet-layer5");

namespace {

enum DEPTHWISE_CONV_OP {
  DEPTHWISE_CONV_OP_FWD = 0,
  DEPTHWISE_CONV_OP_BACKPROP_INPUT = 1,
  DEPTHWISE_CONV_OP_BACKPROP_FILTER = 2
};

}  // namespace

static void BM_ConvFloatDepthwise(int iters, int batch, int rows, int cols,
                                  int in_depth, int depth_multiplier,
                                  int out_depth, int filter_rows,
                                  int filter_cols, DEPTHWISE_CONV_OP op,
                                  int num_threads, int stride, Padding padding,
                                  bool use_gpu, const string& label) {
  if (!IsGoogleCudaEnabled() && use_gpu) {
    testing::SetLabel(
        strings::StrCat("Skipping GPU test (no --config=cuda): ", label));
    return;
  }
  testing::SetLabel(label);

  // Set the number of threads
  SessionOptions options;
  options.config.set_intra_op_parallelism_threads(num_threads);

  // We set up a graph for computing convolution.
  GraphDef graph;

  // For this, we need an input tensor and a filter tensor.
  // Compute the output size.
  int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
  TF_CHECK_OK(GetWindowedOutputSize(rows, filter_rows, stride, padding,
                                    &out_rows, &pad_rows));
  TF_CHECK_OK(GetWindowedOutputSize(cols, filter_cols, stride, padding,
                                    &out_cols, &pad_cols));

  int64 num_ops = 0;
  if (op == DEPTHWISE_CONV_OP_FWD) {
    // Counting the number of floating point operations (both MUL and ADD)
    // Forward computation:
    // BATCH x OUT_ROW X OUT_COL X FLTR_ROW X FLTR_COL X DEPTH_MULT X IN_DEPTH
    // We multiply by two since there are multiplications and additions.
    num_ops = static_cast<int64>(batch * out_rows * out_cols) *
              static_cast<int64>(filter_rows * filter_cols) *
              static_cast<int64>(in_depth * depth_multiplier) * 2;
  } else {
    // Backward computation: both input and filter backprop take the same
    // amount of computation:
    // BATCH x IN_ROW X IN_COL X FLTR_ROW X FLTR_COL X DEPTH_MULT X IN_DEPTH
    // We multiply by two since there are multiplications and additions.
    // We divide by stride squared to approximate the affect of decreasing
    // number of bprop output points per bprop input point with increasing
    // stride.
    num_ops = (static_cast<int64>(batch * rows * cols) *
               static_cast<int64>(filter_rows * filter_cols) *
               static_cast<int64>(in_depth * depth_multiplier) * 2) /
              (stride * stride);
  }

  // FIXME
  SetConstOp("input", {batch, rows, cols, in_depth}, DT_FLOAT,
             graph.add_node());
  SetConstOp("depthwise_filter",
             {filter_rows, filter_cols, in_depth, depth_multiplier}, DT_FLOAT,
             graph.add_node());
  SetConstOp("output_backprop", {batch, out_rows, out_cols, out_depth},
             DT_FLOAT, graph.add_node());
  SetConstSizesOp("input_sizes",
                  std::vector<int32>({batch, rows, cols, in_depth}),
                  graph.add_node());
  SetConstSizesOp("filter_sizes",
                  std::vector<int32>(
                      {filter_rows, filter_cols, in_depth, depth_multiplier}),
                  graph.add_node());

  // Now add the convolution op
  NodeDef* conv = graph.add_node();
  switch (op) {
    case DEPTHWISE_CONV_OP_FWD:
      TF_CHECK_OK(NodeDefBuilder("depthwise_conv2d", "DepthwiseConv2dNative")
                      .Input("input", 0, DT_FLOAT)
                      .Input("depthwise_filter", 0, DT_FLOAT)
                      .Attr("strides", {1, stride, stride, 1})
                      .Attr("padding", padding == VALID ? "VALID" : "SAME")
                      .Finalize(conv));
      break;
    case DEPTHWISE_CONV_OP_BACKPROP_INPUT:
      TF_CHECK_OK(NodeDefBuilder("depthwise_conv2d_backprop_input",
                                 "DepthwiseConv2dNativeBackpropInput")
                      .Input("input_sizes", 0, DT_INT32)
                      .Input("depthwise_filter", 0, DT_FLOAT)
                      .Input("output_backprop", 0, DT_FLOAT)
                      .Attr("strides", {1, stride, stride, 1})
                      .Attr("padding", padding == VALID ? "VALID" : "SAME")
                      .Finalize(conv));
      break;
    case DEPTHWISE_CONV_OP_BACKPROP_FILTER:
      TF_CHECK_OK(NodeDefBuilder("depthwise_conv2d_backprop_filter",
                                 "DepthwiseConv2dNativeBackpropFilter")
                      .Input("input", 0, DT_FLOAT)
                      .Input("filter_sizes", 0, DT_INT32)
                      .Input("output_backprop", 0, DT_FLOAT)
                      .Attr("strides", {1, stride, stride, 1})
                      .Attr("padding", padding == VALID ? "VALID" : "SAME")
                      .Finalize(conv));
      break;
  }
  Graph* g = new Graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph, g));

  string device = use_gpu ? "gpu" : "cpu";
  testing::UseRealTime();
  test::Benchmark(device, g, &options).Run(iters);
  testing::ItemsProcessed(num_ops * iters);
}

// BS: batch_size
// R: tensor_in_rows
// C: tensor_in_cols
// ID: input_depth
// DM: depth_multiplier
// OD: output_depth
// KR: kernel_rows
// KC: kernel_cols
// STR: stride
// PAD: padding

#define BM_ConvFloatDepthwiseFwd(BS, R, C, ID, DM, OD, KR, KC, STR, PAD,    \
                                 LABEL)                                     \
  static void BM_ConvFloatDepthwiseFwdCPU1_##LABEL(int iters) {             \
    BM_ConvFloatDepthwise(                                                  \
        iters, BS, R, C, ID, DM, OD, KR, KC, DEPTHWISE_CONV_OP_FWD, 1, STR, \
        PAD, false,                                                         \
        strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", DM, "_", OD, "_", \
                        KR, "_", KC, "_", STR, "_", PAD, "_cpu1"));         \
  }                                                                         \
  static void BM_ConvFloatDepthwiseFwdCPU4_##LABEL(int iters) {             \
    BM_ConvFloatDepthwise(                                                  \
        iters, BS, R, C, ID, DM, OD, KR, KC, DEPTHWISE_CONV_OP_FWD, 4, STR, \
        PAD, false,                                                         \
        strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", DM, "_", OD, "_", \
                        KR, "_", KC, "_", STR, "_", PAD, "_cpu4"));         \
  }                                                                         \
  static void BM_ConvFloatDepthwiseFwdGPU_##LABEL(int iters) {              \
    BM_ConvFloatDepthwise(                                                  \
        iters, BS, R, C, ID, DM, OD, KR, KC, DEPTHWISE_CONV_OP_FWD, 1, STR, \
        PAD, true,                                                          \
        strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", DM, "_", OD, "_", \
                        KR, "_", KC, "_", STR, "_", PAD, "_gpu"));          \
  }                                                                         \
  BENCHMARK(BM_ConvFloatDepthwiseFwdCPU1_##LABEL);                          \
  BENCHMARK(BM_ConvFloatDepthwiseFwdCPU4_##LABEL);                          \
  BENCHMARK(BM_ConvFloatDepthwiseFwdGPU_##LABEL);

// The configurations below are mostly from mobilenet models.
BM_ConvFloatDepthwiseFwd(32, 112, 112, 3, 8, 24, 3, 3, 1, SAME, conv0);
BM_ConvFloatDepthwiseFwd(32, 112, 112, 64, 1, 64, 3, 3, 1, SAME, conv1);
BM_ConvFloatDepthwiseFwd(32, 56, 56, 128, 1, 128, 3, 3, 1, SAME, conv2);
BM_ConvFloatDepthwiseFwd(32, 56, 56, 128, 1, 128, 3, 3, 2, SAME, conv3);
BM_ConvFloatDepthwiseFwd(32, 28, 28, 128, 1, 128, 3, 3, 1, SAME, conv4);
BM_ConvFloatDepthwiseFwd(32, 14, 14, 512, 1, 512, 3, 3, 1, SAME, conv5);
BM_ConvFloatDepthwiseFwd(32, 7, 7, 1024, 1, 1024, 3, 3, 1, SAME, conv6);
// Benchmarks with different stride and padding options.
BM_ConvFloatDepthwiseFwd(32, 112, 112, 3, 8, 24, 3, 3, 2, SAME, conv7);
BM_ConvFloatDepthwiseFwd(32, 112, 112, 3, 8, 24, 3, 3, 2, VALID, conv8);

#define BM_ConvFloatDepthwiseBk(BS, R, C, ID, DM, OD, KR, KC, STR, PAD, LABEL) \
  static void BM_ConvFloatDepthwiseBkInCPU1_##LABEL(int iters) {               \
    BM_ConvFloatDepthwise(                                                     \
        iters, BS, R, C, ID, DM, OD, KR, KC, DEPTHWISE_CONV_OP_BACKPROP_INPUT, \
        1, STR, PAD, false,                                                    \
        strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", DM, "_", OD, "_",    \
                        KR, "_", KC, "_", STR, "_", PAD, "_cpu1"));            \
  }                                                                            \
  static void BM_ConvFloatDepthwiseBkInCPU4_##LABEL(int iters) {               \
    BM_ConvFloatDepthwise(                                                     \
        iters, BS, R, C, ID, DM, OD, KR, KC, DEPTHWISE_CONV_OP_BACKPROP_INPUT, \
        4, STR, PAD, false,                                                    \
        strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", DM, "_", OD, "_",    \
                        KR, "_", KC, "_", STR, "_", PAD, "_cpu4"));            \
  }                                                                            \
  static void BM_ConvFloatDepthwiseBkInGPU_##LABEL(int iters) {                \
    BM_ConvFloatDepthwise(                                                     \
        iters, BS, R, C, ID, DM, OD, KR, KC, DEPTHWISE_CONV_OP_BACKPROP_INPUT, \
        4, STR, PAD, true,                                                     \
        strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", DM, "_", OD, "_",    \
                        KR, "_", KC, "_", STR, "_", PAD, "_gpu"));             \
  }                                                                            \
  static void BM_ConvFloatDepthwiseBkFilterCPU1_##LABEL(int iters) {           \
    BM_ConvFloatDepthwise(                                                     \
        iters, BS, R, C, ID, DM, OD, KR, KC,                                   \
        DEPTHWISE_CONV_OP_BACKPROP_FILTER, 1, STR, PAD, false,                 \
        strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", DM, "_", OD, "_",    \
                        KR, "_", KC, "_", STR, "_", PAD, "_cpu1"));            \
  }                                                                            \
  static void BM_ConvFloatDepthwiseBkFilterCPU4_##LABEL(int iters) {           \
    BM_ConvFloatDepthwise(                                                     \
        iters, BS, R, C, ID, DM, OD, KR, KC,                                   \
        DEPTHWISE_CONV_OP_BACKPROP_FILTER, 4, STR, PAD, false,                 \
        strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", DM, "_", OD, "_",    \
                        KR, "_", KC, "_", STR, "_", PAD, "_cpu4"));            \
  }                                                                            \
  static void BM_ConvFloatDepthwiseBkFilterGPU_##LABEL(int iters) {            \
    BM_ConvFloatDepthwise(                                                     \
        iters, BS, R, C, ID, DM, OD, KR, KC,                                   \
        DEPTHWISE_CONV_OP_BACKPROP_FILTER, 4, STR, PAD, true,                  \
        strings::StrCat(BS, "_", R, "_", C, "_", ID, "_", DM, "_", OD, "_",    \
                        KR, "_", KC, "_", STR, "_", PAD, "_gpu"));             \
  }                                                                            \
  BENCHMARK(BM_ConvFloatDepthwiseBkInCPU1_##LABEL);                            \
  BENCHMARK(BM_ConvFloatDepthwiseBkInCPU4_##LABEL);                            \
  BENCHMARK(BM_ConvFloatDepthwiseBkInGPU_##LABEL);                             \
  BENCHMARK(BM_ConvFloatDepthwiseBkFilterCPU1_##LABEL);                        \
  BENCHMARK(BM_ConvFloatDepthwiseBkFilterCPU4_##LABEL);                        \
  BENCHMARK(BM_ConvFloatDepthwiseBkFilterGPU_##LABEL)

// The configurations below are mostly from mobilenet models.
BM_ConvFloatDepthwiseBk(32, 112, 112, 3, 8, 24, 3, 3, 1, SAME, conv0);
BM_ConvFloatDepthwiseBk(32, 112, 112, 64, 1, 64, 3, 3, 1, SAME, conv1);
BM_ConvFloatDepthwiseBk(32, 56, 56, 128, 1, 128, 3, 3, 1, SAME, conv2);
BM_ConvFloatDepthwiseBk(32, 56, 56, 128, 1, 128, 3, 3, 2, SAME, conv3);
BM_ConvFloatDepthwiseBk(32, 28, 28, 128, 1, 128, 3, 3, 1, SAME, conv4);
BM_ConvFloatDepthwiseBk(32, 14, 14, 512, 1, 512, 3, 3, 1, SAME, conv5);
BM_ConvFloatDepthwiseBk(32, 7, 7, 1024, 1, 1024, 3, 3, 1, SAME, conv6);
// Benchmarks with different stride and padding options, varying depth
// multiplier.
BM_ConvFloatDepthwiseBk(32, 112, 112, 3, 8, 24, 3, 3, 2, SAME, conv7);
BM_ConvFloatDepthwiseBk(32, 112, 112, 3, 8, 24, 3, 3, 2, VALID, conv8);

// Vary depth multiplier.
BM_ConvFloatDepthwiseBk(32, 112, 112, 1, 24, 24, 3, 3, 1, SAME, conv9);
BM_ConvFloatDepthwiseBk(32, 112, 112, 2, 12, 24, 3, 3, 1, SAME, conv10);
BM_ConvFloatDepthwiseBk(32, 112, 112, 3, 8, 24, 3, 3, 1, SAME, conv11);
BM_ConvFloatDepthwiseBk(32, 112, 112, 8, 3, 24, 3, 3, 1, SAME, conv12);
BM_ConvFloatDepthwiseBk(32, 112, 112, 12, 2, 24, 3, 3, 1, SAME, conv13);
BM_ConvFloatDepthwiseBk(32, 112, 112, 24, 1, 24, 3, 3, 1, SAME, conv14);

static void BM_LRNFloat(int iters, int depth, int cols, int rows,
                        int batch_size, int range, int num_threads,
                        const string& label) {
  tensorflow::testing::StopTiming();
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  thread::ThreadPool threadpool(Env::Default(), "test", num_threads);
  EigenThreadPoolWrapper wrapper(&threadpool);
  Eigen::ThreadPoolDevice eigen_cpu_device(&wrapper, num_threads);
  device->set_eigen_cpu_device(&eigen_cpu_device);

  gtl::InlinedVector<TensorValue, 4> inputs;
  TensorShape shape({batch_size, rows, cols, depth});

  Tensor input(DT_FLOAT, shape);
  test::FillIota<float>(&input, 1.0);
  inputs.push_back({nullptr, &input});

  // Convolution op.
  NodeDef lrn_node_def;
  TF_CHECK_OK(NodeDefBuilder("lrn_op", "LRN")
                  .Input("input", 0, DT_FLOAT)
                  .Attr("depth_radius", range)
                  .Attr("bias", 1.0)
                  .Attr("alpha", 0.1)
                  .Attr("beta", 0.5)
                  .Finalize(&lrn_node_def));

  Status status;
  std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                              cpu_allocator(), lrn_node_def,
                                              TF_GRAPH_DEF_VERSION, &status));
  TF_CHECK_OK(status);

  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &inputs;
  params.op_kernel = op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  std::unique_ptr<OpKernelContext> context(new OpKernelContext(&params));

  op->Compute(context.get());
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    delete context->release_output(0).tensor;
    op->Compute(context.get());
  }
  tensorflow::testing::StopTiming();
  testing::ItemsProcessed(context->mutable_output(0)->NumElements() * iters *
                          (2 * range + 1) * 2);
  testing::SetLabel(label);
}

#define BM_LRNFloatFwdCPU(DEPTH, COLS, ROWS, BATCH, RANGE, THREADS, LABEL)   \
  static void                                                                \
      BM_LRNFloat_##DEPTH##_##COLS##_##ROWS##_##BATCH##_##RANGE##_##THREADS( \
          int iters) {                                                       \
    BM_LRNFloat(iters, DEPTH, COLS, ROWS, BATCH, RANGE, THREADS, LABEL);     \
  }                                                                          \
  BENCHMARK(                                                                 \
      BM_LRNFloat_##DEPTH##_##COLS##_##ROWS##_##BATCH##_##RANGE##_##THREADS)

// clang-format off
//                DEPTH, COLS, ROWS, BATCH, RANGE, THREADS, LABEL
BM_LRNFloatFwdCPU(64,    56,   56,   32,    5,     1,       "lrn 1 thread");
BM_LRNFloatFwdCPU(192,   28,   28,   64,    2,     1,       "lrn 1 thread");
BM_LRNFloatFwdCPU(192,   56,   56,   32,    5,     1,       "lrn 1 thread");
BM_LRNFloatFwdCPU(64,    56,   56,   32,    5,     4,       "lrn 4 threads");
BM_LRNFloatFwdCPU(192,   28,   28,   64,    2,     4,       "lrn 4 threads");
BM_LRNFloatFwdCPU(192,   56,   56,   32,    5,     4,       "lrn 4 threads");
BM_LRNFloatFwdCPU(64,    56,   56,   32,    5,     8,       "lrn 8 threads");
BM_LRNFloatFwdCPU(192,   28,   28,   64,    2,     8,       "lrn 8 threads");
BM_LRNFloatFwdCPU(192,   56,   56,   32,    5,     8,       "lrn 8 threads");
// clang-format on

/*
AvgPooling Op
*/
static void BM_AvgPool(int iters, int batch_size, int rows, int cols, int depth,
                       int kernel_rows, int kernel_cols, int stride,
                       Padding padding, int num_threads, const string& label) {
  tensorflow::testing::StopTiming();
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  thread::ThreadPool threadpool(Env::Default(), "test", num_threads);
  EigenThreadPoolWrapper wrapper(&threadpool);
  Eigen::ThreadPoolDevice eigen_cpu_device(&wrapper, num_threads);
  device->set_eigen_cpu_device(&eigen_cpu_device);

  gtl::InlinedVector<TensorValue, 4> inputs;
  TensorShape shape1({batch_size, rows, cols, depth});
  Tensor input1(DT_FLOAT, shape1);
  test::FillIota<float>(&input1, 1.0);
  inputs.push_back({nullptr, &input1});

  // AvgPooling op.
  NodeDef avgpool_node_def;
  CHECK_EQ(kernel_rows, kernel_cols);
  Status status = NodeDefBuilder("avgpool_op", "AvgPool")
                      .Input(FakeInput(DT_FLOAT))
                      .Attr("ksize", {1, kernel_rows, kernel_cols, 1})
                      .Attr("strides", {1, stride, stride, 1})
                      .Attr("padding", padding == VALID ? "VALID" : "SAME")
                      .Finalize(&avgpool_node_def);
  TF_CHECK_OK(status);

  std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                              cpu_allocator(), avgpool_node_def,
                                              TF_GRAPH_DEF_VERSION, &status));
  TF_CHECK_OK(status);
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &inputs;
  params.op_kernel = op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  std::unique_ptr<OpKernelContext> avgpool_context(
      new OpKernelContext(&params));

  op->Compute(avgpool_context.get());
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    delete avgpool_context->release_output(0).tensor;
    op->Compute(avgpool_context.get());
  }
  tensorflow::testing::StopTiming();
  testing::ItemsProcessed(avgpool_context->mutable_output(0)->NumElements() *
                          iters);
  testing::SetLabel(label);
}

// BS: batch_size
// IR: input_rows
// IC: input_cols
// ND: node_depth
// KR: kernel_rows
// KC: kernel_cols
// ST: stride. We use the same stride for both directions.
// PT: padding
#define BM_AvgPoolFwdCPU(BS, IR, IC, ND, KR, KC, ST, PT, TH, LABEL)            \
  static void                                                                  \
      BM_AvgPool_##BS##_##IR##_##IC##_##ND##_##KR##_##KC##_##ST##_##PT##_##TH( \
          int iters) {                                                         \
    BM_AvgPool(iters, BS, IR, IC, ND, KR, KC, ST, PT, TH, LABEL);              \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_AvgPool_##BS##_##IR##_##IC##_##ND##_##KR##_##KC##_##ST##_##PT##_##TH)

// Labels are taken from the 2014-July-24 version of imagenet
BM_AvgPoolFwdCPU(32, 112, 112, 64, 3, 3, 2, VALID, 1, "avgpool0_VALID");
BM_AvgPoolFwdCPU(32, 56, 56, 192, 3, 3, 2, VALID, 1, "avgpool1_VALID");
BM_AvgPoolFwdCPU(32, 28, 28, 352, 3, 3, 2, VALID, 1, "avgpool4_VALID");
BM_AvgPoolFwdCPU(32, 14, 14, 576, 3, 3, 2, VALID, 1, "avgpool10_VALID");
BM_AvgPoolFwdCPU(32, 112, 112, 64, 3, 3, 2, SAME, 1, "avgpool0_SAME");
BM_AvgPoolFwdCPU(32, 56, 56, 192, 3, 3, 2, SAME, 1, "avgpool1_SAME");
BM_AvgPoolFwdCPU(32, 28, 28, 352, 3, 3, 2, SAME, 1, "avgpool4_SAME");
BM_AvgPoolFwdCPU(32, 14, 14, 576, 3, 3, 2, SAME, 1, "avgpool10_SAME");
BM_AvgPoolFwdCPU(32, 112, 112, 64, 3, 3, 2, VALID, 4, "avgpool0_VALID");
BM_AvgPoolFwdCPU(32, 56, 56, 192, 3, 3, 2, VALID, 4, "avgpool1_VALID");
BM_AvgPoolFwdCPU(32, 28, 28, 352, 3, 3, 2, VALID, 4, "avgpool4_VALID");
BM_AvgPoolFwdCPU(32, 14, 14, 576, 3, 3, 2, VALID, 4, "avgpool10_VALID");
BM_AvgPoolFwdCPU(32, 112, 112, 64, 3, 3, 2, SAME, 4, "avgpool0_SAME");
BM_AvgPoolFwdCPU(32, 56, 56, 192, 3, 3, 2, SAME, 4, "avgpool1_SAME");
BM_AvgPoolFwdCPU(32, 28, 28, 352, 3, 3, 2, SAME, 4, "avgpool4_SAME");
BM_AvgPoolFwdCPU(32, 14, 14, 576, 3, 3, 2, SAME, 4, "avgpool10_SAME");

static void BM_AvgPoolBk(int iters, int batch_size, int rows, int cols,
                         int depth, int kernel_rows, int kernel_cols,
                         int stride, Padding padding, int num_threads,
                         const string& label) {
  tensorflow::testing::StopTiming();
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  thread::ThreadPool threadpool(Env::Default(), "test", num_threads);
  EigenThreadPoolWrapper wrapper(&threadpool);
  Eigen::ThreadPoolDevice eigen_cpu_device(&wrapper, num_threads);
  device->set_eigen_cpu_device(&eigen_cpu_device);

  gtl::InlinedVector<TensorValue, 4> inputs;

  int64 out_height, out_width, pad_rows, pad_cols;
  TF_CHECK_OK(GetWindowedOutputSize(rows, kernel_rows, stride, padding,
                                    &out_height, &pad_rows));
  TF_CHECK_OK(GetWindowedOutputSize(cols, kernel_cols, stride, padding,
                                    &out_width, &pad_cols));
  TensorShape output_shape({batch_size, out_height, out_width, depth});
  TensorShape shape2({4});
  Tensor input_shape_tensor(DT_INT32, shape2);
  int32 input_dims[] = {batch_size, rows, cols, depth};
  for (int i = 0; i < 4; i++) {
    input_shape_tensor.flat<int32>()(i) = input_dims[i];
  }
  inputs.push_back({nullptr, &input_shape_tensor});

  Tensor output_backprop(DT_FLOAT, output_shape);
  test::FillIota<float>(&output_backprop, 11.0);
  inputs.push_back({nullptr, &output_backprop});

  // AvgPoolGrad op.
  NodeDef avgpool_grad_node_def;
  Status status = NodeDefBuilder("avgpool_grad_op", "AvgPoolGrad")
                      .Input(FakeInput())
                      .Input(FakeInput(DT_FLOAT))
                      .Attr("ksize", {1, kernel_rows, kernel_cols, 1})
                      .Attr("strides", {1, stride, stride, 1})
                      .Attr("padding", padding == VALID ? "VALID" : "SAME")
                      .Finalize(&avgpool_grad_node_def);
  TF_CHECK_OK(status);
  std::unique_ptr<OpKernel> op(
      CreateOpKernel(DEVICE_CPU, nullptr, cpu_allocator(),
                     avgpool_grad_node_def, TF_GRAPH_DEF_VERSION, &status));
  TF_CHECK_OK(status);
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &inputs;
  params.op_kernel = op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  std::unique_ptr<OpKernelContext> avgpool_context(
      new OpKernelContext(&params));

  op->Compute(avgpool_context.get());
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    delete avgpool_context->release_output(0).tensor;
    op->Compute(avgpool_context.get());
  }
  tensorflow::testing::StopTiming();
  testing::ItemsProcessed(avgpool_context->mutable_output(0)->NumElements() *
                          iters);
  testing::SetLabel(label);
}

// BS: batch_size
// IR: input_rows
// IC: input_cols
// ND: node_depth
// KR: kernel_rows
// KC: kernel_cols
// ST: stride. We use the same stride for both directions.
// PT: padding
// The resulted symbol is too long. Need to use two macros to fit in 80-chars
#define BM_AvgPoolBkCPU(BS, IR, IC, ND, KR, KC, ST, PT, TH, LABEL)               \
  static void                                                                    \
      BM_AvgPoolBk_##BS##_##IR##_##IC##_##ND##_##KR##_##KC##_##ST##_##PT##_##TH( \
          int iters) {                                                           \
    BM_AvgPoolBk(iters, BS, IR, IC, ND, KR, KC, ST, PT, TH, LABEL);              \
  }                                                                              \
  BENCHMARK(                                                                     \
      BM_AvgPoolBk_##BS##_##IR##_##IC##_##ND##_##KR##_##KC##_##ST##_##PT##_##TH)

// Shapes taken from the 2015/05/16 inception model
BM_AvgPoolBkCPU(32, 35, 35, 192, 3, 3, 1, SAME, 1, "avgpool_grad0_SAME");
BM_AvgPoolBkCPU(32, 35, 35, 256, 3, 3, 1, SAME, 1, "avgpool_grad1_SAME");
BM_AvgPoolBkCPU(32, 17, 17, 768, 3, 3, 1, SAME, 1, "avgpool_grad2_SAME");
BM_AvgPoolBkCPU(32, 17, 17, 1024, 3, 3, 1, SAME, 1, "avgpool_grad3_SAME");
BM_AvgPoolBkCPU(32, 17, 17, 1152, 3, 3, 1, SAME, 1, "avgpool_grad4_SAME");
BM_AvgPoolBkCPU(32, 17, 17, 1216, 3, 3, 1, SAME, 1, "avgpool_grad5_SAME");
BM_AvgPoolBkCPU(32, 17, 17, 1248, 5, 5, 3, VALID, 1, "avgpool_grad6_VALID");
BM_AvgPoolBkCPU(32, 8, 8, 1760, 3, 3, 1, SAME, 1, "avgpool_grad7_SAME");
BM_AvgPoolBkCPU(32, 8, 8, 2048, 8, 8, 1, VALID, 1, "avgpool_grad8_VALID");

/*
MaxPooling Op
*/
static void BM_MaxPool(int iters, int batch_size, int rows, int cols, int depth,
                       int kernel_rows, int kernel_cols, int stride,
                       Padding padding, int num_threads, const string& label) {
  tensorflow::testing::StopTiming();
  SessionOptions options;
  options.config.set_intra_op_parallelism_threads(num_threads);

  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", options, "/job:a/replica:0/task:0"));

  thread::ThreadPool threadpool(Env::Default(), "test", num_threads);
  EigenThreadPoolWrapper wrapper(&threadpool);
  Eigen::ThreadPoolDevice eigen_cpu_device(&wrapper, num_threads);
  device->set_eigen_cpu_device(&eigen_cpu_device);

  gtl::InlinedVector<TensorValue, 4> inputs;
  TensorShape shape1({batch_size, rows, cols, depth});
  Tensor input1(DT_FLOAT, shape1);
  test::FillIota<float>(&input1, 1.0);
  inputs.push_back({nullptr, &input1});

  // MaxPooling op.
  NodeDef maxpool_node_def;
  CHECK_EQ(kernel_rows, kernel_cols);
  Status status = NodeDefBuilder("maxpool_op", "MaxPool")
                      .Input(FakeInput())
                      .Attr("ksize", {1, kernel_rows, kernel_cols, 1})
                      .Attr("strides", {1, stride, stride, 1})
                      .Attr("padding", padding == VALID ? "VALID" : "SAME")
                      .Finalize(&maxpool_node_def);
  TF_CHECK_OK(status);
  std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                              cpu_allocator(), maxpool_node_def,
                                              TF_GRAPH_DEF_VERSION, &status));
  TF_CHECK_OK(status);
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &inputs;
  params.op_kernel = op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  std::unique_ptr<OpKernelContext> maxpool_context(
      new OpKernelContext(&params));

  op->Compute(maxpool_context.get());
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    delete maxpool_context->release_output(0).tensor;
    op->Compute(maxpool_context.get());
  }
  tensorflow::testing::StopTiming();
  testing::ItemsProcessed(maxpool_context->mutable_output(0)->NumElements() *
                          iters);
  testing::SetLabel(label);
}

// BS: batch_size
// IR: input_rows
// IC: input_cols
// ND: node_depth
// KR: kernel_rows
// KC: kernel_cols
// ST: stride. We use the same stride for both directions.
// PT: padding
#define BM_MaxPoolFwdCPU(BS, IR, IC, ND, KR, KC, ST, PT, TH, LABEL)            \
  static void                                                                  \
      BM_MaxPool_##BS##_##IR##_##IC##_##ND##_##KR##_##KC##_##ST##_##PT##_##TH( \
          int iters) {                                                         \
    BM_MaxPool(iters, BS, IR, IC, ND, KR, KC, ST, PT, TH, LABEL);              \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_MaxPool_##BS##_##IR##_##IC##_##ND##_##KR##_##KC##_##ST##_##PT##_##TH)

// Labels are taken from the 2014-July-24 version of imagenet
/* TODO XXX
BM_MaxPoolFwdCPU(32, 112, 112, 64, 3, 3, 2, VALID, 1, "maxpool0_VALID");
BM_MaxPoolFwdCPU(32, 56, 56, 192, 3, 3, 2, VALID, 1, "maxpool1_VALID");
BM_MaxPoolFwdCPU(32, 28, 28, 352, 3, 3, 2, VALID, 1, "maxpool4_VALID");
BM_MaxPoolFwdCPU(32, 14, 14, 576, 3, 3, 2, VALID, 1, "maxpool10_VALID");
BM_MaxPoolFwdCPU(32, 112, 112, 64, 3, 3, 2, SAME, 1, "maxpool0_SAME");
BM_MaxPoolFwdCPU(32, 56, 56, 192, 3, 3, 2, SAME, 1, "maxpool1_SAME");
BM_MaxPoolFwdCPU(32, 28, 28, 352, 3, 3, 2, SAME, 1, "maxpool4_SAME");
BM_MaxPoolFwdCPU(32, 14, 14, 576, 3, 3, 2, SAME, 1, "maxpool10_SAME");
*/
BM_MaxPoolFwdCPU(32, 112, 112, 64, 3, 3, 2, VALID, 4, "maxpool0_VALID");
BM_MaxPoolFwdCPU(32, 56, 56, 192, 3, 3, 2, VALID, 4, "maxpool1_VALID");
BM_MaxPoolFwdCPU(32, 28, 28, 352, 3, 3, 2, VALID, 4, "maxpool4_VALID");
BM_MaxPoolFwdCPU(32, 14, 14, 576, 3, 3, 2, VALID, 4, "maxpool10_VALID");
BM_MaxPoolFwdCPU(32, 112, 112, 64, 3, 3, 2, SAME, 4, "maxpool0_SAME");
BM_MaxPoolFwdCPU(32, 56, 56, 192, 3, 3, 2, SAME, 4, "maxpool1_SAME");
BM_MaxPoolFwdCPU(32, 28, 28, 352, 3, 3, 2, SAME, 4, "maxpool4_SAME");
BM_MaxPoolFwdCPU(32, 14, 14, 576, 3, 3, 2, SAME, 4, "maxpool10_SAME");

static void BM_MaxPoolBk(int iters, int batch_size, int rows, int cols,
                         int depth, int kernel_rows, int kernel_cols,
                         int stride, Padding padding, int num_threads,
                         bool use_gpu, const string& label) {
  auto root = Scope::NewRootScope().ExitOnError();

  int64 out_height, out_width, pad_rows, pad_cols;
  TF_CHECK_OK(GetWindowedOutputSize(rows, kernel_rows, stride, padding,
                                    &out_height, &pad_rows));
  TF_CHECK_OK(GetWindowedOutputSize(cols, kernel_cols, stride, padding,
                                    &out_width, &pad_cols));

  Tensor input_data(DT_FLOAT, TensorShape({batch_size, rows, cols, depth}));
  input_data.flat<float>().setRandom();

  Tensor output_data(DT_FLOAT,
                     TensorShape({batch_size, out_height, out_width, depth}));
  output_data.flat<float>().setRandom();

  Tensor output_diff(DT_FLOAT,
                     TensorShape({batch_size, out_height, out_width, depth}));
  output_diff.flat<float>().setRandom();

  CHECK_EQ(kernel_rows, kernel_cols);
  ops::internal::MaxPoolGrad(root, input_data, output_data, output_diff,
                             {1, kernel_rows, kernel_cols, 1} /* ksize */,
                             {1, stride, stride, 1} /* stride */,
                             padding == VALID ? "VALID" : "SAME");
  TF_CHECK_OK(root.status());
  Graph* g = new Graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(g));
  string device = use_gpu ? "gpu" : "cpu";
  testing::UseRealTime();
  test::Benchmark(device, g).Run(iters);

  testing::ItemsProcessed(batch_size * rows * cols * depth * iters);
  testing::SetLabel(label);
}

// BS: batch_size
// IR: input_rows
// IC: input_cols
// ND: node_depth
// KR: kernel_rows
// KC: kernel_cols
// ST: stride. We use the same stride for both directions.
// PT: padding
// The resulted symbol is too long. Need to use two macros to fit in 80-chars
// clang-format off
#define BM_MaxPoolBkGPU(BS, IR, IC, ND, KR, KC, ST, PT, TH, LABEL)             \
  static void                                                                  \
      BM_MaxPoolBk_GPU_##BS##_##IR##_##IC##_##ND##_##KR##_##KC##_##ST##_       \
          ##PT##_##TH(                                                         \
          int iters) {                                                         \
    BM_MaxPoolBk(iters, BS, IR, IC, ND, KR, KC, ST, PT, TH, true, LABEL);      \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_MaxPoolBk_GPU_##BS##_##IR##_##IC##_##ND##_##KR##_##KC##_##ST##_       \
          ##PT##_##TH)                                                         \

#define BM_MaxPoolBkCPU(BS, IR, IC, ND, KR, KC, ST, PT, TH, LABEL)             \
  static void                                                                  \
      BM_MaxPoolBk_CPU_##BS##_##IR##_##IC##_##ND##_##KR##_##KC##_##ST##_       \
          ##PT##_##TH(                                                         \
          int iters) {                                                         \
    BM_MaxPoolBk(iters, BS, IR, IC, ND, KR, KC, ST, PT, TH, false, LABEL);     \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_MaxPoolBk_CPU_##BS##_##IR##_##IC##_##ND##_##KR##_##KC##_##ST##_       \
          ##PT##_##TH)
// clang-format on

// Shapes taken from the 2015/05/16 inception model
BM_MaxPoolBkGPU(32, 147, 147, 64, 3, 3, 2, VALID, 1, "maxpool_grad0_VALID");
BM_MaxPoolBkGPU(32, 71, 71, 192, 3, 3, 2, VALID, 1, "maxpool_grad1_VALID");
BM_MaxPoolBkGPU(32, 35, 35, 288, 3, 3, 2, VALID, 1, "maxpool_grad2_VALID");
BM_MaxPoolBkGPU(32, 17, 17, 1248, 3, 3, 2, VALID, 1, "maxpool_grad3_VALID");
BM_MaxPoolBkGPU(32, 8, 8, 2048, 3, 3, 2, VALID, 1, "maxpool_grad4_VALID");

BM_MaxPoolBkCPU(32, 147, 147, 64, 3, 3, 2, VALID, 1, "maxpool_grad0_VALID");
BM_MaxPoolBkCPU(32, 71, 71, 192, 3, 3, 2, VALID, 1, "maxpool_grad1_VALID");
BM_MaxPoolBkCPU(32, 35, 35, 288, 3, 3, 2, VALID, 1, "maxpool_grad2_VALID");
BM_MaxPoolBkCPU(32, 17, 17, 1248, 3, 3, 2, VALID, 1, "maxpool_grad3_VALID");
BM_MaxPoolBkCPU(32, 8, 8, 2048, 3, 3, 2, VALID, 1, "maxpool_grad4_VALID");

/*
Relu Op
Run benchmark with:
*/
static void BM_ReluFloat(int iters, int batch_size, int rows, int cols,
                         int depth, int num_threads, const string& label) {
  tensorflow::testing::StopTiming();
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  thread::ThreadPool threadpool(Env::Default(), "test", num_threads);
  EigenThreadPoolWrapper wrapper(&threadpool);
  Eigen::ThreadPoolDevice eigen_cpu_device(&wrapper, num_threads);
  device->set_eigen_cpu_device(&eigen_cpu_device);

  gtl::InlinedVector<TensorValue, 4> inputs;
  TensorShape shape1({batch_size, rows, cols, depth});
  Tensor input1(DT_FLOAT, shape1);
  test::FillIota<float>(&input1, 1.0);
  inputs.push_back({nullptr, &input1});

  // Reluing op.
  NodeDef relu_node_def;
  Status status = NodeDefBuilder("relu_op", "Relu")
                      .Input(FakeInput(DT_FLOAT))
                      .Finalize(&relu_node_def);
  TF_CHECK_OK(status);
  std::unique_ptr<OpKernel> op(CreateOpKernel(DEVICE_CPU, device.get(),
                                              cpu_allocator(), relu_node_def,
                                              TF_GRAPH_DEF_VERSION, &status));
  TF_CHECK_OK(status);
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = &inputs;
  params.op_kernel = op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  std::unique_ptr<OpKernelContext> relu_context(new OpKernelContext(&params));

  op->Compute(relu_context.get());
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    delete relu_context->release_output(0).tensor;
    op->Compute(relu_context.get());
  }
  tensorflow::testing::StopTiming();
  testing::ItemsProcessed(relu_context->mutable_output(0)->NumElements() *
                          iters);
  testing::SetLabel(label);
}

// BS: batch_size
// IR: input_rows
// IC: input_cols
// ND: node_depth
#define BM_Relu(BS, IR, IC, ND, TH, LABEL)                               \
  static void BM_ReluFloat_##BS##_##IR##_##IC##_##ND##_##TH(int iters) { \
    BM_ReluFloat(iters, BS, IR, IC, ND, TH, LABEL);                      \
  }                                                                      \
  BENCHMARK(BM_ReluFloat_##BS##_##IR##_##IC##_##ND##_##TH)

BM_Relu(32, 112, 112, 64, 1, "relu0");
BM_Relu(32, 56, 56, 192, 1, "relu1");
BM_Relu(32, 28, 28, 352, 1, "relu4");
BM_Relu(32, 14, 14, 576, 1, "relu10");
BM_Relu(32, 112, 112, 64, 4, "relu0");
BM_Relu(32, 56, 56, 192, 4, "relu1");
BM_Relu(32, 28, 28, 352, 4, "relu4");
BM_Relu(32, 14, 14, 576, 4, "relu10");

static void BM_ImageNetSoftmaxFwd(int iters, int batch_size, int node_depth,
                                  int num_threads, bool use_gpu,
                                  const string& label) {
  auto root = Scope::NewRootScope().ExitOnError();

  Tensor input(DT_FLOAT, TensorShape({batch_size, node_depth}));
  input.flat<float>().setRandom();

  auto softmax = ops::Softmax(root, input);

  TF_CHECK_OK(root.status());
  Graph* g = new Graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(g));
  string device = use_gpu ? "gpu" : "cpu";
  SessionOptions opts;
  opts.config.set_inter_op_parallelism_threads(1);
  opts.config.set_intra_op_parallelism_threads(num_threads);
  opts.config.set_use_per_session_threads(true);
  opts.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);
  testing::UseRealTime();
  test::Benchmark(device, g, &opts).Run(iters);
  testing::ItemsProcessed(batch_size * node_depth * iters);
  testing::SetLabel(label);
}

#define BM_ImageNetSoftmaxFwd(BATCH_SIZE, NODE_DEPTH, TH, GPU, LABEL)     \
  static void                                                             \
      BM_ImageNetSoftmaxFwd_##BATCH_SIZE##_##NODE_DEPTH##_##TH##_##GPU(   \
          int iters) {                                                    \
    BM_ImageNetSoftmaxFwd(iters, BATCH_SIZE, NODE_DEPTH, TH, GPU, LABEL); \
  }                                                                       \
  BENCHMARK(BM_ImageNetSoftmaxFwd_##BATCH_SIZE##_##NODE_DEPTH##_##TH##_##GPU)

// Labels are taken from the 2014-July-24 version of imagenet
BM_ImageNetSoftmaxFwd(32, 1008, 1, false, "softmax32");
BM_ImageNetSoftmaxFwd(128, 1008, 1, false, "softmax128");
BM_ImageNetSoftmaxFwd(32, 1008, 4, false, "softmax32");
BM_ImageNetSoftmaxFwd(128, 1008, 4, false, "softmax128");
BM_ImageNetSoftmaxFwd(32, 1008, 1, true, "softmax32");
BM_ImageNetSoftmaxFwd(128, 1008, 1, true, "softmax128");
BM_ImageNetSoftmaxFwd(8192, 1024, 1, true, "softmax32");
BM_ImageNetSoftmaxFwd(8192, 32768, 1, true, "softmax128");

static void BM_TopK(int iters, int rows, int cols, int k, int num_threads,
                    bool use_gpu, const string& label) {
  testing::StopTiming();
  auto root = Scope::NewRootScope().ExitOnError();

  Tensor input(DT_FLOAT, TensorShape({rows, cols}));
  input.flat<float>().setRandom();

  Tensor input_k(DT_INT32, TensorShape({}));
  input_k.scalar<int32>()() = k;

  auto top_k = ops::TopK(root, input, input_k, ops::TopK::Sorted(true));

  TF_CHECK_OK(root.status());
  Graph* g = new Graph(OpRegistry::Global());
  TF_CHECK_OK(root.ToGraph(g));
  string device = use_gpu ? "gpu" : "cpu";
  SessionOptions opts;
  opts.config.set_inter_op_parallelism_threads(1);
  opts.config.set_intra_op_parallelism_threads(num_threads);
  opts.config.set_use_per_session_threads(true);
  opts.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);
  testing::UseRealTime();
  testing::StartTiming();
  test::Benchmark(device, g, &opts).Run(iters);
  testing::ItemsProcessed(rows * cols * iters);
  testing::SetLabel(label);
}

// IR: input_rows
// IC: input_cols
// IK: k
// TH: number of threads
#define BM_TopKGPU(IR, IC, IK, TH, LABEL)                        \
  static void BM_TopK_GPU_##IR##_##IC##_##IK##_##TH(int iters) { \
    BM_TopK(iters, IR, IC, IK, TH, true, LABEL);                 \
  }                                                              \
  BENCHMARK(BM_TopK_GPU_##IR##_##IC##_##IK##_##TH)

#define BM_TopKCPU(IR, IC, IK, TH, LABEL)                        \
  static void BM_TopK_CPU_##IR##_##IC##_##IK##_##TH(int iters) { \
    BM_TopK(iters, IR, IC, IK, TH, false, LABEL);                \
  }                                                              \
  BENCHMARK(BM_TopK_CPU_##IR##_##IC##_##IK##_##TH)

// clang-format on

BM_TopKCPU(1, 100, 1, 16, "topk_r_1_c_100_k_1_th_16");
BM_TopKCPU(1, 100, 2, 16, "topk_r_1_c_100_k_2_th_16");
BM_TopKCPU(1, 100, 10, 16, "topk_r_1_c_100_k_10_th_16");
BM_TopKCPU(1, 100, 50, 16, "topk_r_1_c_100_k_50_th_16");
BM_TopKCPU(1, 100, 100, 16, "topk_r_1_c_100_k_100_th_16");
BM_TopKCPU(32, 100, 1, 16, "topk_r_32_c_100_k_1_th_16");
BM_TopKCPU(32, 100, 2, 16, "topk_r_32_c_100_k_2_th_16");
BM_TopKCPU(32, 100, 10, 16, "topk_r_32_c_100_k_10_th_16");
BM_TopKCPU(32, 100, 50, 16, "topk_r_32_c_100_k_50_th_16");
BM_TopKCPU(32, 100, 100, 16, "topk_r_32_c_100_k_100_th_16");
BM_TopKCPU(128, 100, 1, 16, "topk_r_128_c_100_k_1_th_16");
BM_TopKCPU(128, 100, 2, 16, "topk_r_128_c_100_k_2_th_16");
BM_TopKCPU(128, 100, 10, 16, "topk_r_128_c_100_k_10_th_16");
BM_TopKCPU(128, 100, 50, 16, "topk_r_128_c_100_k_50_th_16");
BM_TopKCPU(128, 100, 100, 16, "topk_r_128_c_100_k_100_th_16");
BM_TopKCPU(128, 1000, 1, 16, "topk_r_128_c_1000_k_1_th_16");
BM_TopKCPU(128, 1000, 2, 16, "topk_r_128_c_1000_k_2_th_16");
BM_TopKCPU(128, 1000, 10, 16, "topk_r_128_c_1000_k_10_th_16");
BM_TopKCPU(128, 1000, 50, 16, "topk_r_128_c_1000_k_50_th_16");
BM_TopKCPU(128, 1000, 100, 16, "topk_r_128_c_1000_k_100_th_16");
BM_TopKCPU(128, 1000, 500, 16, "topk_r_128_c_1000_k_500_th_16");
BM_TopKCPU(128, 1000, 1000, 16, "topk_r_128_c_1000_k_1000_th_16");

// From NMT Codebase:
//   batch_sizes: 16, 128
//   vocab_sizes: 10000 for small dataset, 35000 for large.
//   beam_widths: 1, 2, 5, 10
BM_TopKCPU(16, 10000, 10000, 16, "topk_nmt_r_16_c_10000_k_10000_th_16");
BM_TopKCPU(16, 20000, 20000, 16, "topk_nmt_r_16_c_20000_k_20000_th_16");
BM_TopKCPU(16, 50000, 50000, 16, "topk_nmt_r_16_c_50000_k_50000_th_16");
BM_TopKCPU(16, 100000, 100000, 16, "topk_nmt_r_16_c_100000_k_100000_th_16");
BM_TopKCPU(16, 35000, 35000, 16, "topk_nmt_r_16_c_35000_k_35000_th_16");
BM_TopKCPU(16, 70000, 70000, 16, "topk_nmt_r_16_c_70000_k_70000_th_16");
BM_TopKCPU(16, 175000, 175000, 16, "topk_nmt_r_16_c_175000_k_175000_th_16");
BM_TopKCPU(16, 350000, 350000, 16, "topk_nmt_r_16_c_350000_k_350000_th_16");
BM_TopKCPU(128, 10000, 10000, 16, "topk_nmt_r_128_c_10000_k_10000_th_16");
BM_TopKCPU(128, 20000, 20000, 16, "topk_nmt_r_128_c_20000_k_20000_th_16");
BM_TopKCPU(128, 50000, 50000, 16, "topk_nmt_r_128_c_50000_k_50000_th_16");
BM_TopKCPU(128, 100000, 100000, 16, "topk_nmt_r_128_c_100000_k_100000_th_16");
BM_TopKCPU(128, 35000, 35000, 16, "topk_nmt_r_128_c_35000_k_35000_th_16");
BM_TopKCPU(128, 70000, 70000, 16, "topk_nmt_r_128_c_70000_k_70000_th_16");
BM_TopKCPU(128, 175000, 175000, 16, "topk_nmt_r_128_c_175000_k_175000_th_16");
BM_TopKCPU(128, 350000, 350000, 16, "topk_nmt_r_128_c_350000_k_350000_th_16");

}  // namespace tensorflow

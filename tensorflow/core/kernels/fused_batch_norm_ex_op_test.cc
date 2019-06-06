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

#include "absl/algorithm/container.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

template <typename T, typename U>
class FusedBatchNormExOpTestBase : public OpsTestBase {
 public:
  FusedBatchNormExOpTestBase() {
    setenv("TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT", "1", 1 /* replace */);
  }

 protected:
  using GraphRunner = std::function<void(
      const Tensor& input_data, const Tensor& scale_data,
      const Tensor& offset_data, const Tensor& mean_data,
      const Tensor& var_data, const Tensor& side_input_data, Tensor* out)>;

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the output Tensor. Optional `fetch_node` parameter
  // allows to define a fetch node directly using a NodeDef for the ops that are
  // not supported by the C++ Api.
  // TODO(ezhulenev): RunAndFetch defined in FusedConv2D and FusedMatMul tests.
  // Add a base class for all FusedABC kernels and remove code duplication.
  void RunAndFetch(const tensorflow::Scope& root, const string& fetch,
                   Tensor* output, bool allow_gpu_device,
                   const NodeDef* fetch_node = nullptr) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    if (fetch_node) {
      *graph.add_node() = *fetch_node;
    }

    // We really want to make sure that graph executed exactly as we passed it
    // to the session, so we disable various optimizations.
    tensorflow::SessionOptions session_options;

    // Disable common runtime constant folding.
    session_options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_opt_level(OptimizerOptions::L0);

    // Disable Grappler optimizations for tests.
    tensorflow::RewriterConfig* cfg =
        session_options.config.mutable_graph_options()
            ->mutable_rewrite_options();
    cfg->set_constant_folding(tensorflow::RewriterConfig::OFF);
    cfg->set_layout_optimizer(tensorflow::RewriterConfig::OFF);
    cfg->set_remapping(tensorflow::RewriterConfig::OFF);

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(session_options));

    std::vector<DeviceAttributes> available_devices;
    TF_ASSERT_OK(session->ListDevices(&available_devices))
        << "Failed to get available session devices";

    // Check if session has an available GPU device.
    const bool has_gpu_device =
        absl::c_any_of(available_devices, [](const DeviceAttributes& device) {
          return device.device_type() == DEVICE_GPU;
        });

    // Some of the `FusedABC` ops are implemented only for CPU, and in this test
    // we don't want to compare GPU vs CPU numbers, so place all nodes on CPU in
    // this case.
    const bool place_all_on_gpu = allow_gpu_device && has_gpu_device;

    const string device = place_all_on_gpu ? "/device:GPU:0" : "/device:CPU:0";
    for (NodeDef& mutable_node : *graph.mutable_node()) {
      mutable_node.set_device(device);
    }

    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &unfused_tensors));

    *output = unfused_tensors[0];
  }

  void RunFusedBatchNorm(const Tensor& input_data, const Tensor& scale_data,
                         const Tensor& offset_data, const Tensor& mean_data,
                         const Tensor& var_data, const Tensor& side_input_data,
                         const TensorFormat data_format, bool is_training,
                         bool has_side_input, const string& activation_mode,
                         Tensor* output, float epsilon = 0.1f) {
    Scope root = tensorflow::Scope::NewRootScope();

    ops::FusedBatchNormV2 fbn = ops::FusedBatchNormV2(
        root.WithOpName("fused_batch_norm"),
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data)),
        ops::Const(root.WithOpName("scale"), Input::Initializer(scale_data)),
        ops::Const(root.WithOpName("offset"), Input::Initializer(offset_data)),
        ops::Const(root.WithOpName("mean"), Input::Initializer(mean_data)),
        ops::Const(root.WithOpName("var"), Input::Initializer(var_data)),
        ops::FusedBatchNormV2::IsTraining(is_training)
            .Epsilon(epsilon)
            .DataFormat(ToString(data_format)));

    Output with_side_input;
    if (has_side_input) {
      with_side_input =
          ops::Add(root.WithOpName("with_side_input"), fbn.y,
                   ops::Const(root.WithOpName("side_input"),
                              Input::Initializer(side_input_data)));
    } else {
      with_side_input =
          ops::Identity(root.WithOpName("with_side_input"), fbn.y);
    }

    if (activation_mode == "Relu") {
      ops::Relu(root.WithOpName("with_activation"), with_side_input);
    } else {
      ops::Identity(root.WithOpName("with_activation"), with_side_input);
    }

    RunAndFetch(root, "with_activation", output, /*allow_gpu_device=*/true);
  }

  void RunFusedBatchNormEx(const Tensor& input_data, const Tensor& scale_data,
                           const Tensor& offset_data, const Tensor& mean_data,
                           const Tensor& var_data,
                           const Tensor& side_input_data,
                           const TensorFormat data_format, bool is_training,
                           bool has_side_input, const string& activation_mode,
                           Tensor* output, float epsilon = 0.1f) {
    Scope root = tensorflow::Scope::NewRootScope();

    DataType t_dtype = DataTypeToEnum<T>::v();
    DataType u_dtype = DataTypeToEnum<U>::v();

    Output input =
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data));
    Output scale =
        ops::Const(root.WithOpName("scale"), Input::Initializer(scale_data));
    Output offset =
        ops::Const(root.WithOpName("offset"), Input::Initializer(offset_data));
    Output mean =
        ops::Const(root.WithOpName("mean"), Input::Initializer(mean_data));
    Output var =
        ops::Const(root.WithOpName("var"), Input::Initializer(var_data));
    Output side_input = ops::Const(root.WithOpName("side_input"),
                                   Input::Initializer(side_input_data));

    int num_side_inputs = 0;
    std::vector<NodeDefBuilder::NodeOut> side_inputs;

    if (has_side_input) {
      num_side_inputs = 1;
      side_inputs.push_back({side_input.name(), 0, t_dtype});
    }

    NodeDef fused_batch_norm_ex;
    TF_EXPECT_OK(NodeDefBuilder("fused_batch_norm_ex", "_FusedBatchNormEx")
                     .Input({input.name(), 0, t_dtype})
                     .Input({scale.name(), 0, u_dtype})
                     .Input({offset.name(), 0, u_dtype})
                     .Input({mean.name(), 0, u_dtype})
                     .Input({var.name(), 0, u_dtype})
                     .Input(side_inputs)
                     .Attr("T", t_dtype)
                     .Attr("U", u_dtype)
                     .Attr("data_format", ToString(data_format))
                     .Attr("epsilon", epsilon)
                     .Attr("activation_mode", activation_mode)
                     .Attr("num_side_inputs", num_side_inputs)
                     .Attr("is_training", is_training)
                     .Finalize(&fused_batch_norm_ex));

    RunAndFetch(root, fused_batch_norm_ex.name(), output,
                /*allow_gpu_device=*/true, &fused_batch_norm_ex);
  }

  void VerifyTensorsNear(int batch, int height, int width, int channels,
                         TensorFormat data_format, bool is_training,
                         const GraphRunner& run_default,
                         const GraphRunner& run_fused) {
    DataType t_dtype = DataTypeToEnum<T>::v();
    DataType u_dtype = DataTypeToEnum<U>::v();

    TensorShape input_shape =
        data_format == FORMAT_NHWC
            ? TensorShape({batch, height, width, channels})
            : TensorShape({batch, channels, height, width});

    Tensor input(t_dtype, input_shape);
    input.flat<T>().setRandom();

    Tensor scale(u_dtype, {channels});
    scale.flat<U>().setRandom();

    Tensor offset(u_dtype, {channels});
    offset.flat<U>().setRandom();

    Tensor mean(u_dtype, {channels});
    mean.flat<U>().setRandom();

    Tensor var(u_dtype, {channels});
    var.flat<U>().setRandom();

    Tensor empty(u_dtype, {0});

    Tensor fused_batch_norm;
    Tensor fused_batch_norm_ex;

    Tensor side_input(t_dtype, input_shape);
    side_input.flat<T>().setRandom();

    run_default(input, scale, offset, is_training ? empty : mean,
                is_training ? empty : var, side_input, &fused_batch_norm);

    // Write some garbage to the `fused_batch_norm_ex` first to make sure
    // that fused kernel actually writes correct results to memory.
    run_default(side_input, scale, offset, is_training ? empty : mean,
                is_training ? empty : var, input, &fused_batch_norm_ex);

    run_fused(input, scale, offset, is_training ? empty : mean,
              is_training ? empty : var, side_input, &fused_batch_norm_ex);

    ASSERT_EQ(fused_batch_norm.dtype(), fused_batch_norm_ex.dtype());
    ASSERT_EQ(fused_batch_norm.shape(), fused_batch_norm_ex.shape());

    test::ExpectClose(fused_batch_norm, fused_batch_norm_ex, 1e-2);
  }

  // Verifies that computing FusedBatchNormOp+{SideInput}+{Activation} is
  // identical to FusedBatchNormExOp[fused_ops={SideInput, Activation}].
  void VerifyFusedBatchNormEx(int batch, int height, int width, int channels,
                              TensorFormat data_format, bool is_training,
                              bool has_side_input,
                              const string& activation_mode) {
    const GraphRunner run_default =
        [&](const Tensor& input_data, const Tensor& scale_data,
            const Tensor& offset_data, const Tensor& mean_data,
            const Tensor& var_data, const Tensor& side_input_data,
            Tensor* out) {
          this->RunFusedBatchNorm(input_data, scale_data, offset_data,
                                  mean_data, var_data, side_input_data,
                                  data_format, is_training, has_side_input,
                                  activation_mode, out);
        };

    const GraphRunner run_inference =
        [&](const Tensor& input_data, const Tensor& scale_data,
            const Tensor& offset_data, const Tensor& mean_data,
            const Tensor& var_data, const Tensor& side_input_data,
            Tensor* out) {
          this->RunFusedBatchNormEx(input_data, scale_data, offset_data,
                                    mean_data, var_data, side_input_data,
                                    data_format, is_training, has_side_input,
                                    activation_mode, out);
        };

    VerifyTensorsNear(batch, height, width, channels, data_format, is_training,
                      run_default, run_inference);
  }
};

template <typename T>
using FusedBatchNormExOpTest =
    FusedBatchNormExOpTestBase<T, float>;  // scale is always float

constexpr bool kInTraining = true;     // is_training == true
constexpr bool kNoSideInput = false;   // side_input == false
constexpr bool kWithSideInput = true;  // side_input == true

TYPED_TEST_SUITE_P(FusedBatchNormExOpTest);

TYPED_TEST_P(FusedBatchNormExOpTest, TrainingInNHWCTest) {
  this->VerifyFusedBatchNormEx(2, 2, 2, 4, FORMAT_NHWC, kInTraining,
                               kNoSideInput, "Identity");
}

TYPED_TEST_P(FusedBatchNormExOpTest, TrainingWithReluInNHWCTest) {
  this->VerifyFusedBatchNormEx(2, 2, 2, 4, FORMAT_NHWC, kInTraining,
                               kNoSideInput, "Relu");
}

TYPED_TEST_P(FusedBatchNormExOpTest, TrainingWithSideInputAndReluInNHWCTest) {
  this->VerifyFusedBatchNormEx(2, 2, 2, 4, FORMAT_NHWC, kInTraining,
                               kWithSideInput, "Relu");
}

REGISTER_TYPED_TEST_SUITE_P(FusedBatchNormExOpTest,      //
                            TrainingInNHWCTest,          //
                            TrainingWithReluInNHWCTest,  //
                            TrainingWithSideInputAndReluInNHWCTest);

#if defined(GOOGLE_CUDA) && (CUDNN_VERSION >= 7402)
using FusedBatchNormExDataTypes = ::testing::Types<Eigen::half>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedBatchNormExOpTest,
                               FusedBatchNormExDataTypes);
#endif  // defined(GOOGLE_CUDA) && (CUDNN_VERSION >= 7402)

// -------------------------------------------------------------------------- //
// Performance benchmarks are below.                                          //
// -------------------------------------------------------------------------- //

using fp16 = Eigen::half;
using SideInputAndActivation = std::pair<bool, string>;

SideInputAndActivation Identity() { return {false, "Identity"}; }
SideInputAndActivation Relu() { return {false, "Relu"}; }
SideInputAndActivation AddAndRelu() { return {true, "Relu"}; }

template <typename T>
static Graph* FusedBatchNormEx(int n, int h, int w, int c,
                               TensorFormat data_format,
                               std::function<SideInputAndActivation()> fn) {
  Graph* g = new Graph(OpRegistry::Global());

  DataType dtype = DataTypeToEnum<T>::value;
  Tensor x_t(dtype, data_format == FORMAT_NHWC ? TensorShape({n, h, w, c})
                                               : TensorShape({n, c, h, w}));
  x_t.flat<T>().setRandom();

  Tensor other_t(DT_FLOAT, TensorShape({c}));
  other_t.flat<float>().setRandom();

  Node* x = test::graph::Constant(g, x_t, "x");
  Node* other = test::graph::Constant(g, other_t, "other");
  Node* empty = test::graph::Constant(g, Tensor(DT_FLOAT, {0}), "empty");

  int num_side_inputs = 0;
  std::vector<NodeBuilder::NodeOut> side_inputs;

  SideInputAndActivation side_input_and_activation = fn();
  bool has_side_input = side_input_and_activation.first;
  string activation_mode = side_input_and_activation.second;

  if (has_side_input) {
    num_side_inputs = 1;
    side_inputs.push_back({x});
  }

  Node* fused_batch_norm;
  TF_CHECK_OK(NodeBuilder(g->NewName("fused_batch_norm"), "_FusedBatchNormEx")
                  .Input(x)
                  .Input(other)        // scale
                  .Input(other)        // offset
                  .Input(empty)        // mean
                  .Input(empty)        // variance
                  .Input(side_inputs)  // side_input
                  .Attr("T", dtype)
                  .Attr("U", DT_FLOAT)
                  .Attr("epsilon", 0.001)
                  .Attr("data_format", ToString(data_format))
                  .Attr("activation_mode", activation_mode)
                  .Attr("num_side_inputs", num_side_inputs)
                  .Attr("is_training", true)
                  .Finalize(g, &fused_batch_norm));

  return g;
}

#define BM_NAME(N, H, W, C, T, FORMAT, A, DEVICE) \
  BM_FusedBatchNorm##_##N##_##H##_##W##_##C##_##FORMAT##_##A##_##T##_##DEVICE

#define BM_FusedBatchNorm(N, H, W, C, T, FORMAT, ACTIVATION, DEVICE)          \
  static void BM_NAME(N, H, W, C, T, FORMAT, ACTIVATION, DEVICE)(int iters) { \
    testing::UseRealTime();                                                   \
    testing::ItemsProcessed(static_cast<int64>(iters) * N * H * W * C);       \
    test::Benchmark(#DEVICE, FusedBatchNormEx<T>(N, H, W, C, FORMAT_##FORMAT, \
                                                 {ACTIVATION}))               \
        .Run(iters);                                                          \
  }                                                                           \
  BENCHMARK(BM_NAME(N, H, W, C, T, FORMAT, ACTIVATION, DEVICE));

#if defined(GOOGLE_CUDA) && (CUDNN_VERSION >= 7402)
BM_FusedBatchNorm(64, 14, 14, 256, fp16, NHWC, Identity, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, NHWC, Relu, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, NHWC, AddAndRelu, gpu);
#endif  // defined(GOOGLE_CUDA) && (CUDNN_VERSION >= 7402)

}  // namespace tensorflow

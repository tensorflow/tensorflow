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
#include "absl/strings/match.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

template <typename T, typename U>
class FusedBatchNormExOpTestBase : public OpsTestBase {
 public:
  FusedBatchNormExOpTestBase() {
    setenv("TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT", "1", 1 /* replace */);
  }

 protected:
  struct FusedBatchNormOutputs {
    Tensor y;
    Tensor batch_mean;
    Tensor batch_variance;
    Tensor reserve_space_1;
    Tensor reserve_space_2;
    Tensor reserve_space_3;
  };

  struct FusedBatchNormGradOutputs {
    Tensor y_backprop;
    Tensor x_backprop;
    Tensor scale_backprop;
    Tensor offset_backprop;
    Tensor reserve_space_4;
    Tensor reserve_space_5;
  };

  using GraphRunner = std::function<void(
      const Tensor& y_backprop, const Tensor& input_data,
      const Tensor& scale_data, const Tensor& offset_data,
      const Tensor& mean_data, const Tensor& var_data,
      const Tensor& side_input_data, FusedBatchNormOutputs* forward,
      FusedBatchNormGradOutputs* backward)>;

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the outputs. Optional `add_nodes` parameter
  // allows to define nodes directly using a NodeDef for the ops that are
  // not supported by the C++ Api.
  // TODO(ezhulenev): RunAndFetch defined in FusedConv2D and FusedMatMul tests.
  // Add a base class for all FusedABC kernels and remove code duplication.
  void RunAndFetch(const tensorflow::Scope& root,
                   const std::vector<string>& fetch,
                   std::vector<Tensor>* outputs, bool allow_gpu_device,
                   const std::vector<const NodeDef*> add_nodes = {}) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    for (const NodeDef* add_node : add_nodes) {
      *graph.add_node() = *add_node;
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
    TF_ASSERT_OK(session->Run({}, fetch, {}, outputs));
  }

  void RunFusedBatchNorm(const Tensor& y_backprop_data,
                         const Tensor& input_data, const Tensor& scale_data,
                         const Tensor& offset_data, const Tensor& mean_data,
                         const Tensor& var_data, const Tensor& side_input_data,
                         const TensorFormat data_format, bool is_training,
                         bool has_side_input, const string& activation_mode,
                         FusedBatchNormOutputs* forward,
                         FusedBatchNormGradOutputs* backward,
                         float epsilon = 0.1f) {
    Scope root = tensorflow::Scope::NewRootScope();

    Output y_backprop = ops::Const(root.WithOpName("y_backprop"),
                                   Input::Initializer(y_backprop_data));
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

    ops::FusedBatchNormV3 fwd = ops::FusedBatchNormV3(
        root.WithOpName("fused_batch_norm"), input, scale, offset, mean, var,
        ops::FusedBatchNormV3::IsTraining(is_training)
            .Epsilon(epsilon)
            .DataFormat(ToString(data_format)));

    Output with_side_input;
    if (has_side_input) {
      with_side_input =
          ops::Add(root.WithOpName("with_side_input"), fwd.y, side_input);
    } else {
      with_side_input =
          ops::Identity(root.WithOpName("with_side_input"), fwd.y);
    }

    Output activation;
    if (activation_mode == "Relu") {
      activation =
          ops::Relu(root.WithOpName("with_activation"), with_side_input);
    } else {
      activation =
          ops::Identity(root.WithOpName("with_activation"), with_side_input);
    }

    Output activation_grad;
    if (activation_mode == "Relu") {
      activation_grad = ops::internal::ReluGrad(
          root.WithOpName("activation_grad"), y_backprop, activation);
    } else {
      activation_grad =
          ops::Identity(root.WithOpName("activation_grad"), y_backprop);
    }

    ops::FusedBatchNormGradV3 bwd = ops::FusedBatchNormGradV3(
        root.WithOpName("fused_batch_norm_grad"), activation_grad, input, scale,
        fwd.reserve_space_1, fwd.reserve_space_2, fwd.reserve_space_3,
        ops::FusedBatchNormGradV3::IsTraining(is_training)
            .Epsilon(epsilon)
            .DataFormat(ToString(data_format)));

    std::vector<Tensor> out_tensors;
    RunAndFetch(
        root,
        {"with_activation:0", "fused_batch_norm:1", "fused_batch_norm:2",
         "fused_batch_norm:3", "fused_batch_norm:4", "fused_batch_norm:5",
         "fused_batch_norm_grad:0", "fused_batch_norm_grad:1",
         "fused_batch_norm_grad:2"},
        &out_tensors, /*allow_gpu_device=*/true);

    forward->y = out_tensors[0];
    forward->batch_mean = out_tensors[1];
    forward->batch_variance = out_tensors[2];
    forward->reserve_space_1 = out_tensors[3];
    forward->reserve_space_2 = out_tensors[4];
    forward->reserve_space_3 = out_tensors[5];

    backward->x_backprop = out_tensors[6];
    backward->scale_backprop = out_tensors[7];
    backward->offset_backprop = out_tensors[8];
  }

  void RunFusedBatchNormEx(const Tensor& y_backprop_data,
                           const Tensor& input_data, const Tensor& scale_data,
                           const Tensor& offset_data, const Tensor& mean_data,
                           const Tensor& var_data,
                           const Tensor& side_input_data,
                           const TensorFormat data_format, bool is_training,
                           bool has_side_input, const string& activation_mode,
                           FusedBatchNormOutputs* forward,
                           FusedBatchNormGradOutputs* backward,
                           float epsilon = 0.1f) {
    Scope root = tensorflow::Scope::NewRootScope();

    DataType t_dtype = DataTypeToEnum<T>::v();
    DataType u_dtype = DataTypeToEnum<U>::v();

    Output y_backprop = ops::Const(root.WithOpName("y_backprop"),
                                   Input::Initializer(y_backprop_data));
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
    Output empty =
        ops::Const(root.WithOpName("empty"),
                   Input::Initializer(Tensor(DataTypeToEnum<U>::value, {0})));

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

    NodeDef fused_batch_norm_grad;
    NodeDef activation_grad;
    std::vector<Tensor> out_tensors;
    std::vector<const NodeDef*> add_nodes;
    if (is_training) {
      TF_EXPECT_OK(
          NodeDefBuilder("fused_batch_norm_grad", "_FusedBatchNormGradEx")
              .Input({y_backprop.name(), 0, t_dtype})
              .Input({input.name(), 0, t_dtype})
              .Input({scale.name(), 0, u_dtype})
              .Input({fused_batch_norm_ex.name(), 3, u_dtype})
              .Input({fused_batch_norm_ex.name(), 4, u_dtype})
              .Input({fused_batch_norm_ex.name(), 5, u_dtype})
              .Input({offset.name(), 0, u_dtype})
              .Input({fused_batch_norm_ex.name(), 0, t_dtype})
              .Attr("T", t_dtype)
              .Attr("U", u_dtype)
              .Attr("data_format", ToString(data_format))
              .Attr("epsilon", epsilon)
              .Attr("activation_mode", activation_mode)
              .Attr("num_side_inputs", num_side_inputs)
              .Attr("is_training", is_training)
              .Finalize(&fused_batch_norm_grad));
      add_nodes = {&fused_batch_norm_ex, &fused_batch_norm_grad};
    } else {
      if (activation_mode == "Relu") {
        TF_EXPECT_OK(NodeDefBuilder("activation_grad", "ReluGrad")
                         .Input({y_backprop.name(), 0, t_dtype})
                         .Input({fused_batch_norm_ex.name(), 0, t_dtype})
                         .Attr("T", t_dtype)
                         .Finalize(&activation_grad));
      } else {
        TF_EXPECT_OK(NodeDefBuilder("activation_grad", "Identity")
                         .Input({y_backprop.name(), 0, t_dtype})
                         .Attr("T", t_dtype)
                         .Finalize(&activation_grad));
      }
      TF_EXPECT_OK(
          NodeDefBuilder("fused_batch_norm_grad", "FusedBatchNormGradV3")
              .Input({activation_grad.name(), 0, t_dtype})
              .Input({input.name(), 0, t_dtype})
              .Input({scale.name(), 0, u_dtype})
              .Input({fused_batch_norm_ex.name(), 3, u_dtype})
              .Input({fused_batch_norm_ex.name(), 4, u_dtype})
              .Input({fused_batch_norm_ex.name(), 5, u_dtype})
              .Attr("T", t_dtype)
              .Attr("U", u_dtype)
              .Attr("data_format", ToString(data_format))
              .Attr("epsilon", epsilon)
              .Attr("is_training", is_training)
              .Finalize(&fused_batch_norm_grad));
      add_nodes = {&fused_batch_norm_ex, &activation_grad,
                   &fused_batch_norm_grad};
    }

    RunAndFetch(root,
                {"fused_batch_norm_ex:0", "fused_batch_norm_ex:1",
                 "fused_batch_norm_ex:2", "fused_batch_norm_ex:3",
                 "fused_batch_norm_ex:4", "fused_batch_norm_ex:5",
                 "fused_batch_norm_grad:0", "fused_batch_norm_grad:1",
                 "fused_batch_norm_grad:2"},
                &out_tensors,
                /*allow_gpu_device=*/true, add_nodes);

    forward->y = out_tensors[0];
    forward->batch_mean = out_tensors[1];
    forward->batch_variance = out_tensors[2];
    forward->reserve_space_1 = out_tensors[3];
    forward->reserve_space_2 = out_tensors[4];
    forward->reserve_space_3 = out_tensors[5];

    backward->x_backprop = out_tensors[6];
    backward->scale_backprop = out_tensors[7];
    backward->offset_backprop = out_tensors[8];
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
    input.flat<T>() -= input.flat<T>().constant(static_cast<T>(0.5));

    Tensor scale(u_dtype, {channels});
    scale.flat<U>().setRandom();

    Tensor offset(u_dtype, {channels});
    offset.flat<U>().setRandom();

    Tensor mean(u_dtype, {channels});
    mean.flat<U>().setRandom();

    Tensor var(u_dtype, {channels});
    var.flat<U>().setRandom();

    Tensor side_input(t_dtype, input_shape);
    side_input.flat<T>().setRandom();
    side_input.flat<T>() += side_input.flat<T>().constant(static_cast<T>(5.0));

    Tensor y_backprop(t_dtype, input_shape);
    y_backprop.flat<T>().setRandom();
    y_backprop.flat<T>() -= y_backprop.flat<T>().constant(static_cast<T>(0.5));

    Tensor empty(u_dtype, {0});

    FusedBatchNormOutputs fbn_forward;
    FusedBatchNormOutputs fbn_ex_forward;

    FusedBatchNormGradOutputs fbn_backward;
    FusedBatchNormGradOutputs fbn_ex_backward;

    run_default(y_backprop, input, scale, offset, is_training ? empty : mean,
                is_training ? empty : var, side_input, &fbn_forward,
                &fbn_backward);

    // Write some garbage to the `fbn_ex_forward` and `fbn_ex_backward` first to
    // make sure that fused kernel actually writes correct results to memory.
    run_default(y_backprop, side_input, scale, offset,
                is_training ? empty : mean, is_training ? empty : var, input,
                &fbn_ex_forward, &fbn_ex_backward);

    run_fused(y_backprop, input, scale, offset, is_training ? empty : mean,
              is_training ? empty : var, side_input, &fbn_ex_forward,
              &fbn_ex_backward);

    std::vector<std::pair<Tensor, Tensor>> tensor_pairs;
    if (is_training) {
      tensor_pairs = {
          {fbn_forward.y, fbn_ex_forward.y},
          {fbn_forward.batch_mean, fbn_ex_forward.batch_mean},
          {fbn_forward.batch_variance, fbn_ex_forward.batch_variance},
          {fbn_forward.reserve_space_1, fbn_ex_forward.reserve_space_1},
          {fbn_forward.reserve_space_2, fbn_ex_forward.reserve_space_2},
          // NOTE(ezhulenev): We deliberately do not check `reserved_space_3`
          // because BatchNormEx with fused side input has different data in it,
          // but we make sure that final gradients are the same.
          {fbn_backward.y_backprop, fbn_ex_backward.y_backprop},
          {fbn_backward.x_backprop, fbn_ex_backward.x_backprop},
          {fbn_backward.scale_backprop, fbn_ex_backward.scale_backprop},
          {fbn_backward.offset_backprop, fbn_ex_backward.offset_backprop},
      };
    } else {
      tensor_pairs = {{fbn_forward.y, fbn_ex_forward.y}};
    }

    for (auto& pair : tensor_pairs) {
      const Tensor& fbn = pair.first;
      const Tensor& fbn_ex = pair.second;

      ASSERT_EQ(fbn.dtype(), fbn_ex.dtype());
      ASSERT_EQ(fbn.shape(), fbn_ex.shape());

      test::ExpectClose(fbn, fbn_ex, 1e-2);
    }
  }

  // Verifies that computing FusedBatchNormOp+{SideInput}+{Activation} is
  // identical to FusedBatchNormExOp[fused_ops={SideInput, Activation}].
  void VerifyFusedBatchNormEx(int batch, int height, int width, int channels,
                              TensorFormat data_format, bool is_training,
                              bool has_side_input,
                              const string& activation_mode) {
    const GraphRunner run_default =
        [&](const Tensor& y_backprop, const Tensor& input_data,
            const Tensor& scale_data, const Tensor& offset_data,
            const Tensor& mean_data, const Tensor& var_data,
            const Tensor& side_input_data, FusedBatchNormOutputs* fwd,
            FusedBatchNormGradOutputs* bwd) {
          this->RunFusedBatchNorm(y_backprop, input_data, scale_data,
                                  offset_data, mean_data, var_data,
                                  side_input_data, data_format, is_training,
                                  has_side_input, activation_mode, fwd, bwd);
        };

    const GraphRunner run_inference =
        [&](const Tensor& y_backprop, const Tensor& input_data,
            const Tensor& scale_data, const Tensor& offset_data,
            const Tensor& mean_data, const Tensor& var_data,
            const Tensor& side_input_data, FusedBatchNormOutputs* fwd,
            FusedBatchNormGradOutputs* bwd) {
          this->RunFusedBatchNormEx(y_backprop, input_data, scale_data,
                                    offset_data, mean_data, var_data,
                                    side_input_data, data_format, is_training,
                                    has_side_input, activation_mode, fwd, bwd);
        };

    VerifyTensorsNear(batch, height, width, channels, data_format, is_training,
                      run_default, run_inference);
  }
};

constexpr bool kInTraining = true;     // is_training == true
constexpr bool kInInference = false;   // is_training == false
constexpr bool kNoSideInput = false;   // side_input == false
constexpr bool kWithSideInput = true;  // side_input == true

// -------------------------------------------------------------------------- //
// FusedBatchNormEx[is_training=true].

#if defined(GOOGLE_CUDA) && (CUDNN_VERSION >= 7402)
template <typename T>
using FusedBatchNormExOpTrainingTest =
    FusedBatchNormExOpTestBase<T, float>;  // scale is always float

TYPED_TEST_SUITE_P(FusedBatchNormExOpTrainingTest);

TYPED_TEST_P(FusedBatchNormExOpTrainingTest, TrainingInNHWCTest) {
  this->VerifyFusedBatchNormEx(4, 28, 28, 256, FORMAT_NHWC, kInTraining,
                               kNoSideInput, "Identity");
}

TYPED_TEST_P(FusedBatchNormExOpTrainingTest, TrainingWithReluInNHWCTest) {
  this->VerifyFusedBatchNormEx(4, 28, 28, 256, FORMAT_NHWC, kInTraining,
                               kNoSideInput, "Relu");
}

TYPED_TEST_P(FusedBatchNormExOpTrainingTest,
             TrainingWithSideInputAndReluInNHWCTest) {
  this->VerifyFusedBatchNormEx(4, 28, 28, 256, FORMAT_NHWC, kInTraining,
                               kWithSideInput, "Relu");
}

REGISTER_TYPED_TEST_SUITE_P(FusedBatchNormExOpTrainingTest,  //
                            TrainingInNHWCTest,              //
                            TrainingWithReluInNHWCTest,      //
                            TrainingWithSideInputAndReluInNHWCTest);

using FusedBatchNormExTrainingDataTypes = ::testing::Types<Eigen::half>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedBatchNormExOpTrainingTest,
                               FusedBatchNormExTrainingDataTypes);
#endif  // defined(GOOGLE_CUDA) && (CUDNN_VERSION >= 7402)

// -------------------------------------------------------------------------- //
// FusedBatchNormEx[is_training=false].

#if defined(GOOGLE_CUDA)
template <typename T>
using FusedBatchNormExOpInferenceTest =
    FusedBatchNormExOpTestBase<T, float>;  // scale is always float

TYPED_TEST_SUITE_P(FusedBatchNormExOpInferenceTest);

TYPED_TEST_P(FusedBatchNormExOpInferenceTest, InferenceInNHWCTest) {
  this->VerifyFusedBatchNormEx(4, 28, 28, 256, FORMAT_NHWC, kInInference,
                               kNoSideInput, "Identity");
}

TYPED_TEST_P(FusedBatchNormExOpInferenceTest, InferenceWithReluInNHWCTest) {
  this->VerifyFusedBatchNormEx(4, 28, 28, 256, FORMAT_NHWC, kInInference,
                               kNoSideInput, "Relu");
}

TYPED_TEST_P(FusedBatchNormExOpInferenceTest,
             InferenceWithSideInputAndReluInNHWCTest) {
  this->VerifyFusedBatchNormEx(4, 28, 28, 256, FORMAT_NHWC, kInInference,
                               kWithSideInput, "Relu");
}

REGISTER_TYPED_TEST_SUITE_P(FusedBatchNormExOpInferenceTest,  //
                            InferenceInNHWCTest,              //
                            InferenceWithReluInNHWCTest,      //
                            InferenceWithSideInputAndReluInNHWCTest);

using FusedBatchNormExInferenceDataTypes = ::testing::Types<Eigen::half, float>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedBatchNormExOpInferenceTest,
                               FusedBatchNormExInferenceDataTypes);
#endif  // defined(GOOGLE_CUDA)

// -------------------------------------------------------------------------- //
// Performance benchmarks are below.                                          //
// -------------------------------------------------------------------------- //

using fp16 = Eigen::half;
using fp32 = float;
using SideInputAndActivation = std::pair<bool, string>;

SideInputAndActivation Identity() { return {false, "Identity"}; }
SideInputAndActivation Relu() { return {false, "Relu"}; }
SideInputAndActivation AddAndRelu() { return {true, "Relu"}; }

template <typename T>
static Graph* FusedBatchNormEx(int n, int h, int w, int c,
                               TensorFormat data_format, bool is_training,
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
                  .Input(other)                        // scale
                  .Input(other)                        // offset
                  .Input(is_training ? empty : other)  // mean
                  .Input(is_training ? empty : other)  // variance
                  .Input(side_inputs)                  // side_input
                  .Attr("T", dtype)
                  .Attr("U", DT_FLOAT)
                  .Attr("epsilon", 0.001)
                  .Attr("data_format", ToString(data_format))
                  .Attr("activation_mode", activation_mode)
                  .Attr("num_side_inputs", num_side_inputs)
                  .Attr("is_training", is_training)
                  .Finalize(g, &fused_batch_norm));

  return g;
}

#define BM_CONCAT(a, b) a##_##b

#define BM_NAME(N, H, W, C, T, FORMAT, IS_TRAINING, A, DEVICE)          \
  BM_CONCAT(BM_FusedBatchNorm##_##DEVICE##_##T##_##N##_##H##_##W##_##C, \
            FORMAT##_##IS_TRAINING##_##A)

#define BM_FusedBatchNorm(N, H, W, C, T, FORMAT, IS_TRAINING, ACTIVATION,    \
                          DEVICE)                                            \
  static void BM_NAME(N, H, W, C, T, FORMAT, IS_TRAINING, ACTIVATION,        \
                      DEVICE)(::testing::benchmark::State & state) {         \
    test::Benchmark(#DEVICE,                                                 \
                    FusedBatchNormEx<T>(N, H, W, C, FORMAT_##FORMAT,         \
                                        IS_TRAINING, {ACTIVATION}),          \
                    /*old_benchmark_api*/ false)                             \
        .Run(state);                                                         \
    state.SetItemsProcessed(state.iterations() * N * H * W * C);             \
  }                                                                          \
  BENCHMARK(BM_NAME(N, H, W, C, T, FORMAT, IS_TRAINING, ACTIVATION, DEVICE)) \
      ->UseRealTime();

#if defined(GOOGLE_CUDA) && (CUDNN_VERSION >= 7402)
BM_FusedBatchNorm(64, 14, 14, 256, fp16, NHWC, true, Identity, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, NHWC, true, Relu, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, NHWC, true, AddAndRelu, gpu);
#endif  // defined(GOOGLE_CUDA) && (CUDNN_VERSION >= 7402)

#if defined(GOOGLE_CUDA)
BM_FusedBatchNorm(64, 14, 14, 256, fp16, NHWC, false, Identity, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, NHWC, false, Relu, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, NHWC, false, AddAndRelu, gpu);

BM_FusedBatchNorm(64, 14, 14, 256, fp16, NCHW, false, Identity, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, NCHW, false, Relu, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp16, NCHW, false, AddAndRelu, gpu);

BM_FusedBatchNorm(64, 14, 14, 256, fp32, NHWC, false, Identity, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp32, NHWC, false, Relu, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp32, NHWC, false, AddAndRelu, gpu);

BM_FusedBatchNorm(64, 14, 14, 256, fp32, NCHW, false, Identity, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp32, NCHW, false, Relu, gpu);
BM_FusedBatchNorm(64, 14, 14, 256, fp32, NCHW, false, AddAndRelu, gpu);
#endif  // defined(GOOGLE_CUDA)

}  // namespace tensorflow

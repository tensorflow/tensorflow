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

#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

template <typename T, typename U>
class FusedLayerNormOpTestBase : public OpsTestBase {
 protected:
  struct FusedLayerNormOutputs {
    Tensor y;
    Tensor reserve_space_1;
    Tensor reserve_space_2;
  };

  struct FusedLayerNormGradOutputs {
    Tensor x_backprop;
    Tensor scale_backprop;
    Tensor offset_backprop;
  };

  using GraphRunner = std::function<void(
      const Tensor& y_backprop, const Tensor& input_data,
      const Tensor& scale_data, const Tensor& offset_data,
      FusedLayerNormOutputs* forward, FusedLayerNormGradOutputs* backward)>;

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the outputs.
  void RunAndFetch(const tensorflow::Scope& root,
                   const std::vector<string>& fetch,
                   std::vector<Tensor>* outputs, bool allow_gpu_device) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

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

  void RunFusedLayerNorm(const Tensor& y_backprop_data,
                         const Tensor& input_data, const Tensor& scale_data,
                         const Tensor& offset_data,
                         FusedLayerNormOutputs* forward,
                         FusedLayerNormGradOutputs* backward, bool use_gpu,
                         float epsilon = 0.001f) {
    Scope root = tensorflow::Scope::NewRootScope();

    Output y_backprop = ops::Const(root.WithOpName("y_backprop"),
                                   Input::Initializer(y_backprop_data));
    Output input =
        ops::Const(root.WithOpName("input"), Input::Initializer(input_data));
    Output scale =
        ops::Const(root.WithOpName("scale"), Input::Initializer(scale_data));
    Output offset =
        ops::Const(root.WithOpName("offset"), Input::Initializer(offset_data));

    ops::FusedLayerNorm fwd =
        ops::FusedLayerNorm(root.WithOpName("fused_layer_norm"), input, scale,
                            offset, ops::FusedLayerNorm::Epsilon(epsilon));

    ops::FusedLayerNormGrad bwd = ops::FusedLayerNormGrad(
        root.WithOpName("fused_layer_norm_grad"), y_backprop, input, scale,
        fwd.reserve_space_1, fwd.reserve_space_2,
        ops::FusedLayerNormGrad::Epsilon(epsilon));

    std::vector<Tensor> out_tensors;
    RunAndFetch(root,
                {"fused_layer_norm:0", "fused_layer_norm:1",
                 "fused_layer_norm:2", "fused_layer_norm_grad:0",
                 "fused_layer_norm_grad:1", "fused_layer_norm_grad:2"},
                &out_tensors, /*allow_gpu_device=*/use_gpu);

    forward->y = out_tensors[0];
    forward->reserve_space_1 = out_tensors[1];
    forward->reserve_space_2 = out_tensors[2];

    backward->x_backprop = out_tensors[3];
    backward->scale_backprop = out_tensors[4];
    backward->offset_backprop = out_tensors[5];
  }

  void VerifyTensorsNear(int batches, int features, const GraphRunner& run_cpu,
                         const GraphRunner& run_gpu) {
    DataType t_dtype = DataTypeToEnum<T>::v();
    DataType u_dtype = DataTypeToEnum<U>::v();

    TensorShape input_shape = TensorShape({batches, features});

    Tensor input(t_dtype, input_shape);
    input.flat<T>().setRandom();
    input.flat<T>() -= input.flat<T>().constant(static_cast<T>(0.5));

    Tensor scale(u_dtype, {features});
    scale.flat<U>().setRandom();

    Tensor offset(u_dtype, {features});
    offset.flat<U>().setRandom();

    Tensor y_backprop(t_dtype, input_shape);
    y_backprop.flat<T>().setRandom();
    y_backprop.flat<T>() -= y_backprop.flat<T>().constant(static_cast<T>(0.5));

    FusedLayerNormOutputs fln_forward_cpu;
    FusedLayerNormOutputs fln_forward_gpu;

    FusedLayerNormGradOutputs fln_backward_cpu;
    FusedLayerNormGradOutputs fln_backward_gpu;

    run_cpu(y_backprop, input, scale, offset, &fln_forward_cpu,
            &fln_backward_cpu);

    run_gpu(y_backprop, input, scale, offset, &fln_forward_gpu,
            &fln_backward_gpu);

    std::vector<std::pair<Tensor, Tensor>> tensor_pairs;
    tensor_pairs = {
        {fln_forward_cpu.y, fln_forward_gpu.y},
        {fln_forward_cpu.reserve_space_1, fln_forward_gpu.reserve_space_1},
        {fln_forward_cpu.reserve_space_2, fln_forward_gpu.reserve_space_2},
        {fln_backward_cpu.x_backprop, fln_backward_gpu.x_backprop},
        {fln_backward_cpu.scale_backprop, fln_backward_gpu.scale_backprop},
        {fln_backward_cpu.offset_backprop, fln_backward_gpu.offset_backprop},
    };

    for (auto& pair : tensor_pairs) {
      const Tensor& fln_cpu = pair.first;
      const Tensor& fln_gpu = pair.second;

      ASSERT_EQ(fln_cpu.dtype(), fln_gpu.dtype());
      ASSERT_EQ(fln_cpu.shape(), fln_gpu.shape());

      test::ExpectClose(fln_cpu, fln_gpu, 1e-2);
    }
  }

  // Verifies that CPU and GPU results are close to each other.
  void VerifyFusedLayerNorm(int batches, int features) {
    const GraphRunner run_cpu =
        [&](const Tensor& y_backprop, const Tensor& input_data,
            const Tensor& scale_data, const Tensor& offset_data,
            FusedLayerNormOutputs* fwd, FusedLayerNormGradOutputs* bwd) {
          this->RunFusedLayerNorm(y_backprop, input_data, scale_data,
                                  offset_data, fwd, bwd, false);
        };

    const GraphRunner run_gpu =
        [&](const Tensor& y_backprop, const Tensor& input_data,
            const Tensor& scale_data, const Tensor& offset_data,
            FusedLayerNormOutputs* fwd, FusedLayerNormGradOutputs* bwd) {
          this->RunFusedLayerNorm(y_backprop, input_data, scale_data,
                                  offset_data, fwd, bwd, true);
        };

    VerifyTensorsNear(batches, features, run_cpu, run_gpu);
  }
};

#if defined(GOOGLE_CUDA)
template <typename T>
using FusedLayerNormOpTrainingTest =
    FusedLayerNormOpTestBase<T, float>;  // scale is always float

TYPED_TEST_SUITE_P(FusedLayerNormOpTrainingTest);

TYPED_TEST_P(FusedLayerNormOpTrainingTest, TrainingLargeRowTest) {
  this->VerifyFusedLayerNorm(4, 32 * 128 * 100 + 10);
}

TYPED_TEST_P(FusedLayerNormOpTrainingTest, TrainingMediumRowTest) {
  this->VerifyFusedLayerNorm(4, 32 * 128 * 100);
}

TYPED_TEST_P(FusedLayerNormOpTrainingTest, TrainingSmallRowTest) {
  this->VerifyFusedLayerNorm(4, 31);
}

TYPED_TEST_P(FusedLayerNormOpTrainingTest, TrainingLargeColTest) {
  this->VerifyFusedLayerNorm(32 * 10000, 127);
}

REGISTER_TYPED_TEST_SUITE_P(FusedLayerNormOpTrainingTest, TrainingLargeRowTest,
                            TrainingMediumRowTest, TrainingSmallRowTest,
                            TrainingLargeColTest);

using FusedLayerNormTrainingDataTypes = ::testing::Types<Eigen::half>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedLayerNormOpTrainingTest,
                               FusedLayerNormTrainingDataTypes);

#endif

class FusedLayerNormOpTest : public OpsTestBase {
 protected:
  void CommonFusedLayerNorm() {
    TF_EXPECT_OK(NodeDefBuilder("layer_norm_op", "FusedLayerNorm")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("epsilon", 0.001)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    AddInputFromArray<float>(TensorShape({3, 1, 4}),
                             {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5});
    AddInputFromArray<float>(TensorShape({4}), {4.0, 4.0, 4.0, 4.0});
    AddInputFromArray<float>(TensorShape({4}), {2.0, 2.0, 2.0, 2.0});

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), DT_FLOAT, TensorShape({3, 1, 4}));
    test::FillValues<float>(&expected, {-3.36, 0.21, 3.79, 7.36, -3.36, 0.21,
                                        3.79, 7.36, -3.36, 0.21, 3.79, 7.36});
    test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);

    Tensor expected_mean(allocator(), DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected_mean, {3.5, 3.5, 3.5});
    test::ExpectTensorNear<float>(expected_mean, *GetOutput(1), 0.01);

    Tensor expected_inv_variance(allocator(), DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected_inv_variance, {0.89, 0.89, 0.89});
    test::ExpectTensorNear<float>(expected_inv_variance, *GetOutput(2), 0.01);
  }
};

TEST_F(FusedLayerNormOpTest, Training) { this->CommonFusedLayerNorm(); }

#if defined(GOOGLE_CUDA)
TEST_F(FusedLayerNormOpTest, TrainingGPU) {
  SetDevice(DEVICE_GPU,
            std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                "GPU", {}, "/job:a/replica:0/task:0")));
  this->CommonFusedLayerNorm();
}
#endif  // defined(GOOGLE_CUDA)

TEST_F(FusedLayerNormOpTest, EmptyInput) {
  TF_EXPECT_OK(NodeDefBuilder("layer_norm_op", "FusedLayerNorm")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("epsilon", 0.001)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({0, 0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({0}), {});

  TF_ASSERT_OK(RunOpKernel());
  EXPECT_EQ(GetOutput(0)->shape(), TensorShape({0, 0}));
}

class FusedLayerNormGradOpTest : public OpsTestBase {
 protected:
  void CommonFusedLayerNormGrad() {
    TF_EXPECT_OK(NodeDefBuilder("layer_norm_grad_op", "FusedLayerNormGrad")
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("epsilon", 0.001)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    AddInputFromArray<float>(TensorShape({2, 1, 6}),
                             {2, 9, -4, 5, 8, 7, 2, 9, -4, 5, 8, 7});
    AddInputFromArray<float>(TensorShape({2, 1, 6}),
                             {1, 7, 4, -3, -11, 13, 1, 7, 4, -3, -11, 13});
    AddInputFromArray<float>(TensorShape({6}), {4.0, 4.0, 4.0, 4.0, 4.0, 4.0});
    AddInputFromArray<float>(TensorShape({2}), {1.83, 1.83});
    AddInputFromArray<float>(TensorShape({2}), {0.13, 0.13});

    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 1, 6}));
    test::FillValues<float>(&expected, {-1.32, 2.43, -4.38, 0.17, 1.58, 1.50,
                                        -1.32, 2.43, -4.38, 0.17, 1.58, 1.50});
    test::ExpectTensorNear<float>(expected, *GetOutput(0), 0.01);

    Tensor expected_dscale(allocator(), DT_FLOAT, TensorShape({6}));
    test::FillValues<float>(&expected_dscale,
                            {-0.43, 12.09, -2.26, -6.28, -26.69, 20.33});
    test::ExpectTensorNear<float>(expected_dscale, *GetOutput(1), 0.01);

    Tensor expected_doffset(allocator(), DT_FLOAT, TensorShape({6}));
    test::FillValues<float>(&expected_doffset, {4, 18, -8, 10, 16, 14});
    test::ExpectTensorNear<float>(expected_doffset, *GetOutput(2), 0.01);
  }
};

TEST_F(FusedLayerNormGradOpTest, Training) { this->CommonFusedLayerNormGrad(); }

#if defined(GOOGLE_CUDA)
TEST_F(FusedLayerNormGradOpTest, TrainingGPU) {
  SetDevice(DEVICE_GPU,
            std::unique_ptr<tensorflow::Device>(DeviceFactory::NewDevice(
                "GPU", {}, "/job:a/replica:0/task:0")));
  this->CommonFusedLayerNormGrad();
}
#endif  // defined(GOOGLE_CUDA)

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

using fp32 = float;
using fp16 = Eigen::half;

template <typename T>
static Graph* FusedLayerNormInference(int n, int d) {
  Graph* g = new Graph(OpRegistry::Global());

  DataType dtype = DataTypeToEnum<T>::value;
  Tensor x_t(dtype, TensorShape({n, d}));
  x_t.flat<T>().setRandom();

  Tensor other_t(DT_FLOAT, TensorShape({d}));
  other_t.flat<float>().setRandom();

  Node* x = test::graph::Constant(g, x_t, "x");
  Node* other = test::graph::Constant(g, other_t, "other");

  Node* fused_layer_norm;
  TF_CHECK_OK(NodeBuilder(g->NewName("fused_layer_norm"), "FusedLayerNorm")
                  .Input(x)
                  .Input(other)  // scale
                  .Input(other)  // offset
                  .Attr("T", dtype)
                  .Attr("U", DT_FLOAT)
                  .Attr("epsilon", 0.001)
                  .Finalize(g, &fused_layer_norm));

  return g;
}

template <typename T>
static Graph* FusedLayerNormGrad(int n, int d) {
  Graph* g = new Graph(OpRegistry::Global());

  DataType dtype = DataTypeToEnum<T>::value;
  TensorShape shape = TensorShape({n, d});

  Tensor y_backprop_t(dtype, shape);
  y_backprop_t.flat<T>().setRandom();

  Tensor x_t(dtype, shape);
  x_t.flat<T>().setRandom();

  Tensor scale_t(DT_FLOAT, TensorShape({d}));
  scale_t.flat<float>().setRandom();

  Tensor other_t(DT_FLOAT, TensorShape({n}));
  other_t.flat<float>().setRandom();

  Node* y_backprop = test::graph::Constant(g, y_backprop_t, "y_backprop");
  Node* x = test::graph::Constant(g, x_t, "x");
  Node* scale = test::graph::Constant(g, scale_t, "scale");
  Node* other = test::graph::Constant(g, other_t, "other");

  Node* fused_layer_norm;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("fused_layer_norm_grad"), "FusedLayerNormGrad")
          .Input(y_backprop)
          .Input(x)
          .Input(scale)
          .Input(other)  // saved_mean
          .Input(other)  // saved_inv_var
          .Attr("T", dtype)
          .Attr("U", DT_FLOAT)
          .Attr("epsilon", 0.001)
          .Finalize(g, &fused_layer_norm));

  return g;
}

#define BM_NAME(NAME, N, D, T, DEVICE) BM_##NAME##_##N##_##D##_##T##_##DEVICE

// -------------------------------------------------------------------------- //
// FusedLayerNorm inference
// -------------------------------------------------------------------------- //
// clang-format off
// NOLINTBEGIN
#define BM_FusedLayerNorm(N, D, T, DEVICE)         \
  static void BM_NAME(FusedLayerNorm, N, D, T, DEVICE)(::testing::benchmark::State & state) {                     \
    test::Benchmark(                                                          \
        #DEVICE,                                                              \
        FusedLayerNormInference<T>(N, D),                                     \
        /*old_benchmark_api*/ false)                                          \
        .Run(state);                                                          \
    state.SetItemsProcessed(state.iterations() * N * D);                      \
  }                                                                           \
  BENCHMARK(                                                                  \
      BM_NAME(FusedLayerNorm, N, D, T, DEVICE))    \
      ->UseRealTime();

// NOLINTEND
// clang-format on

BM_FusedLayerNorm(64, 25600, fp32, cpu);
BM_FusedLayerNorm(64, 25600, fp16, cpu);

#ifdef GOOGLE_CUDA
BM_FusedLayerNorm(64, 25600, fp32, gpu);
BM_FusedLayerNorm(64, 25600, fp16, gpu);
#endif  // GOOGLE_CUDA

#define BM_FusedLayerNormGrad(N, D, T, DEVICE)                       \
  static void BM_NAME(FusedLayerNormGrad, N, D, T,                   \
                      DEVICE)(::testing::benchmark::State & state) { \
    test::Benchmark(#DEVICE, FusedLayerNormGrad<T>(N, D),            \
                    /*old_benchmark_api*/ false)                     \
        .Run(state);                                                 \
    state.SetItemsProcessed(state.iterations() * N * D);             \
  }                                                                  \
  BENCHMARK(BM_NAME(FusedLayerNormGrad, N, D, T, DEVICE))->UseRealTime();

BM_FusedLayerNormGrad(64, 25600, fp32, cpu);
BM_FusedLayerNormGrad(64, 25600, fp16, cpu);

#ifdef GOOGLE_CUDA
BM_FusedLayerNormGrad(64, 25600, fp32, gpu);
BM_FusedLayerNormGrad(64, 25600, fp16, gpu);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow

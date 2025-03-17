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

#include <functional>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

namespace tensorflow {
namespace {

template <typename T>
class FusedMatMulOpTest : public OpsTestBase {
 protected:
  static constexpr auto kTValueType = DataTypeToEnum<T>::value;

  using BiasAddGraphRunner =
      std::function<bool(const Tensor& lhs_data, const Tensor& rhs_data,
                         const Tensor& bias_data, Tensor* out)>;

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the output Tensor. Optional `fetch_node` parameter
  // allows to define a fetch node directly using a NodeDef for the ops that are
  // not supported by the C++ Api.
  void RunAndFetch(const tensorflow::Scope& root, const string& fetch,
                   Tensor* output, bool allow_gpu_device,
                   const NodeDef* fetch_node = nullptr,
                   absl::Status* last_status = nullptr) {
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

    // If fused computation implemented only for CPU, in this test we don't want
    // to compare GPU vs CPU numbers, so place all nodes on CPU in this case.
    const bool place_all_on_gpu = allow_gpu_device && has_gpu_device;

    const string device = place_all_on_gpu ? "/device:GPU:0" : "/device:CPU:0";
    for (NodeDef& mutable_node : *graph.mutable_node()) {
      mutable_node.set_device(device);
    }

    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    auto res = session->Run({}, {fetch}, {}, &unfused_tensors);
    if (last_status != nullptr) {
      *last_status = res;
    } else {
      TF_ASSERT_OK(res);
    }
    if (!unfused_tensors.empty()) {
      *output = unfused_tensors[0];
    }
  }

  void RunMatMulWithBias(const Tensor& lhs_data, const Tensor& rhs_data,
                         const Tensor& bias_data, bool transpose_a,
                         bool transpose_b, Tensor* output,
                         bool allow_gpu_device = false) {
    Scope root = tensorflow::Scope::NewRootScope();

    ops::MatMul matmul = ops::MatMul(
        root.WithOpName("matmul"),
        ops::Const(root.WithOpName("lhs"), Input::Initializer(lhs_data)),
        ops::Const(root.WithOpName("rhs"), Input::Initializer(rhs_data)),
        ops::MatMul::Attrs().TransposeA(transpose_a).TransposeB(transpose_b));

    ops::BiasAdd with_bias = ops::BiasAdd(
        root.WithOpName("with_bias"), matmul,
        ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));

    RunAndFetch(root, "with_bias", output, allow_gpu_device);
  }

  void RunMatMulWithBiasAndActivation(
      const Tensor& lhs_data, const Tensor& rhs_data, const Tensor& bias_data,
      bool transpose_a, bool transpose_b, const string& activation_type,
      Tensor* output, bool allow_gpu_device = false) {
    Scope root = tensorflow::Scope::NewRootScope();

    ops::MatMul matmul = ops::MatMul(
        root.WithOpName("matmul"),
        ops::Const(root.WithOpName("lhs"), Input::Initializer(lhs_data)),
        ops::Const(root.WithOpName("rhs"), Input::Initializer(rhs_data)),
        ops::MatMul::Attrs().TransposeA(transpose_a).TransposeB(transpose_b));

    ops::BiasAdd with_bias = ops::BiasAdd(
        root.WithOpName("with_bias"), matmul,
        ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));

    if (activation_type == "Relu") {
      ops::Relu(root.WithOpName("with_activation"), with_bias);
    } else if (activation_type == "Relu6") {
      ops::Relu6(root.WithOpName("with_activation"), with_bias);
    } else if (activation_type == "Elu") {
      ops::Elu(root.WithOpName("with_activation"), with_bias);
    } else if (activation_type == "LeakyRelu") {
      ops::internal::LeakyRelu(root.WithOpName("with_activation"), with_bias);
    } else if (activation_type == "GeluExact") {
      VLOG(0) << "ERROR: GeluExact is yet not available!!";
      ops::Identity(root.WithOpName("with_activation"), with_bias);
    } else if (activation_type == "Sigmoid") {
      ops::Sigmoid(root.WithOpName("with_activation"), with_bias);
    } else if (activation_type == "Tanh") {
      ops::Tanh(root.WithOpName("with_activation"), with_bias);
    } else {
      ops::Identity(root.WithOpName("with_activation"), with_bias);
    }

    RunAndFetch(root, "with_activation", output, allow_gpu_device);
  }

  void RunFusedMatMulOp(const Tensor& lhs_data, const Tensor& rhs_data,
                        const std::vector<Tensor>& args_data,
                        const std::vector<string>& fused_ops, bool transpose_a,
                        bool transpose_b, Tensor* output,
                        bool allow_gpu_device = false,
                        bool* test_skipped = nullptr) {
    Scope root = tensorflow::Scope::NewRootScope();

    DataType dtype = DataTypeToEnum<T>::v();
    int num_args = static_cast<int>(args_data.size());

    Output lhs =
        ops::Const(root.WithOpName("lhs"), Input::Initializer(lhs_data));
    Output rhs =
        ops::Const(root.WithOpName("rhs"), Input::Initializer(rhs_data));

    std::vector<NodeDefBuilder::NodeOut> args;
    for (int i = 0; i < num_args; ++i) {
      Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                              Input::Initializer(args_data[i]));
      args.emplace_back(arg.name(), 0, dtype);
    }

    NodeDef fused_matmul;
    TF_EXPECT_OK(NodeDefBuilder("fused_matmul", "_FusedMatMul")
                     .Input({lhs.name(), 0, dtype})
                     .Input({rhs.name(), 0, dtype})
                     .Input(args)
                     .Attr("num_args", num_args)
                     .Attr("T", dtype)
                     .Attr("fused_ops", fused_ops)
                     .Attr("transpose_a", transpose_a)
                     .Attr("transpose_b", transpose_b)
                     .Finalize(&fused_matmul));

    absl::Status last_status;
    RunAndFetch(root, fused_matmul.name(), output, allow_gpu_device,
                &fused_matmul, &last_status);

    std::string what = "No algorithm worked!";
    bool skip = absl::StrContains(last_status.message(), what);
    if (test_skipped != nullptr) {
      *test_skipped = skip;
    }
    if (skip) {
      GTEST_SKIP() << what;
    } else {
      TF_ASSERT_OK(last_status);
    }
  }

  void VerifyBiasAddTensorsNear(int m, int k, int n, bool transpose_a,
                                bool transpose_b,
                                const BiasAddGraphRunner& run_default,
                                const BiasAddGraphRunner& run_fused) {
    DataType dtype = DataTypeToEnum<T>::v();

    Tensor lhs(dtype, {transpose_a ? k : m, transpose_a ? m : k});
    lhs.flat<T>() = lhs.flat<T>().setRandom();

    // Add some negative values to filter to properly test Relu.
    Tensor rhs(dtype, {transpose_b ? n : k, transpose_b ? k : n});
    rhs.flat<T>() = rhs.flat<T>().setRandom();
    rhs.flat<T>() -= rhs.flat<T>().constant(static_cast<T>(0.5f));

    // Bias added to the inner dimension.
    const int bias_size = n;
    Tensor bias(dtype, {bias_size});
    bias.flat<T>() = bias.flat<T>().setRandom();
    bias.flat<T>() += bias.flat<T>().constant(static_cast<T>(0.5f));

    Tensor matmul;
    Tensor fused_matmul;

    run_default(lhs, rhs, bias, &matmul);
    bool skipped = run_fused(lhs, rhs, bias, &fused_matmul);

    if (!skipped) {
      ASSERT_EQ(matmul.dtype(), fused_matmul.dtype());
      ASSERT_EQ(matmul.shape(), fused_matmul.shape());

      // use specific rtol value for DT_HALF datatype and the default one for
      // all others
      double atol = this->kTValueType == DT_HALF ? 1e-3 : 1e-5;
      double rtol = this->kTValueType == DT_HALF ? 1e-3 : -1.0;
      test::ExpectClose(matmul, fused_matmul, atol, rtol);
    }
  }

  // Verifies that computing MatMul+BiasAdd in a graph is identical to
  // FusedMatMul.
  void VerifyMatMulWithBias(int m, int k, int n, bool transpose_a,
                            bool transpose_b) {
    VLOG(2) << "=== VerifyMatMulWithBias (" << m << ", " << k << ", " << n
            << ", " << (int)transpose_a << ", " << (int)transpose_b << ") ===";

    const BiasAddGraphRunner run_default =
        [&](const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
          RunMatMulWithBias(input_data, filter_data, bias_data, transpose_a,
                            transpose_b, out, /*allow_gpu_device=*/true);
          return false;
        };

    const BiasAddGraphRunner run_fused =
        [&](const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
          bool skipped = false;
          RunFusedMatMulOp(input_data, filter_data, {bias_data}, {"BiasAdd"},
                           transpose_a, transpose_b, out,
                           /*allow_gpu_device=*/true, &skipped);
          return skipped;
        };

    VerifyBiasAddTensorsNear(m, k, n, transpose_a, transpose_b, run_default,
                             run_fused);
  }

  // Verifies that computing MatMul+BiasAdd+{Activation} in a graph is identical
  // to FusedMatMul.
  void VerifyConv2DWithBiasAndActivation(int m, int k, int n, bool transpose_a,
                                         bool transpose_b,
                                         const string& activation) {
    bool use_gpu_device =
        activation == "Relu" || (this->kTValueType == DT_HALF);
    const BiasAddGraphRunner run_default =
        [&](const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
          RunMatMulWithBiasAndActivation(input_data, filter_data, bias_data,
                                         transpose_a, transpose_b, activation,
                                         out, use_gpu_device);
          return false;
        };

    const BiasAddGraphRunner run_fused =
        [&](const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
          bool skipped = false;
          RunFusedMatMulOp(input_data, filter_data, {bias_data},
                           {"BiasAdd", activation}, transpose_a, transpose_b,
                           out, use_gpu_device, &skipped);
          return skipped;
        };

    VerifyBiasAddTensorsNear(m, k, n, transpose_a, transpose_b, run_default,
                             run_fused);
  }
};

// MatMul with BatchNorm can be tested only with `T=float`, because default
// `FusedBatchNorm` kernel supports only floats for scale, mean and variance.

template <typename T>
class FusedMatMulWithBiasOpTest : public FusedMatMulOpTest<T> {};

TYPED_TEST_SUITE_P(FusedMatMulWithBiasOpTest);

// -------------------------------------------------------------------------- //
// MatMul + BiasAdd + {Activation}                                            //
// -------------------------------------------------------------------------- //

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x128x64) {
  this->VerifyMatMulWithBias(256, 128, 64, false, false);
  this->VerifyMatMulWithBias(256, 128, 64, true, false);
  this->VerifyMatMulWithBias(256, 128, 64, false, true);
  this->VerifyMatMulWithBias(256, 128, 64, true, true);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x256) {
  this->VerifyMatMulWithBias(1, 256, 256, false, false);
  this->VerifyMatMulWithBias(1, 256, 256, true, false);
  this->VerifyMatMulWithBias(1, 256, 256, false, true);
  this->VerifyMatMulWithBias(1, 256, 256, true, true);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x256x1) {
  this->VerifyMatMulWithBias(256, 256, 1, false, false);
  this->VerifyMatMulWithBias(256, 256, 1, true, false);
  this->VerifyMatMulWithBias(256, 256, 1, false, true);
  this->VerifyMatMulWithBias(256, 256, 1, true, true);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x1) {
  this->VerifyMatMulWithBias(1, 256, 1, false, false);
}

static auto GetActivations(DataType dtype) {
  // "GeluExact", "Tanh", "Sigmoid" fusions are only supported for half-float
  // datatype
  switch (dtype) {
    case DT_HALF:
      // TODO: not sure how to add GeluExact op ??
      return std::vector{/*"GeluExact",*/ "Tanh", "Sigmoid"};
    default:
      return std::vector{"Relu", "Relu6", "Elu", "LeakyRelu"};
  }
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x128x64WithActivation) {
  for (const string& activation : GetActivations(this->kTValueType)) {
    this->VerifyConv2DWithBiasAndActivation(256, 128, 64, false, false,
                                            activation);
    this->VerifyConv2DWithBiasAndActivation(256, 128, 64, true, false,
                                            activation);
    this->VerifyConv2DWithBiasAndActivation(256, 128, 64, false, true,
                                            activation);
    this->VerifyConv2DWithBiasAndActivation(256, 128, 64, true, true,
                                            activation);
  }
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x256WithActivation) {
  for (const string& activation : GetActivations(this->kTValueType)) {
    this->VerifyConv2DWithBiasAndActivation(1, 256, 256, false, false,
                                            activation);
  }
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x256x1WithActivation) {
  for (const string& activation : GetActivations(this->kTValueType)) {
    this->VerifyConv2DWithBiasAndActivation(256, 256, 1, false, false,
                                            activation);
  }
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x1WithActivation) {
  for (const string& activation : GetActivations(this->kTValueType)) {
    this->VerifyConv2DWithBiasAndActivation(1, 256, 1, false, false,
                                            activation);
  }
}

REGISTER_TYPED_TEST_SUITE_P(FusedMatMulWithBiasOpTest,       //
                            MatMul256x128x64,                //
                            MatMul1x256x256,                 //
                            MatMul256x256x1,                 //
                            MatMul1x256x1,                   //
                            MatMul256x128x64WithActivation,  //
                            MatMul1x256x256WithActivation,   //
                            MatMul256x256x1WithActivation,   //
                            MatMul1x256x1WithActivation);

// TODO(ezhulenev): Add support for more data types.
using FusedBiasAddDataTypes = ::testing::Types<float, Eigen::half>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedMatMulWithBiasOpTest,
                               FusedBiasAddDataTypes);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* Matmul(int m, int k, int n, bool transpose_a, bool transpose_b,
                     DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
  in1.flat<T>().setRandom();
  test::graph::Matmul(g, test::graph::Constant(g, in0),
                      test::graph::Constant(g, in1), transpose_a, transpose_b);
  return g;
}

#define BM_MatmulDev(M, K, N, TA, TB, T, TFTYPE, DEVICE)                       \
  static void BM_Matmul##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE( \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, Matmul<T>(M, K, N, TA, TB, TFTYPE)).Run(state);   \
    state.SetItemsProcessed(state.iterations() * M * K * N * 2);               \
  }                                                                            \
  BENCHMARK(BM_Matmul##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE)   \
      ->MeasureProcessCPUTime();

#ifdef GOOGLE_CUDA

#define BM_Matmul(M, K, N, TA, TB)                                       \
  BM_MatmulDev(M, K, N, TA, TB, float, DT_FLOAT, cpu);                   \
  BM_MatmulDev(M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64, cpu); \
  BM_MatmulDev(M, K, N, TA, TB, float, DT_FLOAT, gpu);                   \
  BM_MatmulDev(M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64, gpu); \
  /* Uncomment to enable benchmarks for double/complex128: */            \
  // BM_MatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, cpu);                   \
// BM_MatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, cpu); \
// BM_MatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, gpu);                   \
// BM_MatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, gpu);

#else

#define BM_Matmul(M, K, N, TA, TB)                     \
  BM_MatmulDev(M, K, N, TA, TB, float, DT_FLOAT, cpu); \
  BM_MatmulDev(M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64, cpu);

#endif  // GOOGLE_CUDA

// LINT.IfChange

// Batch size of 1 included for inference.
// Typical fully connected layers
BM_Matmul(1, 512, 512, false, false);
BM_Matmul(8, 512, 512, false, false);
BM_Matmul(16, 512, 512, false, false);
BM_Matmul(128, 512, 512, false, false);

BM_Matmul(1, 1024, 1024, false, false);
BM_Matmul(8, 1024, 1024, false, false);
BM_Matmul(16, 1024, 1024, false, false);
BM_Matmul(128, 1024, 1024, false, false);
BM_Matmul(4096, 4096, 4096, false, false);

// Backward for fully connected layers
BM_Matmul(1, 1024, 1024, false, true);
BM_Matmul(8, 1024, 1024, false, true);
BM_Matmul(16, 1024, 1024, false, true);
BM_Matmul(128, 1024, 1024, false, true);

// Forward softmax with large output size
BM_Matmul(1, 200, 10000, false, false);
BM_Matmul(8, 200, 10000, false, false);
BM_Matmul(20, 200, 10000, false, false);
BM_Matmul(20, 200, 20000, false, false);

// Backward softmax with large output size
BM_Matmul(1, 10000, 200, false, true);
BM_Matmul(1, 10000, 200, false, false);
BM_Matmul(8, 10000, 200, false, true);
BM_Matmul(20, 10000, 200, false, true);
BM_Matmul(20, 20000, 200, false, true);

// Test some matrix-vector multiplies.
BM_Matmul(50, 50, 1, false, false);
BM_Matmul(50, 50, 1, true, false);
BM_Matmul(50, 50, 1, false, true);
BM_Matmul(50, 50, 1, true, true);
BM_Matmul(500, 500, 1, false, false);
BM_Matmul(500, 500, 1, true, false);
BM_Matmul(500, 500, 1, false, true);
BM_Matmul(500, 500, 1, true, true);
BM_Matmul(2000, 2000, 1, false, false);
BM_Matmul(2000, 2000, 1, true, false);
BM_Matmul(2000, 2000, 1, false, true);
BM_Matmul(2000, 2000, 1, true, true);

// Test some vector-matrix multiplies.
BM_Matmul(1, 50, 50, false, false);
BM_Matmul(1, 50, 50, true, false);
BM_Matmul(1, 50, 50, false, true);
BM_Matmul(1, 50, 50, true, true);
BM_Matmul(1, 500, 500, false, false);
BM_Matmul(1, 500, 500, true, false);
BM_Matmul(1, 500, 500, false, true);
BM_Matmul(1, 500, 500, true, true);
BM_Matmul(1, 2000, 2000, false, false);
BM_Matmul(1, 2000, 2000, true, false);
BM_Matmul(1, 2000, 2000, false, true);
BM_Matmul(1, 2000, 2000, true, true);

// Test some rank-one products.
BM_Matmul(50, 1, 50, false, false);
BM_Matmul(50, 1, 50, true, false);
BM_Matmul(50, 1, 50, false, true);
BM_Matmul(50, 1, 50, true, true);
BM_Matmul(500, 1, 500, false, false);
BM_Matmul(500, 1, 500, true, false);
BM_Matmul(500, 1, 500, false, true);
BM_Matmul(500, 1, 500, true, true);
BM_Matmul(2000, 1, 2000, false, false);
BM_Matmul(2000, 1, 2000, true, false);
BM_Matmul(2000, 1, 2000, false, true);
BM_Matmul(2000, 1, 2000, true, true);

// LINT.ThenChange(//tensorflow/core/kernels/mkl/mkl_matmul_op_benchmark.cc)

// Benchmarks for batched matmul with broadcasting.
Node* BroadcastTo(Graph* g, Node* input, Node* shape) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "BroadcastTo")
                  .Input(input)
                  .Input(shape)
                  .Finalize(g, &ret));
  return ret;
}

Node* BatchMatmulV2(Graph* g, Node* in0, Node* in1, bool adj_x, bool adj_y) {
  Node* ret;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "BatchMatMulV2")
                  .Input(in0)
                  .Input(in1)
                  .Attr("adj_x", adj_x)
                  .Attr("adj_y", adj_y)
                  .Finalize(g, &ret));
  return ret;
}

template <typename T>
static Graph* BatchMatmul(int b, int m, int k, int n, bool adjoint_a,
                          bool adjoint_b, DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, adjoint_a ? TensorShape({b, k, m}) : TensorShape({b, m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, adjoint_b ? TensorShape({b, n, k}) : TensorShape({b, k, n}));
  in1.flat<T>().setRandom();
  test::graph::BatchMatmul(g, test::graph::Constant(g, in0),
                           test::graph::Constant(g, in1), adjoint_a, adjoint_b);
  return g;
}

template <typename T>
static Graph* BatchMatmulWithBroadcast(int b0, int b1, int m, int k, int n,
                                       bool manual_broadcast, DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, TensorShape({b0, m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, TensorShape({b1, k, n}));
  in1.flat<T>().setRandom();

  Tensor broadcasted_in0_shape(DT_INT64, TensorShape({3}));
  Tensor broadcasted_in1_shape(DT_INT64, TensorShape({3}));

  Node* in0_node = nullptr;
  Node* in1_node = nullptr;
  if (manual_broadcast) {
    for (int i = 0; i < 3; ++i) {
      auto vec0 = broadcasted_in0_shape.vec<int64_t>();
      auto vec1 = broadcasted_in1_shape.vec<int64_t>();
      vec0(i) = (i == 0 ? std::max(b0, b1) : in0.shape().dim_size(i));
      vec1(i) = (i == 0 ? std::max(b0, b1) : in1.shape().dim_size(i));
    }
    in0_node = BroadcastTo(g, test::graph::Constant(g, in0),
                           test::graph::Constant(g, broadcasted_in0_shape));
    in1_node = BroadcastTo(g, test::graph::Constant(g, in1),
                           test::graph::Constant(g, broadcasted_in1_shape));
  } else {
    in0_node = test::graph::Constant(g, in0);
    in1_node = test::graph::Constant(g, in1);
  }

  BatchMatmulV2(g, in0_node, in1_node, false, false);
  return g;
}

// NOLINTBEGIN
// Function names are already longer than 80 chars.
#define BM_BatchMatmulDev(B, M, K, N, TA, TB, T, TFTYPE, DEVICE)                  \
  static void                                                                     \
      BM_BatchMatmul##_##B##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE( \
          ::testing::benchmark::State& state) {                                   \
    test::Benchmark(#DEVICE, BatchMatmul<T>(B, M, K, N, TA, TB, TFTYPE),          \
                    /*old_benchmark_api*/ false)                                  \
        .Run(state);                                                              \
    state.SetItemsProcessed(state.iterations() * B * M * K * N * 2);              \
  }                                                                               \
  BENCHMARK(                                                                      \
      BM_BatchMatmul##_##B##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE) \
      ->MeasureProcessCPUTime();
// NOLINTEND

#define BM_BatchMatmul(B, M, K, N, TA, TB) \
  BM_BatchMatmulDev(B, M, K, N, TA, TB, float, DT_FLOAT, cpu);
// BM_BatchMatmulDev(B, M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64,
// cpu);
//  BM_BatchMatmulDev(B, M, K, N, TA, TB, float, DT_FLOAT, gpu);
/* Uncomment to enable benchmarks for double & complex types: */
// BM_BatchMatmulDev(B, M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64,
// gpu);
// BM_BatchMatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, cpu); \
// BM_BatchMatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, cpu);
// \
// BM_BatchMatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, gpu); \
// BM_BatchMatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, gpu);

// Macro arguments names: --------------------------------------------------- //
//   B1: batch size of LHS
//   B2: batch size of RHS
//    M: outer dimension of LHS
//    K: inner dimensions of LHS and RHS
//    N: outer dimension of RHS
//   MB: boolean indicating whether to use manual broadcasting
//    T: C++ type of scalars (e.g. float, std::complex)
//   TT: TensorFlow type of scalars (e.g. DT_FLOAT, DT_COMPLEX128
//    D: Device (e.g. cpu, gpu)
#define BM_BatchMatmulBCastDev(B1, B2, M, K, N, MB, T, TT, D)                  \
  static void                                                                  \
      BM_BatchMatmulBCast##_##B1##_##B2##_##M##_##K##_##N##_##MB##_##TT##_##D( \
          ::testing::benchmark::State& state) {                                \
    test::Benchmark(#D, BatchMatmulWithBroadcast<T>(B1, B2, M, K, N, MB, TT),  \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(state.iterations() * std::max(B1, B2) * M * K *    \
                            N * 2);                                            \
  }                                                                            \
  BENCHMARK(                                                                   \
      BM_BatchMatmulBCast##_##B1##_##B2##_##M##_##K##_##N##_##MB##_##TT##_##D) \
      ->MeasureProcessCPUTime();

#define BM_BatchMatmulBCast(B1, B2, M, K, N, MB) \
  BM_BatchMatmulBCastDev(B1, B2, M, K, N, MB, float, DT_FLOAT, cpu);

// Typical fully connected layers
BM_BatchMatmulBCast(1, 128, 1, 1024, 1024, true);
BM_BatchMatmulBCast(1, 128, 1, 1024, 1024, false);
BM_BatchMatmulBCast(128, 1, 1, 1024, 1024, true);
BM_BatchMatmulBCast(128, 1, 1, 1024, 1024, false);
BM_BatchMatmulBCast(1, 128, 128, 1024, 1024, true);
BM_BatchMatmulBCast(1, 128, 128, 1024, 1024, false);
BM_BatchMatmulBCast(128, 1, 128, 1024, 1024, true);
BM_BatchMatmulBCast(128, 1, 128, 1024, 1024, false);

// Square matmul.
BM_BatchMatmulBCast(1, 128, 512, 512, 512, true);
BM_BatchMatmulBCast(1, 128, 512, 512, 512, false);
BM_BatchMatmulBCast(128, 1, 512, 512, 512, true);
BM_BatchMatmulBCast(128, 1, 512, 512, 512, false);
BM_BatchMatmulBCast(1, 128, 1024, 1024, 1024, true);
BM_BatchMatmulBCast(1, 128, 1024, 1024, 1024, false);
BM_BatchMatmulBCast(128, 1, 1024, 1024, 1024, true);
BM_BatchMatmulBCast(128, 1, 1024, 1024, 1024, false);

// Matrix-vector multiplies.
BM_BatchMatmulBCast(1, 128, 10000, 200, 1, true);
BM_BatchMatmulBCast(1, 128, 10000, 200, 1, false);
BM_BatchMatmulBCast(128, 1, 10000, 200, 1, true);
BM_BatchMatmulBCast(128, 1, 10000, 200, 1, false);

// Vector-matrix multiplies.
BM_BatchMatmulBCast(1, 128, 1, 200, 10000, true);
BM_BatchMatmulBCast(1, 128, 1, 200, 10000, false);
BM_BatchMatmulBCast(128, 1, 1, 200, 10000, true);
BM_BatchMatmulBCast(128, 1, 1, 200, 10000, false);

// Typical fully connected layers
BM_BatchMatmul(1, 1, 1024, 1024, false, false);
BM_BatchMatmul(1, 8, 1024, 1024, false, false);
BM_BatchMatmul(1, 16, 1024, 1024, false, false);
BM_BatchMatmul(1, 128, 1024, 1024, false, false);
BM_BatchMatmul(2, 1, 1024, 1024, false, false);
BM_BatchMatmul(2, 8, 1024, 1024, false, false);
BM_BatchMatmul(2, 16, 1024, 1024, false, false);
BM_BatchMatmul(2, 128, 1024, 1024, false, false);
BM_BatchMatmul(8, 1, 1024, 1024, false, false);
BM_BatchMatmul(8, 8, 1024, 1024, false, false);
BM_BatchMatmul(8, 16, 1024, 1024, false, false);
BM_BatchMatmul(8, 128, 1024, 1024, false, false);
BM_BatchMatmul(32, 1, 1024, 1024, false, false);
BM_BatchMatmul(32, 8, 1024, 1024, false, false);
BM_BatchMatmul(32, 16, 1024, 1024, false, false);
BM_BatchMatmul(32, 128, 1024, 1024, false, false);

// Square matmul.
BM_BatchMatmul(1, 32, 32, 32, false, false);
BM_BatchMatmul(1, 128, 128, 128, false, false);
BM_BatchMatmul(1, 256, 256, 256, false, false);
BM_BatchMatmul(1, 1024, 1024, 1024, false, false);
BM_BatchMatmul(1, 2048, 2048, 2048, false, false);
BM_BatchMatmul(2, 32, 32, 32, false, false);
BM_BatchMatmul(2, 128, 128, 128, false, false);
BM_BatchMatmul(2, 256, 256, 256, false, false);
BM_BatchMatmul(2, 1024, 1024, 1024, false, false);
BM_BatchMatmul(2, 2048, 2048, 2048, false, false);
BM_BatchMatmul(4, 32, 32, 32, false, false);
BM_BatchMatmul(4, 128, 128, 128, false, false);
BM_BatchMatmul(4, 256, 256, 256, false, false);
BM_BatchMatmul(4, 1024, 1024, 1024, false, false);
BM_BatchMatmul(4, 2048, 2048, 2048, false, false);
BM_BatchMatmul(8, 32, 32, 32, false, false);
BM_BatchMatmul(8, 128, 128, 128, false, false);
BM_BatchMatmul(8, 256, 256, 256, false, false);
BM_BatchMatmul(8, 1024, 1024, 1024, false, false);
BM_BatchMatmul(8, 2048, 2048, 2048, false, false);
BM_BatchMatmul(32, 32, 32, 32, false, false);
BM_BatchMatmul(32, 128, 128, 128, false, false);
BM_BatchMatmul(32, 256, 256, 256, false, false);
BM_BatchMatmul(32, 1024, 1024, 1024, false, false);
BM_BatchMatmul(32, 2048, 2048, 2048, false, false);

// Matrix-vector multiplies.
BM_BatchMatmul(1, 10000, 200, 1, false, false);
BM_BatchMatmul(8, 10000, 200, 1, false, false);
BM_BatchMatmul(32, 10000, 200, 1, false, false);
BM_BatchMatmul(1, 10000, 200, 1, true, false);
BM_BatchMatmul(8, 10000, 200, 1, true, false);
BM_BatchMatmul(32, 10000, 200, 1, true, false);
BM_BatchMatmul(1, 10000, 200, 1, false, true);
BM_BatchMatmul(8, 10000, 200, 1, false, true);
BM_BatchMatmul(32, 10000, 200, 1, false, true);
BM_BatchMatmul(1, 10000, 200, 1, true, true);
BM_BatchMatmul(8, 10000, 200, 1, true, true);
BM_BatchMatmul(32, 10000, 200, 1, true, true);

// Vector-matrix multiplies.
BM_BatchMatmul(1, 1, 200, 10000, false, false);
BM_BatchMatmul(8, 1, 200, 10000, false, false);
BM_BatchMatmul(32, 1, 200, 10000, false, false);
BM_BatchMatmul(1, 1, 200, 10000, true, false);
BM_BatchMatmul(8, 1, 200, 10000, true, false);
BM_BatchMatmul(32, 1, 200, 10000, true, false);
BM_BatchMatmul(1, 1, 200, 10000, false, true);
BM_BatchMatmul(8, 1, 200, 10000, false, true);
BM_BatchMatmul(32, 1, 200, 10000, false, true);
BM_BatchMatmul(1, 1, 200, 10000, true, true);
BM_BatchMatmul(8, 1, 200, 10000, true, true);
BM_BatchMatmul(32, 1, 200, 10000, true, true);

}  // namespace
}  // namespace tensorflow

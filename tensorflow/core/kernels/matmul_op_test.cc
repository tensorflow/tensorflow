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

#include "absl/algorithm/container.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

template <typename T>
class FusedMatMulOpTest : public OpsTestBase {
 protected:
  using BiasAddGraphRunner =
      std::function<void(const Tensor& lhs_data, const Tensor& rhs_data,
                         const Tensor& bias_data, Tensor* out)>;

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the output Tensor. Optional `fetch_node` parameter
  // allows to define a fetch node directly using a NodeDef for the ops that are
  // not supported by the C++ Api.
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

    // If fused computation implemented only for CPU, in this test we don't want
    // to compare GPU vs CPU numbers, so place all nodes on CPU in this case.
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
    } else {
      ops::Identity(root.WithOpName("with_activation"), with_bias);
    }

    RunAndFetch(root, "with_activation", output, allow_gpu_device);
  }

  void RunFusedMatMulOp(const Tensor& lhs_data, const Tensor& rhs_data,
                        const std::vector<Tensor>& args_data,
                        const std::vector<string>& fused_ops, bool transpose_a,
                        bool transpose_b, Tensor* output,
                        bool allow_gpu_device = false) {
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

    RunAndFetch(root, fused_matmul.name(), output, allow_gpu_device,
                &fused_matmul);
  }

  void VerifyBiasAddTensorsNear(int m, int k, int n,
                                const BiasAddGraphRunner& run_default,
                                const BiasAddGraphRunner& run_fused) {
    DataType dtype = DataTypeToEnum<T>::v();

    Tensor lhs(dtype, {m, k});
    lhs.flat<T>() = lhs.flat<T>().setRandom();

    // Add some negative values to filter to properly test Relu.
    Tensor rhs(dtype, {k, n});
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
    run_fused(lhs, rhs, bias, &fused_matmul);

    ASSERT_EQ(matmul.dtype(), fused_matmul.dtype());
    ASSERT_EQ(matmul.shape(), fused_matmul.shape());

    test::ExpectClose(matmul, fused_matmul, /*atol=*/1e-5);
  }

  // Verifies that computing MatMul+BiasAdd in a graph is identical to
  // FusedMatMul.
  void VerifyMatMulWithBias(int m, int k, int n, bool transpose_a,
                            bool transpose_b) {
    const BiasAddGraphRunner run_default =
        [&](const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
          RunMatMulWithBias(input_data, filter_data, bias_data, transpose_a,
                            transpose_b, out);
        };

    const BiasAddGraphRunner run_fused =
        [&](const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
          RunFusedMatMulOp(input_data, filter_data, {bias_data}, {"BiasAdd"},
                           transpose_a, transpose_b, out);
        };

    VerifyBiasAddTensorsNear(m, k, n, run_default, run_fused);
  }

  // Verifies that computing MatMul+BiasAdd+{Activation} in a graph is identical
  // to FusedMatMul.
  void VerifyConv2DWithBiasAndActivation(int m, int k, int n, bool transpose_a,
                                         bool transpose_b,
                                         const string& activation) {
    const BiasAddGraphRunner run_default = [&](const Tensor& input_data,
                                               const Tensor& filter_data,
                                               const Tensor& bias_data,
                                               Tensor* out) {
      RunMatMulWithBiasAndActivation(input_data, filter_data, bias_data,
                                     transpose_a, transpose_b, activation, out);
    };

    const BiasAddGraphRunner run_fused = [&](const Tensor& input_data,
                                             const Tensor& filter_data,
                                             const Tensor& bias_data,
                                             Tensor* out) {
      RunFusedMatMulOp(input_data, filter_data, {bias_data},
                       {"BiasAdd", activation}, transpose_a, transpose_b, out);
    };

    VerifyBiasAddTensorsNear(m, k, n, run_default, run_fused);
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

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x256x256) {
  this->VerifyMatMulWithBias(256, 256, 256, false, false);
  this->VerifyMatMulWithBias(256, 256, 256, true, false);
  this->VerifyMatMulWithBias(256, 256, 256, false, true);
  this->VerifyMatMulWithBias(256, 256, 256, true, true);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x256) {
  this->VerifyMatMulWithBias(1, 256, 256, false, false);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x256x1) {
  this->VerifyMatMulWithBias(256, 256, 1, false, false);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x1) {
  this->VerifyMatMulWithBias(1, 256, 1, false, false);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x256x256WithActivation) {
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBiasAndActivation(256, 256, 256, false, false,
                                            activation);
    this->VerifyConv2DWithBiasAndActivation(256, 256, 256, true, false,
                                            activation);
    this->VerifyConv2DWithBiasAndActivation(256, 256, 256, false, true,
                                            activation);
    this->VerifyConv2DWithBiasAndActivation(256, 256, 256, true, true,
                                            activation);
  }
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x256WithActivation) {
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBiasAndActivation(1, 256, 256, false, false,
                                            activation);
  }
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x256x1WithActivation) {
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBiasAndActivation(256, 256, 1, false, false,
                                            activation);
  }
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x1WithActivation) {
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBiasAndActivation(1, 256, 1, false, false,
                                            activation);
  }
}

REGISTER_TYPED_TEST_SUITE_P(FusedMatMulWithBiasOpTest,        //
                            MatMul256x256x256,                //
                            MatMul1x256x256,                  //
                            MatMul256x256x1,                  //
                            MatMul1x256x1,                    //
                            MatMul256x256x256WithActivation,  //
                            MatMul1x256x256WithActivation,    //
                            MatMul256x256x1WithActivation,    //
                            MatMul1x256x1WithActivation);

// TODO(ezhulenev): Add support for more data types.
using FusedBiasAddDataTypes = ::testing::Types<float>;
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
      int iters) {                                                             \
    testing::UseRealTime();                                                    \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2);        \
    test::Benchmark(#DEVICE, Matmul<T>(M, K, N, TA, TB, TFTYPE)).Run(iters);   \
  }                                                                            \
  BENCHMARK(BM_Matmul##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE);

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

}  // end namespace tensorflow

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL

#include "absl/strings/match.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/math_ops_internal.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stacktrace_handler.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/mkl_util.h"

// Compare the performance of default tensorflow kernels (Eigen) with
// MKL kernels on CPU.
// Before running benchmarks, you need to configure OpenMP environment:
//   export TF_DISABLE_MKL=1 //To avoid Eigen kernels are rewrote by MKL.
//   export KMP_BLOCKTIME=0
//   export OMP_NUM_THREADS=${num_threads}
//   export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
//
// Then you could use below command to test mkl and eigen performance:
// $bazel run --config opt --config=mkl \
//     //tensorflow/core/kernels/mkl:mkl_eltwise_ops_test -- --benchmarks=..
//
//
// ===========================================================================
// If you want to test MKL kernels accuracy with Eigen kernels, you need:
//   export TF_DISABLE_MKL=1 // To avoid Eigen kernels are rewrote by MKL.
// $bazel run --config opt --config=mkl \
//     //tensorflow/core/kernels/mkl:mkl_eltwise_ops_test

namespace tensorflow {

// --------------------------------------------------------------------------//
//  Test Mkl element-wise kernels accuracy compare with Eigen kernel         //
// --------------------------------------------------------------------------//
static const uint8 dummy_tensor[] = {0, 0, 0, 0, 0, 0, 0, 0};
static const TensorShape dummy_shape({8});

using GraphRunner = std::function<void(const Tensor& input,
                                       const string& op_name, Tensor* output)>;

using GraphGradRunner =
    std::function<void(const Tensor& grad, const Tensor& input,
                       const string& op_name, Tensor* output)>;

template <typename T>
class CommonTestUtilities : public OpsTestBase {
 public:
  void PerformConversion(DataType dtype, const Tensor& tensor,
                         const Tensor& mkl_meta_tensor, Tensor* output) {
    // Create an MKL to TF conversion node and execute it
    TF_EXPECT_OK(NodeDefBuilder("mkl_to_tf_op", "_MklToTf")
                     .Input(FakeInput(dtype))     // Input
                     .Input(FakeInput(DT_UINT8))  // Mkl second tensor
                     .Attr("T", dtype)
                     .Attr("_kernel", "MklLayoutDependentOp")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    AddInputFromArray<T>(tensor.shape(), tensor.flat<T>());
    AddInputFromArray<uint8>(mkl_meta_tensor.shape(),
                             mkl_meta_tensor.flat<uint8>());
    TF_ASSERT_OK(RunOpKernel());

    *output = *GetOutput(0);
  }

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the output Tensor.
  static void RunAndFetch(const tensorflow::Scope& root, const string& fetch,
                          Tensor* output) {
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

    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> output_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &output_tensors));

    *output = output_tensors[0];
  }

  void TestBody() {}

  static void VerifyTensorsClose(const GraphRunner& run,
                                 const GraphRunner& run_mkl,
                                 const string& op_name, const bool& is_scalar) {
    float atol = 1e-6, rtol = 1e-6;
    DataType dtype = DataTypeToEnum<T>::v();
    Tensor input;
    if (is_scalar == false) {
      int batch = 1;
      int height = 1;
      int width = 6;
      int depth = 2;

      input = {dtype, {batch, height, width, depth}};
      test::FillValues<T>(&input, {-13.86, -6.86, -2.51, -1.51, -0.53, 0.00,
                                   0.53, 1.51, 2.51, 6.00, 6.86, 13.86});
    } else {
      input = {dtype, {}};
      input.flat<T>().setRandom();
    }
    Tensor output;
    Tensor mkl_output;
    run(input, op_name, &output);
    run_mkl(input, op_name, &mkl_output);

    ASSERT_EQ(output.dtype(), mkl_output.dtype());
    ASSERT_EQ(output.shape(), mkl_output.shape());
    if (dtype == DT_BFLOAT16) {
      rtol = 1e-2;
      atol = 1e-2;
    }
    test::ExpectClose(output, mkl_output, atol, rtol);
  }

  static void VerifyTensorsGradClose(const GraphGradRunner& run,
                                     const GraphGradRunner& run_mkl,
                                     const string& op_name,
                                     const bool& is_scalar) {
    float atol = 1e-6, rtol = 1e-6;
    DataType dtype = DataTypeToEnum<T>::v();
    Tensor input, grad;
    if (is_scalar == false) {
      int batch = 1;
      int height = 1;
      int width = 6;
      int depth = 2;

      grad = {dtype, {batch, height, width, depth}};
      input = {dtype, {batch, height, width, depth}};
      // TODO(Shi,Guangyong) after relu6 backward bug be fixed, add value 6.00.
      test::FillValues<T>(&input, {-13.86, -6.86, -2.51, -1.51, -0.53, 0.00,
                                   0.53, 1.51, 2.51, 6.10, 6.86, 13.86});
    } else {
      input = {dtype, {}};
      grad = {dtype, {}};
      input.flat<T>().setRandom();
    }
    grad.flat<T>().setRandom();

    Tensor output;
    Tensor mkl_output;
    run(grad, input, op_name, &output);
    run_mkl(grad, input, op_name, &mkl_output);

    ASSERT_EQ(output.dtype(), mkl_output.dtype());
    ASSERT_EQ(output.shape(), mkl_output.shape());
    if (dtype == DT_BFLOAT16) {
      rtol = 1e-2;
      atol = 1e-2;
    }
    test::ExpectClose(output, mkl_output, atol, rtol);
  }
};

template <typename T>
class ActivationOpsTest : public OpsTestBase {
 protected:
  void VerifyActivationOps(const string& op_name, const bool& is_scalar) {
    const GraphRunner run = [this](const Tensor& input, const string& op_name,
                                   Tensor* output) {
      auto root = tensorflow::Scope::NewRootScope();
      auto input_op =
          ops::Const(root.WithOpName("input"), Input::Initializer(input));
      Output activation_op;
      if (op_name == "Relu") {
        activation_op = ops::Relu(
            root.WithOpName(strings::StrCat("Default_", op_name)), input_op);
      } else if (op_name == "Relu6") {
        activation_op = ops::Relu6(
            root.WithOpName(strings::StrCat("Default_", op_name)), input_op);
      } else if (op_name == "LeakyRelu") {
        activation_op = ops::internal::LeakyRelu(
            root.WithOpName(strings::StrCat("Default_", op_name)), input_op);
      } else if (op_name == "Elu") {
        activation_op = ops::Elu(
            root.WithOpName(strings::StrCat("Default_", op_name)), input_op);
      } else if (op_name == "Tanh") {
        activation_op = ops::Tanh(
            root.WithOpName(strings::StrCat("Default_", op_name)), input_op);
      }
      auto output_op = ops::Identity(root.WithOpName("output"), activation_op);

      CommonTestUtilities<T>::RunAndFetch(root, "output", output);
    };

    const GraphRunner run_mkl = [this](const Tensor& input,
                                       const string& op_name, Tensor* output) {
      DataType dtype = DataTypeToEnum<T>::v();

      TF_EXPECT_OK(NodeDefBuilder(strings::StrCat("Mkl_", op_name),
                                  strings::StrCat("_Mkl", op_name))
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(DT_UINT8))
                       .Attr("T", dtype)
                       .Attr("_kernel", "MklLayoutDependentOp")
                       .Finalize(node_def()));
      TF_EXPECT_OK(InitOp());

      AddInputFromArray<T>(input.shape(), input.flat<T>());
      AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
      TF_ASSERT_OK(RunOpKernel());

      CommonTestUtilities<T> test_util;
      test_util.PerformConversion(dtype, *GetOutput(0), *GetOutput(1), output);
    };
    CommonTestUtilities<T>::VerifyTensorsClose(run, run_mkl, op_name,
                                               is_scalar);
  }
};

TYPED_TEST_SUITE_P(ActivationOpsTest);

TYPED_TEST_P(ActivationOpsTest, ReluOpTest) {
  this->VerifyActivationOps("Relu", false);
}

TYPED_TEST_P(ActivationOpsTest, ReluOpScalarTest) {
  this->VerifyActivationOps("Relu", true);
}

TYPED_TEST_P(ActivationOpsTest, Relu6OpTest) {
  this->VerifyActivationOps("Relu6", false);
}

TYPED_TEST_P(ActivationOpsTest, Relu6OpScalarTest) {
  this->VerifyActivationOps("Relu6", true);
}

TYPED_TEST_P(ActivationOpsTest, LeakyReluOpTest) {
  this->VerifyActivationOps("LeakyRelu", false);
}

TYPED_TEST_P(ActivationOpsTest, LeakyReluOpScalarTest) {
  this->VerifyActivationOps("LeakyRelu", true);
}

TYPED_TEST_P(ActivationOpsTest, TanhOpTest) {
  this->VerifyActivationOps("Tanh", false);
}

TYPED_TEST_P(ActivationOpsTest, TanhOpScalarTest) {
  this->VerifyActivationOps("Tanh", true);
}

// TODO(Shi,Guangyong) After Mkl Elu kernels bug, Remove test about Elu.
// TYPED_TEST_P(ActivationOpsTest, EluOpTest) {
// this->VerifyActivationOps("Elu", false); }

// TYPED_TEST_P(ActivationOpsTest, EluOpScalarTest) {
// this->VerifyActivationOps("Elu", true); }

REGISTER_TYPED_TEST_SUITE_P(ActivationOpsTest, ReluOpTest, Relu6OpTest,
                            LeakyReluOpTest, /* EluOpTest,*/ TanhOpTest,
                            ReluOpScalarTest, Relu6OpScalarTest,
                            LeakyReluOpScalarTest, TanhOpScalarTest
                            /* EluOpScalarTest,*/);

using ActivationOpsDataTypes = ::testing::Types<float, bfloat16>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, ActivationOpsTest, ActivationOpsDataTypes);

// Test Activation ops gradient accuracy.
template <typename T>
class ActivationGradOpsTest : public OpsTestBase {
 protected:
  void VerifyActivationGradOps(const string& op_name, const bool& is_scalar) {
    const GraphGradRunner run = [this](const Tensor& grad, const Tensor& input,
                                       const string& op_name, Tensor* output) {
      auto root = tensorflow::Scope::NewRootScope();
      auto grad_op =
          ops::Const(root.WithOpName("grad"), Input::Initializer(grad));
      auto input_op =
          ops::Const(root.WithOpName("input"), Input::Initializer(input));
      Output activation_op;
      if (op_name == "ReluGrad") {
        activation_op = ops::internal::ReluGrad(
            root.WithOpName(strings::StrCat("Default_", op_name)), grad,
            input_op);
      } else if (op_name == "Relu6Grad") {
        activation_op = ops::internal::Relu6Grad(
            root.WithOpName(strings::StrCat("Default_", op_name)), grad,
            input_op);
      } else if (op_name == "LeakyReluGrad") {
        activation_op = ops::internal::LeakyReluGrad(
            root.WithOpName(strings::StrCat("Default_", op_name)), grad,
            input_op);
      } else if (op_name == "EluGrad") {
        activation_op = ops::internal::EluGrad(
            root.WithOpName(strings::StrCat("Default_", op_name)), grad,
            input_op);
      } else if (op_name == "TanhGrad") {
        activation_op = ops::internal::TanhGrad(
            root.WithOpName(strings::StrCat("Default_", op_name)), input_op,
            grad);
      }
      auto output_op = ops::Identity(root.WithOpName("output"), activation_op);

      CommonTestUtilities<T>::RunAndFetch(root, "output", output);
    };

    const GraphGradRunner run_mkl = [this](
                                        const Tensor& grad, const Tensor& input,
                                        const string& op_name, Tensor* output) {
      DataType dtype = DataTypeToEnum<T>::v();

      TF_EXPECT_OK(NodeDefBuilder(strings::StrCat("Mkl_", op_name),
                                  strings::StrCat("_Mkl", op_name))
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(dtype))
                       .Input(FakeInput(DT_UINT8))
                       .Input(FakeInput(DT_UINT8))
                       .Attr("T", dtype)
                       .Attr("_kernel", "MklLayoutDependentOp")
                       .Finalize(node_def()));
      TF_EXPECT_OK(InitOp());

      if (op_name == "TanhGrad") {
        AddInputFromArray<T>(input.shape(), input.flat<T>());
        AddInputFromArray<T>(grad.shape(), grad.flat<T>());
      } else {
        AddInputFromArray<T>(grad.shape(), grad.flat<T>());
        AddInputFromArray<T>(input.shape(), input.flat<T>());
      }
      AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
      AddInputFromArray<uint8>(dummy_shape, dummy_tensor);
      TF_ASSERT_OK(RunOpKernel());

      CommonTestUtilities<T> test_util;
      test_util.PerformConversion(dtype, *GetOutput(0), *GetOutput(1), output);
    };
    CommonTestUtilities<T>::VerifyTensorsGradClose(run, run_mkl, op_name,
                                                   is_scalar);
  }
};

TYPED_TEST_SUITE_P(ActivationGradOpsTest);

TYPED_TEST_P(ActivationGradOpsTest, ReluGradOpTest) {
  this->VerifyActivationGradOps("ReluGrad", false);
}

TYPED_TEST_P(ActivationGradOpsTest, ReluGradOpScalarTest) {
  this->VerifyActivationGradOps("ReluGrad", true);
}

TYPED_TEST_P(ActivationGradOpsTest, Relu6GradOpTest) {
  this->VerifyActivationGradOps("Relu6Grad", false);
}

TYPED_TEST_P(ActivationGradOpsTest, Relu6GradOpScalarTest) {
  this->VerifyActivationGradOps("Relu6Grad", true);
}

// TODO(Shi,Guangyong) After Mkl Elu kernels bug, Remove test about Elu.
/*
TYPED_TEST_P(ActivationGradOpsTest, EluGradOpTest) {
  this->VerifyActivationGradOps("EluGrad", false);
}

TYPED_TEST_P(ActivationGradOpsTest, EluGradOpScalarTest) {
  this->VerifyActivationGradOps("EluGrad", true);
}
*/

TYPED_TEST_P(ActivationGradOpsTest, LeakyReluGradOpTest) {
  this->VerifyActivationGradOps("LeakyReluGrad", false);
}

TYPED_TEST_P(ActivationGradOpsTest, LeakyReluGradOpScalarTest) {
  this->VerifyActivationGradOps("LeakyReluGrad", true);
}

TYPED_TEST_P(ActivationGradOpsTest, TanhGradOpTest) {
  this->VerifyActivationGradOps("TanhGrad", false);
}

TYPED_TEST_P(ActivationGradOpsTest, TanhGradOpScalarTest) {
  this->VerifyActivationGradOps("TanhGrad", true);
}

REGISTER_TYPED_TEST_SUITE_P(ActivationGradOpsTest, ReluGradOpTest,
                            Relu6GradOpTest,
                            LeakyReluGradOpTest, /* EluGradOpTest,*/
                            TanhGradOpTest, ReluGradOpScalarTest,
                            Relu6GradOpScalarTest, LeakyReluGradOpScalarTest,
                            TanhGradOpScalarTest
                            /* EluGradOpScalarTest,*/);

using ActivationGradOpsDataTypes = ::testing::Types<float, bfloat16>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, ActivationGradOpsTest,
                               ActivationGradOpsDataTypes);

// --------------------------------------------------------------------------//
// Test Mkl element-wise kernels performance with Eigen                      //
// --------------------------------------------------------------------------//
template <typename T>
static Graph* Activation(const string& op_name, const string& kind,
                         const TensorShape& shape) {
  auto* graph = new Graph(OpRegistry::Global());
  const string node_name = kind + "_" + op_name;
  const bool isForwardOp = !tensorflow::str_util::EndsWith(op_name, "Grad");
  const bool isDefault = (kind == "Default");

  DataType dtype = DataTypeToEnum<T>::v();
  Tensor input_t(dtype, shape);
  input_t.flat<T>().setRandom();
  Node* input = test::graph::Constant(graph, input_t, "input");
  Node* not_mkl_shape =
      test::graph::Constant(graph, GetMklMetaTensor(), "not_mkl");

  Node* activation;
  if (isForwardOp) {
    // Default forward op.
    if (isDefault) {
      TF_CHECK_OK(NodeBuilder(graph->NewName(node_name), op_name)
                      .Input(input)
                      .Attr("T", dtype)
                      .Finalize(graph, &activation));
      return graph;
    }
    // MKL forward op.
    TF_CHECK_OK(NodeBuilder(graph->NewName(node_name), "_Mkl" + op_name)
                    .Input(input)
                    .Input(not_mkl_shape)
                    .Attr("T", dtype)
                    .Attr("_kernel", "MklLayoutDependentOp")
                    .Finalize(graph, &activation));
    return graph;
  }

  // Default backward op.
  Tensor grad_t(dtype, shape);
  grad_t.flat<T>().setRandom();
  Node* grad = test::graph::Constant(graph, grad_t, "grad");
  if (isDefault) {
    TF_CHECK_OK(NodeBuilder(graph->NewName(node_name), op_name)
                    .Input(grad)
                    .Input(input)
                    .Attr("T", dtype)
                    .Finalize(graph, &grad));
    return graph;
  }

  // MKL backward op.
  TF_CHECK_OK(NodeBuilder(graph->NewName(node_name), "_Mkl" + op_name)
                  .Input(grad)
                  .Input(input)
                  .Input(not_mkl_shape)
                  .Input(not_mkl_shape)
                  .Attr("T", dtype)
                  .Attr("_kernel", "MklLayoutDependentOp")
                  .Finalize(graph, &grad));
  return graph;
}

#define BM_Activation(op, kind, A, B, C, D, T, type)                     \
  static void BM_##op##_##kind##_##type##_##A##_##B##_##C##_##D##_##T(   \
      int iters) {                                                       \
    int64 num_computed_elements = (A) * (B) * (C) * (D);                 \
    int64 flops_per_iter = num_computed_elements;                        \
    testing::UseRealTime();                                              \
    testing::ItemsProcessed(static_cast<int64>(iters) * flops_per_iter); \
                                                                         \
    test::Benchmark(#type, Activation<T>(#op, #kind, {A, B, C, D}))      \
        .Run(iters);                                                     \
  }                                                                      \
  BENCHMARK(BM_##op##_##kind##_##type##_##A##_##B##_##C##_##D##_##T)

#define BM(op, A, B, C, D, T, type)                \
  BM_Activation(op, Default, A, B, C, D, T, type); \
  BM_Activation(op, Mkl, A, B, C, D, T, type);

#define TEST_ALL_SIZES(OP, T)      \
  BM(OP, 1, 16, 16, 3, T, cpu);    \
  BM(OP, 1, 32, 32, 16, T, cpu);   \
  BM(OP, 32, 64, 64, 128, T, cpu); \
  BM(OP, 64, 128, 128, 256, T, cpu);

#define TEST_ALL_SIZES_WITH_DTYPE(OP) \
  TEST_ALL_SIZES(OP, float)           \
  TEST_ALL_SIZES(OP, bfloat16)

TEST_ALL_SIZES_WITH_DTYPE(Relu)
TEST_ALL_SIZES_WITH_DTYPE(ReluGrad)
TEST_ALL_SIZES_WITH_DTYPE(Relu6)
TEST_ALL_SIZES_WITH_DTYPE(Relu6Grad)
// TODO(Shi,Guangyong) After Mkl Elu kernels bug, Remove test about Elu.
// TEST_ALL_SIZES_WITH_DTYPE(Elu)
// TEST_ALL_SIZES_WITH_DTYPE(EluGrad)
TEST_ALL_SIZES_WITH_DTYPE(LeakyRelu)
TEST_ALL_SIZES_WITH_DTYPE(LeakyReluGrad)
TEST_ALL_SIZES_WITH_DTYPE(Tanh)
TEST_ALL_SIZES_WITH_DTYPE(TanhGrad)

}  // namespace tensorflow

#endif  // INTEL_MKL

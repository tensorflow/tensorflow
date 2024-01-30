/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#if defined(INTEL_MKL)

#define EIGEN_USE_THREADS

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/mkl/mkl_kernel_util.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

// T: float or bfloat16 used as tensor type of the MatMul and fusion operation.
template <typename T>
class FusedBatchMatMulOpTestBase : public OpsTestBase {
 protected:
  struct FusedOpsAndDims {
    // List of fusions.
    std::vector<string> fused_ops;
    // Tensor dimension associated with the fusions. Currently assuming that
    // each fusion requires no more than one tensor. If some fusion does not
    // require a tensor, e.g., Relu, the tensor dimensions will be {0} implying
    // an an empty tensor.
    std::vector<std::vector<int64_t>> fused_dims;
    // TODO(intel-tf): Add additional field if some fusion needs additional
    // parameters.
  };

  struct FusedOpsAndTensors {
    // List of fusions.
    std::vector<string> fused_ops;
    // Tensors associated with the fusions. Currently assuming that each fusion
    // requires no more than one tensor. If some fusion does not require a
    // tensor, e.g., Relu, the tensor will be an empty tensor.
    std::vector<Tensor> fusion_tensors;
    // TODO(intel-tf): Add additional field if some fusion needs additional
    // parameters.
  };

  using GraphRunner =
      std::function<void(const Tensor& x, const Tensor& y,
                         const FusedOpsAndTensors& fused_ops_and_tensors,
                         Tensor* result, bool adj_x, bool adj_y)>;

  using QuantizedGraphRunner = std::function<void(
      const Tensor& x, const Tensor& y,
      const FusedOpsAndTensors& fused_ops_and_tensors, Tensor* result,
      bool adj_x, bool adj_y, string input_quant_mode, string output_quant_mode,
      bool requantize, float output_min, float output_max)>;

  bool HasQuantizationSupport() {
    return TestCPUFeature(port::CPUFeature::AVX_VNNI_INT8) ||
           TestCPUFeature(port::CPUFeature::AVX512_VNNI) ||
           TestCPUFeature(port::CPUFeature::AMX_INT8);
  }

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the outputs. Optional `add_nodes` parameter
  // allows to define nodes directly using a NodeDef for the ops that are
  // not supported by the C++ Api.
  // TODO(intel-tf): Move RunAndFetch function to a common header file for
  // better reuse in the future.
  void RunAndFetch(const tensorflow::Scope& root,
                   const std::vector<string>& fetch,
                   std::vector<Tensor>* outputs,
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

    const string device = "/device:CPU:0";
    for (NodeDef& mutable_node : *graph.mutable_node()) {
      mutable_node.set_device(device);
    }

    TF_ASSERT_OK(session->Create(graph));
    TF_ASSERT_OK(session->Run({}, fetch, {}, outputs));
  }

  void RunBatchMatMulAndFusedOps(
      const Tensor& x, const Tensor& y,
      const FusedOpsAndTensors& fused_ops_and_tensors, Tensor* result,
      bool adj_x, bool adj_y) {
    Scope root = tensorflow::Scope::NewRootScope();

    Output x_input =
        ops::Const(root.WithOpName("x_input"), Input::Initializer(x));
    Output y_input =
        ops::Const(root.WithOpName("y_input"), Input::Initializer(y));
    Output last_output =
        ops::BatchMatMulV2(root.WithOpName("batch_matmul"), x_input, y_input,
                           ops::BatchMatMulV2::Attrs().AdjX(adj_x).AdjY(adj_y));
    auto& fused_ops = fused_ops_and_tensors.fused_ops;
    auto& fusion_tensors = fused_ops_and_tensors.fusion_tensors;
    for (int i = 0; i < fused_ops.size(); ++i) {
      const string& op = fused_ops[i];
      if (op == "Mul") {
        Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                                Input::Initializer(fusion_tensors[i]));
        last_output = ops::Multiply(root.WithOpName(absl::StrCat("mul_at_", i)),
                                    last_output, arg);
      } else if (op == "Add") {
        Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                                Input::Initializer(fusion_tensors[i]));
        last_output = ops::AddV2(root.WithOpName(absl::StrCat("add_at_", i)),
                                 last_output, arg);
      }
    }
    std::vector<Tensor> outputs;
    RunAndFetch(root, {last_output.name()}, &outputs);
    *result = outputs[0];
  }

  void RunFusedBatchMatMul(const Tensor& x, const Tensor& y,
                           const FusedOpsAndTensors& fused_ops_and_tensors,
                           Tensor* result, bool adj_x, bool adj_y) {
    Scope root = tensorflow::Scope::NewRootScope();

    DataType t_dtype = DataTypeToEnum<T>::v();

    Output x_input =
        ops::Const(root.WithOpName("x_input"), Input::Initializer(x));
    Output y_input =
        ops::Const(root.WithOpName("y_input"), Input::Initializer(y));
    auto& fused_ops = fused_ops_and_tensors.fused_ops;
    auto& fusion_tensors = fused_ops_and_tensors.fusion_tensors;
    int num_fused_inputs = 0;
    std::vector<NodeDefBuilder::NodeOut> fused_inputs;
    for (int i = 0; i < fused_ops.size(); ++i) {
      const string& op = fused_ops[i];
      if (op == "Mul" || op == "Add") {
        Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                                Input::Initializer(fusion_tensors[i]));
        fused_inputs.push_back({arg.name(), 0, t_dtype});
        num_fused_inputs++;
      }
    }
    NodeDef fused_batch_matmul;
    std::vector<const NodeDef*> add_nodes;
    TF_EXPECT_OK(NodeDefBuilder("fused_batch_matmul", "_MklFusedBatchMatMulV2")
                     .Input({x_input.name(), 0, t_dtype})
                     .Input({y_input.name(), 0, t_dtype})
                     .Input(fused_inputs)
                     .Attr("adj_x", adj_x)
                     .Attr("adj_y", adj_y)
                     .Attr("num_args", num_fused_inputs)
                     .Attr("fused_ops", fused_ops)
                     .Finalize(&fused_batch_matmul));
    add_nodes = {&fused_batch_matmul};
    std::vector<Tensor> outputs;
    RunAndFetch(root, {fused_batch_matmul.name()}, &outputs, add_nodes);
    *result = outputs[0];
  }

  void RunQuantizedBatchMatMul(const Tensor& x, const Tensor& y,
                               const FusedOpsAndTensors& fused_ops_and_tensors,
                               Tensor* result, bool adj_x, bool adj_y,
                               string input_quant_mode,
                               string output_quant_mode, bool requantize,
                               float output_min, float output_max) {
    DataType real_dtype = DataTypeToEnum<T>::v();
    // Quantize x and y
    Tensor x_qtensor(DT_QINT8, x.shape());
    Tensor x_min_tensor(DT_FLOAT, TensorShape({}));
    Tensor x_max_tensor(DT_FLOAT, TensorShape({}));
    Tensor y_qtensor(DT_QINT8, y.shape());
    Tensor y_min_tensor(DT_FLOAT, TensorShape({}));
    Tensor y_max_tensor(DT_FLOAT, TensorShape({}));

    MklTestingUtil::GetQuantizationTensors<T>(x, &x_qtensor, DT_QINT8,
                                              input_quant_mode, &x_min_tensor,
                                              &x_max_tensor);
    MklTestingUtil::GetQuantizationTensors<T>(y, &y_qtensor, DT_QINT8,
                                              input_quant_mode, &y_min_tensor,
                                              &y_max_tensor);

    Scope root = tensorflow::Scope::NewRootScope();

    Output x_input =
        ops::Const(root.WithOpName("x_input"), Input::Initializer(x_qtensor));
    Output x_min =
        ops::Const(root.WithOpName("x_min"), Input::Initializer(x_min_tensor));
    Output x_max =
        ops::Const(root.WithOpName("x_max"), Input::Initializer(x_max_tensor));
    Output y_input =
        ops::Const(root.WithOpName("y_input"), Input::Initializer(y_qtensor));
    Output y_min =
        ops::Const(root.WithOpName("y_min"), Input::Initializer(y_min_tensor));
    Output y_max =
        ops::Const(root.WithOpName("y_max"), Input::Initializer(y_max_tensor));
    auto& fused_ops = fused_ops_and_tensors.fused_ops;
    auto& fusion_tensors = fused_ops_and_tensors.fusion_tensors;
    int num_fused_inputs = 0;
    std::vector<NodeDefBuilder::NodeOut> fused_inputs;
    for (int i = 0; i < fused_ops.size(); ++i) {
      const string& op = fused_ops[i];
      if (op == "Mul" || op == "Add") {
        Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                                Input::Initializer(fusion_tensors[i]));
        fused_inputs.push_back({arg.name(), 0, real_dtype});
        num_fused_inputs++;
      }
    }
    NodeDef fused_batch_matmul;
    std::vector<const NodeDef*> add_nodes;
    std::vector<Tensor> outputs;
    std::vector<NodeDefBuilder::NodeOut> inputs;
    inputs.push_back({"x_input", 0, DT_QINT8});
    inputs.push_back({"y_input", 0, DT_QINT8});
    inputs.insert(std::end(inputs), std::begin(fused_inputs),
                  std::end(fused_inputs));
    inputs.push_back({"x_min", 0, DT_FLOAT});
    inputs.push_back({"x_max", 0, DT_FLOAT});
    inputs.push_back({"y_min", 0, DT_FLOAT});
    inputs.push_back({"y_max", 0, DT_FLOAT});

    std::vector<string> extended_fused_ops(fused_ops);
    DataType out_dtype;
    if (requantize) {
      if (output_quant_mode == "SCALED") {
        out_dtype = DT_QINT8;
      } else {
        out_dtype = DT_QUINT8;
      }
    } else {
      out_dtype = real_dtype;
    }
    std::vector<DataType> output_dtypes;
    if (requantize) {
      Output out_min = ops::Const(root.WithOpName("output_min"), output_min);
      Output out_max = ops::Const(root.WithOpName("output_max"), output_max);
      inputs.push_back({"output_min", 0, DT_FLOAT});
      inputs.push_back({"output_max", 0, DT_FLOAT});
      extended_fused_ops.push_back("Requantize");
      output_dtypes = {out_dtype, DT_FLOAT, DT_FLOAT};
    } else {
      extended_fused_ops.push_back("Dequantize");
      output_dtypes = {out_dtype};
    }
    TF_EXPECT_OK(
        NodeDefBuilder("quantized_batch_matmul", "_QuantizedBatchMatMul")
            .Attr("Tdevice_inputs", std::vector<DataType>())
            .Input(FakeInput())
            .Input(inputs)
            .Attr("Tdevice_outputs", std::vector<DataType>())
            .Attr("Thost_outputs", output_dtypes)
            .Attr("T1", DT_QINT8)
            .Attr("T2", DT_QINT8)
            .Attr("Tout", out_dtype)
            .Attr("U", real_dtype)
            .Attr("adj_x", adj_x)
            .Attr("adj_y", adj_y)
            .Attr("fused_ops", extended_fused_ops)
            .Attr("input_quant_mode", input_quant_mode)
            .Attr("output_quant_mode", output_quant_mode)
            .Finalize(&fused_batch_matmul));
    if (requantize) {
      NodeDef dequantize;
      TF_EXPECT_OK(NodeDefBuilder("dequantize", "Dequantize")
                       .Input({"quantized_batch_matmul", 0, out_dtype})
                       .Input({"quantized_batch_matmul", 1, DT_FLOAT})
                       .Input({"quantized_batch_matmul", 2, DT_FLOAT})
                       .Attr("dtype", real_dtype)
                       .Attr("mode", output_quant_mode)
                       .Finalize(&dequantize));
      add_nodes = {&fused_batch_matmul, &dequantize};
      RunAndFetch(root, {dequantize.name()}, &outputs, add_nodes);
    } else {
      add_nodes = {&fused_batch_matmul};
      RunAndFetch(root, {fused_batch_matmul.name()}, &outputs, add_nodes);
    }
    *result = outputs[0];
  }

  template <typename FusedGraphRunner>
  void VerifyTensorsNear(const std::initializer_list<int64_t>& x_dims,
                         const std::initializer_list<int64_t>& y_dims,
                         const FusedOpsAndDims& fused_ops_and_dims,
                         const GraphRunner& run_default,
                         const FusedGraphRunner& run_fused, bool adj_x,
                         bool adj_y, const double atol = 1e-5,
                         // The following arguments are used by quantized fusion
                         string input_quant_mode = "SCALED",
                         string output_quant_mode = "SCALED",
                         bool requantize = false) {
    DataType t_dtype = DataTypeToEnum<T>::v();
    TensorShape x_shape = TensorShape(x_dims);
    TensorShape y_shape = TensorShape(y_dims);

    Tensor x_tensor(t_dtype, x_shape);
    x_tensor.flat<T>().setRandom();
    x_tensor.flat<T>() -= x_tensor.flat<T>().constant(static_cast<T>(0.5));

    Tensor y_tensor(t_dtype, y_shape);
    y_tensor.flat<T>().setRandom();
    y_tensor.flat<T>() -= y_tensor.flat<T>().constant(static_cast<T>(0.5));

    FusedOpsAndTensors fused_ops_and_tensors;
    // Copy fused_ops
    fused_ops_and_tensors.fused_ops = fused_ops_and_dims.fused_ops;
    const auto& fused_ops = fused_ops_and_tensors.fused_ops;  // Alias to field
    const auto& fused_dims = fused_ops_and_dims.fused_dims;   // Alias to field
    auto& fusion_tensors =
        fused_ops_and_tensors.fusion_tensors;  // Initially empty
    for (int i = 0; i < fused_ops.size(); ++i) {
      TensorShape arg_shape = TensorShape(fused_dims[i]);
      Tensor arg_tensor(t_dtype, arg_shape);
      arg_tensor.flat<T>().setRandom();
      arg_tensor.flat<T>() -=
          arg_tensor.flat<T>().constant(static_cast<T>(0.5));
      fusion_tensors.push_back(arg_tensor);
    }
    Tensor default_result;
    run_default(x_tensor, y_tensor, fused_ops_and_tensors, &default_result,
                adj_x, adj_y);

    Tensor fused_result;
    if constexpr (std::is_same<FusedGraphRunner, QuantizedGraphRunner>::value) {
      float output_min = 1.0;
      float output_max = 1.0 + std::numeric_limits<float>::epsilon();
      if (requantize) {
        T min;
        T max;
        MklTestingUtil::ComputeMinMax<T>(default_result, &min, &max);
        output_min = static_cast<float>(min);
        output_max = static_cast<float>(max);
      }
      // Run quantized fusion
      run_fused(x_tensor, y_tensor, fused_ops_and_tensors, &fused_result, adj_x,
                adj_y, input_quant_mode, output_quant_mode, requantize,
                output_min, output_max);
    } else {
      // Run realnumber type fusion
      run_fused(x_tensor, y_tensor, fused_ops_and_tensors, &fused_result, adj_x,
                adj_y);
    }
    std::vector<std::pair<Tensor, Tensor>> tensor_pairs = {
        {default_result, fused_result}};
    for (auto& pair : tensor_pairs) {
      const Tensor& expected = pair.first;
      const Tensor& evaluated = pair.second;

      ASSERT_EQ(expected.dtype(), evaluated.dtype());
      ASSERT_EQ(expected.shape(), evaluated.shape());

      test::ExpectClose(expected, evaluated, atol);
    }
  }

  void GetFusionConfiguration(const std::vector<string>& fused_ops,
                              const int batch_dim0, const int batch_dim1,
                              const int row, const int col,
                              FusedOpsAndDims* fused_ops_and_dims) {
    if (fused_ops.empty()) {
      *fused_ops_and_dims = {fused_ops, {}};
    } else if (fused_ops == std::vector<string>{"Mul"}) {
      *fused_ops_and_dims = {fused_ops, {{} /* scalar multiplicand*/}};
    } else if (fused_ops == std::vector<string>{"Add"}) {
      *fused_ops_and_dims = {fused_ops,
                             {std::vector<int64_t>{batch_dim0, 1, row, col}}};
    } else if (fused_ops == std::vector<string>{"Mul", "Add"}) {
      *fused_ops_and_dims = {
          fused_ops, {{}, std::vector<int64_t>{batch_dim0, 1, row, col}}};
    } else {
      EXPECT_TRUE(false) << absl::StrCat("The fusion: [",
                                         absl::StrJoin(fused_ops, ","),
                                         "] is not supported in this test.");
    }
  }

  void VerifyFusedBatchMatMul(const std::initializer_list<int64_t> x_dims,
                              const std::initializer_list<int64_t> y_dims,
                              const FusedOpsAndDims fused_ops_and_dims,
                              bool adj_x, bool adj_y) {
    const GraphRunner run_default =
        [&](const Tensor& x, const Tensor& y,
            const FusedOpsAndTensors& fused_ops_and_tensors, Tensor* result,
            bool adj_x, bool adj_y) {
          this->RunBatchMatMulAndFusedOps(x, y, fused_ops_and_tensors, result,
                                          adj_x, adj_y);
        };

    const GraphRunner run_fused =
        [&](const Tensor& x, const Tensor& y,
            const FusedOpsAndTensors& fused_ops_and_tensors, Tensor* result,
            bool adj_x, bool adj_y) {
          this->RunFusedBatchMatMul(x, y, fused_ops_and_tensors, result, adj_x,
                                    adj_y);
        };
    const double atol = std::is_same<T, bfloat16>::value ? 1e-2 : 1e-5;
    VerifyTensorsNear<GraphRunner>(x_dims, y_dims, fused_ops_and_dims,
                                   run_default, run_fused, adj_x, adj_y, atol);
  }

  void VerifyQuantizedBatchMatMul(std::vector<string> fused_ops) {
    if (!HasQuantizationSupport()) {
      GTEST_SKIP() << "oneDNN based Quantized ops are not enabled on this CPU.";
    }
    const GraphRunner run_default =
        [&](const Tensor& x, const Tensor& y,
            const FusedOpsAndTensors& fused_ops_and_tensors, Tensor* result,
            bool adj_x, bool adj_y) {
          this->RunBatchMatMulAndFusedOps(x, y, fused_ops_and_tensors, result,
                                          adj_x, adj_y);
        };

    const QuantizedGraphRunner run_quantized =
        [&](const Tensor& x, const Tensor& y,
            const FusedOpsAndTensors& fused_ops_and_tensors, Tensor* result,
            bool adj_x, bool adj_y, string input_quant_mode,
            string output_quant_mode, bool requantize, float output_min,
            float output_max) {
          this->RunQuantizedBatchMatMul(x, y, fused_ops_and_tensors, result,
                                        adj_x, adj_y, input_quant_mode,
                                        output_quant_mode, requantize,
                                        output_min, output_max);
        };
    const double atol = 1e-2;
    constexpr int M = 3;
    constexpr int K = 4;
    constexpr int N = 5;
    constexpr int b0 = 2;
    constexpr int b1 = 2;
    FusedOpsAndDims fused_ops_and_dims;
    GetFusionConfiguration(fused_ops, b0, b1, M, N, &fused_ops_and_dims);
    std::vector<bool> requantization_config;
    for (bool adj_x : {false, true}) {
      for (bool adj_y : {false, true}) {
        std::initializer_list<int64_t> x_dims = {b0, b1, adj_x ? K : M,
                                                 adj_x ? M : K};
        std::initializer_list<int64_t> y_dims = {b0, b1, adj_y ? N : K,
                                                 adj_y ? K : N};
        for (bool requantize : {false, true}) {
          for (string output_quant_mode : {"SCALED", "MIN_FIRST"}) {
            VerifyTensorsNear<QuantizedGraphRunner>(
                x_dims, y_dims, fused_ops_and_dims, run_default, run_quantized,
                adj_x, adj_y, atol, "SCALED", output_quant_mode, requantize);
          }
        }
      }
    }
  }
};

template <typename T>
using FusedBatchMatMulOpTest =
    FusedBatchMatMulOpTestBase<T>;  // additional parameters are float

TYPED_TEST_SUITE_P(FusedBatchMatMulOpTest);

TYPED_TEST_P(FusedBatchMatMulOpTest, MulFusion) {
  const int M = 3;
  const int K = 5;
  const int N = 3;
  for (const bool adj_x : {false, true})
    for (const bool adj_y : {false, true}) {
      std::initializer_list<int64_t> x_dims = {5, 4, adj_x ? K : M,
                                               adj_x ? M : K};
      std::initializer_list<int64_t> y_dims = {5, 4, adj_y ? N : K,
                                               adj_y ? K : N};
      this->VerifyFusedBatchMatMul(x_dims, y_dims,
                                   {{"Mul"}, {{} /*scalar multiplicand*/}},
                                   adj_x, adj_y);
    }
}

TYPED_TEST_P(FusedBatchMatMulOpTest, AddFusion) {
  const int M = 3;
  const int K = 5;
  const int N = 3;
  for (const bool adj_x : {false, true})
    for (const bool adj_y : {false, true}) {
      std::initializer_list<int64_t> x_dims = {5, 4, adj_x ? K : M,
                                               adj_x ? M : K};
      std::initializer_list<int64_t> y_dims = {5, 4, adj_y ? N : K,
                                               adj_y ? K : N};
      this->VerifyFusedBatchMatMul(x_dims, y_dims, {{"Add"}, {{5, 1, M, N}}},
                                   adj_x, adj_y);
    }
}

TYPED_TEST_P(FusedBatchMatMulOpTest, MulAddFusion) {
  const int M = 3;
  const int K = 5;
  const int N = 3;
  for (const bool adj_x : {false, true})
    for (const bool adj_y : {false, true}) {
      std::initializer_list<int64_t> x_dims = {5, 4, adj_x ? K : M,
                                               adj_x ? M : K};
      std::initializer_list<int64_t> y_dims = {5, 4, adj_y ? N : K,
                                               adj_y ? K : N};
      this->VerifyFusedBatchMatMul(
          x_dims, y_dims,
          {{"Mul", "Add"}, {{} /*scalar multiplicand*/, {5, 1, M, N}}}, adj_x,
          adj_y);
    }
}

TYPED_TEST_P(FusedBatchMatMulOpTest, QuantizedNoFusion) {
  this->VerifyQuantizedBatchMatMul({});
}

TYPED_TEST_P(FusedBatchMatMulOpTest, QuantizedMulFusion) {
  this->VerifyQuantizedBatchMatMul({"Mul"});
}

TYPED_TEST_P(FusedBatchMatMulOpTest, QuantizedAddFusion) {
  this->VerifyQuantizedBatchMatMul({"Add"});
}

TYPED_TEST_P(FusedBatchMatMulOpTest, QuantizedMulAddFusion) {
  this->VerifyQuantizedBatchMatMul({"Mul", "Add"});
}

REGISTER_TYPED_TEST_SUITE_P(FusedBatchMatMulOpTest, MulFusion, AddFusion,
                            MulAddFusion, QuantizedNoFusion, QuantizedMulFusion,
                            QuantizedAddFusion, QuantizedMulAddFusion);

using FusedBatchMatMulDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedBatchMatMulOpTest,
                               FusedBatchMatMulDataTypes);

}  // namespace tensorflow

#endif  // INTEL_MKL

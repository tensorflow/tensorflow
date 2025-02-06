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

#include <algorithm>
#include <limits>

#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/nn_ops_internal.h"
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
#include "unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

// The test suite contains different categories of tests.
//    (1) Realnumber (float/bfloat16): The output of _FusedMatMul should be
//    close enough to the final output of the sequence of unfused operations.
//    Only Gelu fusion is included here. All other fusion tests can be found in
//    tensorflow/core/kernels/mkl/mkl_fused_ops_test.cc
//
//    (2) Quantized: Possible fusions are done in _QuantizedMatMul op. The
//    output of
//        quantize --> quantized_op --> dequantize, or
//        quantize --> quantized_op --> requantize --> dequantize
//    should be close (with a higher tolerance) to the final output of the
//    sequence of unfused real number type operations. For the quantized
//    scenario, it is assumed that the first matrix of MatMul op represents
//    feature, while the second matrix represents weight parameters. The feature
//    matrix can be quantized with MIN_FIRST (to QUINT8) or SCALED (to QINT8)
//    mode and always quantized per-tensor. The weight can be quantized with
//    per-tensor or per-channel, only with SCALED mode to QINT8.

// T: float or bfloat16 used as tensor type of the MatMul and fusion operation.
template <typename T>
class FusedMatMulOpsTest : public OpsTestBase {
 private:
  float leakyrelu_alpha_ = 0.2f;

 protected:
  struct FusedOpsAndDims {
    // List of fusions.
    std::vector<string> fused_ops;
    // Tensor dimension associated with the fusions. It is assumed here that
    // each fusion requires no more than one addtional tensor. If some fusion
    // does not require a tensor, e.g., Relu, the tensor dimensions will be {0}
    // implying an an empty tensor.
    std::vector<std::vector<int64_t>> fusion_dims;
  };

  struct FusedOpsAndTensors {
    // List of fusions.
    std::vector<string> fused_ops;
    // Tensors associated with the fusions. It is assumed here that each fusion
    // requires no more than one additional tensor. If some fusion does not
    // require a tensor, e.g., Relu, the tensor will be an empty tensor.
    std::vector<Tensor> fusion_tensors;
  };

  using GraphRunner =
      std::function<void(const Tensor& x, const Tensor& y,
                         const FusedOpsAndTensors& fused_ops_and_tensors,
                         Tensor* result, bool transpose_x, bool transpose_y)>;

  using QuantizedGraphRunner = std::function<void(
      const Tensor& x, const Tensor& y,
      const FusedOpsAndTensors& fused_ops_and_tensors, Tensor* result,
      bool transpose_x, bool transpose_y, string input_quant_mode,
      string output_quant_mode, bool is_bias_quantized, bool is_perchannel,
      bool requantize, float output_min, float output_max)>;

  bool HasQuantizationSupport() {
    return TestCPUFeature(tensorflow::port::CPUFeature::AVX_VNNI_INT8) ||
           TestCPUFeature(tensorflow::port::CPUFeature::AVX512_VNNI) ||
           TestCPUFeature(port::CPUFeature::AMX_INT8);
  }

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the outputs. Optional `add_nodes` parameter
  // allows to define nodes directly using NodeDefBuilder.
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

  Output ActivationOp(Scope& root, string op, Output x, string name) {
    // TODO(intel-tf): Add GeluExact (Erf op based) when the Erf op is enabled
    // for bfloat16. GeluExact with float32 precision test can be found in
    //    tensorflow/python/grappler/remapper_test.py
    if (op == "Relu") {
      return ops::Relu(root.WithOpName(name), x);
    } else if (op == "Relu6") {
      return ops::Relu6(root.WithOpName(name), x);
    } else if (op == "LeakyRelu") {
      return ops::internal::LeakyRelu(
          root.WithOpName(name), x,
          ops::internal::LeakyRelu::Attrs().Alpha(this->leakyrelu_alpha_));
    } else if (op == "Elu") {
      return ops::Elu(root.WithOpName(name), x);
    } else if (op == "Tanh") {
      return ops::Tanh(root.WithOpName(name), x);
    } else if (op == "Sigmoid") {
      return ops::Sigmoid(root.WithOpName(name), x);
    } else if (op == "GeluApproximate") {
      Output three = ops::Const<T>(root.WithOpName("gelu_three"), 3.0f);
      Output empirical =
          ops::Const<T>(root.WithOpName("gelu_empirical"), 0.044715f);
      Output square_root_two_over_pi = ops::Const<T>(
          root.WithOpName("gelu_square_root_two_over_pi"), 0.7978845608028654f);
      Output one = ops::Const<T>(root.WithOpName("gelu_one"), 1.0f);
      Output half = ops::Const<T>(root.WithOpName("gelu_half"), 0.5f);
      Output pow = ops::Pow(root.WithOpName("gelu_pow"), x, three);
      Output mul1 = ops::Multiply(root.WithOpName("gelu_mul1"), empirical, pow);
      Output add1 = ops::AddV2(root.WithOpName("gelu_add1"), x, mul1);
      Output mul2 = ops::Multiply(root.WithOpName("gelu_mul2"),
                                  square_root_two_over_pi, add1);
      Output tanh = ops::Tanh(root.WithOpName("gelu_tanh"), mul2);
      Output add3 = ops::AddV2(root.WithOpName("gelu_add3"), one, tanh);
      Output mul3 = ops::Multiply(root.WithOpName("gelu_mul3"), half, x);
      return ops::Multiply(root.WithOpName(name), mul3, add3);
    } else {
      EXPECT_TRUE(false) << absl::StrCat("The activation: ", op,
                                         " is not supported in this test.");
    }
  }

  void RunMatMulAndFusedOps(const Tensor& x, const Tensor& y,
                            const FusedOpsAndTensors& fused_ops_and_tensors,
                            Tensor* result, bool transpose_x,
                            bool transpose_y) {
    Scope root = tensorflow::Scope::NewRootScope();

    Output x_input =
        ops::Const(root.WithOpName("x_input"), Input::Initializer(x));
    Output y_input =
        ops::Const(root.WithOpName("y_input"), Input::Initializer(y));
    Output last_output = ops::MatMul(
        root.WithOpName("matmul"), x_input, y_input,
        ops::MatMul::Attrs().TransposeA(transpose_x).TransposeB(transpose_y));
    auto& fused_ops = fused_ops_and_tensors.fused_ops;
    auto& fusion_tensors = fused_ops_and_tensors.fusion_tensors;
    for (int i = 0; i < fused_ops.size(); ++i) {
      const string& op = fused_ops[i];
      if (op == "BiasAdd") {
        Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                                Input::Initializer(fusion_tensors[i]));
        last_output = ops::BiasAdd(
            root.WithOpName(absl::StrCat("bias_add_at_", i)), last_output, arg);
      } else if (op == "Relu" || op == "Relu6" || op == "LeakyRelu" ||
                 op == "Elu" || op == "Tanh" || op == "Sigmoid" ||
                 op == "GeluApproximate") {
        last_output =
            ActivationOp(root, op, last_output, absl::StrCat(op, "_at_", i));
      } else if (op == "Add") {
        ASSERT_EQ(x.dtype(), fusion_tensors[i].dtype());
        Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                                Input::Initializer(fusion_tensors[i]));
        last_output = ops::AddV2(root.WithOpName(absl::StrCat("add_at_", i)),
                                 last_output, arg);
      } else {
        EXPECT_TRUE(false) << absl::StrCat("The fusion: [",
                                           absl::StrJoin(fused_ops, ","),
                                           "] is not supported in this test.");
      }
    }
    std::vector<Tensor> outputs;
    RunAndFetch(root, {last_output.name()}, &outputs);
    *result = outputs[0];
  }

  void RunFusedMatMul(const Tensor& x, const Tensor& y,
                      const FusedOpsAndTensors& fused_ops_and_tensors,
                      Tensor* result, bool transpose_x, bool transpose_y) {
    Scope root = tensorflow::Scope::NewRootScope();

    DataType dtype = DataTypeToEnum<T>::v();

    Output x_input =
        ops::Const(root.WithOpName("x_input"), Input::Initializer(x));
    Output y_input =
        ops::Const(root.WithOpName("y_input"), Input::Initializer(y));
    auto& fused_ops = fused_ops_and_tensors.fused_ops;
    auto& fusion_tensors = fused_ops_and_tensors.fusion_tensors;
    int num_fusion_inputs = 0;
    bool has_leaky_relu = false;
    std::vector<NodeDefBuilder::NodeOut> fusion_inputs;
    for (int i = 0; i < fused_ops.size(); ++i) {
      const string& op = fused_ops[i];
      if (op == "BiasAdd") {
        Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                                Input::Initializer(fusion_tensors[i]));
        fusion_inputs.push_back({arg.name(), 0, dtype});
        num_fusion_inputs++;
      } else if (op == "Add") {
        ASSERT_EQ(x.dtype(), fusion_tensors[i].dtype());
        Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                                Input::Initializer(fusion_tensors[i]));
        fusion_inputs.push_back({arg.name(), 0, dtype});
        num_fusion_inputs++;
      } else if (op == "LeakyRelu") {
        has_leaky_relu = true;
      } else {
        bool is_supported = op == "Relu" || op == "Relu6" ||
                            op == "LeakyRelu" || op == "Elu" || op == "Tanh" ||
                            op == "Sigmoid" || op == "GeluApproximate";
        EXPECT_TRUE(is_supported)
            << absl::StrCat("The fusion: [", absl::StrJoin(fused_ops, ","),
                            "] is not supported in this test.");
      }
    }
    NodeDef fused_matmul;
    std::vector<const NodeDef*> add_nodes;
    TF_EXPECT_OK(NodeDefBuilder("fused_batch_matmul", "_MklNativeFusedMatMul")
                     .Input({x_input.name(), 0, dtype})
                     .Input({y_input.name(), 0, dtype})
                     .Input(fusion_inputs)
                     .Attr("transpose_a", transpose_x)
                     .Attr("transpose_b", transpose_y)
                     .Attr("num_args", num_fusion_inputs)
                     .Attr("fused_ops", fused_ops)
                     .Attr("leakyrelu_alpha",
                           has_leaky_relu ? this->leakyrelu_alpha_ : 0.2f)
                     .Attr("_kernel", "MklNameChangeOp")
                     .Finalize(&fused_matmul));
    add_nodes = {&fused_matmul};
    std::vector<Tensor> outputs;
    RunAndFetch(root, {fused_matmul.name()}, &outputs, add_nodes);
    *result = outputs[0];
  }

  // Compute quantized tensor perchannel (aka axis) in SCALED mode for 2D
  // tensor.
  template <bool transpose = false>
  void GetPerchannelQuantizationTensors(const Tensor& input, Tensor* output,
                                        Tensor* min_tensor,
                                        Tensor* max_tensor) {
    ASSERT_EQ(input.dims(), 2);
    ASSERT_EQ(output->dtype(), DT_QINT8);
    constexpr int axis = transpose ? 0 : 1;
    int num_channels = input.dim_size(axis);
    ASSERT_EQ(min_tensor->NumElements(), num_channels);
    ASSERT_EQ(max_tensor->NumElements(), num_channels);

    auto eigen_input_tensor = input.matrix<T>().template cast<float>();
    auto eigen_output_tensor = output->matrix<qint8>();
    std::vector<float> scales(num_channels);
    float* min_tensor_buf = min_tensor->flat<float>().data();
    float* max_tensor_buf = max_tensor->flat<float>().data();
    for (int i = 0; i < num_channels; ++i) {
      auto input_slice = eigen_input_tensor.template chip<axis>(i);
      auto output_slice = eigen_output_tensor.template chip<axis>(i);
      Eigen::Tensor<float, 0, Eigen::RowMajor> min = input_slice.minimum();
      Eigen::Tensor<float, 0, Eigen::RowMajor> max = input_slice.maximum();
      float min_i = min();
      float max_i = max();
      float range = std::max(std::abs(min_i), std::abs(max_i));
      min_tensor_buf[i] = -range;
      max_tensor_buf[i] = range;
      const float scale = 127.0f / range;
      output_slice = (input_slice * scale).round().template cast<qint8>();
    }
  }

  void RunQuantizedMatMul(const Tensor& x, const Tensor& y,
                          const FusedOpsAndTensors& fused_ops_and_tensors,
                          Tensor* result, bool transpose_x, bool transpose_y,
                          string input_quant_mode, string output_quant_mode,
                          bool is_bias_quantized, bool is_perchannel,
                          bool requantize, float output_min, float output_max) {
    // TODO(intel-tf): Extend test with quantized bias
    ASSERT_EQ(is_bias_quantized, false);

    DataType real_dtype = DataTypeToEnum<T>::v();
    DataType qinput_dtype =
        (input_quant_mode == "MIN_FIRST") ? DT_QUINT8 : DT_QINT8;
    // Quantize x and y
    Tensor x_qtensor(qinput_dtype, x.shape());
    Tensor x_min_tensor(DT_FLOAT, TensorShape({}));
    Tensor x_max_tensor(DT_FLOAT, TensorShape({}));
    auto status = MklTestingUtil::GetQuantizationTensors<T>(
        x, &x_qtensor, qinput_dtype, input_quant_mode, &x_min_tensor,
        &x_max_tensor);
    ASSERT_TRUE(status.ok());
    Tensor y_qtensor(DT_QINT8, y.shape());
    const int num_channels = transpose_y ? y.dim_size(0) : y.dim_size(1);
    TensorShape minmax_shape =
        is_perchannel ? TensorShape({num_channels}) : TensorShape({});
    Tensor y_min_tensor(DT_FLOAT, minmax_shape);
    Tensor y_max_tensor(DT_FLOAT, minmax_shape);
    if (is_perchannel) {
      if (transpose_y) {
        GetPerchannelQuantizationTensors<true>(y, &y_qtensor, &y_min_tensor,
                                               &y_max_tensor);
      } else {
        GetPerchannelQuantizationTensors<false>(y, &y_qtensor, &y_min_tensor,
                                                &y_max_tensor);
      }
    } else {
      auto status = MklTestingUtil::GetQuantizationTensors<T>(
          y, &y_qtensor, DT_QINT8, "SCALED", &y_min_tensor, &y_max_tensor);
      ASSERT_TRUE(status.ok());
    }

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
    int num_fusion_inputs = 0;
    std::vector<NodeDefBuilder::NodeOut> fusion_inputs;
    bool has_leaky_relu = false;
    for (int i = 0; i < fused_ops.size(); ++i) {
      const string& op = fused_ops[i];
      if (op == "BiasAdd") {
        Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                                Input::Initializer(fusion_tensors[i]));
        fusion_inputs.push_back({arg.name(), 0, real_dtype});
        num_fusion_inputs++;
      } else if (op == "Add") {
        ASSERT_EQ(real_dtype, fusion_tensors[i].dtype());
        Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                                Input::Initializer(fusion_tensors[i]));
        fusion_inputs.push_back({arg.name(), 0, real_dtype});
        num_fusion_inputs++;
      } else if (op == "LeakyRelu") {
        has_leaky_relu = true;
      }
    }
    NodeDef fused_matmul;
    std::vector<const NodeDef*> add_nodes;
    std::vector<Tensor> outputs;
    std::vector<NodeDefBuilder::NodeOut> inputs;
    inputs.push_back({"x_input", 0, qinput_dtype});
    inputs.push_back({"y_input", 0, DT_QINT8});
    inputs.insert(std::end(inputs), std::begin(fusion_inputs),
                  std::end(fusion_inputs));
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

    TF_EXPECT_OK(NodeDefBuilder("quantized_fused_matmul", "_QuantizedMatMul")
                     .Attr("Tdevice_inputs", std::vector<DataType>())
                     .Input(FakeInput())
                     .Input(inputs)
                     .Attr("Thost_outputs", output_dtypes)
                     .Attr("Tdevice_outputs", std::vector<DataType>())
                     .Attr("T1", qinput_dtype)
                     .Attr("T2", DT_QINT8)
                     .Attr("Tbias", real_dtype)
                     .Attr("Tout", out_dtype)
                     .Attr("U", real_dtype)
                     .Attr("transpose_a", transpose_x)
                     .Attr("transpose_b", transpose_y)
                     .Attr("fused_ops", extended_fused_ops)
                     .Attr("leakyrelu_alpha",
                           has_leaky_relu ? this->leakyrelu_alpha_ : 0.2f)
                     .Attr("input_quant_mode", input_quant_mode)
                     .Attr("output_quant_mode", output_quant_mode)
                     .Finalize(&fused_matmul));
    if (requantize) {
      NodeDef dequantize;
      TF_EXPECT_OK(NodeDefBuilder("dequantize", "Dequantize")
                       .Input({"quantized_fused_matmul", 0, out_dtype})
                       .Input({"quantized_fused_matmul", 1, DT_FLOAT})
                       .Input({"quantized_fused_matmul", 2, DT_FLOAT})
                       .Attr("dtype", real_dtype)
                       .Attr("mode", output_quant_mode)
                       .Finalize(&dequantize));
      add_nodes = {&fused_matmul, &dequantize};
      RunAndFetch(root, {dequantize.name()}, &outputs, add_nodes);
    } else {
      add_nodes = {&fused_matmul};
      RunAndFetch(root, {fused_matmul.name()}, &outputs, add_nodes);
    }
    *result = outputs[0];
  }

  template <typename FusedGraphRunner>
  void VerifyTensorsNear(const std::vector<int64_t>& x_dims,
                         const std::vector<int64_t>& y_dims,
                         const FusedOpsAndDims& fused_ops_and_dims,
                         const GraphRunner& run_default,
                         const FusedGraphRunner& run_fused, bool transpose_x,
                         bool transpose_y, const double atol = 1e-5,
                         // The following arguments are used by quantized fusion
                         string input_quant_mode = "SCALED",
                         string output_quant_mode = "SCALED",
                         bool is_bias_quantized = false,
                         bool is_perchannel = false, bool requantize = false) {
    DataType dtype = DataTypeToEnum<T>::v();
    TensorShape x_shape = TensorShape(x_dims);
    TensorShape y_shape = TensorShape(y_dims);

    Tensor x_tensor(dtype, x_shape);
    x_tensor.flat<T>().setRandom();
    x_tensor.flat<T>() -= x_tensor.flat<T>().constant(static_cast<T>(0.5));

    Tensor y_tensor(dtype, y_shape);
    y_tensor.flat<T>().setRandom();
    y_tensor.flat<T>() -= y_tensor.flat<T>().constant(static_cast<T>(0.5));

    FusedOpsAndTensors fused_ops_and_tensors;
    fused_ops_and_tensors.fused_ops = fused_ops_and_dims.fused_ops;
    const auto& fused_ops = fused_ops_and_tensors.fused_ops;   // Alias to field
    const auto& fusion_dims = fused_ops_and_dims.fusion_dims;  // Alias to field
    auto& fusion_tensors = fused_ops_and_tensors.fusion_tensors;
    for (int i = 0; i < fused_ops.size(); ++i) {
      TensorShape arg_shape = TensorShape(fusion_dims[i]);
      Tensor arg_tensor(dtype, arg_shape);
      arg_tensor.flat<T>().setRandom();
      arg_tensor.flat<T>() -=
          arg_tensor.flat<T>().constant(static_cast<T>(0.5));
      fusion_tensors.push_back(arg_tensor);
    }
    Tensor default_result;
    run_default(x_tensor, y_tensor, fused_ops_and_tensors, &default_result,
                transpose_x, transpose_y);

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
      // Run quantized fusion.
      run_fused(x_tensor, y_tensor, fused_ops_and_tensors, &fused_result,
                transpose_x, transpose_y, input_quant_mode, output_quant_mode,
                is_bias_quantized, is_perchannel, requantize, output_min,
                output_max);
    } else {
      // Run realnumber type fusion.
      run_fused(x_tensor, y_tensor, fused_ops_and_tensors, &fused_result,
                transpose_x, transpose_y);
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
                              const int row, const int col,
                              FusedOpsAndDims* fused_ops_and_dims) {
    if (fused_ops == std::vector<string>{"BiasAdd"}) {
      *fused_ops_and_dims = {fused_ops, {std::vector<int64_t>{col}}};
    } else if (fused_ops == std::vector<string>{"BiasAdd", "Relu"} ||
               fused_ops == std::vector<string>{"BiasAdd", "Relu6"} ||
               fused_ops == std::vector<string>{"BiasAdd", "LeakyRelu"} ||
               fused_ops == std::vector<string>{"BiasAdd", "Elu"} ||
               fused_ops == std::vector<string>{"BiasAdd", "Tanh"} ||
               fused_ops == std::vector<string>{"BiasAdd", "Sigmoid"} ||
               fused_ops == std::vector<string>{"BiasAdd", "GeluApproximate"}) {
      *fused_ops_and_dims = {
          fused_ops, {std::vector<int64_t>{col}, std::vector<int64_t>{0}}};
    } else if (fused_ops == std::vector<string>{"BiasAdd", "Add"}) {
      *fused_ops_and_dims = {
          fused_ops,
          {std::vector<int64_t>{col}, std::vector<int64_t>{row, col}}};
    } else {
      EXPECT_TRUE(false) << absl::StrCat("The fusion: [",
                                         absl::StrJoin(fused_ops, ","),
                                         "] is not supported in this test.");
    }
  }

  void VerifyFusedMatMul(std::vector<string> fused_ops) {
    const GraphRunner run_default =
        [&](const Tensor& x, const Tensor& y,
            const FusedOpsAndTensors& fused_ops_and_tensors, Tensor* result,
            bool transpose_x, bool transpose_y) {
          this->RunMatMulAndFusedOps(x, y, fused_ops_and_tensors, result,
                                     transpose_x, transpose_y);
        };

    const GraphRunner run_fused =
        [&](const Tensor& x, const Tensor& y,
            const FusedOpsAndTensors& fused_ops_and_tensors, Tensor* result,
            bool transpose_x, bool transpose_y) {
          this->RunFusedMatMul(x, y, fused_ops_and_tensors, result, transpose_x,
                               transpose_y);
        };
    const double atol = std::is_same<T, bfloat16>::value ? 1e-2 : 1e-5;
    constexpr int M = 3;
    constexpr int K = 4;
    constexpr int N = 5;
    bool transpose_x = false;  // OpKernel does not support transpose_x.
    std::vector<int64_t> x_dims;
    std::vector<int64_t> y_dims;
    FusedOpsAndDims fused_ops_and_dims;
    GetFusionConfiguration(fused_ops, M, N, &fused_ops_and_dims);
    for (bool transpose_y : {false, true}) {
      x_dims =
          transpose_x ? std::vector<int64_t>{K, M} : std::vector<int64_t>{M, K};
      y_dims =
          transpose_y ? std::vector<int64_t>{N, K} : std::vector<int64_t>{K, N};
      VerifyTensorsNear<GraphRunner>(x_dims, y_dims, fused_ops_and_dims,
                                     run_default, run_fused, transpose_x,
                                     transpose_y, atol);
    }
  }

  // The following test runs with 32 configurations.
  //    (1) input quantization mode : {"MIN_FIRST", "SCALED"}
  //    (2) input quantization mode : {"MIN_FIRST", "SCALED"}
  //    (3) weight quantization per_channel : {false, true}
  //    (4) output is requantized or dequantized:
  //        false: dequantized
  //        true: requantized
  //    (5) weight matrix is transposed : {false, true}
  void VerifyQuantizedMatMul(std::vector<string> fused_ops) {
    if (!HasQuantizationSupport()) {
      GTEST_SKIP() << "oneDNN based Quantized ops are not enabled on this CPU.";
    }
    const GraphRunner run_default =
        [&](const Tensor& x, const Tensor& y,
            const FusedOpsAndTensors& fused_ops_and_tensors, Tensor* result,
            bool transpose_x, bool transpose_y) {
          this->RunMatMulAndFusedOps(x, y, fused_ops_and_tensors, result,
                                     transpose_x, transpose_y);
        };

    const QuantizedGraphRunner run_quantized =
        [&](const Tensor& x, const Tensor& y,
            const FusedOpsAndTensors& fused_ops_and_tensors, Tensor* result,
            bool transpose_x, bool transpose_y, string input_quant_mode,
            string output_quant_mode, bool is_bias_quantized,
            bool is_perchannel, bool requantize, float output_min,
            float output_max) {
          this->RunQuantizedMatMul(
              x, y, fused_ops_and_tensors, result, transpose_x, transpose_y,
              input_quant_mode, output_quant_mode, is_bias_quantized,
              is_perchannel, requantize, output_min, output_max);
        };

    const double atol = 1e-2;
    constexpr int M = 3;
    constexpr int K = 4;
    constexpr int N = 5;
    bool transpose_x = false;  // OpKernel does not support transpose_x.
    std::vector<int64_t> x_dims;
    std::vector<int64_t> y_dims;
    FusedOpsAndDims fused_ops_and_dims;
    GetFusionConfiguration(fused_ops, M, N, &fused_ops_and_dims);
    std::vector<bool> requantization_config;
    if (fused_ops == std::vector<string>{"BiasAdd", "Add"}) {
      // MatMul + BiasAdd + Add + Requantize fusion is not supported yet.
      requantization_config = {false};
    } else {
      requantization_config = {false, true};
    }
    for (bool transpose_y : {false, true}) {
      x_dims =
          transpose_x ? std::vector<int64_t>{K, M} : std::vector<int64_t>{M, K};
      y_dims =
          transpose_y ? std::vector<int64_t>{N, K} : std::vector<int64_t>{K, N};
      for (bool per_channel : {false, true}) {
        for (string input_quant_mode : {"MIN_FIRST", "SCALED"}) {
          for (string output_quant_mode : {"MIN_FIRST", "SCALED"}) {
            for (bool requantize : requantization_config) {
              VerifyTensorsNear<QuantizedGraphRunner>(
                  x_dims, y_dims, fused_ops_and_dims, run_default,
                  run_quantized, transpose_x, transpose_y, atol,
                  input_quant_mode, output_quant_mode, false, per_channel,
                  requantize);
            }
          }
        }
      }
    }
  }
};

TYPED_TEST_SUITE_P(FusedMatMulOpsTest);

// Realnumber typed test.
TYPED_TEST_P(FusedMatMulOpsTest, BiasAddGeluApproximate) {
  this->VerifyFusedMatMul({"BiasAdd", "GeluApproximate"});
}

// The following tests are for quantized fusions.
TYPED_TEST_P(FusedMatMulOpsTest, Quantized_BiasAdd) {
  this->VerifyQuantizedMatMul({"BiasAdd"});
}

TYPED_TEST_P(FusedMatMulOpsTest, Quantized_BiasAddRelu) {
  this->VerifyQuantizedMatMul({"BiasAdd", "Relu"});
}

TYPED_TEST_P(FusedMatMulOpsTest, Quantized_BiasAddRelu6) {
  this->VerifyQuantizedMatMul({"BiasAdd", "Relu6"});
}

TYPED_TEST_P(FusedMatMulOpsTest, Quantized_BiasAddLeakyRelu) {
  this->VerifyQuantizedMatMul({"BiasAdd", "LeakyRelu"});
}

TYPED_TEST_P(FusedMatMulOpsTest, Quantized_BiasAddElu) {
  this->VerifyQuantizedMatMul({"BiasAdd", "Elu"});
}

TYPED_TEST_P(FusedMatMulOpsTest, Quantized_BiasAddTanh) {
  this->VerifyQuantizedMatMul({"BiasAdd", "Tanh"});
}

TYPED_TEST_P(FusedMatMulOpsTest, Quantized_BiasAddSigmoid) {
  this->VerifyQuantizedMatMul({"BiasAdd", "Sigmoid"});
}

TYPED_TEST_P(FusedMatMulOpsTest, Quantized_BiasAddGeluApproximate) {
  this->VerifyQuantizedMatMul({"BiasAdd", "GeluApproximate"});
}

TYPED_TEST_P(FusedMatMulOpsTest, Quantized_BiasAddAdd) {
  this->VerifyQuantizedMatMul({"BiasAdd", "Add"});
}

REGISTER_TYPED_TEST_SUITE_P(FusedMatMulOpsTest, BiasAddGeluApproximate,
                            Quantized_BiasAdd, Quantized_BiasAddRelu,
                            Quantized_BiasAddRelu6, Quantized_BiasAddLeakyRelu,
                            Quantized_BiasAddElu, Quantized_BiasAddTanh,
                            Quantized_BiasAddSigmoid,
                            Quantized_BiasAddGeluApproximate,
                            Quantized_BiasAddAdd);

// TODO(intel-tf): Add bfloat16 to Types when PR#56613 is merged.
using DataTypes = ::testing::Types<float>;

INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedMatMulOpsTest, DataTypes);

}  // namespace tensorflow

#endif  // INTEL_MKL

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
#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/linalg_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/trt_convert_api.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

#if IS_TRT_VERSION_GE(8, 0, 0, 0)

namespace tensorflow {
namespace tensorrt {
namespace convert {

namespace ops = ::tensorflow::ops;
using ::tensorflow::testing::StatusIs;

// This anonymous namespace contains helper functions for instantiating small TF
// building blocks. These are used below to construct specific graph patterns
// which test end-to-end conversion of the TF graph to an explciit-precision
// enabled TensorRT network.
namespace {

enum class ConvEpilogueType {
  kNone,
  kReLU,
  kBatchNorm,
  kReLUBatchnorm,
  kBatchnormReLU
};

std::ostream& operator<<(std::ostream& os, ConvEpilogueType epilogue) {
  switch (epilogue) {
    case ConvEpilogueType::kNone:
      return os << "None";
    case ConvEpilogueType::kReLU:
      return os << "ReLU only";
    case ConvEpilogueType::kBatchNorm:
      return os << "BatchNorm Only";
    case ConvEpilogueType::kReLUBatchnorm:
      return os << "ReLU+Batchnorm";
    case ConvEpilogueType::kBatchnormReLU:
      return os << "BatchNorm+ReLU";
  }
}

std::string DebugString(ConvEpilogueType epilogue) {
  std::stringstream ss;
  ss << epilogue;
  return ss.str();
}

// Adds a 2D 3x3, single channel input with specified data_format. data_format
// must be NHWC,NCHW or NHW.
ops::Placeholder AddInput(Scope scope, int input_idx,
                          const std::string data_format,
                          std::array<int, 3> size_chw = {1, 3, 3}) {
  PartialTensorShape input_shape;
  if (data_format == "NCHW") {
    input_shape =
        PartialTensorShape({1, size_chw[0], size_chw[1], size_chw[2]});
  } else if (data_format == "NHWC") {
    input_shape =
        PartialTensorShape({1, size_chw[1], size_chw[2], size_chw[0]});
  } else if (data_format == "NHW") {
    input_shape = PartialTensorShape({1, size_chw[1], size_chw[2]});
  } else {
    LOG(FATAL) << "Unknown input shape type " << data_format;
  }
  auto input_attrs = ops::Placeholder::Attrs().Shape(input_shape);
  return ops::Placeholder(scope.WithOpName(absl::StrCat("input_", input_idx)),
                          DT_FLOAT, input_attrs);
}

// Adds QDQ op with min = -1.0f, max = 1.0f.
Output AddQDQV2(Scope scope, Input input) {
  // Create scaling factors.
  auto input_min =
      ops::Const<float>(scope.WithOpName("in_min"), -1.0f, TensorShape{});
  auto input_max =
      ops::Const<float>(scope.WithOpName("in_max"), 1.0f, TensorShape{});
  return ops::QuantizeAndDequantizeV2(scope.WithOpName("qdq"), input, input_min,
                                      input_max);
}

Output AddOutput(Scope scope, Output input, int idx, bool add_qdq) {
  Output out = input;
  if (add_qdq) {
    out = AddQDQV2(scope, input);
  }
  return ops::Identity(scope.WithOpName(StrCat("output_", idx)), out);
}

// Adds a 3x3x1x1 Conv2D op and optional bias weights, followed by ReLU
// activation. Puts QDQ between (weights, op). Puts QDQ between (input, op)
// when qdq_on_output=false. Otherwise, puts QDQ between (op, output).
Output AddConv2D(Scope scope, Input input, int in_channels, int out_channels,
                 std::array<int, 2> filter_size = {1, 1},
                 std::array<int, 2> stride = {1, 1},
                 const std::string& data_format = "NCHW", bool with_bias = true,
                 ConvEpilogueType epilogue = ConvEpilogueType::kBatchnormReLU,
                 bool qdq_on_output = false) {
  // Create 3x3 non-quantized weights weights.
  auto weights_const = ops::Const(
      scope.WithOpName("weights"), 1.0f,
      TensorShape({filter_size[0], filter_size[1], in_channels, out_channels}));

  // Add QDQ to input if we don't add QDQ to output.
  auto conv_input =
      !qdq_on_output ? AddQDQV2(scope.WithOpName("qdq_input"), input) : input;

  Output result = ops::Conv2D(
      scope.WithOpName("conv2d"), conv_input, AddQDQV2(scope, weights_const),
      /*strides=*/{1, 1, 1, 1},
      /*padding=*/"SAME", ops::Conv2D::Attrs().DataFormat(data_format));

  if (with_bias) {
    auto bias_const = ops::Const(scope.WithOpName("bias_weights"), 1.0f,
                                 TensorShape({
                                     out_channels,
                                 }));
    result = ops::BiasAdd(scope.WithOpName("bias"), result, bias_const,
                          ops::BiasAdd::Attrs().DataFormat(data_format));
  }

  auto add_bn = [scope, data_format](Input input,
                                     const int channels) -> Output {
    TensorShape constant_shape = TensorShape({channels});
    auto bn_scale =
        ops::Const(scope.WithOpName("bn_scale"), 1.0f, constant_shape);
    auto bn_offset =
        ops::Const(scope.WithOpName("bn_offset"), 1.0f, constant_shape);
    auto bn_mean =
        ops::Const(scope.WithOpName("bn_mean"), 0.1f, TensorShape({channels}));
    auto bn_var =
        ops::Const(scope.WithOpName("bn_var"), 1.0f, TensorShape({channels}));
    Input conv_bn_input = IS_TRT_VERSION_GE(8, 0, 1, 0)
                              ? input
                              : AddQDQV2(scope.WithOpName("qdq_input"), input);
    return ops::FusedBatchNormV3(
               scope.WithOpName("bn"), conv_bn_input, bn_scale, bn_offset,
               bn_mean, bn_var,
               ops::FusedBatchNormV3::Attrs().IsTraining(false).DataFormat(
                   data_format))
        .y;
  };

  switch (epilogue) {
    case ConvEpilogueType::kBatchNorm: {
      result = add_bn(result, out_channels);
      break;
    }
    case ConvEpilogueType::kReLU: {
      result = ops::Relu(scope.WithOpName("relu"), result);
      break;
    }
    case ConvEpilogueType::kReLUBatchnorm: {
      result = ops::Relu(scope.WithOpName("relu"), result);
      result = add_bn(result, out_channels);
      break;
    }
    case ConvEpilogueType::kBatchnormReLU: {
      result = add_bn(result, out_channels);
      result = ops::Relu(scope.WithOpName("relu"), result);
      break;
    }
    case ConvEpilogueType::kNone:
      break;
  }

  if (qdq_on_output) {
    result = AddQDQV2(scope.WithOpName("qdq_out"), result);
  }
  return result;
}

// Adds a batch matrix multiplication V2 operation, which commonly appears in
// fully connected layers. Puts QDQ between (input, op) as well as between
// (weights, op).
ops::BatchMatMulV2 AddMatMul(Scope scope, const std::string& name,
                             Input input) {
  // Add QDQ to input.
  auto input_qdq = AddQDQV2(scope, input);

  // Add 3x3 weights with QDQ.
  auto weights_const =
      ops::Const(scope.WithOpName(name + "_weights"),
                 {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
                 TensorShape({3, 3}));
  auto weights_qdq = AddQDQV2(scope.WithOpName("weights_qdq"), weights_const);
  return ops::BatchMatMulV2(scope.WithOpName(name), input_qdq, weights_qdq);
}
}  // namespace

struct QDQTestOptions {
  bool conv_has_bias{true};

  // TRT7 may have issues with optimizing redundant transpose operations between
  // QDQ and Op introduced by TF-TRT when format is not "NCHW". This allows to
  // test both cases as well as WAR feasibility.
  std::string data_format{"NCHW"};

  // Tests whether placing QDQ on outputs rather than inputs is handled
  // correctly.
  bool qdq_on_output{false};

  // Option for testing whether TRT build succeeds without a final QDQ before
  // the output.
  bool final_qdq{true};

  // Whether to add activations (relu) to conv operations
  ConvEpilogueType conv_epilogue;

  // TF-TRT API Options
  TfTrtConversionParams conversion_params{};
};

std::ostream& operator<<(std::ostream& os, const QDQTestOptions opts) {
  return os << absl::StrCat(
             "QDQTestOptions(conv_has_bias=",
             static_cast<int>(opts.conv_has_bias),
             ", qdq_on_output=", static_cast<int>(opts.qdq_on_output),
             ", data_format=", opts.data_format,
             ", conv_epilogue=", DebugString(opts.conv_epilogue),
             ", final_qdq=", opts.final_qdq, ")");
}

std::vector<QDQTestOptions> EnumerateQDQTestOptions() {
  std::vector<QDQTestOptions> result;
  for (const absl::string_view data_format : {"NCHW", "NHWC"}) {
    for (auto use_bias : {true, false}) {
      for (auto qdq_on_output : {false, true}) {
        // For now, always append a QDQ before output. For small single-op tests
        // (besides QDQ), TensorRT7 sometimes has trouble.
        for (auto final_qdq : {true, false}) {
          for (auto conv_epilogue :
               {ConvEpilogueType::kReLU, ConvEpilogueType::kNone,
                ConvEpilogueType::kBatchnormReLU}) {
            // Currently batch norm converter only supports NHWC.
            if (data_format == "NHWC" &&
                (conv_epilogue == ConvEpilogueType::kBatchnormReLU ||
                 conv_epilogue == ConvEpilogueType::kBatchNorm ||
                 conv_epilogue == ConvEpilogueType::kBatchnormReLU)) {
              continue;
            }
            QDQTestOptions opts{};
            opts.conv_has_bias = use_bias;
            opts.data_format = data_format;
            opts.qdq_on_output = qdq_on_output;
            opts.final_qdq = final_qdq;
            opts.conv_epilogue = conv_epilogue;
            result.push_back(opts);
          }
        }
      }
    }
  }
  return result;
}

// This class is a test fixture for running graph conversion and evaluating
// numerical results.
class QDQExplicitTest : public ::testing::Test,
                        public ::testing::WithParamInterface<QDQTestOptions> {
 public:
  static StatusOr<PartialTensorShape> GetShape(const std::string& name,
                                               const GraphShapeInfo& shapes) {
    TRT_ENSURE(shapes.find(name) != shapes.end());
    TRT_ENSURE(shapes.at(name).size() == 1);
    return shapes.at(name)[0].shape;
  }

  StatusOr<MetaGraphDef> GetModel(const GraphDef& graph_def,
                                  const std::vector<const NodeDef*>& inputs,
                                  const std::vector<const NodeDef*>& outputs,
                                  const GraphShapeInfo& shapes) {
    TRT_ENSURE(!inputs.empty());
    TRT_ENSURE(!outputs.empty());

    MetaGraphDef out;
    out.mutable_graph_def()->CopyFrom(graph_def);

    SignatureDef signature_def;
    auto& mutable_inputs = *signature_def.mutable_inputs();
    for (int i = 0; i < inputs.size(); i++) {
      std::string input_name = inputs[i]->name();
      auto& input = mutable_inputs[input_name];
      input.set_name(input_name);
      input.set_dtype(DT_FLOAT);
      TRT_ENSURE(shapes.find(input_name) != shapes.end());
      TRT_ENSURE(shapes.at(input_name).size() == 1);
      PartialTensorShape input_shape = shapes.at(input_name)[0].shape;
      input_shape.AsProto(input.mutable_tensor_shape());
    }

    auto& mutable_outputs = *signature_def.mutable_outputs();
    for (int i = 0; i < outputs.size(); i++) {
      std::string output_name = outputs[i]->name();
      auto& output = mutable_outputs[output_name];
      output.set_name(output_name);
      output.set_dtype(DT_FLOAT);
      TRT_ENSURE(shapes.find(output_name) != shapes.end());
      TRT_ENSURE(shapes.at(output_name).size() == 1);
      PartialTensorShape output_shape = shapes.at(output_name)[0].shape;
      output_shape.AsProto(output.mutable_tensor_shape());
    }

    (*out.mutable_signature_def())["serving_default"] = signature_def;
    return out;
  }

  // Confirms that we have a TRT node with the correct attributes.
  static Status CheckTrtNode(const GraphDef& converted_graph_def) {
    int n_trt_ops = 0;
    string op_name{"TRTEngineOp"};
    for (const auto& node : converted_graph_def.node()) {
      if (op_name == node.op()) {
        n_trt_ops++;
        const auto& attr = node.attr();
        TRT_ENSURE(attr.at("static_engine").b());
        VLOG(2) << "Found serialized segment with size "
                << attr.at("serialized_segment").s().size();
        TRT_ENSURE(!attr.at("serialized_segment").s().empty());
      }
    }
    TRT_ENSURE(n_trt_ops == 1);
    return OkStatus();
  }

  Status ConvertAndRun(Scope* scope) {
    std::vector<const NodeDef*> inputs;
    std::vector<const NodeDef*> outputs;

    GraphDef gdef;
    TF_RETURN_IF_ERROR(scope->ToGraphDef(&gdef));

    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    TF_RETURN_IF_ERROR(scope->ToGraph(graph.get()));

    GraphShapeInfo shape_info;
    TF_RETURN_IF_ERROR(InferShapes(graph.get(), /*arg_shapes=*/{},
                                   /*fnlib_def=*/nullptr, &shape_info));

    for (const NodeDef& node : gdef.node()) {
      if (absl::StartsWith(node.name(), "input_")) {
        inputs.push_back(&node);
      } else if (absl::StartsWith(node.name(), "output_")) {
        outputs.push_back(&node);
      }
    }

    StatusOr<MetaGraphDef> meta_graph_def =
        GetModel(gdef, inputs, outputs, shape_info);
    TRT_ENSURE_OK(meta_graph_def);

    // Create a list of input tensors, they will be used to build the engines.
    std::vector<Tensor> input_tensors;
    std::vector<std::string> input_names;
    for (const auto& input : inputs) {
      input_names.push_back(input->name());

      StatusOr<PartialTensorShape> input_shape =
          GetShape(input->name(), shape_info);
      TRT_ENSURE_OK(input_shape);

      TensorShape shape;
      input_shape->AsTensorShape(&shape);
      Tensor tensor(DT_FLOAT, shape);
      test::FillIota(&tensor, 1.0f);
      input_tensors.push_back(tensor);
    }

    std::vector<std::string> output_names;
    for (const auto& output : outputs) {
      output_names.push_back(output->name());
    }

    TfTrtConversionParams conversion_params;
    conversion_params.allow_build_at_runtime = true;
    conversion_params.precision_mode = TrtPrecisionMode::INT8;
    conversion_params.use_calibration = false;
    conversion_params.convert_to_static_engine = true;
    TRT_ENSURE(input_names.size() == input_tensors.size());
    StatusOr<GraphDef> converted_gdef = tensorrt::ConvertAndBuild(
        meta_graph_def->graph_def(), input_names, output_names, {input_tensors},
        conversion_params);
    TRT_ENSURE_OK(converted_gdef);
    return CheckTrtNode(*converted_gdef);
  }

 protected:
  TfTrtConversionParams params_;
  TrtUniquePtrType<nvinfer1::ICudaEngine> engine_;
};

class TestQDQSuite : public QDQExplicitTest {};

#define EXPECT_QDQ_ON_OUTPUT_FAILURE(params, scope)                  \
  if ((params).qdq_on_output) {                                      \
    EXPECT_THAT(ConvertAndRun(&(scope)), StatusIs(error::INTERNAL)); \
    return;                                                          \
  }
#define EXPECT_NO_FINAL_QDQ_FAILURE(params, scope)                   \
  if (!(params).final_qdq) {                                         \
    EXPECT_THAT(ConvertAndRun(&(scope)), StatusIs(error::INTERNAL)); \
    return;                                                          \
  }

#define EXPECT_BUILD_OK(scope) TF_EXPECT_OK(ConvertAndRun(&(scope)))

#define POLICY_TRT7(params, scope)               \
  if (!IS_TRT_VERSION_GE(8, 0, 0, 0)) {          \
    EXPECT_QDQ_ON_OUTPUT_FAILURE(params, scope); \
    EXPECT_NO_FINAL_QDQ_FAILURE(params, scope);  \
    EXPECT_BUILD_OK(scope);                      \
  }

#define POLICY_TRT8(params, scope)                                          \
  if (IS_TRT_VERSION_GE(8, 0, 0, 0)) {                                      \
    if (((params).conv_epilogue == ConvEpilogueType::kBatchNorm ||          \
         (params).conv_epilogue == ConvEpilogueType::kBatchnormReLU ||      \
         (params).conv_epilogue == ConvEpilogueType::kReLUBatchnorm) &&     \
        (params).data_format == "NHWC") {                                   \
      EXPECT_THAT(ConvertAndRun(&(scope)), StatusIs(error::UNIMPLEMENTED)); \
      return;                                                               \
    }                                                                       \
    EXPECT_BUILD_OK(scope);                                                 \
  }

#define SKIP_TRT7(x)                           \
  if (!IS_TRT_VERSION_GE(8, 0, 0, 0) && (x)) { \
    GTEST_SKIP();                              \
  }

// Tests single convolution operation conversion.
TEST_P(TestQDQSuite, TestConv2DBasic) {
  SKIP_TRT7(GetParam().qdq_on_output);
  SKIP_TRT7(GetParam().data_format != "NCHW");
  SKIP_TRT7(!GetParam().final_qdq);

  Scope scope = Scope::NewRootScope();
  auto input = AddInput(scope, 0, GetParam().data_format, {3, 28, 28});

  Output out = input;
  const int num_conv = 1;
  std::array<int, 2> in_channels = {3, 16};
  std::array<int, 2> out_channels = {16, 32};
  for (int i = 0; i < num_conv; i++) {
    out = AddConv2D(scope.WithOpName(absl::StrCat("conv_", i)), out,
                    in_channels[i], out_channels[i], /*filter_size=*/{3, 3},
                    /*stride=*/{1, 1}, GetParam().data_format,
                    GetParam().conv_has_bias, GetParam().conv_epilogue,
                    GetParam().qdq_on_output);
  }
  out = AddOutput(scope, out, 0, GetParam().final_qdq);
  POLICY_TRT7(GetParam(), scope);
  POLICY_TRT8(GetParam(), scope);
}

// Tests single convolution operation conversion.
TEST_P(TestQDQSuite, TestMatMulBasic) {
  // Some param's don't apply, so pick one combination and skip otherwise.
  if (GetParam().data_format != "NCHW" || !GetParam().conv_has_bias ||
      GetParam().qdq_on_output ||
      GetParam().conv_epilogue != ConvEpilogueType::kReLU) {
    GTEST_SKIP();
  }
  Scope scope = Scope::NewRootScope();
  auto input = AddInput(scope, 0, "NHW");
  auto matmul_op = AddMatMul(scope, "matmul", input);
  auto out = AddOutput(scope, matmul_op, 0, GetParam().final_qdq);

  TF_EXPECT_OK(ConvertAndRun(&scope));
}

// A single input goes through two different Conv2D. Outputs of Conv2D are
// added together, with QQQ on both branches of ADD.
TEST_P(TestQDQSuite, AddBothBranchesQDQConvSingleInput) {
  SKIP_TRT7(!GetParam().final_qdq);
  SKIP_TRT7(GetParam().data_format != "NCHW");

  Scope scope = Scope::NewRootScope();
  auto input1 = AddInput(scope, 0, GetParam().data_format,
                         /*size_chw=*/{3, 28, 28});

  auto conv1 =
      AddConv2D(scope, input1, 3, 16, /*filter_size=*/{3, 3}, /*stride=*/{1, 1},
                GetParam().data_format, GetParam().conv_has_bias,
                GetParam().conv_epilogue, GetParam().qdq_on_output);

  auto conv2 =
      AddConv2D(scope, input1, 3, 16, /*filter_size=*/{3, 3}, /*stride=*/
                {1, 1}, GetParam().data_format, GetParam().conv_has_bias,
                GetParam().conv_epilogue, GetParam().qdq_on_output);

  // In the case of "qdq on output", we don't need to add QDQ.
  auto add =
      ops::Add(scope.WithOpName("add"),
               !GetParam().qdq_on_output ? AddQDQV2(scope, conv1) : conv1,
               !GetParam().qdq_on_output ? AddQDQV2(scope, conv2) : conv2);

  auto conv3 =
      AddConv2D(scope.WithOpName("conv3"), conv2, 16, 16, {1, 1}, {1, 1},
                GetParam().data_format, GetParam().conv_has_bias,
                GetParam().conv_epilogue, GetParam().qdq_on_output);

  auto out =
      AddOutput(scope.WithOpName("output"), conv3, 0, GetParam().final_qdq);

  POLICY_TRT7(GetParam(), scope);
  POLICY_TRT8(GetParam(), scope);
}

// Tests adding a single tensor to itself, with QQQ on both branches of ADD.
TEST_P(TestQDQSuite, AddBothBranchesQDQMultipleInput) {
  // TRT7 QDQ optimizer makes single-input restriction.
  SKIP_TRT7(true);

  Scope scope = Scope::NewRootScope();
  auto input1 = AddInput(scope, 0, GetParam().data_format);
  auto input2 = AddInput(scope, 1, GetParam().data_format);
  auto add =
      ops::Add(scope.WithOpName("add"),
               !GetParam().qdq_on_output ? AddQDQV2(scope, input1) : input1,
               !GetParam().qdq_on_output ? AddQDQV2(scope, input2) : input2);
  auto output = AddOutput(scope, add, 0, true);
  TF_EXPECT_OK(ConvertAndRun(&scope));
}

// Tests Conv-MaxPool combination
TEST_P(TestQDQSuite, TestConvMaxpool) {
  SKIP_TRT7(!GetParam().final_qdq);
  SKIP_TRT7(GetParam().data_format != "NCHW");

  Scope scope = Scope::NewRootScope();
  auto input = AddInput(scope, 0, GetParam().data_format,
                        /*size_chw=*/{3, 28, 28});
  auto conv1 =
      AddConv2D(scope, input, 3, 16, /*filter_size=*/{3, 3}, /*stride=*/{1, 1},
                GetParam().data_format, GetParam().conv_has_bias,
                GetParam().conv_epilogue, GetParam().qdq_on_output);
  ops::MaxPool maxpool =
      ops::MaxPool(scope.WithOpName("maxpool"),
                   AddQDQV2(scope.WithOpName("mp_qdq_in"), conv1), {1, 1, 1, 1},
                   {1, 1, 1, 1}, "SAME",
                   ops::MaxPool::Attrs().DataFormat(GetParam().data_format));
  auto output =
      AddOutput(scope.WithOpName("output"), maxpool, 0, GetParam().final_qdq);
  POLICY_TRT7(GetParam(), scope);
  POLICY_TRT8(GetParam(), scope);
}

// Tests QDQ(Conv(QDQ(MaxPool(Conv(QDQ(x))))))
TEST_P(TestQDQSuite, TestConvMaxpoolConv) {
  SKIP_TRT7(!GetParam().final_qdq);
  SKIP_TRT7(GetParam().data_format != "NCHW");

  Scope scope = Scope::NewRootScope();
  auto input = AddInput(scope, 0, GetParam().data_format,
                        /*size_chw=*/{3, 28, 28});
  auto conv1 =
      AddConv2D(scope, input, 3, 16, /*filter_size=*/{3, 3}, /*stride=*/{1, 1},
                GetParam().data_format, GetParam().conv_has_bias,
                GetParam().conv_epilogue, GetParam().qdq_on_output);
  ops::MaxPool maxpool =
      ops::MaxPool(scope.WithOpName("maxpool"),
                   AddQDQV2(scope.WithOpName("mp_qdq_in"), conv1), {1, 1, 1, 1},
                   {1, 1, 1, 1}, "SAME",
                   ops::MaxPool::Attrs().DataFormat(GetParam().data_format));
  auto conv2 = AddConv2D(scope, maxpool, 16, 16, {3, 3}, {1, 1},
                         GetParam().data_format, GetParam().conv_has_bias,
                         GetParam().conv_epilogue, GetParam().qdq_on_output);
  auto output =
      AddOutput(scope.WithOpName("out"), conv2, 0, GetParam().final_qdq);
  POLICY_TRT7(GetParam(), scope);
  POLICY_TRT8(GetParam(), scope);
}

INSTANTIATE_TEST_SUITE_P(TestQDQSuiteInst, TestQDQSuite,
                         ::testing::ValuesIn(EnumerateQDQTestOptions()));

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // IS_TRT_VERSION_GE(8, 0, 0, 0)
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

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

#include "tensorflow/compiler/tf2tensorrt/experimental/trt_convert_api.h"

#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;

namespace tensorflow {
namespace tensorrt {

constexpr TrtConversionParams DefaultTestConversionParams = {
    1 << 20,  // max workspace size
    TrtPrecisionMode::FP32,
    3,      // minimum_segment_size
    1,      // max_cached_engines
    false,  // use_calibration
    false,  // use_dynamic_shape
    ProfileStrategy::kRangeOptimal,
    true,  // allow_build_at_runtime
};

const std::array<std::vector<std::vector<int64>>, 2> DefaultInputShapes = {
    {{{1, 2}}, {{1, 2}, {4, 2}}}};

struct TestParam {
  TrtConversionParams conv_params = DefaultTestConversionParams;
  bool use_variable = false;
  errors::Code expected_code = errors::OK;
  string expected_msg_substr = "";
};

#define SKIP_CONVERSION_ON_VALIDATION_ERROR() \
  if (param_.expected_code != errors::OK) {   \
    return;                                   \
  }

class TrtGraphConverterTest
    : public ::testing::TestWithParam<
          std::tuple<TestParam, std::vector<std::vector<int64>>,
                     TrtPrecisionMode, bool, bool, bool>> {
 protected:
  TrtGraphConverterTest()
      : param_(std::get<0>(GetParam())),
        input_shapes_(std::get<1>(GetParam())) {
    // Apply conversion param overrides
    param_.conv_params.precision_mode = std::get<2>(GetParam());
    param_.conv_params.use_calibration = std::get<3>(GetParam());
    param_.conv_params.use_dynamic_shape = std::get<4>(GetParam());
    param_.conv_params.allow_build_at_runtime = std::get<5>(GetParam());
    Reset();
  }

  void Reset() {
    PartialTensorShape shape({-1, 2});
    input_tensors_ = GetInputTensors(input_shapes_);
    calibration_tensors_ = GetInputTensors(input_shapes_);
    graph_ = GetGraphDef(shape);
    auto status_or_converter = TrtGraphConverter::Create(
        graph_, {"input"}, {"output"}, param_.conv_params);
    EXPECT_THAT(
        status_or_converter.status(),
        StatusIs(param_.expected_code, HasSubstr(param_.expected_msg_substr)));
    if (param_.expected_code == errors::OK) {
      converter_ = std::move(status_or_converter.value());
    }
  }

  // Returns the following graph: output = input * [42, 137] + input
  GraphDef GetGraphDef(PartialTensorShape input_shape) {
    Scope root = Scope::NewRootScope();
    Output c;
    c = ops::Const(root.WithOpName("my_const"), {{42.0f, 137.0f}});
    Output v;
    if (param_.use_variable) {
      Output v_handle = ops::VarHandleOp(root.WithOpName("my_var"),
                                         DataType::DT_FLOAT, {1, 2});
      v = ops::ReadVariableOp(root.WithOpName("my_var/Read/ReadVariableOp"),
                              v_handle, DataType::DT_FLOAT);
      auto v_init =
          ops::AssignVariableOp(root.WithOpName("my_var/init"), v_handle, c);
    } else {
      v = c;
    }
    const auto attrs = ops::Placeholder::Shape(input_shape);
    auto x = ops::Placeholder(root.WithOpName("input"), DT_FLOAT, attrs);
    auto y = ops::Mul(root.WithOpName("my_mul"), x, v);
    auto z = ops::Add(root.WithOpName("my_add"), x, y);
    auto q = ops::Identity(root.WithOpName("output"), z);

    GraphDef out;
    TF_CHECK_OK(root.ToGraphDef(&out));
    return out;
  }

  // Creates a list of input tensors, they will be used to build the engines.
  std::vector<std::vector<Tensor>> GetInputTensors(
      const std::vector<std::vector<int64>>& input_shapes) {
    std::vector<std::vector<Tensor>> input_tensors;
    for (const auto& shape : input_shapes) {
      Tensor tensor(DT_FLOAT, TensorShape(shape));
      test::FillIota(&tensor, 1.0f);
      input_tensors.push_back({tensor});
    }
    return input_tensors;
  }

  // Confirms that we have a TRT node with the correct attributes.
  void CheckTrtNode(const GraphDef& converted_graph_def) {
    int n_trt_ops = 0;
    string op_name{"TRTEngineOp"};
    for (const auto& node : converted_graph_def.node()) {
      if (!op_name.compare(node.op())) {
        n_trt_ops++;
      }
    }
    EXPECT_EQ(n_trt_ops, 1);
  }

  void RunAndCompareResults() {
    // Create a session to execute the original graph.
    Session* p_session = nullptr;
    TF_EXPECT_OK(NewSession(SessionOptions(), &p_session));
    std::unique_ptr<tensorflow::Session> session(p_session);
    TF_EXPECT_OK(session->Create(graph_));

    // Run models and compare the output.
    for (const std::vector<Tensor>& input : input_tensors_) {
      std::vector<Tensor> outputs;
      TF_EXPECT_OK(
          session->Run({{"input", input.at(0)}}, {"output"}, {}, &outputs));

      std::vector<Tensor> trt_outputs;
      TF_EXPECT_OK(converter_->session->Run({{"input", input.at(0)}}, {"output"}, {},
                                    &trt_outputs));
      ASSERT_EQ(outputs.size(), 1);
      ASSERT_EQ(trt_outputs.size(), 1);
      tensorflow::test::ExpectEqual(outputs.at(0), trt_outputs.at(0));
    }
  }

  TestParam param_;
  std::vector<std::vector<int64>> input_shapes_;
  std::vector<std::vector<Tensor>> input_tensors_;
  std::vector<std::vector<Tensor>> calibration_tensors_;
  std::unique_ptr<TrtGraphConverter> converter_;
  GraphDef graph_;
};

// Base class for FP32 and FP16 tests
typedef TrtGraphConverterTest TrtGraphConverter_FP32_FP16_Test;
// Base class for INT8 tests
typedef TrtGraphConverterTest TrtGraphConverter_INT8_Test;

// Parameters for base FP32 and FP16 tests
INSTANTIATE_TEST_CASE_P(
    TrtGraphConverterTestInstantiation, TrtGraphConverter_FP32_FP16_Test,
    ::testing::Combine(
        ::testing::Values(TestParam(),
                          // Validation failure case for conversion params.
                          TestParam{
                              TrtConversionParams{
                                  1 << 20,  // max workspace size
                                  TrtPrecisionMode::FP32,
                                  -2,     // minimum_segment_size
                                  1,      // max_cached_engines
                                  true,   // use_calibration
                                  false,  // use_dynamic_shape
                                  ProfileStrategy::kOptimal,
                                  true,  // allow_build_at_runtime
                              },
                              false,  // use_variable
                              errors::Code::INVALID_ARGUMENT,
                              "Minimum segment size should be positive or -1"},
                          // Validation failure case for non-frozen graph.
                          TestParam{
                              TrtConversionParams{
                                  1 << 20,  // max workspace size
                                  TrtPrecisionMode::FP32,
                                  3,      // minimum_segment_size
                                  1,      // max_cached_engines
                                  true,   // use_calibration
                                  false,  // use_dynamic_shape
                                  ProfileStrategy::kOptimal,
                                  true,  // allow_build_at_runtime
                              },
                              true,  // use_variable
                              errors::Code::INVALID_ARGUMENT,
                              "Input graph must be frozen"}),
        ::testing::ValuesIn(DefaultInputShapes),
        ::testing::Values(TrtPrecisionMode::FP32, TrtPrecisionMode::FP16),
        ::testing::Values(false), ::testing::Values(false, true),
        ::testing::Values(false, true)));

// Parameters for base INT8 tests
INSTANTIATE_TEST_CASE_P(
    TrtGraphConverterTestInstantiation, TrtGraphConverter_INT8_Test,
    ::testing::Combine(::testing::Values(TestParam()),
                       ::testing::ValuesIn(DefaultInputShapes),
                       ::testing::Values(TrtPrecisionMode::INT8),
                       ::testing::Values(true), ::testing::Values(false, true),
                       ::testing::Values(false, true)));

TEST_P(TrtGraphConverter_FP32_FP16_Test, BuildOffline) {
  SKIP_CONVERSION_ON_VALIDATION_ERROR();
  TF_EXPECT_OK(converter_->Convert().status());
  StatusOr<GraphDef> result = converter_->Build(input_tensors_);
  TF_ASSERT_OK(result.status());
  const GraphDef& converted_graph_def = result.value();
  CheckTrtNode(converted_graph_def);
  RunAndCompareResults();
}

TEST_P(TrtGraphConverter_FP32_FP16_Test, BuildAtRuntime) {
  SKIP_CONVERSION_ON_VALIDATION_ERROR();
  StatusOr<GraphDef> result = converter_->Convert();
  TF_ASSERT_OK(result.status());
  const GraphDef& converted_graph_def = result.value();
  CheckTrtNode(converted_graph_def);
  RunAndCompareResults();
}

TEST_P(TrtGraphConverter_INT8_Test, BuildOffline) {
  SKIP_CONVERSION_ON_VALIDATION_ERROR();
  TF_EXPECT_OK(converter_->Convert(calibration_tensors_).status());
  StatusOr<GraphDef> result = converter_->Build(input_tensors_);
  TF_ASSERT_OK(result.status());
  const GraphDef& converted_graph_def = result.value();
  CheckTrtNode(converted_graph_def);
  RunAndCompareResults();
}

TEST_P(TrtGraphConverter_INT8_Test, BuildAtRuntime) {
  SKIP_CONVERSION_ON_VALIDATION_ERROR();
  StatusOr<GraphDef> result = converter_->Convert(calibration_tensors_);
  TF_ASSERT_OK(result.status());
  const GraphDef& converted_graph_def = result.value();
  CheckTrtNode(converted_graph_def);
  RunAndCompareResults();
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

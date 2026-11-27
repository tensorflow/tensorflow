/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {
namespace {

struct RmsNormModelConfig {
  std::vector<int32_t> input_shape;
  std::optional<std::vector<int32_t>> weight_shape;
  std::optional<std::vector<int32_t>> output_shape;
  bool optional_weight_slot = false;
  float epsilon = 1e-6f;
  std::optional<std::vector<uint8_t>> raw_attributes;
};

std::vector<char> CreateRmsNormModel(const RmsNormModelConfig& config) {
  const std::vector<int32_t>& input_shape = config.input_shape;
  const std::vector<int32_t>& output_shape =
      config.output_shape.has_value() ? *config.output_shape : input_shape;
  const bool with_weight = config.weight_shape.has_value();

  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes;
  operator_codes.push_back(
      CreateOperatorCode(builder, BuiltinOperator_STABLEHLO_COMPOSITE));

  std::vector<flatbuffers::Offset<Buffer>> buffers;
  buffers.push_back(CreateBuffer(builder, builder.CreateVector({})));

  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.push_back(CreateTensor(
      builder, builder.CreateVector<int32_t>(input_shape), TensorType_FLOAT32));
  std::vector<int32_t> op_inputs = {0};
  std::vector<int32_t> subgraph_inputs = {0};
  int32_t output_tensor_idx = 1;
  if (with_weight) {
    tensors.push_back(CreateTensor(
        builder, builder.CreateVector<int32_t>(*config.weight_shape),
        TensorType_FLOAT32));
    op_inputs.push_back(1);
    subgraph_inputs.push_back(1);
    output_tensor_idx = 2;
  } else if (config.optional_weight_slot) {
    op_inputs.push_back(kTfLiteOptionalTensor);
  }
  tensors.push_back(CreateTensor(builder,
                                 builder.CreateVector<int32_t>(output_shape),
                                 TensorType_FLOAT32));

  std::vector<uint8_t> attr_buf;
  if (config.raw_attributes.has_value()) {
    attr_buf = *config.raw_attributes;
  } else {
    flexbuffers::Builder fbb;
    fbb.Map([&]() { fbb.Float("epsilon", config.epsilon); });
    fbb.Finish();
    attr_buf = fbb.GetBuffer();
  }

  auto composite_options = CreateStableHLOCompositeOptionsDirect(
      builder, "odml.rms_norm", /*decomposition_subgraph_index=*/1, &attr_buf);

  std::vector<int32_t> op_outputs = {output_tensor_idx};
  std::vector<flatbuffers::Offset<Operator>> operators;
  operators.push_back(CreateOperator(
      builder, /*opcode_index=*/0, builder.CreateVector<int32_t>(op_inputs),
      builder.CreateVector<int32_t>(op_outputs), BuiltinOptions_NONE, 0,
      /*custom_options=*/0, CustomOptionsFormat_FLEXBUFFERS,
      /*mutating_variable_inputs=*/0, /*intermediates=*/0,
      /*large_custom_options_offset=*/0, /*large_custom_options_size=*/0,
      BuiltinOptions2_StableHLOCompositeOptions, composite_options.Union()));

  auto subgraph0 =
      CreateSubGraph(builder, builder.CreateVector(tensors),
                     builder.CreateVector<int32_t>(subgraph_inputs),
                     builder.CreateVector<int32_t>(op_outputs),
                     builder.CreateVector(operators));

  // Subgraph 1: dummy decomposition subgraph (replaced by XNNPACK delegate).
  std::vector<flatbuffers::Offset<Tensor>> sub1_tensors;
  sub1_tensors.push_back(CreateTensor(
      builder, builder.CreateVector<int32_t>(input_shape), TensorType_FLOAT32));
  std::vector<int32_t> sub1_inputs = {0};
  int32_t sub1_output_idx = 0;
  if (with_weight) {
    sub1_tensors.push_back(CreateTensor(
        builder, builder.CreateVector<int32_t>(*config.weight_shape),
        TensorType_FLOAT32));
    sub1_inputs.push_back(1);
  } else if (config.optional_weight_slot) {
    sub1_inputs.push_back(kTfLiteOptionalTensor);
  }
  if (config.output_shape.has_value()) {
    sub1_output_idx = static_cast<int32_t>(sub1_tensors.size());
    sub1_tensors.push_back(
        CreateTensor(builder, builder.CreateVector<int32_t>(output_shape),
                     TensorType_FLOAT32));
  }
  std::vector<int32_t> sub1_outputs = {sub1_output_idx};
  std::vector<flatbuffers::Offset<Operator>> sub1_ops;
  auto subgraph1 = CreateSubGraph(builder, builder.CreateVector(sub1_tensors),
                                  builder.CreateVector<int32_t>(sub1_inputs),
                                  builder.CreateVector<int32_t>(sub1_outputs),
                                  builder.CreateVector(sub1_ops));

  auto description = builder.CreateString("odml.rms_norm composite model");
  auto model = CreateModel(builder, TFLITE_SCHEMA_VERSION,
                           builder.CreateVector(operator_codes),
                           builder.CreateVector({subgraph0, subgraph1}),
                           description, builder.CreateVector(buffers));
  builder.Finish(model);

  return std::vector<char>(
      reinterpret_cast<const char*>(builder.GetBufferPointer()),
      reinterpret_cast<const char*>(builder.GetBufferPointer() +
                                    builder.GetSize()));
}

void ExpectRejectedByDelegate(const RmsNormModelConfig& config) {
  std::vector<char> buffer = CreateRmsNormModel(config);
  const Model* model = GetModel(buffer.data());

  TfLiteXNNPackDelegateOptions options = TfLiteXNNPackDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&options),
               TfLiteXNNPackDelegateDelete);
  ASSERT_NE(delegate, nullptr);

  std::unique_ptr<Interpreter> interpreter;
  auto resolver =
      ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates();
  ASSERT_EQ(InterpreterBuilder(model, resolver)(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  // In :xnnpack_delegate_test_mode, unsupported nodes fail delegation with
  // kTfLiteDelegateError instead of silently falling back.
  EXPECT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()),
            kTfLiteDelegateError);
}

void RunRmsNormTest(
    const std::vector<int32_t>& input_shape, bool with_weight, float epsilon,
    bool force_fp16, bool optional_weight_slot = false,
    std::optional<std::vector<int32_t>> weight_shape = std::nullopt) {
  RmsNormModelConfig config;
  config.input_shape = input_shape;
  if (with_weight) {
    config.weight_shape = weight_shape.has_value()
                              ? *weight_shape
                              : std::vector<int32_t>{input_shape.back()};
  }
  config.optional_weight_slot = optional_weight_slot;
  config.epsilon = epsilon;
  std::vector<char> buffer = CreateRmsNormModel(config);
  const Model* model = GetModel(buffer.data());

  TfLiteXNNPackDelegateOptions options = TfLiteXNNPackDelegateOptionsDefault();
  if (force_fp16) {
    options.flags |= TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16;
  }
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&options),
               TfLiteXNNPackDelegateDelete);
  ASSERT_NE(delegate, nullptr);

  std::unique_ptr<Interpreter> interpreter;
  auto resolver =
      ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates();
  ASSERT_EQ(InterpreterBuilder(model, resolver)(&interpreter), kTfLiteOk);
  ASSERT_NE(interpreter, nullptr);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);

  // Verify the composite node was delegated into a single XNNPACK partition.
  ASSERT_EQ(interpreter->execution_plan().size(), 1);
  const int node_idx = interpreter->execution_plan()[0];
  const auto* node_and_reg = interpreter->node_and_registration(node_idx);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_STREQ(node_and_reg->second.custom_name, "TfLiteXNNPackDelegate");

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  const int32_t channels = input_shape.back();
  const int32_t total_elts = NumElements(interpreter->input_tensor(0));
  const int32_t outer = total_elts / channels;

  float* input_data = interpreter->typed_input_tensor<float>(0);
  for (int32_t i = 0; i < total_elts; ++i) {
    input_data[i] = dist(rng);
  }
  std::vector<float> weight_data(channels, 1.0f);
  if (with_weight) {
    float* w_ptr = interpreter->typed_input_tensor<float>(1);
    for (int32_t c = 0; c < channels; ++c) {
      weight_data[c] = dist(rng);
      w_ptr[c] = weight_data[c];
    }
  }

  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  const float* output_data = interpreter->typed_output_tensor<float>(0);
  const float tol = force_fp16 ? 2e-2f : 1e-5f;
  for (int32_t o = 0; o < outer; ++o) {
    double sum_sq = 0.0;
    for (int32_t c = 0; c < channels; ++c) {
      double v = input_data[o * channels + c];
      sum_sq += v * v;
    }
    double inv_rms = 1.0 / std::sqrt(sum_sq / channels + epsilon);
    for (int32_t c = 0; c < channels; ++c) {
      float expected = static_cast<float>(input_data[o * channels + c] *
                                          inv_rms * weight_data[c]);
      EXPECT_NEAR(output_data[o * channels + c], expected,
                  tol * std::max(1.0f, std::abs(expected)));
    }
  }
}

TEST(OdmlRmsNormTest, FP32WithWeight3D) {
  RunRmsNormTest({1, 8, 64}, /*with_weight=*/true, /*epsilon=*/1e-6f,
                 /*force_fp16=*/false);
}

TEST(OdmlRmsNormTest, FP32WithoutWeight2D) {
  RunRmsNormTest({4, 32}, /*with_weight=*/false, /*epsilon=*/1e-5f,
                 /*force_fp16=*/false);
}

TEST(OdmlRmsNormTest, FP32WithWeight4D) {
  RunRmsNormTest({2, 3, 5, 16}, /*with_weight=*/true, /*epsilon=*/1e-6f,
                 /*force_fp16=*/false);
}

TEST(OdmlRmsNormTest, ForceFP16WithWeight3D) {
  RunRmsNormTest({1, 8, 64}, /*with_weight=*/true, /*epsilon=*/1e-5f,
                 /*force_fp16=*/true);
}

TEST(OdmlRmsNormTest, ForceFP16WithoutWeight2D) {
  RunRmsNormTest({4, 32}, /*with_weight=*/false, /*epsilon=*/1e-5f,
                 /*force_fp16=*/true);
}

TEST(OdmlRmsNormTest, OptionalWeightTensorTreatedAsUnweighted) {
  RunRmsNormTest({4, 32}, /*with_weight=*/false, /*epsilon=*/1e-5f,
                 /*force_fp16=*/false, /*optional_weight_slot=*/true);
}

TEST(OdmlRmsNormTest, BroadcastWeightWithLeadingOne) {
  RunRmsNormTest({2, 8, 64}, /*with_weight=*/true, /*epsilon=*/1e-6f,
                 /*force_fp16=*/false, /*optional_weight_slot=*/false,
                 /*weight_shape=*/std::vector<int32_t>{1, 64});
}

TEST(OdmlRmsNormTest, MismatchedOutputShapeNotDelegated) {
  RmsNormModelConfig config;
  config.input_shape = {1, 8, 64};
  config.output_shape = std::vector<int32_t>{1, 8, 32};
  ExpectRejectedByDelegate(config);
}

TEST(OdmlRmsNormTest, IncompatibleWeightShapeNotDelegated) {
  RmsNormModelConfig config;
  config.input_shape = {1, 8, 64};
  config.weight_shape = std::vector<int32_t>{32};
  ExpectRejectedByDelegate(config);
}

TEST(OdmlRmsNormTest, HigherRankWeightNotDelegated) {
  RmsNormModelConfig config;
  config.input_shape = {4, 32};
  config.weight_shape = std::vector<int32_t>{1, 4, 32};
  ExpectRejectedByDelegate(config);
}

TEST(OdmlRmsNormTest, MalformedFlexbufferAttributesNotDelegated) {
  RmsNormModelConfig config;
  config.input_shape = {4, 32};
  config.raw_attributes = std::vector<uint8_t>{0xFF, 0xFF, 0xFF};
  ExpectRejectedByDelegate(config);
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite

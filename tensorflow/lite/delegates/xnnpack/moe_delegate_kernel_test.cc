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
#include "tensorflow/lite/delegates/xnnpack/moe_delegate_kernel.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {
namespace xnnpack {
namespace {

constexpr int kNumExperts = 2;
constexpr int kNumActiveExperts = 1;
constexpr int kModelDim = 8;
constexpr int kHiddenDim = 4;
constexpr int kNumTokens = 2;

// gate/ff1 are [hidden_dim, num_experts, 1, model_dim]; linear is the
// transpose. One scale per row is per-output-channel quantization.
constexpr int kGateFf1Rows = kHiddenDim * kNumExperts;
constexpr int kLinearRows = kModelDim * kNumExperts;

std::vector<uint8_t> MoeCustomData(const std::string& weight_type,
                                   const std::string& activation) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_experts", kNumExperts);
    fbb.Int("num_active_experts", kNumActiveExperts);
    fbb.Int("model_dim", kModelDim);
    fbb.Int("hidden_dim", kHiddenDim);
    fbb.String("weight_type", weight_type);
    fbb.String("activation", activation);
    fbb.Bool("renormalized_top_weights", true);
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

// Builds a single-node interpreter holding one custom "moe" op so that
// MoeExpertsDelegateKernel::IsSupported can be called against a real
// TfLiteContext. The op is never invoked, so the tensor contents are
// irrelevant; only types, shapes and constness matter.
class MoeNodeBuilder {
 public:
  MoeNodeBuilder() : interpreter_(std::make_unique<Interpreter>()) {
    registration_ = {nullptr, nullptr, nullptr, nullptr};
    registration_.builtin_code = kTfLiteBuiltinCustom;
    registration_.custom_name = "moe";
    registration_.version = 1;
  }

  void AddInput(TfLiteType type, const std::vector<int>& shape) {
    interpreter_->AddTensors(1);
    interpreter_->SetTensorParametersReadWrite(
        next_tensor_, type, "", shape, {kTfLiteNoQuantization, nullptr});
    inputs_.push_back(next_tensor_++);
  }

  void AddConstInput(TfLiteType type, const std::vector<int>& shape,
                     size_t num_bytes) {
    buffers_.push_back(std::vector<char>(num_bytes, 0));
    interpreter_->AddTensors(1);
    interpreter_->SetTensorParametersReadOnly(
        next_tensor_, type, "", shape, {kTfLiteNoQuantization, nullptr},
        buffers_.back().data(), buffers_.back().size());
    inputs_.push_back(next_tensor_++);
  }

  void AddOutput(TfLiteType type, const std::vector<int>& shape) {
    interpreter_->AddTensors(1);
    interpreter_->SetTensorParametersReadWrite(
        next_tensor_, type, "", shape, {kTfLiteNoQuantization, nullptr});
    outputs_.push_back(next_tensor_++);
  }

  void SetCustomData(const std::vector<uint8_t>& data) {
    custom_data_.assign(data.begin(), data.end());
  }

  void Finish() {
    interpreter_->SetInputs(inputs_);
    interpreter_->SetOutputs(outputs_);
    interpreter_->AddNodeWithParameters(inputs_, outputs_, custom_data_.data(),
                                        custom_data_.size(), nullptr,
                                        &registration_);
  }

  TfLiteStatus IsSupported() {
    const auto* pair = interpreter_->node_and_registration(0);
    return MoeExpertsDelegateKernel::IsSupported(
        interpreter_->primary_subgraph().context(), &pair->first, &pair->second,
        /*node_index=*/0);
  }

 private:
  std::unique_ptr<Interpreter> interpreter_;
  TfLiteRegistration registration_;
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  std::vector<char> custom_data_;
  // Backing storage for the constant tensors, which are referenced rather than
  // copied by the interpreter.
  std::vector<std::vector<char>> buffers_;
  int next_tensor_ = 0;
};

// Assembles the ten inputs of an int8 moe op. `*_scales` is the total element
// count of each scale tensor: `rows` is per-output-channel and `rows * n` is
// blockwise with n blocks along the input axis.
void BuildInt8Moe(MoeNodeBuilder& builder, int gate_scales, int ff1_scales,
                  int linear_scales,
                  const std::string& activation = "gelu_tanh") {
  builder.AddInput(kTfLiteFloat32, {1, 1, kNumTokens, kModelDim});
  builder.AddInput(kTfLiteFloat32, {1, 1, kNumTokens, kNumActiveExperts});
  builder.AddInput(kTfLiteInt32, {1, 1, kNumTokens, kNumActiveExperts});

  builder.AddConstInput(kTfLiteInt8, {kHiddenDim, kNumExperts, 1, kModelDim},
                        kGateFf1Rows * kModelDim * sizeof(int8_t));
  builder.AddConstInput(kTfLiteFloat32, {gate_scales},
                        gate_scales * sizeof(float));
  builder.AddConstInput(kTfLiteInt8, {kHiddenDim, kNumExperts, 1, kModelDim},
                        kGateFf1Rows * kModelDim * sizeof(int8_t));
  builder.AddConstInput(kTfLiteFloat32, {ff1_scales},
                        ff1_scales * sizeof(float));
  builder.AddConstInput(kTfLiteInt8, {kModelDim, kNumExperts, 1, kHiddenDim},
                        kLinearRows * kHiddenDim * sizeof(int8_t));
  builder.AddConstInput(kTfLiteFloat32, {linear_scales},
                        linear_scales * sizeof(float));
  builder.AddConstInput(kTfLiteFloat32, {kNumExperts},
                        kNumExperts * sizeof(float));

  builder.AddOutput(kTfLiteFloat32, {1, 1, kNumTokens, kModelDim});
  builder.SetCustomData(MoeCustomData("int8", activation));
  builder.Finish();
}

TEST(MoeDelegateKernelIsSupportedTest, AcceptsPerChannelScales) {
  MoeNodeBuilder builder;
  BuildInt8Moe(builder, kGateFf1Rows, kGateFf1Rows, kLinearRows);
  EXPECT_EQ(builder.IsSupported(), kTfLiteOk);
}

// Four scales per row, i.e. the input axis split into four blocks. This is the
// case the delegate could not previously represent.
TEST(MoeDelegateKernelIsSupportedTest, AcceptsBlockwiseScales) {
  MoeNodeBuilder builder;
  BuildInt8Moe(builder, kGateFf1Rows * 4, kGateFf1Rows * 4, kLinearRows * 4);
  EXPECT_EQ(builder.IsSupported(), kTfLiteOk);
}

// Different projections may be blocked differently, since gate/ff1 and linear
// have different input axes.
TEST(MoeDelegateKernelIsSupportedTest, AcceptsMixedBlockCounts) {
  MoeNodeBuilder builder;
  BuildInt8Moe(builder, kGateFf1Rows * 2, kGateFf1Rows, kLinearRows * 4);
  EXPECT_EQ(builder.IsSupported(), kTfLiteOk);
}

// A scale count that is not a whole number of scales per row cannot describe
// any blocking.
TEST(MoeDelegateKernelIsSupportedTest, RejectsScaleCountNotMultipleOfRows) {
  MoeNodeBuilder builder;
  BuildInt8Moe(builder, kGateFf1Rows + 1, kGateFf1Rows, kLinearRows);
  EXPECT_EQ(builder.IsSupported(), kTfLiteError);
}

TEST(MoeDelegateKernelIsSupportedTest, RejectsEmptyScaleTensor) {
  MoeNodeBuilder builder;
  BuildInt8Moe(builder, kGateFf1Rows, kGateFf1Rows, 0);
  EXPECT_EQ(builder.IsSupported(), kTfLiteError);
}

// The exporter now labels the activation "gelu_tanh"; the legacy "gelu"
// spelling has to keep working for already-converted models.
TEST(MoeDelegateKernelIsSupportedTest, AcceptsBothGeluSpellings) {
  MoeNodeBuilder tanh_builder;
  BuildInt8Moe(tanh_builder, kGateFf1Rows, kGateFf1Rows, kLinearRows,
               /*activation=*/"gelu_tanh");
  EXPECT_EQ(tanh_builder.IsSupported(), kTfLiteOk);

  MoeNodeBuilder legacy_builder;
  BuildInt8Moe(legacy_builder, kGateFf1Rows, kGateFf1Rows, kLinearRows,
               /*activation=*/"gelu");
  EXPECT_EQ(legacy_builder.IsSupported(), kTfLiteOk);
}

TEST(MoeDelegateKernelIsSupportedTest, RejectsUnknownActivation) {
  MoeNodeBuilder builder;
  BuildInt8Moe(builder, kGateFf1Rows, kGateFf1Rows, kLinearRows,
               /*activation=*/"silu");
  EXPECT_EQ(builder.IsSupported(), kTfLiteError);
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite

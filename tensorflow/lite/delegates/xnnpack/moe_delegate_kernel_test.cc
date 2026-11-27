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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"  // from @XNNPACK
#include "absl/algorithm/container.h"
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

std::vector<uint8_t> MoeCustomData(
    const std::string& weight_type, const std::string& activation,
    std::optional<float> swiglu_alpha = std::nullopt,
    std::optional<float> swiglu_limit = std::nullopt) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_experts", kNumExperts);
    fbb.Int("num_active_experts", kNumActiveExperts);
    fbb.Int("model_dim", kModelDim);
    fbb.Int("hidden_dim", kHiddenDim);
    fbb.String("weight_type", weight_type);
    fbb.String("activation", activation);
    if (swiglu_alpha.has_value()) fbb.Float("swiglu_alpha", *swiglu_alpha);
    if (swiglu_limit.has_value()) fbb.Float("swiglu_limit", *swiglu_limit);
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
    buffers_.reserve(16);
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

  void AddConstFloatInput(const std::vector<int>& shape,
                          const std::vector<float>& values) {
    buffers_.emplace_back(values.size() * sizeof(float));
    std::memcpy(buffers_.back().data(), values.data(),
                values.size() * sizeof(float));
    interpreter_->AddTensors(1);
    interpreter_->SetTensorParametersReadOnly(
        next_tensor_, kTfLiteFloat32, "", shape,
        {kTfLiteNoQuantization, nullptr}, buffers_.back().data(),
        buffers_.back().size());
    inputs_.push_back(next_tensor_++);
  }

  Interpreter* interpreter() { return interpreter_.get(); }

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
                  const std::string& activation = "gelu_tanh",
                  std::optional<float> swiglu_alpha = std::nullopt,
                  std::optional<float> swiglu_limit = std::nullopt) {
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
  builder.SetCustomData(
      MoeCustomData("int8", activation, swiglu_alpha, swiglu_limit));
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

TEST(MoeDelegateKernelIsSupportedTest, AcceptsSiluActivation) {
  MoeNodeBuilder builder;
  BuildInt8Moe(builder, kGateFf1Rows, kGateFf1Rows, kLinearRows,
               /*activation=*/"silu");
  EXPECT_EQ(builder.IsSupported(), kTfLiteOk);
}

// gpt-oss's clamped SwiGLU; alpha and limit default to gpt-oss's 1.702 / 7.
TEST(MoeDelegateKernelIsSupportedTest, AcceptsSwigluOai) {
  MoeNodeBuilder default_builder;
  BuildInt8Moe(default_builder, kGateFf1Rows, kGateFf1Rows, kLinearRows,
               /*activation=*/"swiglu_oai");
  EXPECT_EQ(default_builder.IsSupported(), kTfLiteOk);

  MoeNodeBuilder explicit_builder;
  BuildInt8Moe(explicit_builder, kGateFf1Rows, kGateFf1Rows, kLinearRows,
               /*activation=*/"swiglu_oai", /*swiglu_alpha=*/1.702f,
               /*swiglu_limit=*/7.0f);
  EXPECT_EQ(explicit_builder.IsSupported(), kTfLiteOk);
}

TEST(MoeDelegateKernelIsSupportedTest, RejectsNonPositiveSwigluLimit) {
  MoeNodeBuilder builder;
  BuildInt8Moe(builder, kGateFf1Rows, kGateFf1Rows, kLinearRows,
               /*activation=*/"swiglu_oai", /*swiglu_alpha=*/1.702f,
               /*swiglu_limit=*/0.0f);
  EXPECT_EQ(builder.IsSupported(), kTfLiteError);
}

TEST(MoeDelegateKernelIsSupportedTest, RejectsNonFiniteSwigluAlpha) {
  MoeNodeBuilder builder;
  BuildInt8Moe(builder, kGateFf1Rows, kGateFf1Rows, kLinearRows,
               /*activation=*/"swiglu_oai",
               /*swiglu_alpha=*/std::numeric_limits<float>::quiet_NaN(),
               /*swiglu_limit=*/7.0f);
  EXPECT_EQ(builder.IsSupported(), kTfLiteError);
}

TEST(MoeDelegateKernelIsSupportedTest, RejectsUnknownActivation) {
  MoeNodeBuilder builder;
  BuildInt8Moe(builder, kGateFf1Rows, kGateFf1Rows, kLinearRows,
               /*activation=*/"relu");
  EXPECT_EQ(builder.IsSupported(), kTfLiteError);
}

TEST(MoeDelegateKernelInvokeTest, ExecutesSiluSingleAndMultiToken) {
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);

  // Test both single-token (token 0 -> expert 0) and multi-token (tokens 1, 2
  // -> expert 1) routing paths with SiLU activation: silu(x) = x / (1 + e^-x).
  constexpr int kTokens = 3;
  MoeNodeBuilder builder;
  builder.AddInput(kTfLiteFloat32, {1, 1, kTokens, kModelDim});
  builder.AddInput(kTfLiteFloat32, {1, 1, kTokens, kNumActiveExperts});
  builder.AddInput(kTfLiteInt32, {1, 1, kTokens, kNumActiveExperts});

  std::vector<float> gate_w(kGateFf1Rows * kModelDim, 0.0f);
  std::vector<float> ff1_w(kGateFf1Rows * kModelDim, 0.0f);
  std::vector<float> linear_w(kLinearRows * kHiddenDim, 0.0f);
  // Set row (out=0, expert=e) -> row_idx = 0 * kNumExperts + e = e, col 0 = 1.
  for (int e = 0; e < kNumExperts; ++e) {
    gate_w[e * kModelDim + 0] = 1.0f;
    ff1_w[e * kModelDim + 0] = 2.0f;
    linear_w[e * kHiddenDim + 0] = 1.0f;
  }
  builder.AddConstFloatInput({kHiddenDim, kNumExperts, 1, kModelDim}, gate_w);
  builder.AddConstFloatInput({kHiddenDim, kNumExperts, 1, kModelDim}, ff1_w);
  builder.AddConstFloatInput({kModelDim, kNumExperts, 1, kHiddenDim}, linear_w);
  builder.AddConstFloatInput({kNumExperts}, {1.0f, 1.5f});

  builder.AddOutput(kTfLiteFloat32, {1, 1, kTokens, kModelDim});
  builder.SetCustomData(MoeCustomData("fp32", "silu"));
  builder.Finish();

  Interpreter* interp = builder.interpreter();
  ASSERT_EQ(interp->AllocateTensors(), kTfLiteOk);

  float* src = interp->typed_input_tensor<float>(0);
  std::fill(src, src + kTokens * kModelDim, 0.0f);
  src[0 * kModelDim + 0] = -1.0f;  // token 0 -> expert 0 (single-token path)
  src[1 * kModelDim + 0] = 1.0f;   // token 1 -> expert 1 (multi-token path)
  src[2 * kModelDim + 0] = -2.0f;  // token 2 -> expert 1 (multi-token path)

  float* top_weights = interp->typed_input_tensor<float>(1);
  top_weights[0] = 1.0f;
  top_weights[1] = 0.5f;
  top_weights[2] = 2.0f;

  int32_t* top_indices = interp->typed_input_tensor<int32_t>(2);
  top_indices[0] = 0;
  top_indices[1] = 1;
  top_indices[2] = 1;

  TfLiteIntArray* nodes = TfLiteIntArrayCreate(1);
  nodes->data[0] = 0;
  TfLiteDelegateParams params{};
  params.nodes_to_replace = nodes;

  TfLiteContext* ctx = interp->primary_subgraph().context();
  ctx->GetNodeAndRegistration = [](struct TfLiteContext* context,
                                   int node_index, TfLiteNode** node,
                                   TfLiteRegistration** registration) {
    auto* subgraph = static_cast<Subgraph*>(context->impl_);
    const auto* pair = subgraph->node_and_registration(node_index);
    *node = const_cast<TfLiteNode*>(&pair->first);
    *registration = const_cast<TfLiteRegistration*>(&pair->second);
    return kTfLiteOk;
  };
  auto kernel = MoeExpertsDelegateKernel::Create(ctx, &params, nullptr);
  TfLiteIntArrayFree(nodes);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(ctx), kTfLiteOk);
  ASSERT_EQ(kernel->Invoke(ctx), kTfLiteOk);

  auto silu = [](float x) { return x / (1.0f + std::exp(-x)); };
  const float* out = interp->typed_output_tensor<float>(0);
  EXPECT_NEAR(out[0 * kModelDim + 0], silu(-1.0f) * (-2.0f) * 1.0f * 1.0f,
              1e-5f);
  EXPECT_NEAR(out[1 * kModelDim + 0], silu(1.0f) * 2.0f * 1.5f * 0.5f, 1e-5f);
  EXPECT_NEAR(out[2 * kModelDim + 0], silu(-2.0f) * (-4.0f) * 1.5f * 2.0f,
              1e-5f);
}

// gpt-oss's clamped SwiGLU in double: (clamp(ff1, -limit, limit) + 1) * g *
// sigmoid(alpha * g) with g = min(gate, limit).
double SwigluOaiReference(double gate, double ff1, double alpha, double limit) {
  const double g = std::min(gate, limit);
  const double u = std::clamp(ff1, -limit, limit);
  return (u + 1.0) * g / (1.0 + std::exp(-alpha * g));
}

// Multiples of 1/16 in [-1, 1] from a fixed LCG. A product of two of them is a
// multiple of 1/256, so a projection over 32 of them, scaled by a power of two,
// is exact in fp32 whatever the summation order.
std::vector<float> GridValues(size_t count, uint32_t seed) {
  std::vector<float> values(count);
  uint32_t state = seed;
  for (float& value : values) {
    state = state * 1664525u + 1013904223u;
    value =
        static_cast<float>(static_cast<int>((state >> 16) % 33) - 16) / 16.0f;
  }
  return values;
}

// Runs an fp32 moe op with activation="swiglu_oai" through the kernel and
// compares every output with the op computed in double from the same values.
// Expert 0 gets one token, which takes the kernel's single-token path; experts
// 1 and 2 get several tokens each, which take the batched path. Unset options
// must act as gpt-oss's alpha 1.702 and limit 7.
void ExpectSwigluOaiMatchesReference(std::optional<float> swiglu_alpha,
                                     std::optional<float> swiglu_limit) {
  ASSERT_EQ(xnn_initialize(nullptr), xnn_status_success);

  constexpr int kExperts = 3;
  constexpr int kActive = 2;
  constexpr int kModel = 32;
  constexpr int kHidden = 24;
  constexpr int kTokens = 4;
  constexpr int kRoutes[kTokens][kActive] = {{0, 1}, {1, 2}, {2, 1}, {1, 2}};
  constexpr float kRouteWeights[kTokens][kActive] = {
      {0.75f, 0.25f}, {0.5f, 0.5f}, {0.625f, 0.375f}, {0.875f, 0.125f}};
  // Powers of two keep the inputs exact; 4 takes many projections past 7.
  constexpr float kTokenScales[kTokens] = {4.0f, 1.0f, 2.0f, 4.0f};
  const std::vector<float> per_expert_scale = {1.0f, 0.75f, 1.25f};
  const double alpha = swiglu_alpha.value_or(1.702f);
  const double limit = swiglu_limit.value_or(7.0f);

  const std::vector<float> gate_w = GridValues(kHidden * kExperts * kModel, 1);
  const std::vector<float> ff1_w = GridValues(kHidden * kExperts * kModel, 2);
  const std::vector<float> linear_w =
      GridValues(kModel * kExperts * kHidden, 3);
  std::vector<float> src = GridValues(kTokens * kModel, 4);
  for (int t = 0; t < kTokens; ++t) {
    for (int k = 0; k < kModel; ++k) {
      src[t * kModel + k] *= kTokenScales[t];
    }
  }

  MoeNodeBuilder builder;
  builder.AddInput(kTfLiteFloat32, {1, 1, kTokens, kModel});
  builder.AddInput(kTfLiteFloat32, {1, 1, kTokens, kActive});
  builder.AddInput(kTfLiteInt32, {1, 1, kTokens, kActive});
  builder.AddConstFloatInput({kHidden, kExperts, 1, kModel}, gate_w);
  builder.AddConstFloatInput({kHidden, kExperts, 1, kModel}, ff1_w);
  builder.AddConstFloatInput({kModel, kExperts, 1, kHidden}, linear_w);
  builder.AddConstFloatInput({kExperts}, per_expert_scale);
  builder.AddOutput(kTfLiteFloat32, {1, 1, kTokens, kModel});
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_experts", kExperts);
    fbb.Int("num_active_experts", kActive);
    fbb.Int("model_dim", kModel);
    fbb.Int("hidden_dim", kHidden);
    fbb.String("weight_type", "fp32");
    fbb.String("activation", "swiglu_oai");
    if (swiglu_alpha.has_value()) fbb.Float("swiglu_alpha", *swiglu_alpha);
    if (swiglu_limit.has_value()) fbb.Float("swiglu_limit", *swiglu_limit);
  });
  fbb.Finish();
  builder.SetCustomData(fbb.GetBuffer());
  builder.Finish();

  Interpreter* interp = builder.interpreter();
  ASSERT_EQ(interp->AllocateTensors(), kTfLiteOk);
  absl::c_copy(src, interp->typed_input_tensor<float>(0));
  float* top_weights = interp->typed_input_tensor<float>(1);
  int32_t* top_indices = interp->typed_input_tensor<int32_t>(2);
  for (int t = 0; t < kTokens; ++t) {
    for (int r = 0; r < kActive; ++r) {
      top_weights[t * kActive + r] = kRouteWeights[t][r];
      top_indices[t * kActive + r] = kRoutes[t][r];
    }
  }

  TfLiteIntArray* nodes = TfLiteIntArrayCreate(1);
  nodes->data[0] = 0;
  TfLiteDelegateParams params{};
  params.nodes_to_replace = nodes;
  TfLiteContext* ctx = interp->primary_subgraph().context();
  ctx->GetNodeAndRegistration = [](struct TfLiteContext* context,
                                   int node_index, TfLiteNode** node,
                                   TfLiteRegistration** registration) {
    auto* subgraph = static_cast<Subgraph*>(context->impl_);
    const auto* pair = subgraph->node_and_registration(node_index);
    *node = const_cast<TfLiteNode*>(&pair->first);
    *registration = const_cast<TfLiteRegistration*>(&pair->second);
    return kTfLiteOk;
  };
  auto kernel = MoeExpertsDelegateKernel::Create(ctx, &params, nullptr);
  TfLiteIntArrayFree(nodes);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(ctx), kTfLiteOk);
  ASSERT_EQ(kernel->Invoke(ctx), kTfLiteOk);

  // The reference, the sum of |terms| of each output, and per path (0: the
  // single-token expert 0, 1: the batched experts) how many gate and ff1
  // values lie above the limit, below -limit and within it.
  std::vector<double> expected(kTokens * kModel, 0.0);
  std::vector<double> magnitude(kTokens * kModel, 0.0);
  int gate_sides[2][3] = {};
  int ff1_sides[2][3] = {};
  auto side = [&](double v) { return v > limit ? 0 : (v < -limit ? 1 : 2); };
  for (int t = 0; t < kTokens; ++t) {
    for (int r = 0; r < kActive; ++r) {
      const int e = kRoutes[t][r];
      const int path = e == 0 ? 0 : 1;
      const double route_scale =
          static_cast<double>(per_expert_scale[e]) * kRouteWeights[t][r];
      std::vector<double> hidden(kHidden);
      for (int j = 0; j < kHidden; ++j) {
        const int row = (j * kExperts + e) * kModel;
        double gate = 0.0;
        double ff1 = 0.0;
        for (int k = 0; k < kModel; ++k) {
          gate += static_cast<double>(gate_w[row + k]) * src[t * kModel + k];
          ff1 += static_cast<double>(ff1_w[row + k]) * src[t * kModel + k];
        }
        ++gate_sides[path][side(gate)];
        ++ff1_sides[path][side(ff1)];
        hidden[j] = SwigluOaiReference(gate, ff1, alpha, limit);
      }
      for (int o = 0; o < kModel; ++o) {
        const int row = (o * kExperts + e) * kHidden;
        for (int j = 0; j < kHidden; ++j) {
          const double term = route_scale * linear_w[row + j] * hidden[j];
          expected[t * kModel + o] += term;
          magnitude[t * kModel + o] += std::abs(term);
        }
      }
    }
  }
  for (int path = 0; path < 2; ++path) {
    for (int s = 0; s < 3; ++s) {
      EXPECT_GT(gate_sides[path][s], 0) << "path " << path << ", side " << s;
      EXPECT_GT(ff1_sides[path][s], 0) << "path " << path << ", side " << s;
    }
  }

  // The gate and ff1 projections are exact (see GridValues). In fp32
  // (u = 2^-24) the activation is good to a few u and the 24-term down
  // projection to 24 u of the sum of |terms|, so 1e-5 of that sum is a wide
  // margin, while a wrong alpha, limit or formula moves outputs by far more.
  const float* out = interp->typed_output_tensor<float>(0);
  for (int i = 0; i < kTokens * kModel; ++i) {
    EXPECT_NEAR(out[i], expected[i], 1e-5 * magnitude[i]) << "output " << i;
  }
}

TEST(MoeDelegateKernelInvokeTest, SwigluOaiMatchesReferenceWithDefaultOptions) {
  ExpectSwigluOaiMatchesReference(/*swiglu_alpha=*/std::nullopt,
                                  /*swiglu_limit=*/std::nullopt);
}

TEST(MoeDelegateKernelInvokeTest,
     SwigluOaiMatchesReferenceWithExplicitOptions) {
  ExpectSwigluOaiMatchesReference(/*swiglu_alpha=*/1.25f,
                                  /*swiglu_limit=*/2.5f);
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite

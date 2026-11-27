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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"  // from @XNNPACK
#include "absl/algorithm/container.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
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
// TfLiteContext. For IsSupported the tensor contents are irrelevant; only
// types, shapes and constness matter. Run invokes the op.
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

  // Adds a constant tensor holding `bytes` without making it a node input, so
  // that one tensor can be passed as two inputs (see AddTensorInput).
  int AddConstTensor(TfLiteType type, const std::vector<int>& shape,
                     std::vector<char> bytes) {
    buffers_.push_back(std::move(bytes));
    interpreter_->AddTensors(1);
    interpreter_->SetTensorParametersReadOnly(
        next_tensor_, type, "", shape, {kTfLiteNoQuantization, nullptr},
        buffers_.back().data(), buffers_.back().size());
    return next_tensor_++;
  }

  void AddTensorInput(int tensor) { inputs_.push_back(tensor); }

  TfLiteStatus IsSupported() {
    const auto* pair = interpreter_->node_and_registration(0);
    return MoeExpertsDelegateKernel::IsSupported(
        interpreter_->primary_subgraph().context(), &pair->first, &pair->second,
        /*node_index=*/0);
  }

  // Runs the node through the XNNPACK delegate, which hands a moe node to
  // MoeExpertsDelegateKernel, and returns the output.
  std::vector<float> Run(const std::vector<float>& src,
                         const std::vector<float>& top_weights,
                         const std::vector<int32_t>& top_indices) {
    delegate_.reset(TfLiteXNNPackDelegateCreate(/*options=*/nullptr));
    if (interpreter_->ModifyGraphWithDelegate(delegate_.get()) != kTfLiteOk ||
        interpreter_->AllocateTensors() != kTfLiteOk) {
      ADD_FAILURE() << "could not apply the XNNPACK delegate";
      return {};
    }
    const std::vector<int>& plan = interpreter_->execution_plan();
    if (plan.size() != 1 ||
        interpreter_->node_and_registration(plan[0])->first.delegate !=
            delegate_.get()) {
      ADD_FAILURE() << "the XNNPACK delegate did not take the moe node";
      return {};
    }
    absl::c_copy(src, interpreter_->typed_tensor<float>(inputs_[0]));
    absl::c_copy(top_weights, interpreter_->typed_tensor<float>(inputs_[1]));
    absl::c_copy(top_indices, interpreter_->typed_tensor<int32_t>(inputs_[2]));
    if (interpreter_->Invoke() != kTfLiteOk) {
      ADD_FAILURE() << "the moe node did not run";
      return {};
    }
    const float* output = interpreter_->typed_tensor<float>(outputs_[0]);
    return std::vector<float>(output, output + src.size());
  }

 private:
  // Declared before the interpreter, which has to be destroyed first.
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate_{nullptr, &TfLiteXNNPackDelegateDelete};
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

TEST(MoeDelegateKernelIsSupportedTest, AcceptsSiluActivation) {
  MoeNodeBuilder builder;
  BuildInt8Moe(builder, kGateFf1Rows, kGateFf1Rows, kLinearRows,
               /*activation=*/"silu");
  EXPECT_EQ(builder.IsSupported(), kTfLiteOk);
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

// Where a node that is run takes its gate weights from.
enum class Gate {
  kFf1,        // the ff1 tensors themselves
  kCopyOfFf1,  // separate tensors holding the ff1 values
  kOther,      // separate tensors holding other values
};

// Deterministic values in [-1, 1].
float Pattern(int i, int salt) {
  return static_cast<float>((i * 37 + salt * 101) % 41 - 20) / 20.0f;
}

template <typename T>
std::vector<char> AsBytes(const std::vector<T>& values) {
  std::vector<char> bytes(values.size() * sizeof(T));
  if (!values.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}

// Adds a `rows` x `cols` expert weight tensor of `weight_type` (int4 packs two
// values into each int8 byte) and its per-row scales. Returns the tensor
// indices of the weights and the scales (-1 for fp32, which has no scales).
std::pair<int, int> AddExpertWeight(MoeNodeBuilder& builder,
                                    const std::string& weight_type, int rows,
                                    int cols, int salt) {
  const int count = rows * cols;
  if (weight_type == "fp32") {
    std::vector<float> weights(count);
    for (int i = 0; i < count; ++i) {
      weights[i] = Pattern(i, salt);
    }
    return {builder.AddConstTensor(kTfLiteFloat32, {count}, AsBytes(weights)),
            -1};
  }
  const bool is_int8 = weight_type == "int8";
  const int max_q = is_int8 ? 127 : 7;
  std::vector<int8_t> weights(is_int8 ? count : (count + 1) / 2, 0);
  for (int i = 0; i < count; ++i) {
    const int q = static_cast<int>(std::lround(Pattern(i, salt) * max_q));
    if (is_int8) {
      weights[i] = static_cast<int8_t>(q);
    } else {
      weights[i / 2] =
          static_cast<int8_t>(weights[i / 2] | ((q & 0xF) << (4 * (i % 2))));
    }
  }
  std::vector<float> scales(rows);
  for (int r = 0; r < rows; ++r) {
    scales[r] = (0.5f + 0.1f * static_cast<float>(r % 5)) / max_q;
  }
  const int weight_tensor = builder.AddConstTensor(
      kTfLiteInt8, {static_cast<int>(weights.size())}, AsBytes(weights));
  return {weight_tensor,
          builder.AddConstTensor(kTfLiteFloat32, {rows}, AsBytes(scales))};
}

// Runs one gelu_tanh moe node with one active expert per token, token t going
// to expert top_indices[t], and returns its output.
std::vector<float> RunMoe(const std::string& weight_type, Gate gate,
                          const std::vector<int32_t>& top_indices) {
  const int tokens = static_cast<int>(top_indices.size());
  MoeNodeBuilder builder;
  builder.AddInput(kTfLiteFloat32, {1, 1, tokens, kModelDim});
  builder.AddInput(kTfLiteFloat32, {1, 1, tokens, kNumActiveExperts});
  builder.AddInput(kTfLiteInt32, {1, 1, tokens, kNumActiveExperts});

  const std::pair<int, int> ff1 = AddExpertWeight(
      builder, weight_type, kGateFf1Rows, kModelDim, /*salt=*/1);
  const std::pair<int, int> gate_weight =
      gate == Gate::kFf1
          ? ff1
          : AddExpertWeight(builder, weight_type, kGateFf1Rows, kModelDim,
                            /*salt=*/gate == Gate::kCopyOfFf1 ? 1 : 2);
  const std::pair<int, int> linear = AddExpertWeight(
      builder, weight_type, kLinearRows, kHiddenDim, /*salt=*/3);
  const int per_expert_scale = builder.AddConstTensor(
      kTfLiteFloat32, {kNumExperts}, AsBytes(std::vector<float>{1.0f, 0.5f}));

  const bool is_quantized = weight_type != "fp32";
  for (const std::pair<int, int>& weight : {gate_weight, ff1, linear}) {
    builder.AddTensorInput(weight.first);
    if (is_quantized) {
      builder.AddTensorInput(weight.second);
    }
  }
  builder.AddTensorInput(per_expert_scale);
  builder.AddOutput(kTfLiteFloat32, {1, 1, tokens, kModelDim});
  builder.SetCustomData(MoeCustomData(weight_type, "gelu_tanh"));
  builder.Finish();

  std::vector<float> src(tokens * kModelDim);
  for (int i = 0; i < tokens * kModelDim; ++i) {
    src[i] = Pattern(i, /*salt=*/4);
  }
  std::vector<float> top_weights(tokens);
  for (int t = 0; t < tokens; ++t) {
    top_weights[t] = 1.0f - 0.25f * static_cast<float>(t);
  }
  return builder.Run(src, top_weights, top_indices);
}

// A node whose gate input is its ff1 input, as a model with ungated experts
// passes them, has to give the output of a node with a separate gate tensor
// holding the same values. The node with other gate values is the control
// that the output depends on the gate at all.
void ExpectGateIsFf1MatchesSeparateGate(
    const std::vector<int32_t>& top_indices) {
  for (const char* weight_type : {"fp32", "int8", "int4"}) {
    SCOPED_TRACE(weight_type);
    const std::vector<float> separate =
        RunMoe(weight_type, Gate::kCopyOfFf1, top_indices);
    const std::vector<float> aliased =
        RunMoe(weight_type, Gate::kFf1, top_indices);
    const std::vector<float> other =
        RunMoe(weight_type, Gate::kOther, top_indices);
    ASSERT_EQ(separate.size(), top_indices.size() * kModelDim);
    ASSERT_EQ(aliased.size(), separate.size());
    ASSERT_EQ(other.size(), separate.size());
    float max_output = 0.0f;
    float max_other_difference = 0.0f;
    for (size_t i = 0; i < separate.size(); ++i) {
      EXPECT_NEAR(aliased[i], separate[i],
                  1e-5f * std::max(1.0f, std::fabs(separate[i])))
          << "output " << i;
      max_output = std::max(max_output, std::fabs(separate[i]));
      max_other_difference =
          std::max(max_other_difference, std::fabs(other[i] - separate[i]));
    }
    EXPECT_GT(max_output, 1e-2f);
    EXPECT_GT(max_other_difference, 1e-2f);
  }
}

// Several tokens routed to one expert take the batched path (one dynamic fully
// connected call for all of them).
TEST(MoeDelegateKernelInvokeTest, GateIsFf1BatchedTokens) {
  ExpectGateIsFf1MatchesSeparateGate({1, 1, 1});
}

// One token per expert takes the single-token path.
TEST(MoeDelegateKernelInvokeTest, GateIsFf1SingleTokens) {
  ExpectGateIsFf1MatchesSeparateGate({1, 0});
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite

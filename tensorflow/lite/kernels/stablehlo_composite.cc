/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/control_flow_common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_composite {

struct State {
  int32_t subgraph_index;
  bool subgraph_has_dynamic_output_tensors = false;
  std::string name;
  int q_num_heads = 0;
  int kv_num_heads = 0;
  float scale = 0.0f;
  int chunk_size = 64;
  bool use_chunked_prefill = false;
  enum class Activation { kNone, kSilu };
  Activation activation = Activation::kNone;
  enum class UpdateRule { kLinear, kGated, kDelta, kGatedDelta };
  UpdateRule update_rule = UpdateRule::kGatedDelta;
  int num_groups = 0;
  enum class DeltaTransform { kNone, kSoftplus };
  DeltaTransform delta_transform = DeltaTransform::kNone;
  bool has_delta_min = false;
  bool has_delta_max = false;
  float delta_min = 0.0f;
  float delta_max = 0.0f;
};

constexpr char kCausalConvWithState1d[] = "odml.causal_conv_with_state_1d";
constexpr char kRecurrentLinearAttention[] = "odml.recurrent_linear_attention";
constexpr char kSelectiveStateSpace[] = "odml.selective_state_space";

TfLiteIntArray* NewSize(std::initializer_list<int> dims) {
  TfLiteIntArray* size = TfLiteIntArrayCreate(dims.size());
  int i = 0;
  for (int dim : dims) {
    size->data[i++] = dim;
  }
  return size;
}

bool IsSupportedQuantizedType(TfLiteType type) {
  return type == kTfLiteInt8 || type == kTfLiteUInt8 || type == kTfLiteInt16 ||
         type == kTfLiteInt32;
}

TfLiteStatus EnsureFloatOrQuantized(TfLiteContext* context,
                                    const TfLiteTensor* tensor) {
  TF_LITE_ENSURE(context, tensor != nullptr);
  TF_LITE_ENSURE(context, tensor->type == kTfLiteFloat32 ||
                              IsSupportedQuantizedType(tensor->type));
  if (IsSupportedQuantizedType(tensor->type)) {
    TF_LITE_ENSURE(context,
                   tensor->params.scale != 0.0f ||
                       tensor->quantization.type == kTfLiteAffineQuantization);
  }
  return kTfLiteOk;
}

float QuantizationScale(const TfLiteTensor* tensor, int channel = 0) {
  if (tensor->quantization.type == kTfLiteAffineQuantization &&
      tensor->quantization.params != nullptr) {
    const auto* quantization =
        reinterpret_cast<const TfLiteAffineQuantization*>(
            tensor->quantization.params);
    if (quantization->scale != nullptr && quantization->scale->size > 0) {
      const int scale_index =
          quantization->scale->size == 1
              ? 0
              : std::min(channel, quantization->scale->size - 1);
      return quantization->scale->data[scale_index];
    }
  }
  return tensor->params.scale;
}

int QuantizationZeroPoint(const TfLiteTensor* tensor, int channel = 0) {
  if (tensor->quantization.type == kTfLiteAffineQuantization &&
      tensor->quantization.params != nullptr) {
    const auto* quantization =
        reinterpret_cast<const TfLiteAffineQuantization*>(
            tensor->quantization.params);
    if (quantization->zero_point != nullptr &&
        quantization->zero_point->size > 0) {
      const int zero_point_index =
          quantization->zero_point->size == 1
              ? 0
              : std::min(channel, quantization->zero_point->size - 1);
      return quantization->zero_point->data[zero_point_index];
    }
  }
  return tensor->params.zero_point;
}

template <typename T>
T ClampTo(int64_t value) {
  const int64_t min_value = static_cast<int64_t>(std::numeric_limits<T>::min());
  const int64_t max_value = static_cast<int64_t>(std::numeric_limits<T>::max());
  return static_cast<T>(std::max(min_value, std::min(max_value, value)));
}

float ReadTensorValue(const TfLiteTensor* tensor, int index, int channel = 0) {
  switch (tensor->type) {
    case kTfLiteFloat32:
      return GetTensorData<float>(tensor)[index];
    case kTfLiteInt8:
      return (static_cast<int>(GetTensorData<int8_t>(tensor)[index]) -
              QuantizationZeroPoint(tensor, channel)) *
             QuantizationScale(tensor, channel);
    case kTfLiteUInt8:
      return (static_cast<int>(GetTensorData<uint8_t>(tensor)[index]) -
              QuantizationZeroPoint(tensor, channel)) *
             QuantizationScale(tensor, channel);
    case kTfLiteInt16:
      return (static_cast<int>(GetTensorData<int16_t>(tensor)[index]) -
              QuantizationZeroPoint(tensor, channel)) *
             QuantizationScale(tensor, channel);
    case kTfLiteInt32:
      return (static_cast<int64_t>(GetTensorData<int32_t>(tensor)[index]) -
              QuantizationZeroPoint(tensor, channel)) *
             QuantizationScale(tensor, channel);
    default:
      return 0.0f;
  }
}

void WriteTensorValue(TfLiteTensor* tensor, int index, float value,
                      int channel = 0) {
  switch (tensor->type) {
    case kTfLiteFloat32:
      GetTensorData<float>(tensor)[index] = value;
      break;
    case kTfLiteInt8: {
      const int64_t quantized = static_cast<int64_t>(
          std::llround(value / QuantizationScale(tensor, channel)) +
          QuantizationZeroPoint(tensor, channel));
      GetTensorData<int8_t>(tensor)[index] = ClampTo<int8_t>(quantized);
      break;
    }
    case kTfLiteUInt8: {
      const int64_t quantized = static_cast<int64_t>(
          std::llround(value / QuantizationScale(tensor, channel)) +
          QuantizationZeroPoint(tensor, channel));
      GetTensorData<uint8_t>(tensor)[index] = ClampTo<uint8_t>(quantized);
      break;
    }
    case kTfLiteInt16: {
      const int64_t quantized = static_cast<int64_t>(
          std::llround(value / QuantizationScale(tensor, channel)) +
          QuantizationZeroPoint(tensor, channel));
      GetTensorData<int16_t>(tensor)[index] = ClampTo<int16_t>(quantized);
      break;
    }
    case kTfLiteInt32: {
      const int64_t quantized = static_cast<int64_t>(
          std::llround(value / QuantizationScale(tensor, channel)) +
          QuantizationZeroPoint(tensor, channel));
      GetTensorData<int32_t>(tensor)[index] = ClampTo<int32_t>(quantized);
      break;
    }
    default:
      break;
  }
}

void ParseOdmlCompositeAttributes(State* state, const uint8_t* attributes,
                                  size_t attributes_size) {
  if (attributes == nullptr || attributes_size == 0) return;
  const flexbuffers::Map map =
      flexbuffers::GetRoot(attributes, attributes_size).AsMap();

  const flexbuffers::Reference q_num_heads = map["q_num_heads"];
  if (q_num_heads.IsInt()) state->q_num_heads = q_num_heads.AsInt32();

  const flexbuffers::Reference kv_num_heads = map["kv_num_heads"];
  if (kv_num_heads.IsInt()) state->kv_num_heads = kv_num_heads.AsInt32();

  const flexbuffers::Reference scale = map["scale"];
  if (scale.IsFloat()) state->scale = scale.AsFloat();

  const flexbuffers::Reference chunk_size = map["chunk_size"];
  if (chunk_size.IsInt() && chunk_size.AsInt32() > 0) {
    state->chunk_size = chunk_size.AsInt32();
  }
  const flexbuffers::Reference use_chunked_prefill = map["use_chunked_prefill"];
  if (use_chunked_prefill.IsBool()) {
    state->use_chunked_prefill = use_chunked_prefill.AsBool();
  }

  const flexbuffers::Reference activation = map["activation"];
  if (activation.IsString()) {
    const std::string activation_value = activation.AsString().str();
    if (activation_value == "silu" || activation_value == "swish") {
      state->activation = State::Activation::kSilu;
    }
  }

  const flexbuffers::Reference update_rule = map["update_rule"];
  if (update_rule.IsString()) {
    const std::string rule = update_rule.AsString().str();
    if (rule == "linear") {
      state->update_rule = State::UpdateRule::kLinear;
    } else if (rule == "gated") {
      state->update_rule = State::UpdateRule::kGated;
    } else if (rule == "delta") {
      state->update_rule = State::UpdateRule::kDelta;
    } else if (rule == "gated_delta") {
      state->update_rule = State::UpdateRule::kGatedDelta;
    }
  }

  const flexbuffers::Reference num_groups = map["num_groups"];
  if (num_groups.IsInt()) state->num_groups = num_groups.AsInt32();

  const flexbuffers::Reference delta_transform = map["delta_transform"];
  if (delta_transform.IsString()) {
    const std::string transform = delta_transform.AsString().str();
    if (transform == "softplus") {
      state->delta_transform = State::DeltaTransform::kSoftplus;
    }
  }

  const flexbuffers::Reference delta_softplus = map["delta_softplus"];
  if (delta_softplus.IsBool() && delta_softplus.AsBool()) {
    state->delta_transform = State::DeltaTransform::kSoftplus;
  }

  const flexbuffers::Reference delta_min = map["delta_min"];
  if (delta_min.IsFloat()) {
    state->has_delta_min = true;
    state->delta_min = delta_min.AsFloat();
  } else if (delta_min.IsInt()) {
    state->has_delta_min = true;
    state->delta_min = static_cast<float>(delta_min.AsInt64());
  }

  const flexbuffers::Reference delta_max = map["delta_max"];
  if (delta_max.IsFloat()) {
    state->has_delta_max = true;
    state->delta_max = delta_max.AsFloat();
  } else if (delta_max.IsInt()) {
    state->has_delta_max = true;
    state->delta_max = static_cast<float>(delta_max.AsInt64());
  }
}

int LastDim(const TfLiteTensor* tensor) {
  return SizeOfDimension(tensor, NumDimensions(tensor) - 1);
}

TfLiteStatus EnsureMaskType(TfLiteContext* context,
                            const TfLiteTensor* tensor) {
  TF_LITE_ENSURE(context, tensor != nullptr);
  TF_LITE_ENSURE(context, tensor->type == kTfLiteBool ||
                              tensor->type == kTfLiteFloat32 ||
                              IsSupportedQuantizedType(tensor->type));
  return kTfLiteOk;
}

bool ReadMaskValue(const TfLiteTensor* tensor, int index) {
  if (tensor->type == kTfLiteBool) {
    return GetTensorData<bool>(tensor)[index];
  }
  return ReadTensorValue(tensor, index) != 0.0f;
}

bool IsCausalConvWeightChannelMajor(const TfLiteTensor* weight, int channels) {
  return SizeOfDimension(weight, 0) == channels &&
         SizeOfDimension(weight, 1) != channels;
}

TfLiteStatus PrepareCausalConvWithState1d(TfLiteContext* context,
                                          TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->inputs->size >= 2);
  TF_LITE_ENSURE(context, node->inputs->size <= 4);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 2);

  const TfLiteTensor* input;
  const TfLiteTensor* weight;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &weight));
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, input));
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, weight));
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 3);
  TF_LITE_ENSURE_EQ(context, NumDimensions(weight), 2);

  const int batch = SizeOfDimension(input, 0);
  const int seq = SizeOfDimension(input, 1);
  const int channels = SizeOfDimension(input, 2);
  const bool weight_channel_major =
      IsCausalConvWeightChannelMajor(weight, channels);
  const int weight_channels = weight_channel_major ? SizeOfDimension(weight, 0)
                                                   : SizeOfDimension(weight, 1);
  const int kernel = weight_channel_major ? SizeOfDimension(weight, 1)
                                          : SizeOfDimension(weight, 0);
  TF_LITE_ENSURE_EQ(context, channels, weight_channels);
  TF_LITE_ENSURE(context, kernel >= 1);

  const TfLiteTensor* bias = nullptr;
  const TfLiteTensor* past_state = nullptr;
  if (node->inputs->size >= 3 &&
      node->inputs->data[2] != kTfLiteOptionalTensor) {
    const TfLiteTensor* third;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &third));
    if (NumDimensions(third) == 1) {
      bias = third;
    } else {
      past_state = third;
    }
  }
  if (node->inputs->size >= 4 &&
      node->inputs->data[3] != kTfLiteOptionalTensor) {
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 3, &past_state));
  }
  if (bias != nullptr) {
    TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, bias));
    TF_LITE_ENSURE_EQ(context, NumDimensions(bias), 1);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(bias, 0), channels);
  }
  if (past_state != nullptr) {
    TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, past_state));
    TF_LITE_ENSURE_EQ(context, NumDimensions(past_state), 3);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(past_state, 0), batch);
    if (weight_channel_major) {
      TF_LITE_ENSURE_EQ(context, SizeOfDimension(past_state, 1), channels);
      TF_LITE_ENSURE_EQ(context, SizeOfDimension(past_state, 2), kernel - 1);
    } else {
      TF_LITE_ENSURE_EQ(context, SizeOfDimension(past_state, 1), kernel - 1);
      TF_LITE_ENSURE_EQ(context, SizeOfDimension(past_state, 2), channels);
    }
  }

  TfLiteTensor* output;
  TfLiteTensor* present_state;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &present_state));
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, output));
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, present_state));
  TF_LITE_ENSURE_OK(
      context,
      context->ResizeTensor(context, output, NewSize({batch, seq, channels})));
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(
                                 context, present_state,
                                 weight_channel_major
                                     ? NewSize({batch, channels, kernel - 1})
                                     : NewSize({batch, kernel - 1, channels})));
  return kTfLiteOk;
}

TfLiteStatus PrepareSelectiveStateSpace(TfLiteContext* context,
                                        TfLiteNode* node) {
  const State* op_state = reinterpret_cast<State*>(node->user_data);
  if (op_state->has_delta_min && op_state->has_delta_max) {
    TF_LITE_ENSURE(context, op_state->delta_min <= op_state->delta_max);
  }
  TF_LITE_ENSURE(context, node->inputs->size >= 6);
  TF_LITE_ENSURE(context, node->inputs->size <= 10);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 2);

  const TfLiteTensor* x;
  const TfLiteTensor* delta;
  const TfLiteTensor* a;
  const TfLiteTensor* b_tensor;
  const TfLiteTensor* c_tensor;
  const TfLiteTensor* past_state = nullptr;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &x));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &delta));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &a));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 3, &b_tensor));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 4, &c_tensor));
  if (node->inputs->data[5] != kTfLiteOptionalTensor) {
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 5, &past_state));
  }
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, x));
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, delta));
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, a));
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, b_tensor));
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, c_tensor));
  if (past_state != nullptr) {
    TF_LITE_ENSURE(context, past_state->type == kTfLiteFloat32);
  }

  TF_LITE_ENSURE(context, NumDimensions(x) == 3 || NumDimensions(x) == 4);
  const int batch = SizeOfDimension(x, 0);
  const int seq = SizeOfDimension(x, 1);
  const int heads = SizeOfDimension(x, 2);
  const int head_dim = NumDimensions(x) == 4 ? SizeOfDimension(x, 3) : 1;
  TF_LITE_ENSURE(context, heads > 0);
  TF_LITE_ENSURE(context, head_dim > 0);

  TF_LITE_ENSURE(context,
                 NumDimensions(delta) == 3 || NumDimensions(delta) == 4);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(delta, 0), batch);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(delta, 1), seq);
  TF_LITE_ENSURE(context, SizeOfDimension(delta, 2) == 1 ||
                              SizeOfDimension(delta, 2) == heads);
  if (NumDimensions(delta) == 4) {
    TF_LITE_ENSURE(context, SizeOfDimension(delta, 3) == 1 ||
                                SizeOfDimension(delta, 3) == head_dim);
  }

  TF_LITE_ENSURE(context,
                 NumDimensions(b_tensor) == 3 || NumDimensions(b_tensor) == 4);
  TF_LITE_ENSURE(context,
                 NumDimensions(c_tensor) == 3 || NumDimensions(c_tensor) == 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(b_tensor), NumDimensions(c_tensor));
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(b_tensor, 0), batch);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(b_tensor, 1), seq);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(c_tensor, 0), batch);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(c_tensor, 1), seq);
  const int groups =
      NumDimensions(b_tensor) == 4 ? SizeOfDimension(b_tensor, 2) : 1;
  const int state_dim = LastDim(b_tensor);
  TF_LITE_ENSURE(context, groups > 0);
  TF_LITE_ENSURE(context, state_dim > 0);
  TF_LITE_ENSURE_EQ(context, LastDim(c_tensor), state_dim);
  if (NumDimensions(c_tensor) == 4) {
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(c_tensor, 2), groups);
  }
  if (op_state->num_groups > 0) {
    TF_LITE_ENSURE_EQ(context, op_state->num_groups, groups);
  }
  TF_LITE_ENSURE_EQ(context, heads % groups, 0);

  TF_LITE_ENSURE(context, NumDimensions(a) >= 1 && NumDimensions(a) <= 3);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(a, 0), heads);
  if (NumDimensions(a) == 2) {
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(a, 1), state_dim);
  } else if (NumDimensions(a) == 3) {
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(a, 1), head_dim);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(a, 2), state_dim);
  }

  if (past_state != nullptr) {
    TF_LITE_ENSURE_EQ(context, NumDimensions(past_state), 4);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(past_state, 0), batch);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(past_state, 1), heads);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(past_state, 2), head_dim);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(past_state, 3), state_dim);
  }

  for (int i = 6; i < node->inputs->size; ++i) {
    if (node->inputs->data[i] == kTfLiteOptionalTensor) continue;
    const TfLiteTensor* tensor;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &tensor));
    if (i == 8 || i == 9) {
      TF_LITE_ENSURE_OK(context, EnsureMaskType(context, tensor));
      TF_LITE_ENSURE_EQ(context, NumDimensions(tensor), 2);
      TF_LITE_ENSURE_EQ(context, SizeOfDimension(tensor, 0), batch);
      TF_LITE_ENSURE_EQ(context, SizeOfDimension(tensor, 1), seq);
      continue;
    }
    TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, tensor));
    TF_LITE_ENSURE(context,
                   NumDimensions(tensor) == 1 || NumDimensions(tensor) == 2);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(tensor, 0), heads);
    if (NumDimensions(tensor) == 2) {
      TF_LITE_ENSURE_EQ(context, SizeOfDimension(tensor, 1), head_dim);
    }
  }

  TfLiteTensor* output;
  TfLiteTensor* present_state;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &present_state));
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, output));
  TF_LITE_ENSURE(context, present_state->type == kTfLiteFloat32);
  if (NumDimensions(x) == 4) {
    TF_LITE_ENSURE_OK(
        context, context->ResizeTensor(context, output,
                                       NewSize({batch, seq, heads, head_dim})));
  } else {
    TF_LITE_ENSURE_OK(
        context,
        context->ResizeTensor(context, output, NewSize({batch, seq, heads})));
  }
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(
                                 context, present_state,
                                 NewSize({batch, heads, head_dim, state_dim})));
  return kTfLiteOk;
}

TfLiteStatus PrepareRecurrentLinearAttention(TfLiteContext* context,
                                             TfLiteNode* node) {
  const State* op_state = reinterpret_cast<State*>(node->user_data);
  TF_LITE_ENSURE(context, node->inputs->size >= 3);
  TF_LITE_ENSURE(context, node->inputs->size <= 6);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 2);

  const TfLiteTensor* query;
  const TfLiteTensor* key;
  const TfLiteTensor* value;
  const TfLiteTensor* past_state = nullptr;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &query));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &key));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &value));
  if (node->inputs->size >= 4 &&
      node->inputs->data[3] != kTfLiteOptionalTensor) {
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 3, &past_state));
  }
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, query));
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, key));
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, value));
  if (past_state != nullptr) {
    TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, past_state));
  }
  TF_LITE_ENSURE(context,
                 NumDimensions(query) == 3 || NumDimensions(query) == 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(query), NumDimensions(key));
  TF_LITE_ENSURE_EQ(context, NumDimensions(query), NumDimensions(value));
  if (past_state != nullptr) {
    TF_LITE_ENSURE_EQ(context, NumDimensions(past_state), 4);
  }

  const int batch = SizeOfDimension(query, 0);
  const int seq = SizeOfDimension(query, 1);
  int kv_heads = op_state->kv_num_heads;
  if (kv_heads <= 0 && past_state != nullptr) {
    kv_heads = SizeOfDimension(past_state, 1);
  }
  if (kv_heads <= 0 && NumDimensions(key) == 4) {
    kv_heads = SizeOfDimension(key, 2);
  }
  TF_LITE_ENSURE(context, kv_heads > 0);
  const int q_heads =
      op_state->q_num_heads > 0 ? op_state->q_num_heads : kv_heads;
  const int key_dim =
      past_state != nullptr
          ? SizeOfDimension(past_state, 2)
          : (NumDimensions(key) == 4 ? LastDim(key) : LastDim(key) / kv_heads);
  const int value_dim =
      past_state != nullptr
          ? SizeOfDimension(past_state, 3)
          : (NumDimensions(value) == 4 ? LastDim(value)
                                       : LastDim(value) / kv_heads);
  if (past_state != nullptr) {
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(past_state, 0), batch);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(past_state, 1), kv_heads);
  }
  TF_LITE_ENSURE(context, q_heads >= kv_heads);
  TF_LITE_ENSURE_EQ(context, q_heads % kv_heads, 0);

  if (NumDimensions(query) == 4) {
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(query, 2), q_heads);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(key, 2), kv_heads);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(value, 2), kv_heads);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(query, 3), key_dim);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(key, 3), key_dim);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(value, 3), value_dim);
  } else {
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(query, 2), q_heads * key_dim);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(key, 2), kv_heads * key_dim);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(value, 2), kv_heads * value_dim);
  }

  for (int i = 4; i < node->inputs->size; ++i) {
    if (node->inputs->data[i] == kTfLiteOptionalTensor) continue;
    const TfLiteTensor* tensor;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &tensor));
    TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, tensor));
    TF_LITE_ENSURE_EQ(context, NumDimensions(tensor), 3);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(tensor, 0), batch);
    TF_LITE_ENSURE_EQ(context, SizeOfDimension(tensor, 1), seq);
    TF_LITE_ENSURE(
        context,
        SizeOfDimension(tensor, 2) == 1 ||
            SizeOfDimension(tensor, 2) == kv_heads ||
            SizeOfDimension(tensor, 2) == q_heads ||
            (i == 4 && SizeOfDimension(tensor, 2) == key_dim) ||
            (i == 4 && SizeOfDimension(tensor, 2) == kv_heads * key_dim));
  }

  TfLiteTensor* output;
  TfLiteTensor* present_state;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &present_state));
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, output));
  TF_LITE_ENSURE_OK(context, EnsureFloatOrQuantized(context, present_state));
  if (NumDimensions(query) == 4) {
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(
                                   context, output,
                                   NewSize({batch, seq, q_heads, value_dim})));
  } else {
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(
                                   context, output,
                                   NewSize({batch, seq, q_heads * value_dim})));
  }
  TF_LITE_ENSURE_OK(
      context,
      context->ResizeTensor(context, present_state,
                            NewSize({batch, kv_heads, key_dim, value_dim})));
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* options, size_t options_len) {
  auto data = std::make_unique<State>();
  const TfLiteStablehloCompositeParams* params =
      reinterpret_cast<const TfLiteStablehloCompositeParams*>(options);
  data->subgraph_index = params->subgraph_index;
  if (params->name != nullptr) {
    data->name = params->name;
  }
  ParseOdmlCompositeAttributes(data.get(), params->attributes,
                               params->attributes_size);
  return data.release();
}

void Free(TfLiteContext* context, void* node_data) {
  delete static_cast<State*>(node_data);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  State* op_state = reinterpret_cast<State*>(node->user_data);

  if (op_state->name == kCausalConvWithState1d) {
    return PrepareCausalConvWithState1d(context, node);
  }
  if (op_state->name == kRecurrentLinearAttention) {
    return PrepareRecurrentLinearAttention(context, node);
  }
  if (op_state->name == kSelectiveStateSpace) {
    return PrepareSelectiveStateSpace(context, node);
  }

  TF_LITE_ENSURE(context, node->inputs->size > 0);

  const int num_inputs = node->inputs->size;
  const int num_outputs = node->outputs->size;

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  const auto* subgraphs = this_subgraph->GetSubgraphs();
  TF_LITE_ENSURE(context, op_state->subgraph_index < subgraphs->size());

  Subgraph* decomposition_subgraph =
      (*subgraphs)[op_state->subgraph_index].get();

  TF_LITE_ENSURE_EQ(context, num_inputs,
                    decomposition_subgraph->inputs().size());
  TF_LITE_ENSURE_EQ(context, num_outputs,
                    decomposition_subgraph->outputs().size());

  // Remove unused inputs of subgraph to skip copying unnecessary inputs.
  decomposition_subgraph->RemoveUnusedInputs();

  std::vector<int> node_inputs(node->inputs->data,
                               node->inputs->data + num_inputs);

  // Prepare and check the subgraphs.
  TF_LITE_ENSURE_OK(context,
                    CopyTensorsShapeAndType(context, this_subgraph, node_inputs,
                                            decomposition_subgraph,
                                            decomposition_subgraph->inputs(),
                                            /*resize_subgraph_inputs=*/true));

  // Handle resource input tensors.
  for (int i = 0; i < num_inputs; ++i) {
    int input_idx = decomposition_subgraph->inputs()[i];
    if (input_idx == kTfLiteOptionalTensor) {
      continue;
    }
    TfLiteTensor* subgraph_input = decomposition_subgraph->tensor(input_idx);
    if (!IsResourceOrVariant(subgraph_input)) {
      // Set the allocation type to custom to prevent memory allocation.
      subgraph_input->allocation_type = kTfLiteCustom;
    }
  }

  // Allocate the memory for the subgraph.
  TF_LITE_ENSURE_OK(context, decomposition_subgraph->AllocateTensors());
  op_state->subgraph_has_dynamic_output_tensors |=
      decomposition_subgraph->HasDynamicTensors();

  for (int i = 0; i < num_outputs; ++i) {
    if (node->outputs->data[i] == kTfLiteOptionalTensor) {
      continue;
    }
    TfLiteTensor* output;
    TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, i, &output));
    if (op_state->subgraph_has_dynamic_output_tensors) {
      SetTensorToDynamic(output);
    } else {
      TfLiteTensor* subgraph_output =
          decomposition_subgraph->tensor(decomposition_subgraph->outputs()[i]);
      TfLiteIntArray* output_size = TfLiteIntArrayCopy(subgraph_output->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, output, output_size));
    }
  }
  return kTfLiteOk;
}

float ApplyActivation(State::Activation activation, float value) {
  if (activation == State::Activation::kSilu) {
    return value / (1.0f + std::exp(-value));
  }
  return value;
}

float GetScalarPerHead(const TfLiteTensor* tensor, int batch, int seq,
                       int head) {
  const int last_dim = SizeOfDimension(tensor, 2);
  const int index = (batch * SizeOfDimension(tensor, 1) + seq) * last_dim +
                    (last_dim == 1 ? 0 : head % last_dim);
  return ReadTensorValue(tensor, index, head);
}

int ScalarPerHeadOffset(const TfLiteTensor* tensor, int batch, int seq,
                        int head) {
  const int last_dim = SizeOfDimension(tensor, 2);
  return (batch * SizeOfDimension(tensor, 1) + seq) * last_dim +
         (last_dim == 1 ? 0 : head % last_dim);
}

float GetDecayValue(const TfLiteTensor* decay, int batch, int seq, int head,
                    int key_dim_index, int key_dim) {
  const int last_dim = SizeOfDimension(decay, 2);
  int index_in_last_dim = 0;
  if (last_dim != 1) {
    if (last_dim == key_dim) {
      index_in_last_dim = key_dim_index;
    } else if (last_dim % key_dim == 0) {
      index_in_last_dim = head * key_dim + key_dim_index;
    } else {
      index_in_last_dim = head % last_dim;
    }
  }
  const int index =
      (batch * SizeOfDimension(decay, 1) + seq) * last_dim + index_in_last_dim;
  return ReadTensorValue(decay, index, index_in_last_dim);
}

float Softplus(float value) {
  if (value > 20.0f) return value;
  if (value < -20.0f) return std::exp(value);
  return std::log1p(std::exp(value));
}

float ApplyDeltaTransform(const State* op_state, float value) {
  if (op_state->delta_transform == State::DeltaTransform::kSoftplus) {
    value = Softplus(value);
  }
  if (op_state->has_delta_min) {
    value = std::max(value, op_state->delta_min);
  }
  if (op_state->has_delta_max) {
    value = std::min(value, op_state->delta_max);
  }
  return value;
}

int SelectiveStateSpaceXIndex(const TfLiteTensor* x, int b, int t, int h,
                              int p) {
  const int seq = SizeOfDimension(x, 1);
  const int heads = SizeOfDimension(x, 2);
  if (NumDimensions(x) == 3) {
    return (b * seq + t) * heads + h;
  }
  const int head_dim = SizeOfDimension(x, 3);
  return ((b * seq + t) * heads + h) * head_dim + p;
}

int SelectiveStateSpaceOutputIndex(const TfLiteTensor* output, int b, int t,
                                   int h, int p) {
  const int seq = SizeOfDimension(output, 1);
  const int heads = SizeOfDimension(output, 2);
  if (NumDimensions(output) == 3) {
    return (b * seq + t) * heads + h;
  }
  const int head_dim = SizeOfDimension(output, 3);
  return ((b * seq + t) * heads + h) * head_dim + p;
}

int SelectiveStateSpaceStateIndex(int heads, int head_dim, int state_dim, int b,
                                  int h, int p, int n) {
  return ((b * heads + h) * head_dim + p) * state_dim + n;
}

int SelectiveStateSpaceGroupIndex(int heads, int groups, int head) {
  const int heads_per_group = heads / groups;
  return std::min(groups - 1, head / heads_per_group);
}

float SelectiveStateSpaceDeltaValue(const TfLiteTensor* delta, int b, int t,
                                    int h, int p) {
  const int seq = SizeOfDimension(delta, 1);
  const int delta_heads = SizeOfDimension(delta, 2);
  const int dh = delta_heads == 1 ? 0 : h;
  if (NumDimensions(delta) == 3) {
    return ReadTensorValue(delta, (b * seq + t) * delta_heads + dh, dh);
  }
  const int delta_head_dim = SizeOfDimension(delta, 3);
  const int dp = delta_head_dim == 1 ? 0 : p;
  const int channel = dh * delta_head_dim + dp;
  return ReadTensorValue(
      delta, ((b * seq + t) * delta_heads + dh) * delta_head_dim + dp, channel);
}

float SelectiveStateSpaceAValue(const TfLiteTensor* a, int h, int p, int n) {
  if (NumDimensions(a) == 1) {
    return ReadTensorValue(a, h, h);
  }
  if (NumDimensions(a) == 2) {
    const int state_dim = SizeOfDimension(a, 1);
    return ReadTensorValue(a, h * state_dim + n, h * state_dim + n);
  }
  const int head_dim = SizeOfDimension(a, 1);
  const int state_dim = SizeOfDimension(a, 2);
  return ReadTensorValue(a, (h * head_dim + p) * state_dim + n,
                         (h * head_dim + p) * state_dim + n);
}

float SelectiveStateSpaceBCValue(const TfLiteTensor* tensor, int b, int t,
                                 int group, int n) {
  const int seq = SizeOfDimension(tensor, 1);
  if (NumDimensions(tensor) == 3) {
    const int state_dim = SizeOfDimension(tensor, 2);
    return ReadTensorValue(tensor, (b * seq + t) * state_dim + n, n);
  }
  const int groups = SizeOfDimension(tensor, 2);
  const int state_dim = SizeOfDimension(tensor, 3);
  return ReadTensorValue(tensor,
                         ((b * seq + t) * groups + group) * state_dim + n,
                         group * state_dim + n);
}

float SelectiveStateSpacePerHeadDimValue(const TfLiteTensor* tensor, int h,
                                         int p) {
  if (NumDimensions(tensor) == 1) {
    return ReadTensorValue(tensor, h, h);
  }
  const int head_dim = SizeOfDimension(tensor, 1);
  return ReadTensorValue(tensor, h * head_dim + p, h * head_dim + p);
}

struct CausalConvWithState1dFloatTask : cpu_backend_threadpool::Task {
  CausalConvWithState1dFloatTask(const float* input_data,
                                 const float* weight_data,
                                 const float* bias_data, const float* past_data,
                                 float* output_data, int job_start, int job_end,
                                 int seq, int channels, int kernel,
                                 int state_len, bool weight_channel_major,
                                 State::Activation activation)
      : input_data(input_data),
        weight_data(weight_data),
        bias_data(bias_data),
        past_data(past_data),
        output_data(output_data),
        job_start(job_start),
        job_end(job_end),
        seq(seq),
        channels(channels),
        kernel(kernel),
        state_len(state_len),
        weight_channel_major(weight_channel_major),
        activation(activation) {}

  void Run() override {
    int b = job_start / (seq * channels);
    int rem = job_start - b * seq * channels;
    int t = rem / channels;
    int c = rem - t * channels;
    for (int job = job_start; job < job_end; ++job) {
      const float* batch_input = input_data + b * seq * channels;
      float acc = bias_data == nullptr ? 0.0f : bias_data[c];
      if (kernel == 4) {
        const float x0 =
            t >= 3 ? batch_input[(t - 3) * channels + c] : PastValue(b, t, c);
        const float x1 = t >= 2 ? batch_input[(t - 2) * channels + c]
                                : PastValue(b, t + 1, c);
        const float x2 = t >= 1 ? batch_input[(t - 1) * channels + c]
                                : PastValue(b, t + 2, c);
        const float x3 = batch_input[t * channels + c];
        acc += x0 * weight_data[WeightIndex(0, c)] +
               x1 * weight_data[WeightIndex(1, c)] +
               x2 * weight_data[WeightIndex(2, c)] +
               x3 * weight_data[WeightIndex(3, c)];
      } else {
        for (int k = 0; k < kernel; ++k) {
          const int source_t = t - (state_len - k);
          float x = 0.0f;
          if (source_t >= 0) {
            x = batch_input[source_t * channels + c];
          } else if (past_data != nullptr) {
            const int state_t = source_t + state_len;
            if (state_t >= 0) {
              x = past_data[StateIndex(b, state_t, c)];
            }
          }
          acc += x * weight_data[WeightIndex(k, c)];
        }
      }
      output_data[(b * seq + t) * channels + c] =
          ApplyActivation(activation, acc);

      ++c;
      if (c == channels) {
        c = 0;
        ++t;
        if (t == seq) {
          t = 0;
          ++b;
        }
      }
    }
  }

  int WeightIndex(int k, int c) const {
    return weight_channel_major ? c * kernel + k : k * channels + c;
  }

  int StateIndex(int b, int s, int c) const {
    return weight_channel_major ? (b * channels + c) * state_len + s
                                : (b * state_len + s) * channels + c;
  }

  float PastValue(int b, int state_t, int c) const {
    if (past_data == nullptr || state_t >= state_len) {
      return 0.0f;
    }
    return past_data[StateIndex(b, state_t, c)];
  }

  const float* input_data;
  const float* weight_data;
  const float* bias_data;
  const float* past_data;
  float* output_data;
  int job_start;
  int job_end;
  int seq;
  int channels;
  int kernel;
  int state_len;
  bool weight_channel_major;
  State::Activation activation;
};

TfLiteStatus EvalCausalConvWithState1d(TfLiteContext* context,
                                       TfLiteNode* node) {
  const State* op_state = reinterpret_cast<State*>(node->user_data);
  const TfLiteTensor* input;
  const TfLiteTensor* weight;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &weight));

  const TfLiteTensor* bias = nullptr;
  const TfLiteTensor* past_state = nullptr;
  if (node->inputs->size >= 3 &&
      node->inputs->data[2] != kTfLiteOptionalTensor) {
    const TfLiteTensor* third;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &third));
    if (NumDimensions(third) == 1) {
      bias = third;
    } else {
      past_state = third;
    }
  }
  if (node->inputs->size >= 4 &&
      node->inputs->data[3] != kTfLiteOptionalTensor) {
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 3, &past_state));
  }

  TfLiteTensor* output;
  TfLiteTensor* present_state;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &present_state));

  const int batch = SizeOfDimension(input, 0);
  const int seq = SizeOfDimension(input, 1);
  const int channels = SizeOfDimension(input, 2);
  const bool weight_channel_major =
      IsCausalConvWeightChannelMajor(weight, channels);
  const int kernel = weight_channel_major ? SizeOfDimension(weight, 1)
                                          : SizeOfDimension(weight, 0);
  const int state_len = kernel - 1;
  auto weight_index = [&](int k, int c) {
    return weight_channel_major ? c * kernel + k : k * channels + c;
  };
  auto state_index = [&](int b, int s, int c) {
    return weight_channel_major ? (b * channels + c) * state_len + s
                                : (b * state_len + s) * channels + c;
  };
  const bool all_float =
      input->type == kTfLiteFloat32 && weight->type == kTfLiteFloat32 &&
      output->type == kTfLiteFloat32 && present_state->type == kTfLiteFloat32 &&
      (bias == nullptr || bias->type == kTfLiteFloat32) &&
      (past_state == nullptr || past_state->type == kTfLiteFloat32);

  if (all_float) {
    const float* input_data = GetTensorData<float>(input);
    const float* weight_data = GetTensorData<float>(weight);
    const float* bias_data =
        bias == nullptr ? nullptr : GetTensorData<float>(bias);
    const float* past_data =
        past_state == nullptr ? nullptr : GetTensorData<float>(past_state);
    float* output_data = GetTensorData<float>(output);
    float* present_data = GetTensorData<float>(present_state);

    const int output_size = batch * seq * channels;
    CpuBackendContext* cpu_backend_context =
        CpuBackendContext::GetFromContext(context);
    const int max_threads = cpu_backend_context->max_num_threads();
    constexpr int kMinConvWorkPerThread = 32768;
    int thread_count = 1;
    if (output_size * kernel >= kMinConvWorkPerThread && max_threads > 1) {
      thread_count = std::min(output_size, max_threads);
    }
    if (thread_count == 1) {
      if (kernel == 4) {
        for (int b = 0; b < batch; ++b) {
          const float* batch_input = input_data + b * seq * channels;
          float* batch_output = output_data + b * seq * channels;

          for (int t = 0; t < seq; ++t) {
            const float* input_0 =
                t >= 3 ? batch_input + (t - 3) * channels : nullptr;
            const float* input_1 =
                t >= 2 ? batch_input + (t - 2) * channels : nullptr;
            const float* input_2 =
                t >= 1 ? batch_input + (t - 1) * channels : nullptr;
            const float* input_3 = batch_input + t * channels;
            const int past_0 = t;
            const int past_1 = t + 1;
            const int past_2 = t + 2;
            float* out = batch_output + t * channels;
            for (int c = 0; c < channels; ++c) {
              const float x0 = input_0 != nullptr
                                   ? input_0[c]
                                   : (past_data != nullptr && past_0 < 3
                                          ? past_data[state_index(b, past_0, c)]
                                          : 0.0f);
              const float x1 = input_1 != nullptr
                                   ? input_1[c]
                                   : (past_data != nullptr && past_1 < 3
                                          ? past_data[state_index(b, past_1, c)]
                                          : 0.0f);
              const float x2 = input_2 != nullptr
                                   ? input_2[c]
                                   : (past_data != nullptr && past_2 < 3
                                          ? past_data[state_index(b, past_2, c)]
                                          : 0.0f);
              float acc = bias_data == nullptr ? 0.0f : bias_data[c];
              acc += x0 * weight_data[weight_index(0, c)] +
                     x1 * weight_data[weight_index(1, c)] +
                     x2 * weight_data[weight_index(2, c)] +
                     input_3[c] * weight_data[weight_index(3, c)];
              out[c] = ApplyActivation(op_state->activation, acc);
            }
          }
        }
      } else {
        for (int b = 0; b < batch; ++b) {
          for (int t = 0; t < seq; ++t) {
            for (int c = 0; c < channels; ++c) {
              float acc = bias_data == nullptr ? 0.0f : bias_data[c];
              for (int k = 0; k < kernel; ++k) {
                const int source_t = t - (state_len - k);
                float x = 0.0f;
                if (source_t >= 0) {
                  x = input_data[(b * seq + source_t) * channels + c];
                } else if (past_data != nullptr) {
                  const int state_t = source_t + state_len;
                  if (state_t >= 0) {
                    x = past_data[state_index(b, state_t, c)];
                  }
                }
                acc += x * weight_data[weight_index(k, c)];
              }
              output_data[(b * seq + t) * channels + c] =
                  ApplyActivation(op_state->activation, acc);
            }
          }
        }
      }
    } else {
      std::vector<CausalConvWithState1dFloatTask> tasks;
      tasks.reserve(thread_count);
      int job_start = 0;
      for (int i = 0; i < thread_count; ++i) {
        const int job_end =
            job_start + (output_size - job_start) / (thread_count - i);
        tasks.emplace_back(input_data, weight_data, bias_data, past_data,
                           output_data, job_start, job_end, seq, channels,
                           kernel, state_len, weight_channel_major,
                           op_state->activation);
        job_start = job_end;
      }
      cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                      cpu_backend_context);
    }

    for (int b = 0; b < batch; ++b) {
      for (int s = 0; s < state_len; ++s) {
        const int source_t = seq - state_len + s;
        for (int c = 0; c < channels; ++c) {
          float x = 0.0f;
          if (source_t >= 0) {
            x = input_data[(b * seq + source_t) * channels + c];
          } else if (past_data != nullptr) {
            x = past_data[state_index(b, source_t + state_len, c)];
          }
          present_data[state_index(b, s, c)] = x;
        }
      }
    }
    return kTfLiteOk;
  }

  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < seq; ++t) {
      for (int c = 0; c < channels; ++c) {
        float acc = bias == nullptr ? 0.0f : ReadTensorValue(bias, c, c);
        for (int k = 0; k < kernel; ++k) {
          const int source_t = t - (state_len - k);
          float x = 0.0f;
          if (source_t >= 0) {
            x = ReadTensorValue(input, (b * seq + source_t) * channels + c, c);
          } else if (past_state != nullptr) {
            const int state_t = source_t + state_len;
            if (state_t >= 0) {
              x = ReadTensorValue(past_state, state_index(b, state_t, c), c);
            }
          }
          acc += x * ReadTensorValue(weight, weight_index(k, c), c);
        }
        WriteTensorValue(output, (b * seq + t) * channels + c,
                         ApplyActivation(op_state->activation, acc), c);
      }
    }

    for (int s = 0; s < state_len; ++s) {
      const int source_t = seq - state_len + s;
      for (int c = 0; c < channels; ++c) {
        float x = 0.0f;
        if (source_t >= 0) {
          x = ReadTensorValue(input, (b * seq + source_t) * channels + c, c);
        } else if (past_state != nullptr) {
          x = ReadTensorValue(past_state,
                              state_index(b, source_t + state_len, c), c);
        }
        WriteTensorValue(present_state, state_index(b, s, c), x, c);
      }
    }
  }
  return kTfLiteOk;
}

struct RecurrentLinearAttentionFloatTask : cpu_backend_threadpool::Task {
  RecurrentLinearAttentionFloatTask(
      const float* query_data, const float* key_data, const float* value_data,
      const float* decay_data, const float* beta_data, float* output_data,
      float* state_data, int job_start, int job_end, int seq, int kv_heads,
      int q_heads, int q_heads_per_kv_head, int key_dim, int value_dim,
      int chunk_size, int decay_last_dim, int beta_last_dim, float scale,
      bool has_gate, bool has_delta, bool has_headwise_decay)
      : query_data(query_data),
        key_data(key_data),
        value_data(value_data),
        decay_data(decay_data),
        beta_data(beta_data),
        output_data(output_data),
        state_data(state_data),
        job_start(job_start),
        job_end(job_end),
        seq(seq),
        kv_heads(kv_heads),
        q_heads(q_heads),
        q_heads_per_kv_head(q_heads_per_kv_head),
        key_dim(key_dim),
        value_dim(value_dim),
        chunk_size(chunk_size),
        decay_last_dim(decay_last_dim),
        beta_last_dim(beta_last_dim),
        scale(scale),
        has_gate(has_gate),
        has_delta(has_delta),
        has_headwise_decay(has_headwise_decay) {}

  void Run() override {
    std::vector<float> kv_mem(value_dim);
    std::vector<float> delta(value_dim);
    std::vector<float> output_accumulator(std::max(1, q_heads_per_kv_head) *
                                          value_dim);
    std::vector<float> query_key_values(std::max(1, q_heads_per_kv_head));
    std::vector<float> query_values(std::max(1, q_heads_per_kv_head));
    float* const kv_mem_data = kv_mem.data();
    float* const delta_data = delta.data();
    float* const output_accumulator_data = output_accumulator.data();
    float* const query_key_values_data = query_key_values.data();
    float* const query_values_data = query_values.data();
    const bool fuse_headwise_decay = has_gate && decay_data != nullptr &&
                                     has_headwise_decay &&
                                     q_heads_per_kv_head == 1;

    for (int job = job_start; job < job_end; ++job) {
      const int b = job / kv_heads;
      const int h = job - b * kv_heads;
      for (int chunk_start = 0; chunk_start < seq; chunk_start += chunk_size) {
        const int chunk_end = std::min(seq, chunk_start + chunk_size);
        for (int t = chunk_start; t < chunk_end; ++t) {
          const float beta_value =
              has_delta && beta_data != nullptr
                  ? beta_data[ScalarPerHeadOffset(beta_last_dim, b, t, h)]
                  : 1.0f;
          const float headwise_decay_value =
              fuse_headwise_decay ? std::exp(decay_data[ScalarPerHeadOffset(
                                        decay_last_dim, b, t, h)])
                                  : 1.0f;

          if (has_gate && decay_data != nullptr) {
            if (has_headwise_decay) {
              if (!fuse_headwise_decay) {
                const float decay_value = std::exp(
                    decay_data[ScalarPerHeadOffset(decay_last_dim, b, t, h)]);
                float* state_matrix = state_data + SIndex(b, h, 0, 0);
                const int state_matrix_size = key_dim * value_dim;
                for (int i = 0; i < state_matrix_size; ++i) {
                  state_matrix[i] *= decay_value;
                }
              }
            } else {
              for (int kd = 0; kd < key_dim; ++kd) {
                float* row = state_data + SIndex(b, h, kd, 0);
                const float decay_value = std::exp(DecayValue(b, t, h, kd));
                for (int vd = 0; vd < value_dim; ++vd) {
                  row[vd] *= decay_value;
                }
              }
            }
          }

          const float* key_vec = key_data + KVIndex(b, t, h, key_dim);
          if (q_heads_per_kv_head == 1) {
            const int qh = h;
            float* output_vec = output_data + OutIndex(b, t, qh);
            std::fill(kv_mem_data, kv_mem_data + value_dim, 0.0f);
            std::fill(output_vec, output_vec + value_dim, 0.0f);
            const float* query_vec = query_data + QIndex(b, t, qh);
            float query_key = 0.0f;
            if (fuse_headwise_decay) {
              for (int kd = 0; kd < key_dim; ++kd) {
                const float k = key_vec[kd];
                const float q = query_vec[kd] * scale;
                query_key += q * k;
                float* const row = state_data + SIndex(b, h, kd, 0);
                for (int vd = 0; vd < value_dim; ++vd) {
                  const float state_value = row[vd] * headwise_decay_value;
                  row[vd] = state_value;
                  kv_mem_data[vd] += state_value * k;
                  output_vec[vd] += state_value * q;
                }
              }
            } else {
              for (int kd = 0; kd < key_dim; ++kd) {
                const float k = key_vec[kd];
                const float q = query_vec[kd] * scale;
                query_key += q * k;
                const float* const row = state_data + SIndex(b, h, kd, 0);
                for (int vd = 0; vd < value_dim; ++vd) {
                  const float state_value = row[vd];
                  kv_mem_data[vd] += state_value * k;
                  output_vec[vd] += state_value * q;
                }
              }
            }

            const float* value_vec = value_data + KVIndex(b, t, h, value_dim);
            for (int vd = 0; vd < value_dim; ++vd) {
              const float v = value_vec[vd];
              delta_data[vd] =
                  has_delta ? (v - kv_mem_data[vd]) * beta_value : v;
              output_vec[vd] += query_key * delta_data[vd];
            }

            for (int kd = 0; kd < key_dim; ++kd) {
              const float k = key_vec[kd];
              float* row = state_data + SIndex(b, h, kd, 0);
              for (int vd = 0; vd < value_dim; ++vd) {
                row[vd] += k * delta_data[vd];
              }
            }
          } else {
            std::fill(kv_mem_data, kv_mem_data + value_dim, 0.0f);
            std::fill(output_accumulator_data,
                      output_accumulator_data + q_heads_per_kv_head * value_dim,
                      0.0f);
            std::fill(query_key_values_data,
                      query_key_values_data + q_heads_per_kv_head, 0.0f);
            for (int kd = 0; kd < key_dim; ++kd) {
              const float k = key_vec[kd];
              const float* row = state_data + SIndex(b, h, kd, 0);
              for (int q_offset = 0; q_offset < q_heads_per_kv_head;
                   ++q_offset) {
                const int qh = h * q_heads_per_kv_head + q_offset;
                const float* query_vec = query_data + QIndex(b, t, qh);
                const float q = query_vec[kd] * scale;
                query_values_data[q_offset] = q;
                query_key_values_data[q_offset] += q * k;
              }
              for (int vd = 0; vd < value_dim; ++vd) {
                const float state_value = row[vd];
                kv_mem_data[vd] += state_value * k;
                for (int q_offset = 0; q_offset < q_heads_per_kv_head;
                     ++q_offset) {
                  output_accumulator_data[q_offset * value_dim + vd] +=
                      state_value * query_values_data[q_offset];
                }
              }
            }

            const float* value_vec = value_data + KVIndex(b, t, h, value_dim);
            for (int vd = 0; vd < value_dim; ++vd) {
              const float v = value_vec[vd];
              delta_data[vd] =
                  has_delta ? (v - kv_mem_data[vd]) * beta_value : v;
            }

            for (int q_offset = 0; q_offset < q_heads_per_kv_head; ++q_offset) {
              const int qh = h * q_heads_per_kv_head + q_offset;
              float* output_vec = output_data + OutIndex(b, t, qh);
              for (int vd = 0; vd < value_dim; ++vd) {
                output_vec[vd] =
                    output_accumulator_data[q_offset * value_dim + vd] +
                    query_key_values_data[q_offset] * delta_data[vd];
              }
            }

            for (int kd = 0; kd < key_dim; ++kd) {
              const float k = key_vec[kd];
              float* row = state_data + SIndex(b, h, kd, 0);
              for (int vd = 0; vd < value_dim; ++vd) {
                row[vd] += k * delta_data[vd];
              }
            }
          }
        }
      }
    }
  }

  int QIndex(int b, int t, int h) const {
    return ((b * seq + t) * q_heads + h) * key_dim;
  }

  int KVIndex(int b, int t, int h, int dim) const {
    return ((b * seq + t) * kv_heads + h) * dim;
  }

  int OutIndex(int b, int t, int h) const {
    return ((b * seq + t) * q_heads + h) * value_dim;
  }

  int SIndex(int b, int h, int kd, int vd) const {
    return ((b * kv_heads + h) * key_dim + kd) * value_dim + vd;
  }

  int ScalarPerHeadOffset(int last_dim, int b, int t, int h) const {
    return (b * seq + t) * last_dim + (last_dim == 1 ? 0 : h % last_dim);
  }

  float DecayValue(int b, int t, int h, int kd) const {
    int index_in_last_dim = 0;
    if (decay_last_dim != 1) {
      if (decay_last_dim == key_dim) {
        index_in_last_dim = kd;
      } else if (decay_last_dim % key_dim == 0) {
        index_in_last_dim = h * key_dim + kd;
      } else {
        index_in_last_dim = h % decay_last_dim;
      }
    }
    return decay_data[(b * seq + t) * decay_last_dim + index_in_last_dim];
  }

  const float* query_data;
  const float* key_data;
  const float* value_data;
  const float* decay_data;
  const float* beta_data;
  float* output_data;
  float* state_data;
  int job_start;
  int job_end;
  int seq;
  int kv_heads;
  int q_heads;
  int q_heads_per_kv_head;
  int key_dim;
  int value_dim;
  int chunk_size;
  int decay_last_dim;
  int beta_last_dim;
  float scale;
  bool has_gate;
  bool has_delta;
  bool has_headwise_decay;
};

TfLiteStatus EvalRecurrentLinearAttention(TfLiteContext* context,
                                          TfLiteNode* node) {
  const State* op_state = reinterpret_cast<State*>(node->user_data);
  const TfLiteTensor* query;
  const TfLiteTensor* key;
  const TfLiteTensor* value;
  const TfLiteTensor* past_state = nullptr;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &query));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &key));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &value));
  if (node->inputs->size >= 4 &&
      node->inputs->data[3] != kTfLiteOptionalTensor) {
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 3, &past_state));
  }
  const TfLiteTensor* decay = nullptr;
  const TfLiteTensor* beta = nullptr;
  if (node->inputs->size >= 5 &&
      node->inputs->data[4] != kTfLiteOptionalTensor) {
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 4, &decay));
  }
  if (node->inputs->size >= 6 &&
      node->inputs->data[5] != kTfLiteOptionalTensor) {
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 5, &beta));
  }

  TfLiteTensor* output;
  TfLiteTensor* present_state;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &present_state));

  const int batch = SizeOfDimension(query, 0);
  const int seq = SizeOfDimension(query, 1);
  const int kv_heads = SizeOfDimension(present_state, 1);
  const int q_heads =
      op_state->q_num_heads > 0 ? op_state->q_num_heads : kv_heads;
  const int key_dim = SizeOfDimension(present_state, 2);
  const int value_dim = SizeOfDimension(present_state, 3);
  const float scale = op_state->scale == 0.0f
                          ? 1.0f / std::sqrt(static_cast<float>(key_dim))
                          : op_state->scale;
  const int q_heads_per_kv_head = q_heads / kv_heads;
  const bool has_gate = op_state->update_rule == State::UpdateRule::kGated ||
                        op_state->update_rule == State::UpdateRule::kGatedDelta;
  const bool has_delta =
      op_state->update_rule == State::UpdateRule::kDelta ||
      op_state->update_rule == State::UpdateRule::kGatedDelta;
  const int decay_last_dim = decay == nullptr ? 0 : SizeOfDimension(decay, 2);
  const bool has_headwise_decay =
      decay != nullptr && (decay_last_dim == 1 || decay_last_dim == kv_heads ||
                           decay_last_dim == q_heads);

  auto q_index = [&](int b, int t, int h) {
    return ((b * seq + t) * q_heads + h) * key_dim;
  };
  auto kv_index = [&](int b, int t, int h, int dim) {
    return ((b * seq + t) * kv_heads + h) * dim;
  };
  auto out_index = [&](int b, int t, int h) {
    return ((b * seq + t) * q_heads + h) * value_dim;
  };
  auto s_index = [&](int b, int h, int kd, int vd) {
    return ((b * kv_heads + h) * key_dim + kd) * value_dim + vd;
  };

  std::vector<float> kv_mem(value_dim);
  std::vector<float> delta(value_dim);
  const bool all_float =
      query->type == kTfLiteFloat32 && key->type == kTfLiteFloat32 &&
      value->type == kTfLiteFloat32 && output->type == kTfLiteFloat32 &&
      present_state->type == kTfLiteFloat32 &&
      (past_state == nullptr || past_state->type == kTfLiteFloat32) &&
      (decay == nullptr || decay->type == kTfLiteFloat32) &&
      (beta == nullptr || beta->type == kTfLiteFloat32);

  if (all_float) {
    const float* query_data = GetTensorData<float>(query);
    const float* key_data = GetTensorData<float>(key);
    const float* value_data = GetTensorData<float>(value);
    const float* past_data =
        past_state == nullptr ? nullptr : GetTensorData<float>(past_state);
    const float* decay_data =
        decay == nullptr ? nullptr : GetTensorData<float>(decay);
    const float* beta_data =
        beta == nullptr ? nullptr : GetTensorData<float>(beta);
    float* output_data = GetTensorData<float>(output);
    float* state_data = GetTensorData<float>(present_state);

    if (past_data == nullptr) {
      std::fill(state_data, state_data + NumElements(present_state), 0.0f);
    } else {
      std::copy(past_data, past_data + NumElements(past_state), state_data);
    }

    const int chunk_size = std::max(1, op_state->chunk_size);
    const bool use_chunked_gated_delta =
        op_state->use_chunked_prefill && chunk_size > 1 && seq > 1 &&
        has_gate && has_delta && decay_data != nullptr &&
        beta_data != nullptr && has_headwise_decay && q_heads_per_kv_head > 0;
    if (use_chunked_gated_delta) {
      const int max_chunk = std::min(seq, chunk_size);
      std::vector<float> cumulative_decay(max_chunk);
      std::vector<float> decay_mask(max_chunk * max_chunk);
      std::vector<float> attn_transform(max_chunk * max_chunk);
      std::vector<float> attn_row(max_chunk);
      std::vector<float> value_transformed(max_chunk * value_dim);
      std::vector<float> k_cumdecay(max_chunk * key_dim);
      std::vector<float> v_new(max_chunk * value_dim);
      auto chunk_index = [max_chunk](int row, int col) {
        return row * max_chunk + col;
      };
      auto chunk_value_index = [value_dim](int row, int col) {
        return row * value_dim + col;
      };
      auto chunk_key_index = [key_dim](int row, int col) {
        return row * key_dim + col;
      };

      for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < kv_heads; ++h) {
          for (int chunk_start = 0; chunk_start < seq;
               chunk_start += chunk_size) {
            const int chunk_len = std::min(chunk_size, seq - chunk_start);

            float decay_sum = 0.0f;
            for (int i = 0; i < chunk_len; ++i) {
              const int t = chunk_start + i;
              decay_sum += decay_data[ScalarPerHeadOffset(decay, b, t, h)];
              cumulative_decay[i] = decay_sum;
            }

            std::fill(decay_mask.begin(), decay_mask.end(), 0.0f);
            for (int i = 0; i < chunk_len; ++i) {
              for (int j = 0; j <= i; ++j) {
                decay_mask[chunk_index(i, j)] =
                    std::exp(cumulative_decay[i] - cumulative_decay[j]);
              }
            }

            std::fill(attn_transform.begin(), attn_transform.end(), 0.0f);
            for (int i = 1; i < chunk_len; ++i) {
              const int ti = chunk_start + i;
              const float* key_i = key_data + kv_index(b, ti, h, key_dim);
              const float beta_i =
                  beta_data[ScalarPerHeadOffset(beta, b, ti, h)];
              for (int j = 0; j < i; ++j) {
                const int tj = chunk_start + j;
                const float* key_j = key_data + kv_index(b, tj, h, key_dim);
                float key_dot = 0.0f;
                for (int kd = 0; kd < key_dim; ++kd) {
                  key_dot += key_i[kd] * beta_i * key_j[kd];
                }
                attn_transform[chunk_index(i, j)] =
                    -key_dot * decay_mask[chunk_index(i, j)];
              }

              for (int j = 0; j < i; ++j) {
                attn_row[j] = attn_transform[chunk_index(i, j)];
              }
              for (int j = 0; j < i; ++j) {
                float value = attn_row[j];
                for (int k = 0; k < i; ++k) {
                  value += attn_row[k] * attn_transform[chunk_index(k, j)];
                }
                attn_transform[chunk_index(i, j)] = value;
              }
            }
            for (int i = 0; i < chunk_len; ++i) {
              attn_transform[chunk_index(i, i)] = 1.0f;
            }

            std::fill(value_transformed.begin(), value_transformed.end(), 0.0f);
            std::fill(k_cumdecay.begin(), k_cumdecay.end(), 0.0f);
            for (int i = 0; i < chunk_len; ++i) {
              for (int j = 0; j <= i; ++j) {
                const int tj = chunk_start + j;
                const float transform = attn_transform[chunk_index(i, j)];
                const float beta_j =
                    beta_data[ScalarPerHeadOffset(beta, b, tj, h)];
                const float* key_j = key_data + kv_index(b, tj, h, key_dim);
                const float* value_j =
                    value_data + kv_index(b, tj, h, value_dim);
                for (int vd = 0; vd < value_dim; ++vd) {
                  value_transformed[chunk_value_index(i, vd)] +=
                      transform * value_j[vd] * beta_j;
                }
                const float k_decay = beta_j * std::exp(cumulative_decay[j]);
                for (int kd = 0; kd < key_dim; ++kd) {
                  k_cumdecay[chunk_key_index(i, kd)] +=
                      transform * key_j[kd] * k_decay;
                }
              }
            }

            std::fill(v_new.begin(), v_new.end(), 0.0f);
            for (int i = 0; i < chunk_len; ++i) {
              for (int vd = 0; vd < value_dim; ++vd) {
                float v_prime = 0.0f;
                for (int kd = 0; kd < key_dim; ++kd) {
                  v_prime += k_cumdecay[chunk_key_index(i, kd)] *
                             state_data[s_index(b, h, kd, vd)];
                }
                v_new[chunk_value_index(i, vd)] =
                    value_transformed[chunk_value_index(i, vd)] - v_prime;
              }
            }

            for (int i = 0; i < chunk_len; ++i) {
              const int ti = chunk_start + i;
              const float state_decay = std::exp(cumulative_decay[i]);
              for (int q_offset = 0; q_offset < q_heads_per_kv_head;
                   ++q_offset) {
                const int qh = h * q_heads_per_kv_head + q_offset;
                const float* query_vec = query_data + q_index(b, ti, qh);
                float* output_vec = output_data + out_index(b, ti, qh);
                std::fill(output_vec, output_vec + value_dim, 0.0f);
                for (int kd = 0; kd < key_dim; ++kd) {
                  const float q = query_vec[kd] * scale;
                  const float inter_q = q * state_decay;
                  const float* state_row = state_data + s_index(b, h, kd, 0);
                  for (int vd = 0; vd < value_dim; ++vd) {
                    output_vec[vd] += inter_q * state_row[vd];
                  }
                }
                for (int j = 0; j <= i; ++j) {
                  const int tj = chunk_start + j;
                  const float* key_j = key_data + kv_index(b, tj, h, key_dim);
                  float query_key = 0.0f;
                  for (int kd = 0; kd < key_dim; ++kd) {
                    query_key += query_vec[kd] * scale * key_j[kd];
                  }
                  const float attn = query_key * decay_mask[chunk_index(i, j)];
                  for (int vd = 0; vd < value_dim; ++vd) {
                    output_vec[vd] += attn * v_new[chunk_value_index(j, vd)];
                  }
                }
              }
            }

            const float final_decay = std::exp(cumulative_decay[chunk_len - 1]);
            for (int kd = 0; kd < key_dim; ++kd) {
              float* state_row = state_data + s_index(b, h, kd, 0);
              for (int vd = 0; vd < value_dim; ++vd) {
                state_row[vd] *= final_decay;
              }
            }
            for (int i = 0; i < chunk_len; ++i) {
              const int ti = chunk_start + i;
              const float key_decay = std::exp(cumulative_decay[chunk_len - 1] -
                                               cumulative_decay[i]);
              const float* key_i = key_data + kv_index(b, ti, h, key_dim);
              for (int kd = 0; kd < key_dim; ++kd) {
                float* state_row = state_data + s_index(b, h, kd, 0);
                const float k = key_i[kd] * key_decay;
                for (int vd = 0; vd < value_dim; ++vd) {
                  state_row[vd] += k * v_new[chunk_value_index(i, vd)];
                }
              }
            }
          }
        }
      }
      return kTfLiteOk;
    }

    const int job_count = batch * kv_heads;
    if (job_count <= 0) {
      return kTfLiteOk;
    }
    CpuBackendContext* cpu_backend_context =
        CpuBackendContext::GetFromContext(context);
    const int max_threads = cpu_backend_context->max_num_threads();
    const int thread_count =
        std::max(1, std::min(job_count, std::max(1, max_threads)));
    std::vector<RecurrentLinearAttentionFloatTask> tasks;
    tasks.reserve(thread_count);
    int job_start = 0;
    for (int i = 0; i < thread_count; ++i) {
      const int job_end =
          job_start + (job_count - job_start) / (thread_count - i);
      tasks.emplace_back(query_data, key_data, value_data, decay_data,
                         beta_data, output_data, state_data, job_start, job_end,
                         seq, kv_heads, q_heads, q_heads_per_kv_head, key_dim,
                         value_dim, chunk_size, decay_last_dim,
                         beta == nullptr ? 0 : SizeOfDimension(beta, 2), scale,
                         has_gate, has_delta, has_headwise_decay);
      job_start = job_end;
    }
    if (thread_count == 1 || max_threads <= 1) {
      tasks[0].Run();
    } else {
      cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                      cpu_backend_context);
    }
    return kTfLiteOk;
  }

  std::vector<float> state(NumElements(present_state), 0.0f);
  if (past_state != nullptr) {
    for (int i = 0; i < NumElements(past_state); ++i) {
      state[i] = ReadTensorValue(past_state, i);
    }
  }

  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < seq; ++t) {
      for (int h = 0; h < kv_heads; ++h) {
        const float beta_value = has_delta && beta != nullptr
                                     ? GetScalarPerHead(beta, b, t, h)
                                     : 1.0f;

        for (int kd = 0; kd < key_dim; ++kd) {
          if (!has_gate || decay == nullptr) continue;
          const float decay_value =
              std::exp(GetDecayValue(decay, b, t, h, kd, key_dim));
          for (int vd = 0; vd < value_dim; ++vd) {
            state[s_index(b, h, kd, vd)] *= decay_value;
          }
        }

        std::fill(kv_mem.begin(), kv_mem.end(), 0.0f);
        const int key_base = kv_index(b, t, h, key_dim);
        for (int kd = 0; kd < key_dim; ++kd) {
          const float k = ReadTensorValue(key, key_base + kd, h * key_dim + kd);
          for (int vd = 0; vd < value_dim; ++vd) {
            kv_mem[vd] += state[s_index(b, h, kd, vd)] * k;
          }
        }

        const int value_base = kv_index(b, t, h, value_dim);
        for (int vd = 0; vd < value_dim; ++vd) {
          const float v =
              ReadTensorValue(value, value_base + vd, h * value_dim + vd);
          delta[vd] = has_delta ? (v - kv_mem[vd]) * beta_value : v;
        }

        for (int kd = 0; kd < key_dim; ++kd) {
          const float k = ReadTensorValue(key, key_base + kd, h * key_dim + kd);
          for (int vd = 0; vd < value_dim; ++vd) {
            state[s_index(b, h, kd, vd)] += k * delta[vd];
          }
        }

        for (int qh = h * q_heads_per_kv_head;
             qh < (h + 1) * q_heads_per_kv_head; ++qh) {
          const int query_base = q_index(b, t, qh);
          const int output_base = out_index(b, t, qh);
          for (int vd = 0; vd < value_dim; ++vd) {
            float acc = 0.0f;
            for (int kd = 0; kd < key_dim; ++kd) {
              const float q =
                  ReadTensorValue(query, query_base + kd, qh * key_dim + kd);
              acc += state[s_index(b, h, kd, vd)] * q * scale;
            }
            WriteTensorValue(output, output_base + vd, acc,
                             qh * value_dim + vd);
          }
        }
      }
    }
  }

  for (int i = 0; i < NumElements(present_state); ++i) {
    WriteTensorValue(present_state, i, state[i]);
  }
  return kTfLiteOk;
}

TfLiteStatus EvalSelectiveStateSpace(TfLiteContext* context, TfLiteNode* node) {
  const State* op_state = reinterpret_cast<State*>(node->user_data);
  const TfLiteTensor* x;
  const TfLiteTensor* delta;
  const TfLiteTensor* a;
  const TfLiteTensor* b_tensor;
  const TfLiteTensor* c_tensor;
  const TfLiteTensor* past_state = nullptr;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &x));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &delta));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &a));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 3, &b_tensor));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 4, &c_tensor));
  if (node->inputs->data[5] != kTfLiteOptionalTensor) {
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 5, &past_state));
  }

  const TfLiteTensor* d = nullptr;
  const TfLiteTensor* delta_bias = nullptr;
  const TfLiteTensor* token_mask = nullptr;
  const TfLiteTensor* reset_mask = nullptr;
  if (node->inputs->size >= 7 &&
      node->inputs->data[6] != kTfLiteOptionalTensor) {
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 6, &d));
  }
  if (node->inputs->size >= 8 &&
      node->inputs->data[7] != kTfLiteOptionalTensor) {
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 7, &delta_bias));
  }
  if (node->inputs->size >= 9 &&
      node->inputs->data[8] != kTfLiteOptionalTensor) {
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 8, &token_mask));
  }
  if (node->inputs->size >= 10 &&
      node->inputs->data[9] != kTfLiteOptionalTensor) {
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 9, &reset_mask));
  }

  TfLiteTensor* output;
  TfLiteTensor* present_state;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &present_state));

  const int batch = SizeOfDimension(x, 0);
  const int seq = SizeOfDimension(x, 1);
  const int heads = SizeOfDimension(x, 2);
  const int head_dim = NumDimensions(x) == 4 ? SizeOfDimension(x, 3) : 1;
  const int groups =
      NumDimensions(b_tensor) == 4 ? SizeOfDimension(b_tensor, 2) : 1;
  const int state_dim = LastDim(b_tensor);

  std::vector<float> state(NumElements(present_state), 0.0f);
  if (past_state != nullptr) {
    const float* past_data = GetTensorData<float>(past_state);
    std::copy(past_data, past_data + NumElements(past_state), state.begin());
  }

  auto zero_batch_state = [&](int b) {
    for (int h = 0; h < heads; ++h) {
      for (int p = 0; p < head_dim; ++p) {
        float* row = state.data() + SelectiveStateSpaceStateIndex(
                                        heads, head_dim, state_dim, b, h, p, 0);
        std::fill(row, row + state_dim, 0.0f);
      }
    }
  };

  auto write_zero_output = [&](int b, int t) {
    for (int h = 0; h < heads; ++h) {
      for (int p = 0; p < head_dim; ++p) {
        WriteTensorValue(output,
                         SelectiveStateSpaceOutputIndex(output, b, t, h, p),
                         0.0f, h * head_dim + p);
      }
    }
  };

  for (int b = 0; b < batch; ++b) {
    for (int t = 0; t < seq; ++t) {
      if (reset_mask != nullptr &&
          ReadMaskValue(reset_mask, b * SizeOfDimension(reset_mask, 1) + t)) {
        zero_batch_state(b);
      }

      if (token_mask != nullptr &&
          !ReadMaskValue(token_mask, b * SizeOfDimension(token_mask, 1) + t)) {
        write_zero_output(b, t);
        continue;
      }

      for (int h = 0; h < heads; ++h) {
        const int group = SelectiveStateSpaceGroupIndex(heads, groups, h);
        for (int p = 0; p < head_dim; ++p) {
          const float x_value = ReadTensorValue(
              x, SelectiveStateSpaceXIndex(x, b, t, h, p), h * head_dim + p);
          float dt = SelectiveStateSpaceDeltaValue(delta, b, t, h, p);
          if (delta_bias != nullptr) {
            dt += SelectiveStateSpacePerHeadDimValue(delta_bias, h, p);
          }
          dt = ApplyDeltaTransform(op_state, dt);

          float y = 0.0f;
          for (int n = 0; n < state_dim; ++n) {
            const float a_value = SelectiveStateSpaceAValue(a, h, p, n);
            const float b_value =
                SelectiveStateSpaceBCValue(b_tensor, b, t, group, n);
            const float c_value =
                SelectiveStateSpaceBCValue(c_tensor, b, t, group, n);
            const int state_index = SelectiveStateSpaceStateIndex(
                heads, head_dim, state_dim, b, h, p, n);
            const float updated = std::exp(dt * a_value) * state[state_index] +
                                  dt * b_value * x_value;
            state[state_index] = updated;
            y += updated * c_value;
          }
          if (d != nullptr) {
            y += SelectiveStateSpacePerHeadDimValue(d, h, p) * x_value;
          }
          WriteTensorValue(output,
                           SelectiveStateSpaceOutputIndex(output, b, t, h, p),
                           y, h * head_dim + p);
        }
      }
    }
  }

  float* present_data = GetTensorData<float>(present_state);
  std::copy(state.begin(), state.end(), present_data);
  return kTfLiteOk;
}

// Evaluate the COMPOSITE op when the subgraph has dynamic outputs.
TfLiteStatus Eval_dynamic(TfLiteContext* context, TfLiteNode* node,
                          Subgraph* this_subgraph,
                          Subgraph* decomposition_subgraph) {
  TF_LITE_ENSURE_OK(context, decomposition_subgraph->AllocateTensors());
  const int num_inputs = node->inputs->size;
  const int num_outputs = node->outputs->size;
  const int* const start = node->inputs->data;
  std::vector<int> node_inputs(start, start + num_inputs);
  // node->inputs -> subgraph->inputs
  TF_LITE_ENSURE_OK(
      context, DeepOrShallowCopyTensorsShapeTypeData(
                   context, node, this_subgraph, node_inputs,
                   decomposition_subgraph, decomposition_subgraph->inputs()));

  // Invoke decomposition_subgraph subgraph
  TF_LITE_ENSURE_OK(context, decomposition_subgraph->Invoke());
  for (int tensor_index : decomposition_subgraph->outputs()) {
    decomposition_subgraph->EnsureTensorDataIsReadable(tensor_index);
  }

  // subgraph->outputs -> node->outputs
  TF_LITE_ENSURE_OK(context,
                    DeepCopyTensorsShapeTypeData(
                        context, node, decomposition_subgraph,
                        decomposition_subgraph->outputs(), this_subgraph,
                        TfLiteIntArrayView(node->outputs), true));

  for (int i = 0; i < num_outputs; ++i) {
    const int input_pos = OutputIsInput(decomposition_subgraph->outputs()[i],
                                        decomposition_subgraph->inputs());
    if (input_pos != -1) {
      TfLiteTensor* this_input =
          this_subgraph->tensor(node->inputs->data[input_pos]);
      TfLiteTensor* this_output = this_subgraph->tensor(node->outputs->data[i]);
      TfLiteTensorCopy(this_input, this_output);
    }
  }
  return kTfLiteOk;
}

// Evaluate the COMPOSITE op when the subgraph has static outputs.
TfLiteStatus Eval_static(TfLiteContext* context, TfLiteNode* node,
                         Subgraph* this_subgraph,
                         Subgraph* decomposition_subgraph) {
  const int num_inputs = node->inputs->size;
  const int num_outputs = node->outputs->size;
  const int* const start = node->inputs->data;
  std::vector<int> node_inputs(start, start + num_inputs);
  for (int i = 0; i < num_outputs; ++i) {
    int output_idx = decomposition_subgraph->outputs()[i];
    if (output_idx == kTfLiteOptionalTensor) continue;
    TfLiteTensor* subgraph_output = decomposition_subgraph->tensor(output_idx);
    if (!IsResourceOrVariant(subgraph_output) &&
        !IsConstantTensor(subgraph_output)) {
      subgraph_output->allocation_type = kTfLiteCustom;
    }
  }
  // node->inputs -> subgraph->inputs
  TF_LITE_ENSURE_OK(
      context, DeepOrShallowCopyTensorsShapeTypeData(
                   context, node, this_subgraph, node_inputs,
                   decomposition_subgraph, decomposition_subgraph->inputs()));

  TF_LITE_ENSURE_OK(
      context,
      CopyTensorsShapeAndType(context, decomposition_subgraph,
                              decomposition_subgraph->outputs(), this_subgraph,
                              TfLiteIntArrayView(node->outputs), false));
  for (int i = 0; i < num_outputs; ++i) {
    TfLiteTensor* this_output = this_subgraph->tensor(node->outputs->data[i]);
    TfLiteTensor* subgraph_output =
        decomposition_subgraph->tensor(decomposition_subgraph->outputs()[i]);
    if (decomposition_subgraph->outputs()[i] == kTfLiteOptionalTensor) {
      TfLiteTensor* this_input = this_subgraph->tensor(node->inputs->data[i]);
      TfLiteTensorResizeMaybeCopy(this_input->bytes, this_output, false);
      TfLiteTensorCopy(this_input, this_output);
    } else {
      const int input_pos = OutputIsInput(decomposition_subgraph->outputs()[i],
                                          decomposition_subgraph->inputs());
      if (input_pos != -1) {
        TfLiteTensor* this_input =
            this_subgraph->tensor(node->inputs->data[input_pos]);
        TfLiteTensorResizeMaybeCopy(this_input->bytes, this_output, false);
        TfLiteTensorCopy(this_input, this_output);
      } else if (IsConstantTensor(subgraph_output)) {
        TfLiteTensorCopy(subgraph_output, this_output);
      } else {
        subgraph_output->data = this_output->data;
      }
    }
  }

  // Invoke subgraph
  TF_LITE_ENSURE_OK(context, decomposition_subgraph->Invoke());
  for (int tensor_index : decomposition_subgraph->outputs()) {
    decomposition_subgraph->EnsureTensorDataIsReadable(tensor_index);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  State* op_state = reinterpret_cast<State*>(node->user_data);
  if (op_state->name == kCausalConvWithState1d) {
    return EvalCausalConvWithState1d(context, node);
  }
  if (op_state->name == kRecurrentLinearAttention) {
    return EvalRecurrentLinearAttention(context, node);
  }
  if (op_state->name == kSelectiveStateSpace) {
    return EvalSelectiveStateSpace(context, node);
  }

  Subgraph* this_subgraph = reinterpret_cast<Subgraph*>(context->impl_);
  auto* subgraphs = this_subgraph->GetSubgraphs();
  Subgraph* decomposition_subgraph =
      (*subgraphs)[op_state->subgraph_index].get();

  if (op_state->subgraph_has_dynamic_output_tensors) {
    TF_LITE_ENSURE_OK(context, Eval_dynamic(context, node, this_subgraph,
                                            decomposition_subgraph));
  } else {
    TF_LITE_ENSURE_OK(context, Eval_static(context, node, this_subgraph,
                                           decomposition_subgraph));
  }

  if (!this_subgraph->ShouldPreserveAllTensors()) {
    TF_LITE_ENSURE_OK(context, decomposition_subgraph->ReleaseMemory());
  }

  return kTfLiteOk;
}

}  // namespace stablehlo_composite

TfLiteRegistration* Register_STABLEHLO_COMPOSITE() {
  static TfLiteRegistration r = {/*.init=*/stablehlo_composite::Init,
                                 /*.free=*/stablehlo_composite::Free,
                                 /*.prepare=*/stablehlo_composite::Prepare,
                                 /*.invoke=*/stablehlo_composite::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

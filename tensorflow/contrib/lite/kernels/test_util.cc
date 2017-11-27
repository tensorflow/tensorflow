/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/contrib/lite/kernels/test_util.h"

#include "tensorflow/contrib/lite/version.h"
#include "tensorflow/core/platform/logging.h"

namespace tflite {

using ::testing::FloatNear;
using ::testing::Matcher;

namespace {
template <typename T>
std::pair<float, int32_t> QuantizationParams(float f_min, float f_max) {
  // These are required by many quantized operations.
  CHECK_LE(f_min, 0);
  CHECK_GE(f_max, 0);
  T q_min = std::numeric_limits<T>::min();
  T q_max = std::numeric_limits<T>::max();
  float range = q_max - q_min;
  float scale = (f_max - f_min) / range;
  int32_t zero_point = std::min(
      q_max,
      std::max(q_min, static_cast<T>(std::round(q_min - f_min / scale))));
  return {scale, zero_point};
}
}  // namespace

std::vector<Matcher<float>> ArrayFloatNear(const std::vector<float>& values,
                                           float max_abs_error) {
  std::vector<Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float& v : values) {
    matchers.emplace_back(FloatNear(v, max_abs_error));
  }
  return matchers;
}

int SingleOpModel::AddTensor(TensorData t) {
  int id = tensors_.size();

  // This is slightly different depending on whether we are adding a
  // quantized or a regular tensor.
  bool is_quantized = (t.min != 0 || t.max != 0 || t.scale != 0);

  flatbuffers::Offset<QuantizationParameters> q_params = 0;

  if (is_quantized) {
    if (t.min != 0 || t.max != 0) {
      if (t.type == TensorType_UINT8) {
        std::tie(t.scale, t.zero_point) =
            QuantizationParams<uint8_t>(t.min, t.max);
      } else if (t.type == TensorType_INT32) {
        std::tie(t.scale, t.zero_point) =
            QuantizationParams<int32_t>(t.min, t.max);
      } else {
        LOG(FATAL) << "No support for the requested quantized type";
      }
      t.min = 0;
      t.max = 0;
    }

    q_params = CreateQuantizationParameters(
        builder_, /*min=*/0, /*max=*/0, builder_.CreateVector<float>({t.scale}),
        builder_.CreateVector<int64_t>({t.zero_point}));
  }

  tensors_.push_back(CreateTensor(builder_, builder_.CreateVector<int>({}),
                                  t.type, /*buffer=*/0,
                                  /*name=*/0, q_params));

  tensor_data_[id] = t;

  return id;
}

int SingleOpModel::AddInput(const TensorData& t) {
  int id = AddTensor(t);
  inputs_.push_back(id);
  return id;
}

int SingleOpModel::AddNullInput() {
  int id = kOptionalTensor;
  inputs_.push_back(id);
  return id;
}

int SingleOpModel::AddOutput(const TensorData& t) {
  int id = AddTensor(t);
  outputs_.push_back(id);
  return id;
}

void SingleOpModel::SetBuiltinOp(BuiltinOperator type,
                                 BuiltinOptions builtin_options_type,
                                 flatbuffers::Offset<void> builtin_options) {
  opcodes_.push_back(CreateOperatorCode(builder_, type, 0));
  operators_.push_back(CreateOperator(
      builder_, /*opcode_index=*/0, builder_.CreateVector<int32_t>(inputs_),
      builder_.CreateVector<int32_t>(outputs_), builtin_options_type,
      builtin_options,
      /*custom_options=*/0, CustomOptionsFormat_FLEXBUFFERS));
}

void SingleOpModel::SetCustomOp(
    const string& name, const std::vector<uint8_t>& custom_option,
    const std::function<TfLiteRegistration*()>& registeration) {
  custom_registrations_[name] = registeration;
  opcodes_.push_back(
      CreateOperatorCodeDirect(builder_, BuiltinOperator_CUSTOM, name.data()));
  operators_.push_back(CreateOperator(
      builder_, /*opcode_index=*/0, builder_.CreateVector<int32_t>(inputs_),
      builder_.CreateVector<int32_t>(outputs_), BuiltinOptions_NONE, 0,
      builder_.CreateVector<uint8_t>(custom_option),
      CustomOptionsFormat_FLEXBUFFERS));
}

void SingleOpModel::BuildInterpreter(
    std::vector<std::vector<int>> input_shapes) {
  auto opcodes = builder_.CreateVector(opcodes_);
  auto operators = builder_.CreateVector(operators_);
  auto tensors = builder_.CreateVector(tensors_);
  auto inputs = builder_.CreateVector<int32_t>(inputs_);
  auto outputs = builder_.CreateVector<int32_t>(outputs_);
  // Create a single subgraph
  std::vector<flatbuffers::Offset<SubGraph>> subgraphs;
  auto subgraph = CreateSubGraph(builder_, tensors, inputs, outputs, operators);
  subgraphs.push_back(subgraph);
  auto subgraphs_flatbuffer = builder_.CreateVector(subgraphs);

  std::vector<flatbuffers::Offset<Buffer>> buffers_vec;
  auto buffers = builder_.CreateVector(buffers_vec);
  auto description = builder_.CreateString("programmatic model");
  builder_.Finish(CreateModel(builder_, TFLITE_SCHEMA_VERSION, opcodes,
                              subgraphs_flatbuffer, description, buffers));

  auto* model = GetModel(builder_.GetBufferPointer());

  ops::builtin::BuiltinOpResolver builtins;
  for (const auto& reg : custom_registrations_) {
    builtins.AddCustom(reg.first.data(), reg.second());
  }
  InterpreterBuilder(model, builtins)(&interpreter_);

  CHECK(interpreter_ != nullptr);

  int i = 0;
  for (const auto& shape : input_shapes) {
    int input_idx = interpreter_->inputs()[i++];
    if (input_idx == kOptionalTensor) continue;
    CHECK(interpreter_->ResizeInputTensor(input_idx, shape) == kTfLiteOk);
  }
  CHECK(interpreter_->AllocateTensors() == kTfLiteOk)
      << "Cannot allocate tensors";
}

void SingleOpModel::Invoke() { CHECK(interpreter_->Invoke() == kTfLiteOk); }

int32_t SingleOpModel::GetTensorSize(int index) const {
  TfLiteTensor* t = interpreter_->tensor(index);
  CHECK(t);
  int total_size = 1;
  for (int i = 0; i < t->dims->size; ++i) {
    total_size *= t->dims->data[i];
  }
  return total_size;
}

}  // namespace tflite

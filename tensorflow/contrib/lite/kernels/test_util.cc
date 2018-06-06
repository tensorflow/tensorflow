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

std::vector<Matcher<float>> ArrayFloatNear(const std::vector<float>& values,
                                           float max_abs_error) {
  std::vector<Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float& v : values) {
    matchers.emplace_back(FloatNear(v, max_abs_error));
  }
  return matchers;
}

int SingleOpModel::AddInput(const TensorData& t) {
  int id = AddTensor<float>(t, {});
  inputs_.push_back(id);
  return id;
}

int SingleOpModel::AddNullInput() {
  int id = kOptionalTensor;
  inputs_.push_back(id);
  return id;
}

int SingleOpModel::AddOutput(const TensorData& t) {
  int id = AddTensor<float>(t, {});
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
    const std::function<TfLiteRegistration*()>& registration) {
  custom_registrations_[name] = registration;
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

  auto buffers = builder_.CreateVector(buffers_);
  auto description = builder_.CreateString("programmatic model");
  builder_.Finish(CreateModel(builder_, TFLITE_SCHEMA_VERSION, opcodes,
                              subgraphs_flatbuffer, description, buffers));

  auto* model = GetModel(builder_.GetBufferPointer());

  if (!resolver_) {
    auto resolver = new ops::builtin::BuiltinOpResolver();
    for (const auto& reg : custom_registrations_) {
      resolver->AddCustom(reg.first.data(), reg.second());
    }
    resolver_ = std::unique_ptr<OpResolver>(resolver);
  }
  CHECK(InterpreterBuilder(model, *resolver_)(&interpreter_) == kTfLiteOk);

  CHECK(interpreter_ != nullptr);

  int i = 0;
  for (const auto& shape : input_shapes) {
    int input_idx = interpreter_->inputs()[i++];
    if (input_idx == kOptionalTensor) continue;
    if (shape.empty()) continue;
    CHECK(interpreter_->ResizeInputTensor(input_idx, shape) == kTfLiteOk);
  }

  // Modify delegate with function.
  if (apply_delegate_fn_) {
    apply_delegate_fn_(interpreter_.get());
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

template <>
std::vector<string> SingleOpModel::ExtractVector(int index) {
  TfLiteTensor* tensor_ptr = interpreter_->tensor(index);
  CHECK(tensor_ptr != nullptr);
  const int num_strings = GetStringCount(tensor_ptr);
  std::vector<string> result;
  result.reserve(num_strings);
  for (int i = 0; i < num_strings; ++i) {
    const auto str = GetString(tensor_ptr, i);
    result.emplace_back(str.str, str.len);
  }
  return result;
}
}  // namespace tflite

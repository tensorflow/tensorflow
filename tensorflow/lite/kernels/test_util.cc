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
#include "tensorflow/lite/kernels/test_util.h"

#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/nnapi/acceleration_test_util.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/acceleration_test_util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/tools/versioning/op_version.h"
#include "tensorflow/lite/version.h"

namespace tflite {

using ::testing::FloatNear;
using ::testing::Matcher;

namespace {

// Whether to enable (global) use of NNAPI. Note that this will typically
// be set via a command-line flag.
static bool force_use_nnapi = false;

TfLiteDelegate* TestNnApiDelegate() {
  static TfLiteDelegate* delegate = [] {
    StatefulNnApiDelegate::Options options;
    // In Android Q, the NNAPI delegate avoids delegation if the only device
    // is the reference CPU. However, for testing purposes, we still want
    // delegation coverage, so force use of this reference path.
    options.accelerator_name = "nnapi-reference";
    return new StatefulNnApiDelegate(options);
  }();
  return delegate;
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

std::vector<Matcher<std::complex<float>>> ArrayComplex64Near(
    const std::vector<std::complex<float>>& values, float max_abs_error) {
  std::vector<Matcher<std::complex<float>>> matchers;
  matchers.reserve(values.size());
  for (const std::complex<float>& v : values) {
    matchers.emplace_back(
        AllOf(::testing::Property(&std::complex<float>::real,
                                  FloatNear(v.real(), max_abs_error)),
              ::testing::Property(&std::complex<float>::imag,
                                  FloatNear(v.imag(), max_abs_error))));
  }
  return matchers;
}

int SingleOpModel::AddInput(const TensorData& t, bool is_variable) {
  int id = 0;
  if (t.per_channel_quantization) {
    id = AddTensorPerChannelQuant(t);
  } else {
    id = AddTensor<float>(t, {}, is_variable);
  }
  inputs_.push_back(id);
  return id;
}

int SingleOpModel::AddIntermediate(TensorType type,
                                   const std::vector<float>& scale,
                                   const std::vector<int64_t>& zero_point) {
  // Currently supports only int16 intermediate types.
  // TODO(jianlijianli): make use of the type.
  int id = tensors_.size();
  flatbuffers::Offset<QuantizationParameters> q_params =
      CreateQuantizationParameters(builder_, /*min=*/0, /*max=*/0,
                                   builder_.CreateVector<float>(scale),
                                   builder_.CreateVector<int64_t>(zero_point));
  tensors_.push_back(CreateTensor(builder_, builder_.CreateVector<int>({}),
                                  type,
                                  /*buffer=*/0,
                                  /*name=*/0, q_params, false));
  intermediates_.push_back(id);
  return id;
}

int SingleOpModel::AddNullInput() {
  int id = kTfLiteOptionalTensor;
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
  opcodes_.push_back(CreateOperatorCode(builder_, type, 0, 0));
  operators_.push_back(CreateOperator(
      builder_, /*opcode_index=*/0, builder_.CreateVector<int32_t>(inputs_),
      builder_.CreateVector<int32_t>(outputs_), builtin_options_type,
      builtin_options,
      /*custom_options=*/0, CustomOptionsFormat_FLEXBUFFERS, 0,
      builder_.CreateVector<int32_t>(intermediates_)));
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

void SingleOpModel::BuildInterpreter(std::vector<std::vector<int>> input_shapes,
                                     int num_threads,
                                     bool allow_fp32_relax_to_fp16,
                                     bool apply_delegate) {
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

  uint8_t* buffer_pointer = builder_.GetBufferPointer();
  UpdateOpVersion(buffer_pointer);

  if (!resolver_) {
    auto resolver = new ops::builtin::BuiltinOpResolver();
    for (const auto& reg : custom_registrations_) {
      resolver->AddCustom(reg.first.data(), reg.second());
    }
    resolver_ = std::unique_ptr<OpResolver>(resolver);
  }
  CHECK(InterpreterBuilder(GetModel(buffer_pointer), *resolver_)(
            &interpreter_, num_threads) == kTfLiteOk);

  CHECK(interpreter_ != nullptr);

  for (size_t i = 0; i < input_shapes.size(); ++i) {
    const int input_idx = interpreter_->inputs()[i];
    if (input_idx == kTfLiteOptionalTensor) continue;
    const auto& shape = input_shapes[i];
    if (shape.empty()) continue;
    CHECK(interpreter_->ResizeInputTensor(input_idx, shape) == kTfLiteOk);
  }

  interpreter_->SetAllowFp16PrecisionForFp32(allow_fp32_relax_to_fp16);

  CHECK(interpreter_->AllocateTensors() == kTfLiteOk)
      << "Cannot allocate tensors";
  interpreter_->ResetVariableTensors();

  // In some rare cases a test may need to postpone modifying the graph with
  // a delegate, e.g. if tensors are not fully specified. In such cases the
  // test has to explicitly call ApplyDelegate() when necessary.
  if (apply_delegate) ApplyDelegate();
}

TfLiteStatus SingleOpModel::ApplyDelegate() {
  if (force_use_nnapi) {
    delegate_ = TestNnApiDelegate();
  }

  if (delegate_) {
    return interpreter_->ModifyGraphWithDelegate(delegate_);
  }

  return kTfLiteOk;
}

void SingleOpModel::Invoke() { ASSERT_EQ(interpreter_->Invoke(), kTfLiteOk); }

TfLiteStatus SingleOpModel::InvokeUnchecked() { return interpreter_->Invoke(); }

void SingleOpModel::BuildInterpreter(
    std::vector<std::vector<int>> input_shapes) {
  BuildInterpreter(input_shapes, /*num_threads=*/-1,
                   /*allow_fp32_relax_to_fp16=*/false,
                   /*apply_delegate=*/true);
}

// static
void SingleOpModel::SetForceUseNnapi(bool use_nnapi) {
  force_use_nnapi = use_nnapi;
}

// static
bool SingleOpModel::GetForceUseNnapi() { return force_use_nnapi; }

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
std::vector<string> SingleOpModel::ExtractVector(int index) const {
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

namespace {

// Returns the number of partitions associated, as result of a call to
// ModifyGraphWithDelegate, to the given delegate.
int CountPartitionsDelegatedTo(Subgraph* subgraph,
                               const TfLiteDelegate* delegate) {
  return std::count_if(
      subgraph->nodes_and_registration().begin(),
      subgraph->nodes_and_registration().end(),
      [delegate](
          std::pair<TfLiteNode, TfLiteRegistration> node_and_registration) {
        return node_and_registration.first.delegate == delegate;
      });
}

// Returns the number of partitions associated, as result of a call to
// ModifyGraphWithDelegate, to the given delegate.
int CountPartitionsDelegatedTo(Interpreter* interpreter,
                               const TfLiteDelegate* delegate) {
  int result = 0;
  for (int i = 0; i < interpreter->subgraphs_size(); i++) {
    Subgraph* subgraph = interpreter->subgraph(i);

    result += CountPartitionsDelegatedTo(subgraph, delegate);
  }

  return result;
}

// Returns the number of nodes that will be executed on the CPU
int CountPartitionsExecutedByCpuKernel(const Interpreter* interpreter) {
  int result = 0;
  for (int node_idx : interpreter->execution_plan()) {
    TfLiteNode node;
    TfLiteRegistration reg;
    std::tie(node, reg) = *(interpreter->node_and_registration(node_idx));

    if (node.delegate == nullptr) {
      ++result;
    }
  }

  return result;
}

}  // namespace

void SingleOpModel::ExpectOpAcceleratedWithNnapi(const std::string& test_id) {
  absl::optional<NnapiAccelerationTestParams> validation_params =
      GetNnapiAccelerationTestParam(test_id);
  if (!validation_params.has_value()) {
    return;
  }

  TFLITE_LOG_PROD(TFLITE_LOG_INFO, "Validating acceleration");
  const NnApi* nnapi = NnApiImplementation();
  if (nnapi && nnapi->nnapi_exists &&
      nnapi->android_sdk_version >=
          validation_params.value().MinAndroidSdkVersion()) {
    EXPECT_EQ(
        CountPartitionsDelegatedTo(interpreter_.get(), TestNnApiDelegate()), 1)
        << "Expecting operation to be accelerated but cannot find a partition "
           "associated to the NNAPI delegate";
  }
}

void SingleOpModel::ValidateAcceleration() {
  if (force_use_nnapi) {
    ExpectOpAcceleratedWithNnapi(GetCurrentTestId());
  }
}

int SingleOpModel::CountOpsExecutedByCpuKernel() {
  return CountPartitionsExecutedByCpuKernel(interpreter_.get());
}

SingleOpModel::~SingleOpModel() { ValidateAcceleration(); }

void MultiOpModel::AddBuiltinOp(
    BuiltinOperator type, BuiltinOptions builtin_options_type,
    const flatbuffers::Offset<void>& builtin_options,
    const std::vector<int32_t>& inputs, const std::vector<int32_t>& outputs) {
  opcodes_.push_back(CreateOperatorCode(builder_, type, 0, 0));
  const int opcode_index = opcodes_.size() - 1;
  operators_.push_back(CreateOperator(
      builder_, opcode_index, builder_.CreateVector<int32_t>(inputs),
      builder_.CreateVector<int32_t>(outputs), builtin_options_type,
      builtin_options,
      /*custom_options=*/0, CustomOptionsFormat_FLEXBUFFERS));
}

void MultiOpModel::AddCustomOp(
    const string& name, const std::vector<uint8_t>& custom_option,
    const std::function<TfLiteRegistration*()>& registration,
    const std::vector<int32_t>& inputs, const std::vector<int32_t>& outputs) {
  custom_registrations_[name] = registration;
  opcodes_.push_back(
      CreateOperatorCodeDirect(builder_, BuiltinOperator_CUSTOM, name.data()));
  const int opcode_index = opcodes_.size() - 1;
  operators_.push_back(CreateOperator(
      builder_, opcode_index, builder_.CreateVector<int32_t>(inputs),
      builder_.CreateVector<int32_t>(outputs), BuiltinOptions_NONE, 0,
      builder_.CreateVector<uint8_t>(custom_option),
      CustomOptionsFormat_FLEXBUFFERS));
}

}  // namespace tflite

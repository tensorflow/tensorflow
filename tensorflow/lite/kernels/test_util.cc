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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/nnapi/acceleration_test_util.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/kernels/acceleration_test_util.h"
#include "tensorflow/lite/kernels/test_delegate_providers.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/simple_planner.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/tools/versioning/op_version.h"
#include "tensorflow/lite/version.h"
#include "tsl/platform/logging.h"

namespace tflite {

using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::Matcher;

namespace {

// Converts an integer from the sign-and-magnitude representation to
// the biased representation.  More precisely, let N be 2 to the
// power of (kBitCount - 1), an integer x is represented by the
// unsigned number x + N.
//
// For instance,
//
//   -N + 1 (the most negative number representable using
//          sign-and-magnitude) is represented by 1;
//   0      is represented by N; and
//   N - 1  (the biggest number representable using
//          sign-and-magnitude) is represented by 2N - 1.
//
// Read https://en.wikipedia.org/wiki/Signed_number_representations
// for more details on signed number representations.
uint32_t SignAndMagnitudeToBiased(uint32_t sam) {
  constexpr uint32_t kSignBitMask = 1u << 31;
  if (kSignBitMask & sam) {
    // sam represents a negative number.
    return ~sam + 1;
  } else {
    // sam represents a positive number.
    return kSignBitMask | sam;
  }
}
// Given two numbers in the sign-and-magnitude representation,
// returns the distance between them as an unsigned number.
uint32_t DistanceBetweenSignAndMagnitudeNumbers(uint32_t sam1, uint32_t sam2) {
  uint32_t biased1 = SignAndMagnitudeToBiased(sam1);
  uint32_t biased2 = SignAndMagnitudeToBiased(sam2);
  return (biased1 >= biased2) ? (biased1 - biased2) : (biased2 - biased1);
}
// Returns true if and only if lhs is at most max_ulps ULP's away from rhs.
// In particular, this function:
//
//   - returns true if both numbers are NAN.
//   - returns false if exact one of numbers is NAN.
//   - treats really large numbers as almost equal to infinity.
//   - thinks +0.0 and -0.0 are 0 ULP's apart.
bool AlmostEquals(float lhs, float rhs, uint32_t max_ulps) {
  if (std::isnan(lhs) || std::isnan(rhs)) {
    return std::isnan(lhs) && std::isnan(rhs);
  }

  return DistanceBetweenSignAndMagnitudeNumbers(
             absl::bit_cast<uint32_t>(lhs), absl::bit_cast<uint32_t>(rhs)) <=
         max_ulps;
}

MATCHER_P3(FloatAbsRelNear, value, max_abs_err, max_rel_err, "") {
  auto matcher =
      FloatNear(value, std::max(max_abs_err, std::abs(max_rel_err * value)));
  return ::testing::ExplainMatchResult(matcher, arg, result_listener);
}

MATCHER(Fp16Eq, "") {
  // FP16 only has 10 bits precision while FP32 has 23 bits precision. Thus, to
  // check if results of FP16 are almost equal, we could check the result is
  // within 4 * 2^13 ULPs of FP32, which equals to 4 ULPs of FP16.
  constexpr uint32_t fp16_ulps_in_fp32 = 4 * (1 << 13);
  float actual = std::get<0>(arg);
  float expected = std::get<1>(arg);
  // The minimum exponent of FP16 is 2^-14, which means the minimum ULP of FP16
  // is 2^-24. Therefore, when expected is less than 2^-14, i.e. a subnormal
  // FP16 number, the minimum ULP of FP16 should be used instead of ULP of FP32.
  if (std::abs(expected) < 0x1p-14) {
    return std::abs(actual - expected) <= 4 * 0x1p-24;
  }
  return AlmostEquals(actual, expected, fp16_ulps_in_fp32);
}

}  // namespace

bool AllowFp16PrecisionForFp32() {
  return tflite::KernelTestDelegateProviders::Get()->ConstParams().Get<bool>(
      tflite::KernelTestDelegateProviders::kAllowFp16PrecisionForFp32);
}

Matcher<std::tuple<float, float>> FloatingPointEq() {
  if (AllowFp16PrecisionForFp32()) {
    return Fp16Eq();
  }
  return Eq();
}

Matcher<std::tuple<float, float>> FloatingPointAlmostEq() {
  if (AllowFp16PrecisionForFp32()) {
    return Fp16Eq();
  }
  return FloatEq();
}

std::vector<Matcher<float>> ArrayFloatNear(const std::vector<float>& values,
                                           float max_abs_err,
                                           float fp16_max_abs_err,
                                           float max_rel_err,
                                           float fp16_max_rel_err) {
  if (AllowFp16PrecisionForFp32()) {
    if (fp16_max_abs_err == kFpErrorAuto) {
      max_abs_err = std::max(max_abs_err, std::sqrt(max_abs_err));
    } else {
      max_abs_err = fp16_max_abs_err;
    }
    max_rel_err = fp16_max_rel_err;
  }
  std::vector<Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float& v : values) {
    matchers.emplace_back(FloatAbsRelNear(v, max_abs_err, max_rel_err));
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

int SingleOpModel::AddInput(const TensorData& t) {
  int id = 0;
  if (t.per_block_quantization != 0) {
    id = AddTensorPerBlockQuant(t);
  } else if (t.per_channel_quantization) {
    id = AddTensorPerChannelQuant(t);
  } else {
    id = AddTensor<float>(t, nullptr, 0);
  }
  inputs_.push_back(id);
  return id;
}

int SingleOpModel::AddVariableInput(const TensorData& t) {
  int id = 0;
  if (t.per_channel_quantization) {
    id = AddTensorPerChannelQuant(t);
  } else {
    id = AddTensor<float>(t, nullptr, 0, true);
  }
  inputs_.push_back(id);
  return id;
}

int SingleOpModel::AddIntermediate(TensorType type,
                                   const std::vector<float>& scale,
                                   const std::vector<int64_t>& zero_point) {
  // Currently supports only int16 intermediate types.
  int id = tensors_.size();
  flatbuffers::Offset<QuantizationParameters> q_params =
      CreateQuantizationParameters(builder_, /*min=*/0, /*max=*/0,
                                   builder_.CreateVector<float>(scale),
                                   builder_.CreateVector<int64_t>(zero_point));
  std::vector<int> empty;
  tensors_.push_back(CreateTensor(builder_, builder_.CreateVector<int>(empty),
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
  int id = 0;
  if (t.per_channel_quantization) {
    id = AddTensorPerChannelQuant(t);
  } else {
    id = AddTensor<float>(t, nullptr, 0);
  }
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

void SingleOpModel::SetBuiltinOp(BuiltinOperator type,
                                 BuiltinOptions2 builtin_options_2_type,
                                 flatbuffers::Offset<void> builtin_options_2) {
  opcodes_.push_back(CreateOperatorCode(builder_, type, 0, 0));
  operators_.push_back(CreateOperator(
      builder_, /*opcode_index=*/0, builder_.CreateVector<int32_t>(inputs_),
      builder_.CreateVector<int32_t>(outputs_), tflite::BuiltinOptions_NONE,
      /*builtin_options=*/0, /*custom_options=*/0,
      CustomOptionsFormat_FLEXBUFFERS, /*mutating_variable_inputs=*/0,
      builder_.CreateVector<int32_t>(intermediates_),
      /*large_custom_options_offset=*/0,
      /*large_custom_options_size=*/0, builtin_options_2_type,
      builtin_options_2));
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

void SingleOpModel::AllocateAndDelegate(bool apply_delegate) {
  CHECK(interpreter_->AllocateTensors() == kTfLiteOk)
      << "Cannot allocate tensors";
  interpreter_->ResetVariableTensors();

  // In some rare cases a test may need to postpone modifying the graph with
  // a delegate, e.g. if tensors are not fully specified. In such cases the
  // test has to explicitly call ApplyDelegate() when necessary.
  if (apply_delegate) ApplyDelegate();
}

void SingleOpModel::BuildInterpreter(std::vector<std::vector<int>> input_shapes,
                                     int num_threads,
                                     bool allow_fp32_relax_to_fp16,
                                     bool apply_delegate,
                                     bool allocate_and_delegate,
                                     bool use_simple_allocator) {
  input_shapes_ = input_shapes;
  allow_fp32_relax_to_fp16_ = allow_fp32_relax_to_fp16;
  apply_delegate_ = apply_delegate;
  allocate_and_delegate_ = allocate_and_delegate;

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

  use_simple_allocator |=
      tflite::KernelTestDelegateProviders::Get()->ConstParams().Get<bool>(
          tflite::KernelTestDelegateProviders::kUseSimpleAllocator);

  if (!resolver_) {
    if (!bypass_default_delegates_) {
      // Check if any delegates are specified via the commandline flags. We also
      // assume the intention of the test is to test against a particular
      // delegate, hence bypassing applying TfLite default delegates (i.e. the
      // XNNPACK delegate).
      const auto specified_delegates =
          tflite::KernelTestDelegateProviders::Get()->CreateAllDelegates();
      if (!specified_delegates.empty()) {
        bypass_default_delegates_ = true;
      }
    }
    MutableOpResolver* resolver =
        (bypass_default_delegates_ || use_simple_allocator)
            ? new ops::builtin::BuiltinOpResolverWithoutDefaultDelegates()
            : new ops::builtin::BuiltinOpResolver();
    for (const auto& reg : custom_registrations_) {
      resolver->AddCustom(reg.first.data(), reg.second());
    }
    resolver_ = std::unique_ptr<OpResolver>(resolver);
  }
  CHECK(InterpreterBuilder(GetModel(buffer_pointer), *resolver_)(
            &interpreter_, num_threads) == kTfLiteOk);

  CHECK(interpreter_ != nullptr);

  if (use_simple_allocator) {
    LOG(INFO) << "Use SimplePlanner.\n";
    tflite::Subgraph& primary_subgraph = interpreter_->primary_subgraph();
    auto memory_planner = new SimplePlanner(
        &primary_subgraph.context_,
        std::unique_ptr<GraphInfo>(primary_subgraph.CreateGraphInfo()));
    primary_subgraph.memory_planner_.reset(memory_planner);
    memory_planner->PlanAllocations();
  }

  for (size_t i = 0; i < input_shapes.size(); ++i) {
    const int input_idx = interpreter_->inputs()[i];
    if (input_idx == kTfLiteOptionalTensor) continue;
    const auto& shape = input_shapes[i];
    if (shape.empty()) continue;
    CHECK(interpreter_->ResizeInputTensor(input_idx, shape) == kTfLiteOk);
  }

  interpreter_->SetAllowFp16PrecisionForFp32(allow_fp32_relax_to_fp16);

  if (allocate_and_delegate) {
    AllocateAndDelegate(apply_delegate);
  }
}

TfLiteStatus SingleOpModel::ApplyDelegate() {
  if (delegate_) {
    TFLITE_LOG(WARN) << "Having a manually-set TfLite delegate, and bypassing "
                        "KernelTestDelegateProviders";
    SetDelegateApplicationStatus(
        interpreter_->ModifyGraphWithDelegate(delegate_));
    TF_LITE_ENSURE_STATUS(*GetDelegateApplicationStatus());
    ++num_applied_delegates_;
  } else {
    auto* delegate_providers = tflite::KernelTestDelegateProviders::Get();
    // Most TFLite NNAPI delegation tests have been written to run against the
    // NNAPI CPU path. We'll enable that for tests. However, need to first check
    // if the parameter is present - it will not be if the NNAPI delegate
    // provider is not linked into the test.
    if (delegate_providers->ConstParams().HasParam("disable_nnapi_cpu")) {
      delegate_providers->MutableParams()->Set("disable_nnapi_cpu", false);
    }
    for (auto& one : delegate_providers->CreateAllDelegates()) {
      // The raw ptr always points to the actual TfLiteDegate object.
      auto* delegate_raw_ptr = one.delegate.get();
      SetDelegateApplicationStatus(
          interpreter_->ModifyGraphWithDelegate(std::move(one.delegate)));
      TF_LITE_ENSURE_STATUS(*GetDelegateApplicationStatus());
      // Note: 'delegate_' is always set to the last successfully applied one.
      delegate_ = delegate_raw_ptr;
      ++num_applied_delegates_;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus SingleOpModel::Invoke() { return interpreter_->Invoke(); }

void SingleOpModel::BuildInterpreter(std::vector<std::vector<int>> input_shapes,
                                     bool use_simple_allocator) {
  BuildInterpreter(input_shapes, /*num_threads=*/-1,
                   /*allow_fp32_relax_to_fp16=*/false,
                   /*apply_delegate=*/true, /*allocate_and_delegate=*/true,
                   use_simple_allocator);
}

// static
bool SingleOpModel::GetForceUseNnapi() {
  const auto& delegate_params =
      tflite::KernelTestDelegateProviders::Get()->ConstParams();
  // It's possible this library isn't linked with the nnapi delegate provider
  // lib.
  return delegate_params.HasParam("use_nnapi") &&
         delegate_params.Get<bool>("use_nnapi");
}

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

/*static*/ AccelerationValidator* AccelerationValidator::Get() {
  static AccelerationValidator* const validator = new AccelerationValidator();
  return validator;
}

void AccelerationValidator::AddCallback(Callback callback) {
  callbacks_.push_back(std::move(callback));
}

void AccelerationValidator::Validate(const SingleOpModel& model) const {
  for (const auto& callback : callbacks_) {
    if (callback == nullptr) continue;
    callback(model);
  }
}

void SingleOpModel::ExpectOpAcceleratedWithNnapi(const std::string& test_id) {
  std::optional<NnapiAccelerationTestParams> validation_params =
      GetNnapiAccelerationTestParam(test_id);
  if (!validation_params.has_value()) {
    return;
  }

  // If we have multiple delegates applied, we would skip this check at the
  // moment.
  if (num_applied_delegates_ > 1) {
    TFLITE_LOG(WARN) << "Skipping ExpectOpAcceleratedWithNnapi as "
                     << num_applied_delegates_
                     << " delegates have been successfully applied.";
    return;
  }
  TFLITE_LOG(INFO) << "Validating acceleration";
  const NnApi* nnapi = NnApiImplementation();
  if (nnapi && nnapi->nnapi_exists &&
      nnapi->android_sdk_version >=
          validation_params.value().MinAndroidSdkVersion()) {
    EXPECT_EQ(CountPartitionsDelegatedTo(interpreter_.get(), delegate_), 1)
        << "Expecting operation to be accelerated but cannot find a partition "
           "associated to the NNAPI delegate";
    EXPECT_GT(num_applied_delegates_, 0) << "No delegates were applied.";
  }
}

void SingleOpModel::ValidateAcceleration() {
  if (GetForceUseNnapi()) {
    ExpectOpAcceleratedWithNnapi(GetCurrentTestId());
  }
  AccelerationValidator::Get()->Validate(*this);
}

int SingleOpModel::CountOpsExecutedByCpuKernel() {
  return CountPartitionsExecutedByCpuKernel(interpreter_.get());
}

int SingleOpModel::CountNumberOfDelegatedPartitions() const {
  return CountPartitionsDelegatedTo(interpreter_.get(), delegate_);
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

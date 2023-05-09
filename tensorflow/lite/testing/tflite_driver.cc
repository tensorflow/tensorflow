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
#include "tensorflow/lite/testing/tflite_driver.h"

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#if !defined(__APPLE__)
#include "tensorflow/lite/delegates/flex/delegate.h"
#endif
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/gradient/gradient_ops.h"
#include "tensorflow/lite/kernels/parse_example/parse_example.h"
#include "tensorflow/lite/kernels/perception/perception_ops.h"
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow/lite/kernels/test_delegate_providers.h"
#include "tensorflow/lite/signature_runner.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/join.h"
#include "tensorflow/lite/testing/split.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace testing {

namespace {
const double kRelativeThreshold = 1e-2f;
const double kAbsoluteThreshold = 1e-4f;
const char kDefaultSignatureKey[] = "serving_default";

// For quantized tests, we use a different error measurement from float ones.
// Assumes the baseline is a always a float TF model.
// Error of a quantized model compared to the baseline comes from two sources:
//   1. the math done with quantized inputs, and
//   2. quantization of the output.
// Assumes there is no error introduced by source 1, the theoretical maximum
// error allowed for the output is 0.5 * scale, because scale is equal to the
// size of the quantization bucket.
//
// As a result, we use `scale` as a unit for measuring the quantization error.
// To add the error introduced by source 1 as well, we need to relax the
// multiplier from 0.5 to a larger number, which is model/op dependent.
// The number below is good enough to account for both the two sources of error
// for most quantized op tests to pass.
const int kQuantizationErrorMultiplier = 4;

template <typename T>
void SetTensorData(const std::vector<T>& values, void* data) {
  T* input_ptr = static_cast<T*>(data);
  std::copy(values.begin(), values.end(), input_ptr);
}

bool InterpretAsQuantized(const TfLiteTensor& tensor) {
  if (tensor.quantization.type == kTfLiteNoQuantization) return false;

  // Quantized single-op models with uint8 input/output type are only used for
  // EdgeTPU tests.
  // EdgeTPU tests need to read the quantized values as-is to check for
  // bit-exactness. As a result we don't interpret the tensor as quantized.
  // TODO(b/176121243): Add an option to interpret uint8 buffers as
  // non-quantized type and set if from the child class.
  if (tensor.type == kTfLiteUInt8) return false;

  if (tensor.quantization.params != nullptr) {
    auto* quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(tensor.quantization.params);
    if (quantization->scale != nullptr && quantization->scale->size == 1 &&
        quantization->zero_point != nullptr &&
        quantization->zero_point->size == 1) {
      return true;
    }
  }
  return false;
}
}  // namespace

/* static */
bool TfLiteDriver::InitTestDelegateProviders(int* argc, const char** argv) {
  return tflite::KernelTestDelegateProviders::Get()->InitFromCmdlineArgs(argc,
                                                                         argv);
}

TfLiteDriver::TfLiteDriver(DelegateType delegate_type, bool reference_kernel)
    : delegate_(nullptr, nullptr),
      relative_threshold_(kRelativeThreshold),
      absolute_threshold_(kAbsoluteThreshold),
      quantization_error_multiplier_(kQuantizationErrorMultiplier) {
  if (reference_kernel) {
    resolver_ = std::make_unique<ops::builtin::BuiltinRefOpResolver>();
  } else {
    // TODO(b/168278077): change back to use BuiltinOpResolver after zip tests
    // are fully validated against TfLite delegates.
    resolver_ = std::make_unique<
        ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>();
    ops::builtin::BuiltinOpResolver* builtin_op_resolver_ =
        reinterpret_cast<ops::builtin::BuiltinOpResolver*>(resolver_.get());
    builtin_op_resolver_->AddCustom("IRFFT2D",
                                    tflite::ops::custom::Register_IRFFT2D());
    builtin_op_resolver_->AddCustom("Roll",
                                    tflite::ops::custom::Register_ROLL());
    tflite::ops::custom::AddGradientOps(builtin_op_resolver_);
    tflite::ops::custom::AddParseExampleOp(builtin_op_resolver_);
    tflite::ops::custom::AddPerceptionOps(builtin_op_resolver_);
  }

  switch (delegate_type) {
    case DelegateType::kNone:
      break;
    case DelegateType::kNnapi:
      delegate_ = evaluation::CreateNNAPIDelegate();
      break;
    case DelegateType::kGpu:
      delegate_ = evaluation::CreateGPUDelegate();
      break;
    case DelegateType::kFlex:
#if !defined(__APPLE__)
      delegate_ = FlexDelegate::Create();
#endif
      break;
  }
}

TfLiteDriver::~TfLiteDriver() {
  for (auto t : tensors_to_deallocate_) {
    DeallocateStringTensor(t.second);
  }
}

void TfLiteDriver::AllocateTensors() {
  if (must_allocate_tensors_) {
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
      Invalidate("Failed to allocate tensors");
      return;
    }
    ResetLSTMStateTensors();
    must_allocate_tensors_ = false;
  }
}

void TfLiteDriver::LoadModel(const std::string& bin_file_path,
                             const std::string& signature) {
  if (!IsValid()) return;

  model_ = FlatBufferModel::BuildFromFile(GetFullPath(bin_file_path).c_str());
  if (!model_) {
    Invalidate("Failed to mmap model " + bin_file_path);
    return;
  }
  InterpreterBuilder(*model_, *resolver_)(&interpreter_);
  if (!interpreter_) {
    Invalidate("Failed build interpreter");
    return;
  }
  if (delegate_) {
    if (interpreter_->ModifyGraphWithDelegate(delegate_.get()) != kTfLiteOk) {
      Invalidate("Unable to the build graph using the delegate");
      return;
    }
  } else {
    auto* delegate_providers = tflite::KernelTestDelegateProviders::Get();
    for (auto& one : delegate_providers->CreateAllDelegates()) {
      if (interpreter_->ModifyGraphWithDelegate(std::move(one.delegate)) !=
          kTfLiteOk) {
        Invalidate(
            "Unable to the build graph using the delegate initialized from "
            "tflite::KernelTestDelegateProviders");
        return;
      }
    }
  }

  must_allocate_tensors_ = true;

  signature_runner_ = interpreter_->GetSignatureRunner(signature.c_str());
  if (signature_runner_) {
    signature_inputs_ = interpreter_->signature_inputs(signature.c_str());
    signature_outputs_ = interpreter_->signature_outputs(signature.c_str());
  } else {
    Invalidate("Unable to the fetch signature runner.");
  }
}

void TfLiteDriver::LoadModel(const std::string& bin_file_path) {
  LoadModel(bin_file_path, kDefaultSignatureKey);
}

void TfLiteDriver::ReshapeTensor(const std::string& name,
                                 const std::string& csv_values) {
  if (!IsValid()) return;
  if (signature_runner_->ResizeInputTensor(
          name.c_str(), testing::Split<int>(csv_values, ",")) != kTfLiteOk) {
    Invalidate("Failed to resize input tensor " + name);
    return;
  }
  must_allocate_tensors_ = true;
}

void TfLiteDriver::ResetTensor(const std::string& name) {
  if (!IsValid()) return;
  auto* tensor = signature_runner_->input_tensor(name.c_str());
  memset(tensor->data.raw, 0, tensor->bytes);
}

void TfLiteDriver::Invoke(
    const std::vector<std::pair<std::string, std::string>>& inputs) {
  if (!IsValid()) return;
  for (const auto& input : inputs) {
    SetInput(input.first, input.second);
  }
  if (signature_runner_->Invoke() != kTfLiteOk) {
    Invalidate("Failed to invoke interpreter");
  }
}

std::string TfLiteDriver::ReadOutput(const std::string& name) {
  if (!IsValid()) return "";
  return TensorValueToCsvString(signature_runner_->output_tensor(name.c_str()));
}

bool TfLiteDriver::CheckResults(
    const std::vector<std::pair<std::string, std::string>>& expected_outputs,
    const std::vector<std::pair<std::string, std::string>>&
        expected_output_shapes) {
  if (!IsValid()) return false;
  bool success = true;
  for (const auto& output : expected_outputs) {
    SetExpectation(output.first, output.second);
  }
  for (const auto& shape : expected_output_shapes) {
    SetShapeExpectation(shape.first, shape.second);
  }

  for (const auto& p : expected_output_) {
    int id = p.first;
    auto* tensor = interpreter_->tensor(id);
    if (!p.second->Check(/*verbose=*/false, *tensor)) {
      // Do not invalidate anything here. Instead, simply output the
      // differences and return false. Invalidating would prevent all
      // subsequent invocations from running..
      std::cerr << "TfLiteDriver: There were errors in invocation '"
                << GetInvocationId() << "', validating output tensor '" << id
                << "':" << std::endl;
      p.second->Check(/*verbose=*/true, *tensor);
      success = false;
      SetOverallSuccess(false);
    }
  }
  for (const auto& p : expected_output_shape_) {
    int id = p.first;
    auto* tensor = interpreter_->tensor(id);
    if (!p.second->CheckShape(/*verbose=*/false, *tensor)) {
      // Do not invalidate anything here. Instead, simply output the
      // differences and return false. Invalidating would prevent all
      // subsequent invocations from running..
      std::cerr << "TfLiteDriver: There were errors in invocation '"
                << GetInvocationId()
                << "', validating the shape of output tensor '" << id
                << "':" << std::endl;
      p.second->CheckShape(/*verbose=*/true, *tensor);
      success = false;
      SetOverallSuccess(false);
    }
  }
  expected_output_.clear();
  return success;
}

std::vector<std::string> TfLiteDriver::GetOutputNames() {
  if (!IsValid()) return {};
  std::vector<std::string> names;
  for (const auto* name : signature_runner_->output_names()) {
    names.push_back(name);
  }
  return names;
}

void TfLiteDriver::SetInput(const std::string& name,
                            const std::string& csv_values) {
  auto id = signature_inputs_[name];
  auto* tensor = signature_runner_->input_tensor(name.c_str());
  switch (tensor->type) {
    case kTfLiteFloat64: {
      const auto& values = testing::Split<double>(csv_values, ",");
      if (!CheckSizes<double>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteFloat32: {
      const auto& values = testing::Split<float>(csv_values, ",");
      if (!CheckSizes<float>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteInt32: {
      const auto& values = testing::Split<int32_t>(csv_values, ",");
      if (!CheckSizes<int32_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteUInt32: {
      const auto& values = testing::Split<uint32_t>(csv_values, ",");
      if (!CheckSizes<uint32_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteInt64: {
      const auto& values = testing::Split<int64_t>(csv_values, ",");
      if (!CheckSizes<int64_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteUInt64: {
      const auto& values = testing::Split<uint64_t>(csv_values, ",");
      if (!CheckSizes<uint64_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteUInt8: {
      const auto& values = testing::Split<uint8_t>(csv_values, ",");
      if (!CheckSizes<uint8_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteInt8: {
      const auto& values = testing::Split<int8_t>(csv_values, ",");
      if (!CheckSizes<int8_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteInt16: {
      const auto& values = testing::Split<int16_t>(csv_values, ",");
      if (!CheckSizes<int16_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteUInt16: {
      const auto& values = testing::Split<uint16_t>(csv_values, ",");
      if (!CheckSizes<uint16_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteBool: {
      const auto& values = testing::Split<bool>(csv_values, ",");
      if (!CheckSizes<bool>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteString: {
      std::string s = absl::HexStringToBytes(csv_values);

      DeallocateStringTensor(tensors_to_deallocate_[id]);
      AllocateStringTensor(id, s.size(), tensor);
      memcpy(tensor->data.raw, s.data(), s.size());

      break;
    }
    case kTfLiteComplex64: {
      const auto& values = testing::Split<std::complex<float>>(csv_values, ",");
      if (!CheckSizes<std::complex<float>>(tensor->bytes, values.size()))
        return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteComplex128: {
      const auto& values =
          testing::Split<std::complex<double>>(csv_values, ",");
      if (!CheckSizes<std::complex<double>>(tensor->bytes, values.size()))
        return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    case kTfLiteFloat16: {
      const auto& values = testing::Split<Eigen::half>(csv_values, ",");
      for (auto k : values) {
        TFLITE_LOG(INFO) << "input" << k;
      }
      if (!CheckSizes<Eigen::half>(tensor->bytes, values.size())) return;
      SetTensorData(values, tensor->data.raw);
      break;
    }
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ",
                              TfLiteTypeGetName(tensor->type),
                              " in TfLiteDriver::SetInput"));
      return;
  }
}

void TfLiteDriver::SetThreshold(double relative_threshold,
                                double absolute_threshold) {
  relative_threshold_ = relative_threshold;
  absolute_threshold_ = absolute_threshold;
}

void TfLiteDriver::SetQuantizationErrorMultiplier(
    int quantization_error_multiplier) {
  quantization_error_multiplier_ = quantization_error_multiplier;
}

void TfLiteDriver::SetExpectation(const std::string& name,
                                  const std::string& csv_values) {
  auto id = signature_outputs_[name];
  auto* tensor = signature_runner_->output_tensor(name.c_str());
  if (expected_output_.count(id) != 0) {
    Invalidate(absl::StrCat("Overridden expectation for tensor '", id, "'"));
  }
  expected_output_[id] = std::make_unique<DataExpectation>(
      relative_threshold_, absolute_threshold_, quantization_error_multiplier_);

  if (InterpretAsQuantized(*tensor)) {
    expected_output_[id]->SetData<float>(csv_values);
    return;
  }

  switch (tensor->type) {
    case kTfLiteFloat32:
      expected_output_[id]->SetData<float>(csv_values);
      break;
    case kTfLiteInt32:
      expected_output_[id]->SetData<int32_t>(csv_values);
      break;
    case kTfLiteUInt32:
      expected_output_[id]->SetData<uint32_t>(csv_values);
      break;
    case kTfLiteInt64:
      expected_output_[id]->SetData<int64_t>(csv_values);
      break;
    case kTfLiteUInt64:
      expected_output_[id]->SetData<uint64_t>(csv_values);
      break;
    case kTfLiteUInt8:
      expected_output_[id]->SetData<uint8_t>(csv_values);
      break;
    case kTfLiteInt8:
      expected_output_[id]->SetData<int8_t>(csv_values);
      break;
    case kTfLiteUInt16:
      expected_output_[id]->SetData<uint16_t>(csv_values);
      break;
    case kTfLiteInt16:
      expected_output_[id]->SetData<int16_t>(csv_values);
      break;
    case kTfLiteBool:
      expected_output_[id]->SetData<bool>(csv_values);
      break;
    case kTfLiteString:
      expected_output_[id]->SetData<std::string>(csv_values);
      break;
    case kTfLiteFloat64:
      expected_output_[id]->SetData<double>(csv_values);
      break;
    case kTfLiteComplex64:
      expected_output_[id]->SetData<std::complex<float>>(csv_values);
      break;
    case kTfLiteComplex128:
      expected_output_[id]->SetData<std::complex<double>>(csv_values);
      break;
    case kTfLiteFloat16:
      expected_output_[id]->SetData<Eigen::half>(csv_values);
      break;
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ",
                              TfLiteTypeGetName(tensor->type),
                              " in TfLiteDriver::SetExpectation"));
      return;
  }
}

void TfLiteDriver::SetShapeExpectation(const std::string& name,
                                       const std::string& csv_values) {
  auto id = signature_outputs_[name];
  if (expected_output_shape_.count(id) != 0) {
    Invalidate(
        absl::StrCat("Overridden shape expectation for tensor '", id, "'"));
  }
  expected_output_shape_[id] = std::make_unique<ShapeExpectation>(csv_values);
}

void TfLiteDriver::ResetLSTMStateTensors() {
  interpreter_->ResetVariableTensors();
}

std::string TfLiteDriver::TensorValueToCsvString(const TfLiteTensor* tensor) {
  int num_elements = 1;

  for (int i = 0; i < tensor->dims->size; ++i) {
    num_elements *= tensor->dims->data[i];
  }

  switch (tensor->type) {
    case kTfLiteFloat32:
      return JoinDefault(tensor->data.f, num_elements, ",");
    case kTfLiteInt32:
      return JoinDefault(tensor->data.i32, num_elements, ",");
    case kTfLiteUInt32:
      return JoinDefault(tensor->data.u32, num_elements, ",");
    case kTfLiteInt64:
      return JoinDefault(tensor->data.i64, num_elements, ",");
    case kTfLiteUInt64:
      return JoinDefault(tensor->data.u64, num_elements, ",");
    case kTfLiteUInt8:
      return Join(tensor->data.uint8, num_elements, ",");
    case kTfLiteInt8:
      return Join(tensor->data.int8, num_elements, ",");
    case kTfLiteUInt16:
      return Join(tensor->data.ui16, num_elements, ",");
    case kTfLiteInt16:
      return Join(tensor->data.i16, num_elements, ",");
    case kTfLiteBool:
      return JoinDefault(tensor->data.b, num_elements, ",");
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ",
                              TfLiteTypeGetName(tensor->type),
                              " in TfLiteDriver::ReadOutput"));
      return "";
  }
}

}  // namespace testing
}  // namespace tflite

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
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#if !defined(__APPLE__)
#include "tensorflow/lite/delegates/flex/delegate.h"
#endif
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/gradient/gradient_ops.h"
#include "tensorflow/lite/kernels/parse_example/parse_example.h"
#include "tensorflow/lite/kernels/perception/perception_ops.h"
#include "tensorflow/lite/kernels/register.h"
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

// Returns the value in the given position in a tensor.
template <typename T>
T Value(void* data, int index) {
  return static_cast<T*>(data)[index];
}

template <typename T>
void SetTensorData(const std::vector<T>& values, void* data) {
  T* input_ptr = static_cast<T*>(data);
  std::copy(values.begin(), values.end(), input_ptr);
}

// Implement type erasure with unique_ptr with custom deleter
using unique_void_ptr = std::unique_ptr<void, void (*)(void*)>;

template <typename T>
unique_void_ptr make_type_erased_array(size_t size) {
  return unique_void_ptr(static_cast<void*>(new T[size]),
                         [](void* data) { delete[] static_cast<T*>(data); });
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

class TfLiteDriver::DataExpectation {
 public:
  DataExpectation(double relative_threshold, double absolute_threshold,
                  int quantization_error_multiplier)
      : data_(nullptr, nullptr),
        num_elements_(0),
        relative_threshold_(relative_threshold),
        absolute_threshold_(absolute_threshold),
        quantization_error_multiplier_(quantization_error_multiplier) {}

  template <typename T>
  void SetData(const string& csv_values) {
    const auto& values = testing::Split<T>(csv_values, ",");
    num_elements_ = values.size();
    data_ = make_type_erased_array<T>(num_elements_);
    SetTensorData(values, data_.get());
  }

  bool Check(bool verbose, const TfLiteTensor& tensor);

 private:
  bool CompareTwoValuesHelper(float v1, float v2) {
    if (std::isnan(v1) || std::isnan(v2)) {
      return !(std::isnan(v1) && std::isnan(v2));
    }

    float diff = std::abs(v1 - v2);
    bool error_is_large = false;
    // For very small numbers, try absolute error, otherwise go with
    // relative.
    if (std::abs(v2) < relative_threshold_) {
      error_is_large = (diff > absolute_threshold_);
    } else {
      error_is_large = (diff > relative_threshold_ * std::abs(v2));
    }
    return error_is_large;
  }

  bool CompareTwoValuesHelper(double v1, double v2) {
    if (std::isnan(v1) || std::isnan(v2)) {
      return !(std::isnan(v1) && std::isnan(v2));
    }

    double diff = std::abs(v1 - v2);
    bool error_is_large = false;
    // For very small numbers, try absolute error, otherwise go with
    // relative.
    if (std::abs(v2) < relative_threshold_) {
      error_is_large = (diff > absolute_threshold_);
    } else {
      error_is_large = (diff > relative_threshold_ * std::abs(v2));
    }
    return error_is_large;
  }

  bool CompareTwoValues(std::complex<float> v1, std::complex<float> v2) {
    return CompareTwoValues(v1.real(), v2.real()) ||
           CompareTwoValues(v1.imag(), v2.imag());
  }

  bool CompareTwoValues(std::complex<double> v1, std::complex<double> v2) {
    return CompareTwoValues(v1.real(), v2.real()) ||
           CompareTwoValues(v1.imag(), v2.imag());
  }

  bool CompareTwoValues(float v1, float v2) {
    return CompareTwoValuesHelper(v1, v2);
  }

  bool CompareTwoValues(double v1, double v2) {
    return CompareTwoValuesHelper(v1, v2);
  }

  template <typename T, typename TS>
  bool TypedCheck(bool verbose, const TfLiteTensor& tensor) {
    size_t tensor_size = tensor.bytes / sizeof(T);

    if (tensor_size != num_elements_) {
      std::cerr << "Expected a tensor with " << num_elements_
                << " elements, got " << tensor_size << std::endl;
      std::cerr << "while checking tensor " << tensor.name << std::endl;
      return false;
    }

    bool good_output = true;
    for (int i = 0; i < tensor_size; ++i) {
      TS computed = Value<T>(tensor.data.raw, i);
      TS reference = Value<T>(data_.get(), i);
      if (CompareTwoValues(computed, reference)) {
        good_output = false;
        if (verbose) {
          std::cerr << "  Tensor[" << tensor.name << "] index " << i << ": got "
                    << computed << ", but expected " << reference << std::endl;
        }
      }
    }
    return good_output;
  }

  bool TypedCheckString(bool verbose, const TfLiteTensor& tensor);
  bool QuantizedCheck(bool verbose, const TfLiteTensor& tensor);

  unique_void_ptr data_;
  size_t num_elements_;
  double relative_threshold_;
  double absolute_threshold_;
  int quantization_error_multiplier_;
};

class TfLiteDriver::ShapeExpectation {
 public:
  explicit ShapeExpectation(const string& csv_values)
      : shape_(testing::Split<int32_t>(csv_values, ",")) {}

  bool CheckShape(bool verbose, const TfLiteTensor& tensor) {
    bool valid = true;
    if (tensor.dims->size == shape_.size()) {
      for (int i = 0; i < shape_.size(); ++i) {
        if (shape_[i] != tensor.dims->data[i]) {
          valid = false;
        }
      }
    } else {
      valid = false;
    }
    if (!valid && verbose) {
      std::cerr << "Incorrect output shape while checking tensor "
                << tensor.name << std::endl;
      std::cerr << "TFLite output shape: ";
      for (int i = 0; i < tensor.dims->size; ++i) {
        std::cerr << tensor.dims->data[i] << ", ";
      }
      std::cerr << std::endl;
      std::cerr << "Expected output shape: ";
      for (int i = 0; i < shape_.size(); ++i) {
        std::cerr << shape_[i] << ", ";
      }
      std::cerr << std::endl;
    }
    return valid;
  }

 private:
  std::vector<int32_t> shape_;
};

template <>
void TfLiteDriver::DataExpectation::SetData<string>(const string& csv_values) {
  string s = absl::HexStringToBytes(csv_values);
  data_ = make_type_erased_array<char>(s.size());
  memcpy(data_.get(), s.data(), s.size());
}

bool TfLiteDriver::DataExpectation::TypedCheckString(
    bool verbose, const TfLiteTensor& tensor) {
  if (tensor.data.raw == nullptr) {
    if (verbose) {
      std::cerr << "  got empty string" << std::endl;
    }
    return false;
  }
  int expected_num_strings = GetStringCount(data_.get());
  int returned_num_strings = GetStringCount(&tensor);
  if (expected_num_strings != returned_num_strings) {
    if (verbose) {
      std::cerr << "  string count differ: got " << returned_num_strings
                << ", but expected " << expected_num_strings << std::endl;
    }
    return false;
  }
  for (int i = 0; i < returned_num_strings; ++i) {
    auto expected_ref = GetString(data_.get(), i);
    auto returned_ref = GetString(&tensor, i);
    if (expected_ref.len != returned_ref.len) {
      if (verbose) {
        std::cerr << "  index " << i << ": got string of size "
                  << returned_ref.len << ", but expected size "
                  << expected_ref.len << std::endl;
      }
      return false;
    }
    if (strncmp(expected_ref.str, returned_ref.str, returned_ref.len) != 0) {
      if (verbose) {
        std::cerr << "  index " << i << ": strings are different" << std::endl;
      }
      return false;
    }
  }

  return true;
}

bool TfLiteDriver::DataExpectation::QuantizedCheck(bool verbose,
                                                   const TfLiteTensor& tensor) {
  auto* quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(tensor.quantization.params);
  const float scale = quantization->scale->data[0];
  const int32_t zero_point = quantization->zero_point->data[0];

  bool good_result = true;
  int int_size = tensor.type == kTfLiteInt8 ? 1 : 2;
  for (int i = 0; i < tensor.bytes / int_size; i++) {
    int32_t computed =
        tensor.type == kTfLiteInt8 ? tensor.data.int8[i] : tensor.data.i16[i];
    const float dequantized =
        static_cast<float>(scale * (computed - zero_point));
    int error_multiplier = quantization_error_multiplier_;
    // If we are doing int16 symmetric quantization of activations, we need to
    // bump up the potential error. Since the weights are quantized to 8 bits
    // and the activations are 16bits, the output is could be getting
    // effectively 8bit error instead of 16bit error. So we need to multiply the
    // error mulitplier by 255 (the difference in number of values between a
    // 16bit and 8bit number)
    if (tensor.type == kTfLiteInt16) error_multiplier *= 255;
    const float reference = Value<float>(data_.get(), i);
    if (std::abs(dequantized - reference) > error_multiplier * scale) {
      if (verbose) {
        std::cerr << "  index " << i << ": got " << dequantized
                  << ", but expected " << reference << std::endl;
      }
      good_result = false;
    }
  }
  return good_result;
}

bool TfLiteDriver::DataExpectation::Check(bool verbose,
                                          const TfLiteTensor& tensor) {
  if (InterpretAsQuantized(tensor)) {
    return QuantizedCheck(verbose, tensor);
  }

  switch (tensor.type) {
    case kTfLiteFloat32:
      return TypedCheck<float, float>(verbose, tensor);
    case kTfLiteInt32:
      return TypedCheck<int32_t, float>(verbose, tensor);
    case kTfLiteUInt32:
      return TypedCheck<uint32_t, float>(verbose, tensor);
    case kTfLiteInt64:
      return TypedCheck<int64_t, float>(verbose, tensor);
    case kTfLiteUInt64:
      return TypedCheck<uint64_t, float>(verbose, tensor);
    case kTfLiteUInt8:
      return TypedCheck<uint8_t, float>(verbose, tensor);
    case kTfLiteInt8:
      return TypedCheck<int8_t, float>(verbose, tensor);
    case kTfLiteUInt16:
      return TypedCheck<uint16_t, float>(verbose, tensor);
    case kTfLiteInt16:
      return TypedCheck<int16_t, float>(verbose, tensor);
    case kTfLiteBool:
      return TypedCheck<bool, float>(verbose, tensor);
    case kTfLiteString:
      return TypedCheckString(verbose, tensor);
    case kTfLiteComplex64:
      return TypedCheck<std::complex<float>, std::complex<float>>(verbose,
                                                                  tensor);
    case kTfLiteComplex128:
      return TypedCheck<std::complex<double>, std::complex<double>>(verbose,
                                                                    tensor);
    case kTfLiteFloat64:
      return TypedCheck<double, double>(verbose, tensor);
    default:
      fprintf(stderr, "Unsupported type %d in Check\n", tensor.type);
      return false;
  }
}

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
    resolver_.reset(new ops::builtin::BuiltinRefOpResolver);
  } else {
    // TODO(b/168278077): change back to use BuiltinOpResolver after zip tests
    // are fully validated against TfLite delegates.
    resolver_.reset(
        new ops::builtin::BuiltinOpResolverWithoutDefaultDelegates());
    ops::builtin::BuiltinOpResolver* builtin_op_resolver_ =
        reinterpret_cast<ops::builtin::BuiltinOpResolver*>(resolver_.get());
    builtin_op_resolver_->AddCustom("IRFFT2D",
                                    tflite::ops::custom::Register_IRFFT2D());
    builtin_op_resolver_->AddCustom(
        "AvgPool3D", tflite::ops::custom::Register_AVG_POOL_3D());
    builtin_op_resolver_->AddCustom(
        "MaxPool3D", tflite::ops::custom::Register_MAX_POOL_3D());
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

void TfLiteDriver::LoadModel(const string& bin_file_path,
                             const string& signature) {
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

void TfLiteDriver::LoadModel(const string& bin_file_path) {
  LoadModel(bin_file_path, kDefaultSignatureKey);
}

void TfLiteDriver::ReshapeTensor(const string& name, const string& csv_values) {
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
    const std::vector<std::pair<string, string>>& inputs) {
  if (!IsValid()) return;
  for (const auto& input : inputs) {
    SetInput(input.first, input.second);
  }
  if (signature_runner_->Invoke() != kTfLiteOk) {
    Invalidate("Failed to invoke interpreter");
  }
}

string TfLiteDriver::ReadOutput(const string& name) {
  if (!IsValid()) return "";
  return TensorValueToCsvString(signature_runner_->output_tensor(name.c_str()));
}

bool TfLiteDriver::CheckResults(
    const std::vector<std::pair<string, string>>& expected_outputs,
    const std::vector<std::pair<string, string>>& expected_output_shapes) {
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

std::vector<string> TfLiteDriver::GetOutputNames() {
  if (!IsValid()) return {};
  std::vector<string> names;
  for (const auto* name : signature_runner_->output_names()) {
    names.push_back(name);
  }
  return names;
}

void TfLiteDriver::SetInput(const string& name, const string& csv_values) {
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
      string s = absl::HexStringToBytes(csv_values);

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

void TfLiteDriver::SetExpectation(const string& name,
                                  const string& csv_values) {
  auto id = signature_outputs_[name];
  auto* tensor = signature_runner_->output_tensor(name.c_str());
  if (expected_output_.count(id) != 0) {
    Invalidate(absl::StrCat("Overridden expectation for tensor '", id, "'"));
  }
  expected_output_[id].reset(
      new DataExpectation(relative_threshold_, absolute_threshold_,
                          quantization_error_multiplier_));

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
      expected_output_[id]->SetData<string>(csv_values);
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
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ",
                              TfLiteTypeGetName(tensor->type),
                              " in TfLiteDriver::SetExpectation"));
      return;
  }
}

void TfLiteDriver::SetShapeExpectation(const string& name,
                                       const string& csv_values) {
  auto id = signature_outputs_[name];
  if (expected_output_shape_.count(id) != 0) {
    Invalidate(
        absl::StrCat("Overridden shape expectation for tensor '", id, "'"));
  }
  expected_output_shape_[id].reset(new ShapeExpectation(csv_values));
}

void TfLiteDriver::ResetLSTMStateTensors() {
  interpreter_->ResetVariableTensors();
}

string TfLiteDriver::TensorValueToCsvString(const TfLiteTensor* tensor) {
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

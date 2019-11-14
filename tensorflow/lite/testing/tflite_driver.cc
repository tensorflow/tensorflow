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
#include <memory>
#include <vector>

#include "absl/strings/escaping.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/join.h"
#include "tensorflow/lite/testing/split.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace testing {

namespace {
const double kRelativeThreshold = 1e-2f;
const double kAbsoluteThreshold = 1e-4f;

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

}  // namespace

class TfLiteDriver::DataExpectation {
 public:
  DataExpectation(double relative_threshold, double absolute_threshold)
      : data_(nullptr, nullptr),
        num_elements_(0),
        relative_threshold_(relative_threshold),
        absolute_threshold_(absolute_threshold) {}

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

  bool CompareTwoValues(std::complex<float> v1, std::complex<float> v2) {
    return CompareTwoValues(v1.real(), v2.real()) ||
           CompareTwoValues(v1.imag(), v2.imag());
  }

  bool CompareTwoValues(float v1, float v2) {
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
          std::cerr << "  index " << i << ": got " << computed
                    << ", but expected " << reference << std::endl;
        }
      }
    }
    return good_output;
  }

  bool TypedCheckString(bool verbose, const TfLiteTensor& tensor);

  unique_void_ptr data_;
  size_t num_elements_;
  double relative_threshold_;
  double absolute_threshold_;
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

bool TfLiteDriver::DataExpectation::Check(bool verbose,
                                          const TfLiteTensor& tensor) {
  switch (tensor.type) {
    case kTfLiteFloat32:
      return TypedCheck<float, float>(verbose, tensor);
    case kTfLiteInt32:
      return TypedCheck<int32_t, float>(verbose, tensor);
    case kTfLiteInt64:
      return TypedCheck<int64_t, float>(verbose, tensor);
    case kTfLiteUInt8:
      return TypedCheck<uint8_t, float>(verbose, tensor);
    case kTfLiteInt8:
      return TypedCheck<int8_t, float>(verbose, tensor);
    case kTfLiteBool:
      return TypedCheck<bool, float>(verbose, tensor);
    case kTfLiteString:
      return TypedCheckString(verbose, tensor);
    case kTfLiteComplex64:
      return TypedCheck<std::complex<float>, std::complex<float>>(verbose,
                                                                  tensor);
    default:
      fprintf(stderr, "Unsupported type %d in Check\n", tensor.type);
      return false;
  }
}

TfLiteDriver::TfLiteDriver(DelegateType delegate_type, bool reference_kernel)
    : delegate_(nullptr, nullptr),
      relative_threshold_(kRelativeThreshold),
      absolute_threshold_(kAbsoluteThreshold) {
  if (reference_kernel) {
    resolver_.reset(new ops::builtin::BuiltinRefOpResolver);
  } else {
    resolver_.reset(new ops::builtin::BuiltinOpResolver);
    ops::builtin::BuiltinOpResolver* buildinop_resolver_ =
        reinterpret_cast<ops::builtin::BuiltinOpResolver*>(resolver_.get());
    buildinop_resolver_->AddCustom("RFFT2D",
                                   tflite::ops::custom::Register_RFFT2D());
  }

  switch (delegate_type) {
    case DelegateType::kNone:
      break;
    case DelegateType::kNnapi:
      delegate_ = evaluation::CreateNNAPIDelegate();
      break;
    case DelegateType::kGpu:
      delegate_ = evaluation::CreateGPUDelegate(/*model=*/nullptr);
      break;
    case DelegateType::kFlex:
      delegate_ = Interpreter::TfLiteDelegatePtr(
          FlexDelegate::Create().release(), [](TfLiteDelegate* delegate) {
            delete static_cast<tflite::FlexDelegate*>(delegate);
          });
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

void TfLiteDriver::LoadModel(const string& bin_file_path) {
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
  }

  must_allocate_tensors_ = true;
}

void TfLiteDriver::ResetTensor(int id) {
  if (!IsValid()) return;
  auto* tensor = interpreter_->tensor(id);
  memset(tensor->data.raw, 0, tensor->bytes);
}

void TfLiteDriver::ReshapeTensor(int id, const string& csv_values) {
  if (!IsValid()) return;
  if (interpreter_->ResizeInputTensor(
          id, testing::Split<int>(csv_values, ",")) != kTfLiteOk) {
    Invalidate("Failed to resize input tensor " + std::to_string(id));
    return;
  }
  must_allocate_tensors_ = true;
}

void TfLiteDriver::SetInput(int id, const string& csv_values) {
  if (!IsValid()) return;
  auto* tensor = interpreter_->tensor(id);
  switch (tensor->type) {
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
    case kTfLiteInt64: {
      const auto& values = testing::Split<int64_t>(csv_values, ",");
      if (!CheckSizes<int64_t>(tensor->bytes, values.size())) return;
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

void TfLiteDriver::SetExpectation(int id, const string& csv_values) {
  if (!IsValid()) return;
  auto* tensor = interpreter_->tensor(id);
  if (expected_output_.count(id) != 0) {
    Invalidate(absl::StrCat("Overridden expectation for tensor '", id, "'"));
  }
  expected_output_[id].reset(
      new DataExpectation(relative_threshold_, absolute_threshold_));
  switch (tensor->type) {
    case kTfLiteFloat32:
      expected_output_[id]->SetData<float>(csv_values);
      break;
    case kTfLiteInt32:
      expected_output_[id]->SetData<int32_t>(csv_values);
      break;
    case kTfLiteInt64:
      expected_output_[id]->SetData<int64_t>(csv_values);
      break;
    case kTfLiteUInt8:
      expected_output_[id]->SetData<uint8_t>(csv_values);
      break;
    case kTfLiteInt8:
      expected_output_[id]->SetData<int8_t>(csv_values);
      break;
    case kTfLiteBool:
      expected_output_[id]->SetData<bool>(csv_values);
      break;
    case kTfLiteString:
      expected_output_[id]->SetData<string>(csv_values);
      break;
    case kTfLiteComplex64:
      expected_output_[id]->SetData<std::complex<float>>(csv_values);
      break;
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ",
                              TfLiteTypeGetName(tensor->type),
                              " in TfLiteDriver::SetExpectation"));
      return;
  }
}

void TfLiteDriver::SetShapeExpectation(int id, const string& csv_values) {
  if (!IsValid()) return;
  if (expected_output_shape_.count(id) != 0) {
    Invalidate(
        absl::StrCat("Overridden shape expectation for tensor '", id, "'"));
  }
  expected_output_shape_[id].reset(new ShapeExpectation(csv_values));
}

void TfLiteDriver::Invoke() {
  if (!IsValid()) return;
  if (interpreter_->Invoke() != kTfLiteOk) {
    Invalidate("Failed to invoke interpreter");
  }
}

bool TfLiteDriver::CheckResults() {
  if (!IsValid()) return false;
  bool success = true;
  for (const auto& p : expected_output_) {
    int id = p.first;
    auto* tensor = interpreter_->tensor(id);
    if (!p.second->Check(/*verbose=*/false, *tensor)) {
      // Do not invalidate anything here. Instead, simply output the
      // differences and return false. Invalidating would prevent all
      // subsequent invocations from running..
      std::cerr << "There were errors in invocation '" << GetInvocationId()
                << "', output tensor '" << id << "':" << std::endl;
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
      std::cerr << "There were errors in invocation '" << GetInvocationId()
                << "', output tensor '" << id << "':" << std::endl;
      p.second->CheckShape(/*verbose=*/true, *tensor);
      success = false;
      SetOverallSuccess(false);
    }
  }
  expected_output_.clear();
  return success;
}

void TfLiteDriver::ResetLSTMStateTensors() {
  interpreter_->ResetVariableTensors();
}

string TfLiteDriver::ReadOutput(int id) {
  auto* tensor = interpreter_->tensor(id);
  int num_elements = 1;

  for (int i = 0; i < tensor->dims->size; ++i) {
    num_elements *= tensor->dims->data[i];
  }

  switch (tensor->type) {
    case kTfLiteFloat32:
      return JoinDefault(tensor->data.f, num_elements, ",");
    case kTfLiteInt32:
      return JoinDefault(tensor->data.i32, num_elements, ",");
    case kTfLiteInt64:
      return JoinDefault(tensor->data.i64, num_elements, ",");
    case kTfLiteUInt8:
      return Join(tensor->data.uint8, num_elements, ",");
    case kTfLiteInt8:
      return Join(tensor->data.int8, num_elements, ",");
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

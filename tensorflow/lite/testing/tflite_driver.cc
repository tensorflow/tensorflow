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

#include <iostream>

#include "absl/strings/escaping.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/delegates/flex/delegate.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/split.h"

namespace tflite {
namespace testing {

namespace {

// Returns the value in the given position in a tensor.
template <typename T>
T Value(const TfLitePtrUnion& data, int index);
template <>
float Value(const TfLitePtrUnion& data, int index) {
  return data.f[index];
}
template <>
int32_t Value(const TfLitePtrUnion& data, int index) {
  return data.i32[index];
}
template <>
int64_t Value(const TfLitePtrUnion& data, int index) {
  return data.i64[index];
}
template <>
uint8_t Value(const TfLitePtrUnion& data, int index) {
  return data.uint8[index];
}
template <>
bool Value(const TfLitePtrUnion& data, int index) {
  return data.b[index];
}

template <typename T>
void SetTensorData(const std::vector<T>& values, TfLitePtrUnion* data) {
  T* input_ptr = reinterpret_cast<T*>(data->raw);
  for (const T& v : values) {
    *input_ptr = v;
    ++input_ptr;
  }
}

}  // namespace

class TfLiteDriver::Expectation {
 public:
  Expectation() {
    data_.raw = nullptr;
    num_elements_ = 0;
  }
  ~Expectation() { delete[] data_.raw; }
  template <typename T>
  void SetData(const string& csv_values) {
    const auto& values = testing::Split<T>(csv_values, ",");
    num_elements_ = values.size();
    data_.raw = new char[num_elements_ * sizeof(T)];
    SetTensorData(values, &data_);
  }

  template <>
  void SetData<string>(const string& csv_values) {
    string s = absl::HexStringToBytes(csv_values);
    data_.raw = new char[s.size()];
    memcpy(data_.raw, s.data(), s.size());
  }

  bool Check(bool verbose, const TfLiteTensor& tensor) {
    switch (tensor.type) {
      case kTfLiteFloat32:
        return TypedCheck<float>(verbose, tensor);
      case kTfLiteInt32:
        return TypedCheck<int32_t>(verbose, tensor);
      case kTfLiteInt64:
        return TypedCheck<int64_t>(verbose, tensor);
      case kTfLiteUInt8:
        return TypedCheck<uint8_t>(verbose, tensor);
      case kTfLiteBool:
        return TypedCheck<bool>(verbose, tensor);
      case kTfLiteString:
        return TypedCheck<string>(verbose, tensor);
      default:
        fprintf(stderr, "Unsupported type %d in Check\n", tensor.type);
        return false;
    }
  }

 private:
  template <typename T>
  bool TypedCheck(bool verbose, const TfLiteTensor& tensor) {
    // TODO(ahentz): must find a way to configure the tolerance.
    constexpr double kRelativeThreshold = 1e-2f;
    constexpr double kAbsoluteThreshold = 1e-4f;

    size_t tensor_size = tensor.bytes / sizeof(T);

    if (tensor_size != num_elements_) {
      std::cerr << "Expected a tensor with " << num_elements_
                << " elements, got " << tensor_size << std::endl;
      std::cerr << "while checking tensor " << tensor.name << std::endl;
      return false;
    }

    bool good_output = true;
    for (int i = 0; i < tensor_size; ++i) {
      float computed = Value<T>(tensor.data, i);
      float reference = Value<T>(data_, i);
      float diff = std::abs(computed - reference);
      bool error_is_large = false;
      // For very small numbers, try absolute error, otherwise go with
      // relative.
      if (std::abs(reference) < kRelativeThreshold) {
        error_is_large = (diff > kAbsoluteThreshold);
      } else {
        error_is_large = (diff > kRelativeThreshold * std::abs(reference));
      }
      if (error_is_large) {
        good_output = false;
        if (verbose) {
          std::cerr << "  index " << i << ": got " << computed
                    << ", but expected " << reference << std::endl;
        }
      }
    }
    return good_output;
  }

  template <>
  bool TypedCheck<string>(bool verbose, const TfLiteTensor& tensor) {
    if (tensor.data.raw == nullptr) {
      if (verbose) {
        std::cerr << "  got empty string" << std::endl;
      }
      return false;
    }
    int expected_num_strings = GetStringCount(data_.raw);
    int returned_num_strings = GetStringCount(tensor.data.raw);
    if (expected_num_strings != returned_num_strings) {
      if (verbose) {
        std::cerr << "  string count differ: got " << returned_num_strings
                  << ", but expected " << expected_num_strings << std::endl;
      }
      return false;
    }
    for (int i = 0; i < returned_num_strings; ++i) {
      auto expected_ref = GetString(data_.raw, i);
      auto returned_ref = GetString(tensor.data.raw, i);
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
          std::cerr << "  index " << i << ": strings are different"
                    << std::endl;
        }
        return false;
      }
    }

    return true;
  }

  TfLitePtrUnion data_;
  size_t num_elements_;
};

TfLiteDriver::TfLiteDriver(bool use_nnapi, const string& delegate_name,
                           bool reference_kernel)
    : use_nnapi_(use_nnapi) {
  if (reference_kernel) {
    resolver_.reset(new ops::builtin::BuiltinRefOpResolver);
  } else {
    resolver_.reset(new ops::builtin::BuiltinOpResolver);
  }

  if (delegate_name == "FLEX") {
    delegate_ = FlexDelegate::Create();
  }
}

TfLiteDriver::~TfLiteDriver() {
  for (auto t : tensors_to_deallocate_) {
    DeallocateStringTensor(t.second);
  }
  interpreter_.reset();
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
  interpreter_->UseNNAPI(use_nnapi_);

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
      SetTensorData(values, &tensor->data);
      break;
    }
    case kTfLiteInt32: {
      const auto& values = testing::Split<int32_t>(csv_values, ",");
      if (!CheckSizes<int32_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, &tensor->data);
      break;
    }
    case kTfLiteInt64: {
      const auto& values = testing::Split<int64_t>(csv_values, ",");
      if (!CheckSizes<int64_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, &tensor->data);
      break;
    }
    case kTfLiteUInt8: {
      const auto& values = testing::Split<uint8_t>(csv_values, ",");
      if (!CheckSizes<uint8_t>(tensor->bytes, values.size())) return;
      SetTensorData(values, &tensor->data);
      break;
    }
    case kTfLiteBool: {
      const auto& values = testing::Split<bool>(csv_values, ",");
      if (!CheckSizes<bool>(tensor->bytes, values.size())) return;
      SetTensorData(values, &tensor->data);
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

void TfLiteDriver::SetExpectation(int id, const string& csv_values) {
  if (!IsValid()) return;
  auto* tensor = interpreter_->tensor(id);
  if (expected_output_.count(id) != 0) {
    Invalidate(absl::StrCat("Overridden expectation for tensor '", id, "'"));
  }
  expected_output_[id].reset(new Expectation);
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
    case kTfLiteBool:
      expected_output_[id]->SetData<bool>(csv_values);
      break;
    case kTfLiteString:
      expected_output_[id]->SetData<string>(csv_values);
      break;
    default:
      Invalidate(absl::StrCat("Unsupported tensor type ",
                              TfLiteTypeGetName(tensor->type),
                              " in TfLiteDriver::SetExpectation"));
      return;
  }
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
  expected_output_.clear();
  return success;
}

void TfLiteDriver::ResetLSTMStateTensors() {
  interpreter_->ResetVariableTensors();
}

}  // namespace testing
}  // namespace tflite

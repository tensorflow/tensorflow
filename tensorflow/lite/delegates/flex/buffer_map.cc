/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/flex/buffer_map.h"

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/string.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/log_memory.h"

namespace tflite {
namespace flex {
namespace {
// A tensor buffer that is allocated, deallocated and populated by TF Lite.
class BaseTfLiteTensorBuffer : public tensorflow::TensorBuffer {
  using tensorflow::TensorBuffer::TensorBuffer;

  TensorBuffer* root_buffer() override { return this; }
  void FillAllocationDescription(
      tensorflow::AllocationDescription* proto) const override {
    tensorflow::int64 rb = size();
    proto->set_requested_bytes(rb);
    proto->set_allocator_name(tensorflow::cpu_allocator()->Name());
  }

  // Prevents input forwarding from mutating this buffer.
  bool OwnsMemory() const override { return false; }

 protected:
  void LogAllocation() {
    if (tensorflow::LogMemory::IsEnabled() && data() != nullptr) {
      tensorflow::LogMemory::RecordRawAllocation(
          "TfLiteTensorBuffer_New",
          tensorflow::LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID, size(),
          data(), tensorflow::cpu_allocator());
    }
  }
  void LogDeallocation() {
    if (tensorflow::LogMemory::IsEnabled() && data() != nullptr) {
      tensorflow::LogMemory::RecordRawDeallocation(
          "TfLiteTensorBuffer_Delete",
          tensorflow::LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID, data(),
          tensorflow::cpu_allocator(), false);
    }
  }
};

// A tensor buffer for most data types. Numeric types have exactly the same
// representation in TFLITE and TF, so we just need use memcpy().
class TfLiteTensorBuffer : public BaseTfLiteTensorBuffer {
 public:
  explicit TfLiteTensorBuffer(const TfLiteTensor* tensor)
      : BaseTfLiteTensorBuffer(tensorflow::cpu_allocator()->AllocateRaw(
            EIGEN_MAX_ALIGN_BYTES, tensor->bytes)) {
    // TODO(ahentz): if we can guarantee that TF Lite allocated tensors with
    // the same alignment as TensorFlow (EIGEN_MAX_ALIGN_BYTES), then we can
    // potentially eliminate the copy below.
    len_ = tensor->bytes;

    LogAllocation();

    if (data()) {
      std::memcpy(data(), tensor->data.raw, tensor->bytes);
    }
  }

  ~TfLiteTensorBuffer() override {
    LogDeallocation();
    tensorflow::cpu_allocator()->DeallocateRaw(data());
  }

  size_t size() const override { return len_; }

 private:
  size_t len_;
};

// A string buffer. TFLITE string tensor format is different than
// TF's so we need perform the conversion here.
class StringTfLiteTensorBuffer : public BaseTfLiteTensorBuffer {
 public:
  explicit StringTfLiteTensorBuffer(const TfLiteTensor* tensor)
      : StringTfLiteTensorBuffer(tensor, tensor->data.raw != nullptr
                                             ? GetStringCount(tensor->data.raw)
                                             : 0) {}

  ~StringTfLiteTensorBuffer() override {
    LogDeallocation();
    tensorflow::cpu_allocator()->Deallocate<string>(
        static_cast<string*>(data()), num_strings_);
  }

  size_t size() const override { return num_strings_ * sizeof(string); }

 private:
  StringTfLiteTensorBuffer(const TfLiteTensor* tensor, int num_strings)
      : BaseTfLiteTensorBuffer(
            num_strings != 0
                ? tensorflow::cpu_allocator()->Allocate<string>(num_strings)
                : nullptr),
        num_strings_(num_strings) {
    LogAllocation();

    if (data()) {
      string* p = static_cast<string*>(data());
      for (size_t i = 0; i < num_strings_; ++p, ++i) {
        auto ref = GetString(tensor->data.raw, i);
        p->assign(ref.str, ref.len);
      }
    }
  }

  int num_strings_;
};

}  // namespace

BufferMap::BufferMap() {}

BufferMap::~BufferMap() {}

bool BufferMap::HasTensor(int tensor_index) const {
  return id_to_tensor_.count(tensor_index) != 0;
}

bool BufferMap::IsTensorFlowTensor(int tensor_index) const {
  return HasTensor(tensor_index) && owned_by_tf_.count(tensor_index) > 0;
}

tensorflow::Tensor BufferMap::GetTensor(int tensor_index) const {
  return id_to_tensor_.at(tensor_index);
}

void BufferMap::SetFromTfLite(int tensor_index, const TfLiteTensor* tensor) {
  tensorflow::TensorShape shape;
  int num_dims = tensor->dims->size;
  for (int i = 0; i < num_dims; ++i) {
    shape.AddDim(tensor->dims->data[i]);
  }
  // TODO(ahentz): we assume this is a new tensor and allocate a new buffer
  // for it. This is not always the best approach. For example, this might
  // be a reallocation after resizing tensors. In that case it would be
  // preferable to somehow reuse the buffer.
  BaseTfLiteTensorBuffer* buf;
  if (tensor->type == kTfLiteString) {
    buf = new StringTfLiteTensorBuffer(tensor);
  } else {
    buf = new TfLiteTensorBuffer(tensor);
  }
  tensorflow::Tensor t = tensorflow::TensorCApi::MakeTensor(
      GetTensorFlowDataType(tensor->type), shape, buf);
  buf->Unref();

  id_to_tensor_[tensor_index] = std::move(t);
  owned_by_tf_.erase(tensor_index);
}

void BufferMap::SetFromTensorFlow(int tensor_index, tensorflow::Tensor tensor) {
  id_to_tensor_[tensor_index] = std::move(tensor);
  owned_by_tf_.insert(tensor_index);
}

}  // namespace flex
}  // namespace tflite

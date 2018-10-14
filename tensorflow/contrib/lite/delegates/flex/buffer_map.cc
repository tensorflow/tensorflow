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
#include "tensorflow/contrib/lite/delegates/flex/buffer_map.h"

#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/contrib/lite/delegates/flex/util.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/log_memory.h"

namespace tflite {
namespace flex {
namespace {
// A tensor buffer that is allocated, deallocated and populated by TF Lite.
class TfLiteTensorBuffer : public tensorflow::TensorBuffer {
 public:
  explicit TfLiteTensorBuffer(const TfLiteTensor* tensor) {
    len_ = tensor->bytes;
    // TODO(ahentz): if we can guarantee that TF Lite allocated tensors with
    // the same alignment as TensorFlow (EIGEN_MAX_ALIGN_BYTES), then we can
    // potentially eliminate the copy below.
    data_ =
        tensorflow::cpu_allocator()->AllocateRaw(EIGEN_MAX_ALIGN_BYTES, len_);
    if (data_ != nullptr) {
      if (tensorflow::LogMemory::IsEnabled()) {
        tensorflow::LogMemory::RecordRawAllocation(
            "TfLiteTensorBuffer_New",
            tensorflow::LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID, len_,
            data_, tensorflow::cpu_allocator());
      }
      std::memcpy(data_, tensor->data.raw, tensor->bytes);
    }
  }

  ~TfLiteTensorBuffer() override {
    if (tensorflow::LogMemory::IsEnabled() && data_ != nullptr) {
      tensorflow::LogMemory::RecordRawDeallocation(
          "TfLiteTensorBuffer_Delete",
          tensorflow::LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID, data_,
          tensorflow::cpu_allocator(), false);
    }
    tensorflow::cpu_allocator()->DeallocateRaw(data_);
  }

  void* data() const override { return data_; }
  size_t size() const override { return len_; }

  TensorBuffer* root_buffer() override { return this; }
  void FillAllocationDescription(
      tensorflow::AllocationDescription* proto) const override {
    tensorflow::int64 rb = size();
    proto->set_requested_bytes(rb);
    proto->set_allocator_name(tensorflow::cpu_allocator()->Name());
  }

  // Prevents input forwarding from mutating this buffer.
  bool OwnsMemory() const override { return false; }

 private:
  void* data_;
  size_t len_;
};
}  // namespace

BufferMap::BufferMap() {}

BufferMap::~BufferMap() {}

bool BufferMap::HasTensor(int tensor_index) const {
  return id_to_tensor_.count(tensor_index) != 0;
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
  // be a reallocation after resizing tensors. In that case we would be
  // preferable to somehow reuse the buffer.
  auto* buf = new TfLiteTensorBuffer(tensor);
  tensorflow::Tensor t = tensorflow::TensorCApi::MakeTensor(
      GetTensorFlowDataType(tensor->type), shape, buf);
  buf->Unref();

  SetFromTensorFlow(tensor_index, std::move(t));
}

void BufferMap::SetFromTensorFlow(int tensor_index, tensorflow::Tensor tensor) {
  id_to_tensor_[tensor_index] = std::move(tensor);
}

}  // namespace flex
}  // namespace tflite

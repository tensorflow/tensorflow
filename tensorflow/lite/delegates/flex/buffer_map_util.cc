/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/flex/buffer_map_util.h"

#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/typed_allocator.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/delegates/flex/util.h"
#include "tensorflow/lite/experimental/resource/resource_variable.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace flex {

void BaseTfLiteTensorBuffer::FillAllocationDescription(
    tensorflow::AllocationDescription* proto) const {
  int64_t rb = size();
  proto->set_requested_bytes(rb);
  proto->set_allocator_name(tensorflow::cpu_allocator()->Name());
}

void BaseTfLiteTensorBuffer::LogAllocation() {
  if (tensorflow::LogMemory::IsEnabled() && data() != nullptr) {
    tensorflow::LogMemory::RecordRawAllocation(
        "TfLiteTensorBuffer_New",
        tensorflow::LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID, size(),
        data(), tensorflow::cpu_allocator());
  }
}
void BaseTfLiteTensorBuffer::LogDeallocation() {
  if (tensorflow::LogMemory::IsEnabled() && data() != nullptr) {
    tensorflow::LogMemory::RecordRawDeallocation(
        "TfLiteTensorBuffer_Delete",
        tensorflow::LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID, data(),
        tensorflow::cpu_allocator(), false);
  }
}

TfLiteTensorBuffer::TfLiteTensorBuffer(const TfLiteTensor* tensor)
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

TfLiteTensorBuffer::~TfLiteTensorBuffer() {
  LogDeallocation();
  tensorflow::cpu_allocator()->DeallocateRaw(data());
}

StringTfLiteTensorBuffer::StringTfLiteTensorBuffer(const TfLiteTensor* tensor)
    : StringTfLiteTensorBuffer(
          tensor, tensor->data.raw != nullptr ? GetStringCount(tensor) : 0) {}

StringTfLiteTensorBuffer::~StringTfLiteTensorBuffer() {
  LogDeallocation();
  tensorflow::TypedAllocator::Deallocate<tensorflow::tstring>(
      tensorflow::cpu_allocator(), static_cast<tensorflow::tstring*>(data()),
      num_strings_);
}

StringTfLiteTensorBuffer::StringTfLiteTensorBuffer(const TfLiteTensor* tensor,
                                                   int num_strings)
    : BaseTfLiteTensorBuffer(
          num_strings != 0
              ? tensorflow::TypedAllocator::Allocate<tensorflow::tstring>(
                    tensorflow::cpu_allocator(), num_strings,
                    tensorflow::AllocationAttributes())
              : nullptr),
      num_strings_(num_strings) {
  LogAllocation();

  if (data()) {
    tensorflow::tstring* p = static_cast<tensorflow::tstring*>(data());
    for (size_t i = 0; i < num_strings_; ++p, ++i) {
      auto ref = GetString(tensor, i);
      p->assign(ref.str, ref.len);
    }
  }
}

tensorflow::Status SetTfTensorFromTfLite(const TfLiteTensor* tensor,
                                         tensorflow::Tensor* tf_tensor) {
  if (resource::IsBuiltinResource(tensor)) {
    // If this is native TF Lite resource variable, then we create a TF resource
    // tensor where the tensor handle encodes the identifier of the TF Lite
    // resource.
    // This approach assumes that there is only a single model being invoked
    // via the Interpreter instance, so that the resource IDs won't have any
    // collisions. If we plan to support concurrent execution in the future, we
    // should make sure the resource ID being encoded is unique between
    // different executions.
    tensorflow::Tensor t(tensorflow::DT_RESOURCE, tensorflow::TensorShape({}));
    tensorflow::ResourceHandle handle;
    handle.set_name(TfLiteResourceIdentifier(tensor));
    t.flat<tensorflow::ResourceHandle>()(0) = handle;
    *tf_tensor = t;
    return ::tensorflow::OkStatus();
  } else if (IsResourceOrVariant(tensor)) {
    // TODO(b/179094265): This is an experimental implementation, subject to
    // change. This can be re-implemented with life cycle management mechanism
    // like reference counting.
    // In a different subgraph, it can load the TensorFlow tensor pointer of the
    // given TensorFlow Lite tensor, which is stored in the `data` field. The
    // memory management cycle of the shared TensorFlow's tensor will be managed
    // by the buffer maps since the loaded tensors always will be kept in the
    // buffer map.
    //
    // The life cycle of the pointer will be managed by the reference counting
    // in the TensorFlow world and the pointer will be freed when all the buffer
    // maps, who own it, are gone.
    const tensorflow::Tensor** tf_tensor_ptr =
        reinterpret_cast<const tensorflow::Tensor**>(tensor->data.raw);
    *tf_tensor = **tf_tensor_ptr;
    return ::tensorflow::OkStatus();
  }

  tensorflow::TensorShape shape;
  int num_dims = tensor->dims->size;
  for (int i = 0; i < num_dims; ++i) {
    shape.AddDim(tensor->dims->data[i]);
  }
  // TODO(b/152916533): We assume this is a new tensor and allocate a new buffer
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

  *tf_tensor = std::move(t);
  return ::tensorflow::OkStatus();
}

}  // namespace flex
}  // namespace tflite

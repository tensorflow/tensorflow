/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_TF_TENSOR_INTERNAL_H_
#define TENSORFLOW_C_TF_TENSOR_INTERNAL_H_

#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_interface.h"
#include "tensorflow/core/framework/tensor_shape.h"

// Internal structures used by the C API. These are likely to change and should
// not be depended on.

// This struct forms part of the C API's public interface. It must strictly be
// passed to or returned from C functions *by pointer*. Otherwise, changes to
// its internal structure will break the C API's binary interface.
typedef struct TF_Tensor {
  tensorflow::TensorInterface tensor;
} TF_Tensor;

class TF_ManagedBuffer : public tensorflow::TensorBuffer {
 public:
  TF_ManagedBuffer(void* data, size_t len,
                   void (*deallocator)(void* data, size_t len, void* arg),
                   void* deallocator_arg)
      : TensorBuffer(data),
        len_(len),
        deallocator_(deallocator),
        deallocator_arg_(deallocator_arg) {}

  ~TF_ManagedBuffer() override {
    (*deallocator_)(data(), len_, deallocator_arg_);
  }

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
  const size_t len_;
  void (*const deallocator_)(void* data, size_t len, void* arg);
  void* const deallocator_arg_;
};

namespace tensorflow {

class TensorCApi {
 public:
  static TensorBuffer* Buffer(const Tensor& tensor) { return tensor.buf_; }
  static Tensor MakeTensor(TF_DataType type, const TensorShape& shape,
                           TensorBuffer* buf) {
    return Tensor(static_cast<DataType>(type), shape, buf);
  }
};

// Allocates tensor data buffer using specified allocator.
// `operation` is a name for this operation.
void* allocate_tensor(const char* operation, size_t len, Allocator* allocator);

// Deallocates tensor data buffer.
// Defaults to deallocating using CPU allocator. You can pass pointer to
// a different Allocator as `arg`.
void deallocate_buffer(void* data, size_t len, void* arg);
}  // namespace tensorflow

#endif  // TENSORFLOW_C_TF_TENSOR_INTERNAL_H_

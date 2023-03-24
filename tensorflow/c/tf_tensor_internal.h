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

#include <memory>

#include "tensorflow/c/tensor_interface.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/casts.h"

// Internal structures used by the C API. These are likely to change and should
// not be depended on.

// This struct forms part of the C API's public interface. It must strictly be
// passed to or returned from C functions *by pointer*. Otherwise, changes to
// its internal structure will break the C API's binary interface.
typedef struct TF_Tensor {
  tensorflow::AbstractTensorInterface* tensor;
} TF_Tensor;

class TF_ManagedBuffer : public tensorflow::TensorBuffer {
 public:
  TF_ManagedBuffer(void* data, size_t len,
                   void (*deallocator)(void* data, size_t len, void* arg),
                   void* deallocator_arg, bool owns_memory)
      : TensorBuffer(data),
        len_(len),
        deallocator_(deallocator),
        deallocator_arg_(deallocator_arg),
        owns_memory_(owns_memory) {}

  ~TF_ManagedBuffer() override {
    (*deallocator_)(data(), len_, deallocator_arg_);
  }

  size_t size() const override { return len_; }
  TensorBuffer* root_buffer() override { return this; }
  void FillAllocationDescription(
      tensorflow::AllocationDescription* proto) const override {
    int64_t rb = size();
    proto->set_requested_bytes(rb);
    proto->set_allocator_name(tensorflow::cpu_allocator()->Name());
  }

  bool OwnsMemory() const override { return owns_memory_; }

 private:
  const size_t len_;
  void (*const deallocator_)(void* data, size_t len, void* arg);
  void* const deallocator_arg_;
  bool owns_memory_;
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

class TensorInterface : public AbstractTensorInterface {
 public:
  TensorInterface() {}
  explicit TensorInterface(tensorflow::Tensor t) : tensor_(std::move(t)) {}
  ~TensorInterface() override {}

  void Release() override;

  DataType Type() const override;
  int NumDims() const override;
  int64_t Dim(int dim_index) const override;
  int64_t NumElements() const override;
  size_t ByteSize() const override;
  void* Data() const override;
  bool IsAligned() const override;
  bool CanMove() const override;
  std::string SummarizeValue() const override;

  void SetShape(const int64_t* dims, int num_dims);
  Status ToTensor(tensorflow::Tensor* dst) const;
  Status BitcastFrom(const TensorInterface& from, DataType type,
                     const int64_t* new_dims, int num_new_dims);
  Status FromProto(const tensorflow::TensorProto& from);

  tensorflow::Tensor& Tensor() { return tensor_; }

 private:
  tensorflow::Tensor tensor_;
};

inline Tensor& TensorFromInterface(AbstractTensorInterface* tensor) {
  return down_cast<TensorInterface*>(tensor)->Tensor();
}

Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst);

AbstractTensorInterface* TensorInterfaceFromTensor(const Tensor& src,
                                                   Status* status);

TF_Tensor* TF_TensorFromTensor(const Tensor& src, Status* status);

TF_Tensor* TF_TensorFromTensorShallow(const Tensor& src, Status* status);

namespace internal {

struct TFTensorDeleter {
  void operator()(TF_Tensor* tf_tensor) const { TF_DeleteTensor(tf_tensor); }
};

}  // namespace internal

// Struct that wraps TF_Tensor to delete once out of scope.
using TF_TensorPtr = std::unique_ptr<TF_Tensor, internal::TFTensorDeleter>;

}  // namespace tensorflow

#endif  // TENSORFLOW_C_TF_TENSOR_INTERNAL_H_

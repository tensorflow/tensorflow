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

#include "tensorflow/c/tf_tensor.h"

#include <memory>
#include <vector>

#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/platform/casts.h"

using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::TensorBuffer;
using tensorflow::errors::FailedPrecondition;
using tensorflow::errors::InvalidArgument;

namespace tensorflow {
void* allocate_tensor(const char* operation, size_t len, Allocator* allocator) {
  void* data = allocator->AllocateRaw(EIGEN_MAX_ALIGN_BYTES, len);
  if (LogMemory::IsEnabled() && data != nullptr) {
    LogMemory::RecordRawAllocation(
        operation, LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID, len, data,
        allocator);
  }
  return data;
}

void* allocate_tensor(const char* operation, size_t len) {
  return allocate_tensor(operation, len, cpu_allocator());
}

void deallocate_buffer(void* data, size_t len, void* arg) {
  Allocator* allocator = nullptr;
  if (arg == nullptr) {
    allocator = cpu_allocator();
  } else {
    allocator = reinterpret_cast<Allocator*>(arg);
  }
  if (LogMemory::IsEnabled() && data != nullptr) {
    LogMemory::RecordRawDeallocation(
        "TensorFlow C Api", LogMemory::EXTERNAL_TENSOR_ALLOCATION_STEP_ID, data,
        allocator, false);
  }
  allocator->DeallocateRaw(data);
}
}  // namespace tensorflow

namespace {
TF_Tensor* CreateTensor(TF_ManagedBuffer* buf, TF_DataType dtype,
                        const int64_t* dims, int num_dims, size_t len) {
  std::vector<int64_t> dimvec(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    dimvec[i] = static_cast<int64_t>(dims[i]);
  }

  // TODO(gjn): Make the choice of interface a compile-time configuration.
  tensorflow::TensorInterface ret(
      Tensor(static_cast<tensorflow::DataType>(dtype),
             tensorflow::TensorShape(dimvec), buf));
  buf->Unref();
  size_t elem_size = TF_DataTypeSize(dtype);
  if (elem_size > 0 && len < (elem_size * ret.NumElements())) {
    return nullptr;
  }
  return new TF_Tensor{new tensorflow::TensorInterface(ret)};
}
}  // namespace

TF_Tensor* TF_AllocateTensor(TF_DataType dtype, const int64_t* dims,
                             int num_dims, size_t len) {
  void* data = tensorflow::allocate_tensor("TF_AllocateTensor", len,
                                           tensorflow::cpu_allocator());
  TF_ManagedBuffer* buf =
      new TF_ManagedBuffer(data, len, tensorflow::deallocate_buffer,
                           tensorflow::cpu_allocator(), /*owns_memory=*/true);
  return CreateTensor(buf, dtype, dims, num_dims, len);
}

TF_Tensor* TF_NewTensor(TF_DataType dtype, const int64_t* dims, int num_dims,
                        void* data, size_t len,
                        void (*deallocator)(void* data, size_t len, void* arg),
                        void* deallocator_arg) {
  TF_ManagedBuffer* buf = nullptr;
  if (dtype != TF_STRING && dtype != TF_RESOURCE &&
      tensorflow::DataTypeCanUseMemcpy(
          static_cast<tensorflow::DataType>(dtype)) &&
      reinterpret_cast<intptr_t>(data) % std::max(1, EIGEN_MAX_ALIGN_BYTES) !=
          0) {
    // TF_STRING and TF_RESOURCE tensors have a different representation in
    // TF_Tensor than they do in tensorflow::Tensor. So a copy here is a waste
    // (any alignment requirements will be taken care of by TF_TensorToTensor
    // and TF_TensorFromTensor).
    //
    // Other types have the same representation, so copy only if it is safe to
    // do so.
    buf = new TF_ManagedBuffer(tensorflow::allocate_tensor("TF_NewTensor", len),
                               len, tensorflow::deallocate_buffer, nullptr,
                               /*owns_memory=*/true);
    std::memcpy(buf->data(), data, len);
    // Free the original buffer.
    deallocator(data, len, deallocator_arg);
  } else {
    buf = new TF_ManagedBuffer(data, len, deallocator, deallocator_arg,
                               /*owns_memory=*/false);
  }

  return CreateTensor(buf, dtype, dims, num_dims, len);
}

TF_Tensor* TF_TensorMaybeMove(TF_Tensor* t) {
  return t->tensor->CanMove() ? t : nullptr;
}

void TF_DeleteTensor(TF_Tensor* t) {
  if (t == nullptr) {
    return;
  }

  if (t->tensor) {
    t->tensor->Release();
  }

  delete t;
}

TF_DataType TF_TensorType(const TF_Tensor* t) {
  return static_cast<TF_DataType>(t->tensor->Type());
}

void TF_SetShape(TF_Tensor* t, const int64_t* dims, int num_dims) {
  tensorflow::down_cast<tensorflow::TensorInterface*>(t->tensor)->SetShape(
      dims, num_dims);
}

int TF_NumDims(const TF_Tensor* t) { return t->tensor->NumDims(); }

int64_t TF_Dim(const TF_Tensor* t, int dim_index) {
  return t->tensor->Dim(dim_index);
}

size_t TF_TensorByteSize(const TF_Tensor* t) { return t->tensor->ByteSize(); }

void* TF_TensorData(const TF_Tensor* t) { return t->tensor->Data(); }

int64_t TF_TensorElementCount(const TF_Tensor* t) {
  int64_t result = 1;
  int rank = TF_NumDims(t);
  for (int dim = 0; dim < rank; ++dim) {
    result *= TF_Dim(t, dim);
  }
  return result;
}

void TF_TensorBitcastFrom(const TF_Tensor* from, TF_DataType type,
                          TF_Tensor* to, const int64_t* new_dims,
                          int num_new_dims, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
  Status cc_status(
      tensorflow::down_cast<tensorflow::TensorInterface*>(to->tensor)
          ->BitcastFrom(
              *tensorflow::down_cast<const tensorflow::TensorInterface*>(
                  from->tensor),
              static_cast<tensorflow::DataType>(type), new_dims, num_new_dims));
  Set_TF_Status_from_Status(status, cc_status);
}

namespace tensorflow {

void TensorInterface::Release() {
  if (Type() == DT_STRING && NumElements() > 0) {
    TF_TString* data = static_cast<TF_TString*>(Data());
    if (CanMove() && data != nullptr) {
      for (int64_t i = 0; i < NumElements(); ++i) {
        TF_TString_Dealloc(&data[i]);
      }
    }
  }
  delete this;
}

bool TensorInterface::CanMove() const {
  // It is safe to move the Tensor if and only if we own the unique reference to
  // it. In that case, we might as well not delete and reallocate, but a future
  // implementation might need to do so.
  TensorBuffer* buf = tensorflow::TensorCApi::Buffer(tensor_);
  if (buf->RefCountIsOne() && buf->root_buffer()->RefCountIsOne() &&
      buf->OwnsMemory()) {
    return true;
  }
  return false;
}

std::string TensorInterface::SummarizeValue() const {
  return tensor_.SummarizeValue(/*max_entries=*/3, /*print_v2=*/true);
}

DataType TensorInterface::Type() const { return tensor_.dtype(); }

int TensorInterface::NumDims() const { return tensor_.dims(); }

int64_t TensorInterface::Dim(int dim_index) const {
  return static_cast<int64_t>(tensor_.dim_size(dim_index));
}

int64_t TensorInterface::NumElements() const {
  return static_cast<int64_t>(tensor_.NumElements());
}

size_t TensorInterface::ByteSize() const {
  return tensorflow::TensorCApi::Buffer(tensor_)->size();
}

void* TensorInterface::Data() const {
  return tensorflow::TensorCApi::Buffer(tensor_)->data();
}

void TensorInterface::SetShape(const int64_t* dims, int num_dims) {
  tensorflow::TensorShape s;
  for (int i = 0; i < num_dims; ++i) {
    s.AddDim(dims[i]);
  }
  tensor_.set_shape(s);
}

Status TensorInterface::BitcastFrom(const TensorInterface& from, DataType type,
                                    const int64_t* new_dims, int num_new_dims) {
  tensorflow::TensorShape s;
  for (int i = 0; i < num_new_dims; ++i) {
    s.AddDim(new_dims[i]);
  }
  return tensor_.BitcastFrom(from.tensor_, type, s);
}

Status TensorInterface::FromProto(const tensorflow::TensorProto& from) {
  bool success = tensor_.FromProto(from);
  if (success) return OkStatus();
  return errors::InvalidArgument("Unparseable tensor proto");
}

}  // namespace tensorflow

// --------------------------------------------------------------------------

static void DeleteArray(void* data, size_t size, void* arg) {
  DCHECK_EQ(data, arg);
  delete[] reinterpret_cast<char*>(arg);
}

// Create an empty tensor of type 'dtype'. 'shape' can be arbitrary, but has to
// result in a zero-sized tensor.
static TF_Tensor* EmptyTensor(TF_DataType dtype,
                              const tensorflow::TensorShape& shape) {
  static char empty;
  int64_t nelems = 1;
  std::vector<int64_t> dims;
  auto shape_dims = shape.dims();
  dims.reserve(shape_dims);
  for (int i = 0; i < shape_dims; ++i) {
    dims.push_back(shape.dim_size(i));
    nelems *= shape.dim_size(i);
  }
  CHECK_EQ(nelems, 0);
  return TF_NewTensor(
      dtype, reinterpret_cast<const int64_t*>(dims.data()), shape.dims(),
      reinterpret_cast<void*>(&empty), 0, [](void*, size_t, void*) {}, nullptr);
}

namespace tensorflow {

// Non-static for testing.
TF_Tensor* TF_TensorFromTensor(const tensorflow::Tensor& src, Status* status) {
  *status = OkStatus();
  if (!src.IsInitialized()) {
    *status = FailedPrecondition(
        "attempt to use a tensor with an uninitialized value");
    return nullptr;
  }
  if (src.NumElements() == 0) {
    return EmptyTensor(static_cast<TF_DataType>(src.dtype()), src.shape());
  }

  Tensor tensor;
  if (!tensor.CopyFrom(src, src.shape())) {
    return nullptr;
  }
  return new TF_Tensor{new tensorflow::TensorInterface(std::move(tensor))};
}

TF_Tensor* TF_TensorFromTensorShallow(const tensorflow::Tensor& src,
                                      Status* status) {
  *status = OkStatus();
  if (!src.IsInitialized()) {
    *status = FailedPrecondition(
        "attempt to use a tensor with an uninitialized value");
    return nullptr;
  }
  if (src.NumElements() == 0) {
    return EmptyTensor(static_cast<TF_DataType>(src.dtype()), src.shape());
  }
  return new TF_Tensor{new tensorflow::TensorInterface(src)};
}

Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst) {
  return tensorflow::down_cast<const tensorflow::TensorInterface*>(src->tensor)
      ->ToTensor(dst);
}

Status TensorInterface::ToTensor(tensorflow::Tensor* dst) const {
  *dst = tensor_;
  return OkStatus();
}

bool TensorInterface::IsAligned() const { return tensor_.IsAligned(); }

}  // namespace tensorflow

bool TF_TensorIsAligned(const TF_Tensor* t) { return t->tensor->IsAligned(); }

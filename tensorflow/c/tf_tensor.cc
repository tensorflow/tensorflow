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

#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/coding.h"

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


TF_Tensor* TF_AllocateTensor(TF_DataType dtype, const int64_t* dims,
                             int num_dims, size_t len) {
  void* data = tensorflow::allocate_tensor("TF_AllocateTensor", len,
                                           tensorflow::cpu_allocator());
  return TF_NewTensor(dtype, dims, num_dims, data, len,
                      tensorflow::deallocate_buffer,
                      tensorflow::cpu_allocator());
}

TF_Tensor* TF_NewTensor(TF_DataType dtype, const int64_t* dims, int num_dims,
                        void* data, size_t len,
                        void (*deallocator)(void* data, size_t len, void* arg),
                        void* deallocator_arg) {
  std::vector<tensorflow::int64> dimvec(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    dimvec[i] = static_cast<tensorflow::int64>(dims[i]);
  }

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
                               len, tensorflow::deallocate_buffer, nullptr);
    std::memcpy(buf->data(), data, len);
    // Free the original buffer.
    deallocator(data, len, deallocator_arg);
  } else {
    buf = new TF_ManagedBuffer(data, len, deallocator, deallocator_arg);
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
  return new TF_Tensor{std::make_unique<tensorflow::TensorInterface>(ret)};
}

TF_Tensor* TF_TensorMaybeMove(TF_Tensor* t) {
  return t->tensor->CanMove() ? t : nullptr;
}

void TF_DeleteTensor(TF_Tensor* t) { delete t; }

TF_DataType TF_TensorType(const TF_Tensor* t) { return t->tensor->Type(); }

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
      static_cast<tensorflow::TensorInterface*>(to->tensor.get())
          ->BitcastFrom(*static_cast<const tensorflow::TensorInterface*>(
                            from->tensor.get()),
                        type, new_dims, num_new_dims));
  Set_TF_Status_from_Status(status, cc_status);
}

namespace tensorflow {

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

TF_DataType TensorInterface::Type() const {
  return static_cast<TF_DataType>(tensor_.dtype());
}

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

Status TensorInterface::BitcastFrom(const TensorInterface& from,
                                    TF_DataType type, const int64_t* new_dims,
                                    int num_new_dims) {
  tensorflow::TensorShape s;
  for (int i = 0; i < num_new_dims; ++i) {
    s.AddDim(new_dims[i]);
  }
  return tensor_.BitcastFrom(from.tensor_,
                             static_cast<tensorflow::DataType>(type), s);
}

}  // namespace tensorflow

// --------------------------------------------------------------------------
void StringEncode(const char* src, size_t src_len, char* dst) {
  dst = tensorflow::core::EncodeVarint64(dst, src_len);
  memcpy(dst, src, src_len);
}

size_t TF_StringEncode(const char* src, size_t src_len, char* dst,
                       size_t dst_len, TF_Status* status) {
  const size_t sz = TF_StringEncodedSize(src_len);
  if (sz < src_len) {
    Set_TF_Status_from_Status(
        status, InvalidArgument("src string is too large to encode"));
    return 0;
  }
  if (dst_len < sz) {
    Set_TF_Status_from_Status(
        status,
        InvalidArgument("dst_len (", dst_len, ") too small to encode a ",
                        src_len, "-byte string"));
    return 0;
  }
  StringEncode(src, src_len, dst);
  return sz;
}

static Status TF_StringDecode_Impl(const char* src, size_t src_len,
                                   const char** dst, size_t* dst_len) {
  tensorflow::uint64 len64 = 0;
  const char* p = tensorflow::core::GetVarint64Ptr(src, src + src_len, &len64);
  if (p == nullptr) {
    return InvalidArgument("invalid string encoding or truncated src buffer");
  }
  if (len64 > std::numeric_limits<size_t>::max()) {
    return InvalidArgument("encoded string is ", len64,
                           "-bytes, which is too large for this architecture");
  }
  *dst = p;
  *dst_len = static_cast<size_t>(len64);
  return Status::OK();
}

size_t TF_StringDecode(const char* src, size_t src_len, const char** dst,
                       size_t* dst_len, TF_Status* status) {
  Set_TF_Status_from_Status(status,
                            TF_StringDecode_Impl(src, src_len, dst, dst_len));
  if (TF_GetCode(status) != TF_OK) return 0;
  return static_cast<size_t>(*dst - src) + *dst_len;
}

size_t TF_StringEncodedSize(size_t len) {
  return static_cast<size_t>(tensorflow::core::VarintLength(len)) + len;
}

static void DeleteArray(void* data, size_t size, void* arg) {
  DCHECK_EQ(data, arg);
  delete[] reinterpret_cast<char*>(arg);
}

// Create an empty tensor of type 'dtype'. 'shape' can be arbitrary, but has to
// result in a zero-sized tensor.
static TF_Tensor* EmptyTensor(TF_DataType dtype,
                              const tensorflow::TensorShape& shape) {
  static char empty;
  tensorflow::int64 nelems = 1;
  std::vector<tensorflow::int64> dims;
  for (int i = 0; i < shape.dims(); ++i) {
    dims.push_back(shape.dim_size(i));
    nelems *= shape.dim_size(i);
  }
  CHECK_EQ(nelems, 0);
  static_assert(sizeof(int64_t) == sizeof(tensorflow::int64),
                "64-bit int types should match in size");
  return TF_NewTensor(
      dtype, reinterpret_cast<const int64_t*>(dims.data()), shape.dims(),
      reinterpret_cast<void*>(&empty), 0, [](void*, size_t, void*) {}, nullptr);
}

namespace tensorflow {

// Non-static for testing.
TF_Tensor* TF_TensorFromTensor(const tensorflow::Tensor& src, Status* status) {
  *status = tensorflow::Status::OK();
  if (!src.IsInitialized()) {
    *status = FailedPrecondition(
        "attempt to use a tensor with an uninitialized value");
    return nullptr;
  }
  if (src.NumElements() == 0) {
    return EmptyTensor(static_cast<TF_DataType>(src.dtype()), src.shape());
  }
  if (src.dtype() == tensorflow::DT_RESOURCE) {
    if (src.shape().dims() != 0) {
      *status = InvalidArgument(
          "Unexpected non-scalar DT_RESOURCE tensor seen (shape: ",
          src.shape().DebugString(),
          "). Please file a bug at "
          "https://github.com/tensorflow/tensorflow/issues/new, "
          "ideally with a "
          "short code snippet that reproduces this error.");
      return nullptr;
    }
    const string str =
        src.scalar<tensorflow::ResourceHandle>()().SerializeAsString();
    TF_Tensor* t = TF_AllocateTensor(TF_RESOURCE, {}, 0, str.size());
    std::memcpy(TF_TensorData(t), str.c_str(), str.size());
    return t;
  }
  if (src.dtype() != tensorflow::DT_STRING) {
    Tensor tensor;
    if (!tensor.CopyFrom(src, src.shape())) {
      return nullptr;
    }
    return new TF_Tensor{std::make_unique<tensorflow::TensorInterface>(tensor)};
  }
  // DT_STRING tensors require a copying since TF_Tensor.buffer expects a flatly
  // encoded sequence of strings.

  // Compute bytes needed for encoding.
  size_t size = 0;
  const auto& srcarray = src.flat<tstring>();
  for (int i = 0; i < srcarray.size(); ++i) {
    const string& s = srcarray(i);
    // uint64 starting_offset, TF_StringEncode-d string.
    size += sizeof(tensorflow::uint64) + TF_StringEncodedSize(s.size());
  }

  // Encode all strings.
  char* base = new char[size];
  char* data_start = base + sizeof(tensorflow::uint64) * srcarray.size();
  char* dst = data_start;  // Where next string is encoded.
  size_t dst_len = size - static_cast<size_t>(data_start - base);
  tensorflow::uint64* offsets = reinterpret_cast<tensorflow::uint64*>(base);
  for (int i = 0; i < srcarray.size(); ++i) {
    *offsets = (dst - data_start);
    offsets++;
    const string& s = srcarray(i);
    const size_t consumed = TF_StringEncodedSize(s.size());
    StringEncode(s.data(), s.size(), dst);
    dst += consumed;
    dst_len -= consumed;
  }
  if (dst != base + size) {
    *status = InvalidArgument(
        "invalid string tensor encoding (decoded ", (dst - base),
        " bytes, but the tensor is encoded in ", size, " bytes");
    delete[] base;
    return nullptr;
  }

  auto dims = src.shape().dim_sizes();
  std::vector<tensorflow::int64> dimvec(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    dimvec[i] = dims[i];
  }
  static_assert(sizeof(int64_t) == sizeof(tensorflow::int64),
                "64-bit int types should match in size");
  return TF_NewTensor(TF_STRING,
                      reinterpret_cast<const int64_t*>(dimvec.data()),
                      dimvec.size(), base, size, DeleteArray, base);
}

Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst) {
  return static_cast<const tensorflow::TensorInterface*>(src->tensor.get())
      ->ToTensor(dst);
}

Status TensorInterface::ToTensor(Tensor* dst) const {
  if (tensor_.dtype() == DT_RESOURCE) {
    if (tensor_.dims() != 0) {
      return InvalidArgument(
          "Malformed TF_RESOURCE tensor: expected a scalar, got a tensor with "
          "shape ",
          tensor_.shape().DebugString());
    }
    *dst = Tensor(tensorflow::DT_RESOURCE, tensor_.shape());
    if (!dst->scalar<tensorflow::ResourceHandle>()().ParseFromString(
            string(static_cast<const char*>(Data()), ByteSize()))) {
      return InvalidArgument(
          "Malformed TF_RESOURCE tensor: unable to parse resource handle");
    }
    return Status::OK();
  }
  if (tensor_.dtype() != DT_STRING) {
    *dst = tensor_;
    return Status::OK();
  }
  // TF_STRING tensors require copying since Tensor class expects a sequence of
  // string objects.
  const tensorflow::int64 num_elements = tensor_.NumElements();
  const char* input = reinterpret_cast<const char*>(Data());
  const size_t src_size = ByteSize();
  if (static_cast<tensorflow::int64>(src_size / sizeof(tensorflow::uint64)) <
      num_elements) {
    return InvalidArgument(
        "Malformed TF_STRING tensor; too short to hold number of elements");
  }
  const char* data_start = input + sizeof(tensorflow::uint64) * num_elements;
  const char* limit = input + src_size;

  *dst = Tensor(tensor_.dtype(), tensor_.shape());
  auto dstarray = dst->flat<tstring>();
  for (tensorflow::int64 i = 0; i < num_elements; ++i) {
    tensorflow::uint64 offset =
        reinterpret_cast<const tensorflow::uint64*>(input)[i];
    if (static_cast<ptrdiff_t>(offset) >= (limit - data_start)) {
      return InvalidArgument("Malformed TF_STRING tensor; element ", i,
                             " out of range");
    }
    size_t len;
    const char* p;
    const char* srcp = data_start + offset;
    Status status = TF_StringDecode_Impl(srcp, limit - srcp, &p, &len);
    if (!status.ok()) return status;
    dstarray(i).assign(p, len);
  }
  return Status::OK();
}

bool TensorInterface::IsAligned() const { return tensor_.IsAligned(); }

}  // namespace tensorflow

bool TF_TensorIsAligned(const TF_Tensor* t) { return t->tensor->IsAligned(); }

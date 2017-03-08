/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Implementation notes:
//
// Tensor.cc uses a few templated classes and structs to facilitate
// implementation of the Tensor class.
//
// * Buffer<T>: provides the implementation for a typed array T[n].
//   The array is allocated by the given allocator. It runs T's
//   default constructors and destructors when T is not a simple type
//   (e.g., string.), and skips them otherwise.
//
// * Helper<T>: provides various routines given type T.  The routines
//   includes running the constructor and destructor of T[], encoding
//   an decoding T[] into/from a Cord, etc.

#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

// An un-templated base class for Buffer.
class BufferBase : public TensorBuffer {
 public:
  BufferBase(Allocator* alloc) : alloc_(alloc) {}

  TensorBuffer* root_buffer() override { return this; }
  void FillAllocationDescription(AllocationDescription* proto) const override {
    void* data_ptr = data();
    int64 rb = size();
    proto->set_requested_bytes(rb);
    proto->set_allocator_name(alloc_->Name());
    proto->set_ptr(reinterpret_cast<uintptr_t>(data_ptr));
    if (alloc_->TracksAllocationSizes()) {
      int64 ab = alloc_->AllocatedSize(data_ptr);
      proto->set_allocated_bytes(ab);
      int64 id = alloc_->AllocationId(data_ptr);
      if (id > 0) {
        proto->set_allocation_id(id);
      }
      if (RefCountIsOne()) {
        proto->set_has_single_reference(true);
      }
    }
  }

 protected:
  void RecordDeallocation() {
    LogMemory::RecordTensorDeallocation(alloc_->AllocationId(data()),
                                        alloc_->Name());
  }

  Allocator* const alloc_;
};

// Typed ref-counted buffer: T[n].
template <typename T>
class Buffer : public BufferBase {
 public:
  Buffer(Allocator* a, int64 n);
  Buffer(Allocator* a, int64 n, const AllocationAttributes& allocation_attr);

  void* data() const override { return data_; }
  size_t size() const override { return sizeof(T) * elem_; }

 private:
  T* data_;
  int64 elem_;

  ~Buffer() override;

  TF_DISALLOW_COPY_AND_ASSIGN(Buffer);
};

void LogUnexpectedSize(int64 actual, int64 expected) {
  LOG(ERROR) << "Input size was " << actual << " and expected " << expected;
}

// A set of helper functions depending on T.
template <typename T>
struct Helper {
  // By default, we assume T is a simple type (float, int32, etc.)
  static_assert(Allocator::is_simple<T>::value, "T is not a simple type.");
  typedef protobuf::RepeatedField<T> RepeatedFieldType;

  // Encoder of simple type T to a string.  We do a copy.
  template <typename Destination>
  static void Encode(TensorBuffer* in, int64 n, Destination* out) {
    DCHECK_EQ(in->size(), sizeof(T) * n);
    port::AssignRefCounted(StringPiece(in->base<const char>(), in->size()), in,
                           out);
  }

  // Decoder of simple type T. Copy the bytes from "in" into the
  // tensor buffer.
  template <typename Source>
  static TensorBuffer* Decode(Allocator* a, const Source& in, int64 n) {
    if (in.size() != sizeof(T) * n) {
      LogUnexpectedSize(in.size(), sizeof(T) * n);
      return nullptr;
    }
    Buffer<T>* buf = new Buffer<T>(a, n);
    char* data = buf->template base<char>();
    if (data == nullptr) {
      buf->Unref();
      return nullptr;
    }
    port::CopyToArray(in, data);
    return buf;
  }

  // Memory usage.
  static int64 TotalBytes(TensorBuffer* in, int64 n) {
    DCHECK_EQ(in->size(), sizeof(T) * n);
    return in->size();
  }
};

// Helper specialization for string (the only non-simple type we
// support).
template <>
struct Helper<string> {
  // Proto message uses RepeatedFieldType to hold repeated T.
  typedef protobuf::RepeatedPtrField<string> RepeatedFieldType;

  // Encodes "n" elements of type string stored in "in" into Cord
  // "out", which is usually the TensorProto::tensor_content.
  template <typename Destination>
  static void Encode(TensorBuffer* in, int64 n, Destination* out) {
    port::EncodeStringList(in->base<const string>(), n, out);
  }

  // Decodes "n" elements of type string from "in" and constructs a
  // buffer out of it. Returns nullptr if the decoding fails. "in" is
  // usually the TensorProto::tensor_content.
  template <typename Source>
  static TensorBuffer* Decode(Allocator* a, const Source& in, int64 n) {
    Buffer<string>* buf = new Buffer<string>(a, n);
    string* strings = buf->template base<string>();
    if (strings == nullptr) {
      buf->Unref();
      return nullptr;
    }
    if (port::DecodeStringList(in, strings, n)) {
      return buf;
    } else {
      buf->Unref();
      return nullptr;
    }
  }

  // Returns the estimated memory usage of "n" elements of type T
  // stored in buffer "in".
  static int64 TotalBytes(TensorBuffer* in, int n) {
    int64 tot = in->size();
    DCHECK_EQ(tot, sizeof(string) * n);
    const string* p = in->base<const string>();
    for (int i = 0; i < n; ++i, ++p) tot += p->size();
    return tot;
  }
};

template <typename T>
struct ProtoHelper {};

// For a C++ type "T" (float, double, int32, etc.), the repeated field
// "N"_val (float_val, int_val, label_val, etc.) of type "F" (float,
// int32, string, etc) in the TensorProto is used for serializing the
// tensor of type "T".
#define PROTO_TRAITS(T, F, N)                                          \
  template <>                                                          \
  struct ProtoHelper<T> {                                              \
    typedef Helper<F>::RepeatedFieldType FieldType;                    \
    static FieldType::const_iterator Begin(const TensorProto& proto) { \
      return proto.N##_val().begin();                                  \
    }                                                                  \
    static size_t NumElements(const TensorProto& proto) {              \
      return proto.N##_val().size();                                   \
    }                                                                  \
    static void Fill(const T* data, size_t n, TensorProto* proto) {    \
      typename ProtoHelper<T>::FieldType copy(data, data + n);         \
      proto->mutable_##N##_val()->Swap(&copy);                         \
    }                                                                  \
  };
PROTO_TRAITS(float, float, float);
PROTO_TRAITS(double, double, double);
PROTO_TRAITS(int32, int32, int);
PROTO_TRAITS(uint8, int32, int);
PROTO_TRAITS(uint16, int32, int);
PROTO_TRAITS(int16, int32, int);
PROTO_TRAITS(int8, int32, int);
PROTO_TRAITS(int64, int64, int64);
PROTO_TRAITS(bool, bool, bool);
PROTO_TRAITS(string, string, string);
PROTO_TRAITS(qint8, int32, int);
PROTO_TRAITS(quint8, int32, int);
PROTO_TRAITS(qint16, int32, int);
PROTO_TRAITS(quint16, int32, int);
#undef PROTO_TRAITS

template <>
struct ProtoHelper<complex64> {
  typedef Helper<float>::RepeatedFieldType FieldType;
  static const complex64* Begin(const TensorProto& proto) {
    return reinterpret_cast<const complex64*>(proto.scomplex_val().data());
  }
  static size_t NumElements(const TensorProto& proto) {
    return proto.scomplex_val().size() / 2;
  }
  static void Fill(const complex64* data, size_t n, TensorProto* proto) {
    const float* p = reinterpret_cast<const float*>(data);
    FieldType copy(p, p + n * 2);
    proto->mutable_scomplex_val()->Swap(&copy);
  }
};

template <>
struct ProtoHelper<complex128> {
  typedef Helper<double>::RepeatedFieldType FieldType;
  static const complex128* Begin(const TensorProto& proto) {
    return reinterpret_cast<const complex128*>(proto.dcomplex_val().data());
  }
  static size_t NumElements(const TensorProto& proto) {
    return proto.dcomplex_val().size() / 2;
  }
  static void Fill(const complex128* data, size_t n, TensorProto* proto) {
    const double* p = reinterpret_cast<const double*>(data);
    FieldType copy(p, p + n * 2);
    proto->mutable_dcomplex_val()->Swap(&copy);
  }
};

template <>
struct ProtoHelper<qint32> {
  typedef Helper<int32>::RepeatedFieldType FieldType;
  static const qint32* Begin(const TensorProto& proto) {
    return reinterpret_cast<const qint32*>(proto.int_val().data());
  }
  static size_t NumElements(const TensorProto& proto) {
    return proto.int_val().size();
  }
  static void Fill(const qint32* data, size_t n, TensorProto* proto) {
    const int32* p = reinterpret_cast<const int32*>(data);
    FieldType copy(p, p + n);
    proto->mutable_int_val()->Swap(&copy);
  }
};

template <>
struct ProtoHelper<bfloat16> {
  typedef Helper<float>::RepeatedFieldType FieldType;
  static const bfloat16* Begin(const TensorProto& proto) {
    // TODO: Isn't this wrong, given that int_val is 32 bits long?
    return reinterpret_cast<const bfloat16*>(proto.int_val().data());
  }
  static size_t NumElements(const TensorProto& proto) {
    return proto.int_val().size();
  }
  static void Fill(const bfloat16* data, size_t n, TensorProto* proto) {
    proto->mutable_int_val()->Reserve(n);
    for (size_t i = 0; i < n; ++i) {
      proto->mutable_int_val()->AddAlreadyReserved(data[i].value);
    }
  }
};

template <>
struct ProtoHelper<Eigen::half> {
  static void Fill(const Eigen::half* data, size_t n, TensorProto* proto) {
    proto->mutable_half_val()->Reserve(n);
    for (size_t i = 0; i < n; ++i) {
      proto->mutable_half_val()->AddAlreadyReserved(data[i].x);
    }
  }
};

template <typename T>
Buffer<T>::Buffer(Allocator* a, int64 n)
    : BufferBase(a), data_(a->Allocate<T>(n)), elem_(n) {}

template <typename T>
Buffer<T>::Buffer(Allocator* a, int64 n,
                  const AllocationAttributes& allocation_attr)
    : BufferBase(a), data_(a->Allocate<T>(n, allocation_attr)), elem_(n) {}

template <typename T>
Buffer<T>::~Buffer() {
  if (LogMemory::IsEnabled()) {
    RecordDeallocation();
  }
  if (data_) {
    alloc_->Deallocate<T>(data_, elem_);
  }
}

// Allocates a T[n] buffer. Fills in the buffer with repeated values
// in "in".  If "in" has less values than "n", fills the rest of T[n]
// with the last value. If "in" has no values, fills T[n] with the
// default value for T.
//
// This routine is using the typed fields (float_val, etc.) in the
// tenor proto as opposed to the untyped binary representation
// (tensor_content). This is used when we expect the TensorProto is
// used by a client program which may not know how to encode a tensor
// in the compact binary representation.
template <typename T>
TensorBuffer* FromProtoField(Allocator* a, const TensorProto& in, int64 n) {
  CHECK_GT(n, 0);
  Buffer<T>* buf = new Buffer<T>(a, n);
  T* data = buf->template base<T>();
  if (data == nullptr) {
    buf->Unref();
    return nullptr;
  }

  const int64 in_n = ProtoHelper<T>::NumElements(in);
  auto begin = ProtoHelper<T>::Begin(in);
  if (n <= in_n) {
    std::copy_n(begin, n, data);
  } else if (in_n > 0) {
    std::copy_n(begin, in_n, data);
    const T& last = *(data + in_n - 1);
    std::fill_n(data + in_n, n - in_n, last);
  } else {
    std::fill_n(data, n, T());
  }
  return buf;
}

// fp16 is opaque to the protobuf, so we deserialize these identical to uint16
// but with data stored in half_val instead of int_val (ie., we don't use
// ProtoHelper<uint16>).
template <>
TensorBuffer* FromProtoField<Eigen::half>(Allocator* a, const TensorProto& in,
                                          int64 n) {
  CHECK_GT(n, 0);
  Buffer<Eigen::half>* buf = new Buffer<Eigen::half>(a, n);
  uint16* data = buf->template base<uint16>();
  if (data == nullptr) {
    buf->Unref();
    return nullptr;
  }
  const int64 in_n = in.half_val().size();
  auto begin = in.half_val().begin();
  if (n <= in_n) {
    std::copy_n(begin, n, data);
  } else if (in_n > 0) {
    std::copy_n(begin, in_n, data);
    const uint16 last = *(data + in_n - 1);
    std::fill_n(data + in_n, n - in_n, last);
  } else {
    std::fill_n(data, n, 0);
  }
  return buf;
}

// Copies T[n] stored in the buffer "in" into the repeated field in
// "out" corresponding to type T.
template <typename T>
void ToProtoField(const TensorBuffer& in, int64 n, TensorProto* out) {
  const T* data = in.base<const T>();
  // NOTE: T may not the same as
  // ProtoHelper<T>::FieldType::value_type.  E.g., T==int16,
  // ProtoHelper<T>::FieldType::value_type==int32.  If performance is
  // critical, we can specialize T=float and do memcpy directly.
  ProtoHelper<T>::Fill(data, n, out);
}

void RefIfNonNull(core::RefCounted* buf) {
  if (buf) buf->Ref();
}

void UnrefIfNonNull(core::RefCounted* buf) {
  if (buf) buf->Unref();
}

}  // end namespace

Tensor::Tensor() : Tensor(DT_FLOAT) {}

Tensor::Tensor(DataType type) : shape_({0}), buf_(nullptr) { set_dtype(type); }

Tensor::Tensor(DataType type, const TensorShape& shape, TensorBuffer* buf)
    : shape_(shape), buf_(buf) {
  set_dtype(type);
  RefIfNonNull(buf);
}

bool Tensor::IsInitialized() const {
  return (buf_ != nullptr && buf_->data() != nullptr) ||
         shape_.num_elements() == 0;
}

void Tensor::CheckType(DataType expected_dtype) const {
  CHECK_EQ(dtype(), expected_dtype);
}

void Tensor::CheckTypeAndIsAligned(DataType expected_dtype) const {
  CHECK_EQ(dtype(), expected_dtype);
  CHECK(IsAligned());
}

void Tensor::CheckIsAlignedAndSingleElement() const {
  CHECK(IsAligned());
  CHECK_EQ(1, NumElements()) << "Must have a one element tensor";
}

Tensor::~Tensor() { UnrefIfNonNull(buf_); }

void Tensor::CopyFromInternal(const Tensor& other, const TensorShape& shape) {
  CHECK_EQ(shape.num_elements(), other.NumElements());
  shape_ = shape;
  set_dtype(other.dtype());
  if (buf_ != other.buf_) {
    UnrefIfNonNull(buf_);
    buf_ = other.buf_;
    RefIfNonNull(buf_);
  }
}

void Tensor::UnsafeCopyFromInternal(const Tensor& other,
                                    const TensorShape& shape) {
  int in_size = DataTypeSize(other.dtype());
  int out_size = DataTypeSize(shape.data_type());
  CHECK_NE(in_size, 0);
  CHECK_NE(out_size, 0);
  CHECK_EQ(shape.num_elements() * out_size,
           other.shape().num_elements() * in_size);
  shape_ = shape;
  if (buf_ != other.buf_) {
    UnrefIfNonNull(buf_);
    buf_ = other.buf_;
    RefIfNonNull(buf_);
  }
}

// The macro CASES() expands to a switch statement conditioned on
// TYPE_ENUM. Each case expands the STMTS after a typedef for T.
#define SINGLE_ARG(...) __VA_ARGS__
#define CASE(TYPE, STMTS)             \
  case DataTypeToEnum<TYPE>::value: { \
    typedef TYPE T;                   \
    STMTS;                            \
    break;                            \
  }
#define CASES(TYPE_ENUM, STMTS)                       \
  switch (TYPE_ENUM) {                                \
    CASE(float, SINGLE_ARG(STMTS))                    \
    CASE(double, SINGLE_ARG(STMTS))                   \
    CASE(int32, SINGLE_ARG(STMTS))                    \
    CASE(uint8, SINGLE_ARG(STMTS))                    \
    CASE(uint16, SINGLE_ARG(STMTS))                   \
    CASE(int16, SINGLE_ARG(STMTS))                    \
    CASE(int8, SINGLE_ARG(STMTS))                     \
    CASE(string, SINGLE_ARG(STMTS))                   \
    CASE(complex64, SINGLE_ARG(STMTS))                \
    CASE(complex128, SINGLE_ARG(STMTS))               \
    CASE(int64, SINGLE_ARG(STMTS))                    \
    CASE(bool, SINGLE_ARG(STMTS))                     \
    CASE(qint32, SINGLE_ARG(STMTS))                   \
    CASE(quint8, SINGLE_ARG(STMTS))                   \
    CASE(qint8, SINGLE_ARG(STMTS))                    \
    CASE(quint16, SINGLE_ARG(STMTS))                  \
    CASE(qint16, SINGLE_ARG(STMTS))                   \
    CASE(bfloat16, SINGLE_ARG(STMTS))                 \
    CASE(Eigen::half, SINGLE_ARG(STMTS))              \
    case DT_INVALID:                                  \
      LOG(FATAL) << "Type not set";                   \
      break;                                          \
    default:                                          \
      LOG(FATAL) << "Unexpected type: " << TYPE_ENUM; \
      break;                                          \
  }

Tensor::Tensor(Allocator* a, DataType type, const TensorShape& shape)
    : shape_(shape), buf_(nullptr) {
  set_dtype(type);
  CHECK_NOTNULL(a);
  if (shape_.num_elements() > 0 || a->ShouldAllocateEmptyTensors()) {
    CASES(type, buf_ = new Buffer<T>(a, shape.num_elements()));
  }
  if (buf_ != nullptr && buf_->data() != nullptr && LogMemory::IsEnabled()) {
    LogMemory::RecordTensorAllocation("Unknown", LogMemory::UNKNOWN_STEP_ID,
                                      *this);
  }
}

Tensor::Tensor(Allocator* a, DataType type, const TensorShape& shape,
               const AllocationAttributes& allocation_attr)
    : shape_(shape), buf_(nullptr) {
  set_dtype(type);
  CHECK_NOTNULL(a);
  if (shape_.num_elements() > 0 || a->ShouldAllocateEmptyTensors()) {
    CASES(type, buf_ = new Buffer<T>(a, shape.num_elements(), allocation_attr));
  }
  if (!allocation_attr.allocation_will_be_logged && buf_ != nullptr &&
      buf_->data() != nullptr && LogMemory::IsEnabled()) {
    LogMemory::RecordTensorAllocation("Unknown (with attributes)",
                                      LogMemory::UNKNOWN_STEP_ID, *this);
  }
}

Tensor::Tensor(DataType type, const TensorShape& shape)
    : Tensor(cpu_allocator(), type, shape) {}

template <typename T>
class SubBuffer : public TensorBuffer {
 public:
  // This buffer is an alias to buf[delta, delta + n).
  SubBuffer(TensorBuffer* buf, int64 delta, int64 n)
      : root_(buf->root_buffer()), data_(buf->base<T>() + delta), elem_(n) {
    // Sanity check. The caller should ensure the sub buffer is valid.
    CHECK_LE(root_->base<T>(), this->base<T>());
    T* root_limit = root_->base<T>() + root_->size() / sizeof(T);
    CHECK_LE(this->base<T>(), root_limit);
    CHECK_LE(this->base<T>() + n, root_limit);
    // Hold a ref of the underlying root buffer.
    // NOTE: 'buf' is a sub-buffer inside the 'root_' buffer.
    root_->Ref();
  }

  void* data() const override { return data_; }
  size_t size() const override { return sizeof(T) * elem_; }
  TensorBuffer* root_buffer() override { return root_; }
  void FillAllocationDescription(AllocationDescription* proto) const override {
    root_->FillAllocationDescription(proto);
  }

 private:
  TensorBuffer* root_;
  T* data_;
  int64 elem_;

  ~SubBuffer() override { root_->Unref(); }

  TF_DISALLOW_COPY_AND_ASSIGN(SubBuffer);
};

Tensor Tensor::Slice(int64 start, int64 limit) const {
  CHECK_GE(dims(), 1);
  CHECK_LE(0, start);
  CHECK_LE(start, limit);
  int64 dim0_size = shape_.dim_size(0);
  CHECK_LE(limit, dim0_size);
  if ((start == 0) && (limit == dim0_size)) {
    return *this;
  }
  Tensor ret;
  ret.shape_ = shape_;
  ret.set_dtype(dtype());
  ret.buf_ = nullptr;
  if (dim0_size > 0) {
    const int64 elems_per_dim0 = NumElements() / dim0_size;
    const int64 delta = start * elems_per_dim0;
    dim0_size = limit - start;
    ret.shape_.set_dim(0, dim0_size);
    const int64 num_elems = dim0_size * elems_per_dim0;
    if (buf_) {
      DataType dt = dtype();
      CASES(dt, ret.buf_ = new SubBuffer<T>(buf_, delta, num_elems));
    }
  }
  return ret;
}

bool Tensor::FromProto(const TensorProto& proto) {
  return FromProto(cpu_allocator(), proto);
}

bool Tensor::FromProto(Allocator* a, const TensorProto& proto) {
  CHECK_NOTNULL(a);
  TensorBuffer* p = nullptr;
  if (!TensorShape::IsValid(proto.tensor_shape())) return false;
  if (proto.dtype() == DT_INVALID) return false;
  TensorShape shape(proto.tensor_shape());
  const int64 N = shape.num_elements();
  if (N > 0 && proto.dtype()) {
    if (!proto.tensor_content().empty()) {
      const auto& content = proto.tensor_content();
      CASES(proto.dtype(), p = Helper<T>::Decode(a, content, N));
    } else {
      CASES(proto.dtype(), p = FromProtoField<T>(a, proto, N));
    }
    if (p == nullptr) return false;
  }
  shape_ = shape;
  set_dtype(proto.dtype());
  UnrefIfNonNull(buf_);
  buf_ = p;
  // TODO(misard) add tracking of which kernels and steps are calling
  // FromProto.
  if (buf_ != nullptr && buf_->data() != nullptr && LogMemory::IsEnabled()) {
    LogMemory::RecordTensorAllocation("Unknown (from Proto)",
                                      LogMemory::UNKNOWN_STEP_ID, *this);
  }
  return true;
}

void Tensor::AsProtoField(TensorProto* proto) const {
  proto->Clear();
  shape_.AsProto(proto->mutable_tensor_shape());
  proto->set_dtype(dtype());
  if (buf_) {
    CASES(dtype(), ToProtoField<T>(*buf_, shape_.num_elements(), proto));
  }
}

void Tensor::AsProtoTensorContent(TensorProto* proto) const {
  proto->Clear();
  proto->set_dtype(dtype());
  shape_.AsProto(proto->mutable_tensor_shape());
  if (buf_) {
    CASES(dtype(), Helper<T>::Encode(buf_, shape_.num_elements(),
                                     proto->mutable_tensor_content()));
  }
}

size_t Tensor::TotalBytes() const {
  if (shape_.num_elements() == 0) return 0;
  CHECK(buf_) << "null buf_ with non-zero shape size " << shape_.num_elements();
  CASES(dtype(), return Helper<T>::TotalBytes(buf_, shape_.num_elements()));
  return 0;  // Makes compiler happy.
}

bool Tensor::CanUseDMA() const {
  CASES(dtype(), return Allocator::is_simple<T>::value);
  return false;  // Makes compiler happy.
}

#undef CASES
#undef CASE

namespace {
template <typename T>
string SummarizeArray(int64 limit, int64 num_elts, const char* data) {
  string ret;
  const T* array = reinterpret_cast<const T*>(data);
  for (int64 i = 0; i < limit; ++i) {
    if (i > 0) strings::StrAppend(&ret, " ");
    strings::StrAppend(&ret, array[i]);
  }
  if (num_elts > limit) strings::StrAppend(&ret, "...");
  return ret;
}
}  // namespace

string Tensor::SummarizeValue(int64 max_entries) const {
  const int64 num_elts = NumElements();
  size_t limit = std::min(max_entries, num_elts);
  if ((limit > 0) && (buf_ == nullptr)) {
    return strings::StrCat("uninitialized Tensor of ", num_elts,
                           " elements of type ", dtype());
  }
  const char* data = limit > 0 ? tensor_data().data() : nullptr;
  switch (dtype()) {
    case DT_HALF:
      return SummarizeArray<Eigen::half>(limit, num_elts, data);
      break;
    case DT_FLOAT:
      return SummarizeArray<float>(limit, num_elts, data);
      break;
    case DT_DOUBLE:
      return SummarizeArray<double>(limit, num_elts, data);
      break;
    case DT_INT32:
      return SummarizeArray<int32>(limit, num_elts, data);
      break;
    case DT_UINT8:
    case DT_QUINT8:
      return SummarizeArray<uint8>(limit, num_elts, data);
      break;
    case DT_UINT16:
    case DT_QUINT16:
      return SummarizeArray<uint16>(limit, num_elts, data);
      break;
    case DT_INT16:
    case DT_QINT16:
      return SummarizeArray<int16>(limit, num_elts, data);
      break;
    case DT_INT8:
    case DT_QINT8:
      return SummarizeArray<int8>(limit, num_elts, data);
      break;
    case DT_INT64:
      return SummarizeArray<int64>(limit, num_elts, data);
      break;
    case DT_BOOL:
      // TODO(tucker): Is it better to emit "True False..."?  This
      // will emit "1 0..." which is more compact.
      return SummarizeArray<bool>(limit, num_elts, data);
      break;
    default: {
      // All irregular cases
      string ret;
      // TODO(irving): Don't call flat every time around this
      // loop.
      for (int64 i = 0; i < limit; ++i) {
        if (i > 0) strings::StrAppend(&ret, " ");
        switch (dtype()) {
          case DT_STRING:
            strings::StrAppend(&ret, str_util::CEscape(flat<string>()(i)));
            break;
          default:
            // TODO(zhifengc, josh11b): Pretty-print other types (bool,
            // complex64, quantized).
            strings::StrAppend(&ret, "?");
        }
      }
      if (max_entries < num_elts) strings::StrAppend(&ret, "...");
      return ret;
    }
  }
}

StringPiece Tensor::tensor_data() const {
  if (buf_ == nullptr) return StringPiece();  // Don't die for empty tensors
  return StringPiece(static_cast<char*>(buf_->data()), TotalBytes());
}

bool Tensor::SharesBufferWith(const Tensor& b) const {
  CHECK_NE(nullptr, buf_);
  CHECK_NE(nullptr, b.buf_);
  return buf_->root_buffer() == b.buf_->root_buffer();
}

size_t Tensor::BufferHash() const {
  CHECK_NE(nullptr, buf_);
  return std::hash<TensorBuffer*>()(buf_->root_buffer());
}

string Tensor::DebugString() const {
  return strings::StrCat("Tensor<type: ", DataTypeString(dtype()), " shape: ",
                         shape().DebugString(), " values: ", SummarizeValue(3),
                         ">");
}

void Tensor::FillDescription(TensorDescription* description) const {
  description->set_dtype(dtype());
  shape().AsProto(description->mutable_shape());
  if (buf_ != nullptr && buf_->data() != nullptr) {
    buf_->FillAllocationDescription(
        description->mutable_allocation_description());
  }
}

gtl::InlinedVector<int64, 4> Tensor::ComputeFlatInnerDims(
    int64 num_out_dims) const {
  if (num_out_dims == dims()) {
    return shape_.dim_sizes();
  }
  gtl::InlinedVector<int64, 4> out_dims(num_out_dims, 0);
  const int64 num_elements = NumElements();
  int64 prod_out_dims = 1;
  for (int64 out_dim = num_out_dims - 1; out_dim > 0; --out_dim) {
    const int64 in_dim = out_dim + (dims() - num_out_dims);
    out_dims[out_dim] = (in_dim >= dims() || in_dim < 0) ? 1 : dim_size(in_dim);
    prod_out_dims *= out_dims[out_dim];
  }
  if (prod_out_dims != 0) {
    out_dims[0] = num_elements / prod_out_dims;
  } else {
    out_dims[0] = 0;
  }
  return out_dims;
}

gtl::InlinedVector<int64, 4> Tensor::ComputeFlatOuterDims(
    int64 num_out_dims) const {
  if (num_out_dims == dims()) {
    return shape_.dim_sizes();
  }
  gtl::InlinedVector<int64, 4> out_dims(num_out_dims, 0);
  const int64 num_elements = NumElements();
  int64 prod_out_dims = 1;
  for (int64 out_dim = 0; out_dim < num_out_dims - 1; ++out_dim) {
    out_dims[out_dim] = out_dim >= dims() ? 1 : dim_size(out_dim);
    prod_out_dims *= out_dims[out_dim];
  }
  if (prod_out_dims != 0) {
    out_dims[num_out_dims - 1] = num_elements / prod_out_dims;
  } else {
    out_dims[num_out_dims - 1] = 0;
  }
  return out_dims;
}

}  // namespace tensorflow

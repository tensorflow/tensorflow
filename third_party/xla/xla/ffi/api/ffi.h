/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_FFI_API_FFI_H_
#define XLA_FFI_API_FFI_H_

#ifdef XLA_FFI_FFI_H_
#error Two different XLA FFI implementations cannot be included together. \
       See README.md for more details.
#endif  // XLA_FFI_FFI_H_

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "xla/ffi/api/c_api.h"

// IWYU pragma: begin_exports
#include "xla/ffi/api/api.h"
// IWYU pragma: end_exports

namespace xla::ffi {

// All user data types that are passed via the execution context or state must
// be registered with the XLA FFI ahead of time to get unique type id.
using TypeId = XLA_FFI_TypeId;  // NOLINT

enum class DataType : uint8_t {
  INVALID = XLA_FFI_DataType_INVALID,
  PRED = XLA_FFI_DataType_PRED,
  S8 = XLA_FFI_DataType_S8,
  S16 = XLA_FFI_DataType_S16,
  S32 = XLA_FFI_DataType_S32,
  S64 = XLA_FFI_DataType_S64,
  U8 = XLA_FFI_DataType_U8,
  U16 = XLA_FFI_DataType_U16,
  U32 = XLA_FFI_DataType_U32,
  U64 = XLA_FFI_DataType_U64,
  F16 = XLA_FFI_DataType_F16,
  F32 = XLA_FFI_DataType_F32,
  F64 = XLA_FFI_DataType_F64,
  BF16 = XLA_FFI_DataType_BF16,
  C64 = XLA_FFI_DataType_C64,
  C128 = XLA_FFI_DataType_C128,
  TOKEN = XLA_FFI_DataType_TOKEN,
  F8E5M2 = XLA_FFI_DataType_F8E5M2,
  F8E4M3FN = XLA_FFI_DataType_F8E4M3FN,
  F8E4M3B11FNUZ = XLA_FFI_DataType_F8E4M3B11FNUZ,
  F8E5M2FNUZ = XLA_FFI_DataType_F8E5M2FNUZ,
  F8E4M3FNUZ = XLA_FFI_DataType_F8E4M3FNUZ,
};

// Create aliases in ::xla::ffi namespace for all DataTypes, for consistency
// with xla that defines PrimitiveType enums in ::xla namespace.
inline constexpr DataType PRED = DataType::PRED;
inline constexpr DataType S8 = DataType::S8;
inline constexpr DataType S16 = DataType::S16;
inline constexpr DataType S32 = DataType::S32;
inline constexpr DataType S64 = DataType::S64;
inline constexpr DataType U8 = DataType::U8;
inline constexpr DataType U16 = DataType::U16;
inline constexpr DataType U32 = DataType::U32;
inline constexpr DataType U64 = DataType::U64;
inline constexpr DataType F16 = DataType::F16;
inline constexpr DataType F32 = DataType::F32;
inline constexpr DataType F64 = DataType::F64;
inline constexpr DataType BF16 = DataType::BF16;
inline constexpr DataType C64 = DataType::C64;
inline constexpr DataType C128 = DataType::C128;
inline constexpr DataType TOKEN = DataType::TOKEN;
inline constexpr DataType F8E5M2 = DataType::F8E5M2;
inline constexpr DataType F8E4M3FN = DataType::F8E4M3FN;
inline constexpr DataType F8E4M3B11FNUZ = DataType::F8E4M3B11FNUZ;
inline constexpr DataType F8E5M2FNUZ = DataType::F8E5M2FNUZ;
inline constexpr DataType F8E4M3FNUZ = DataType::F8E4M3FNUZ;

inline std::ostream& operator<<(std::ostream& os, const DataType dtype) {
  return os << static_cast<XLA_FFI_DataType>(dtype);
}

constexpr size_t ByteWidth(DataType dtype) {
  switch (dtype) {
    case DataType::INVALID:
    case DataType::TOKEN:
      return 0;
    case DataType::PRED:
      return 1;
    case DataType::S8:
    case DataType::U8:
    case DataType::F8E5M2:
    case DataType::F8E4M3FN:
    case DataType::F8E4M3B11FNUZ:
    case DataType::F8E5M2FNUZ:
    case DataType::F8E4M3FNUZ:
      return 1;
    case DataType::S16:
    case DataType::U16:
    case DataType::F16:
    case DataType::BF16:
      return 2;
    case DataType::S32:
    case DataType::U32:
    case DataType::F32:
      return 4;
    case DataType::S64:
    case DataType::U64:
    case DataType::F64:
      return 8;
    case DataType::C64:
      return 8;
    case DataType::C128:
      return 16;
  }
}

//===----------------------------------------------------------------------===//
// Span is non-owning view into contiguous values of type `T`.
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Replace with `std::span` when C++20 is available.
template <typename T>
class Span {
 public:
  constexpr Span() : data_(nullptr), size_(0) {}

  Span(T* data, size_t size) : data_(data), size_(size) {}
  Span(const std::vector<std::remove_const_t<T>>& vec)  // NOLINT
      : Span(vec.data(), vec.size()) {}

  T& operator[](size_t index) const { return data_[index]; }

  bool operator==(const Span<T>& other) const {
    return size() == other.size() && std::equal(begin(), end(), other.begin());
  }

  T& front() const { return data_[0]; }
  T& back() const { return data_[size_ - 1]; }
  Span<T> first(size_t n) const { return Span<T>(data_, n); }
  Span<T> last(size_t n) const { return Span<T>(data_ + size_ - n, n); }
  size_t size() const { return size_; }

  T* begin() const { return data_; }
  T* end() const { return data_ + size_; }

 private:
  T* data_;
  size_t size_;
};

//===----------------------------------------------------------------------===//
// Error
//===----------------------------------------------------------------------===//

enum class ErrorCode : uint8_t {
  kOk = XLA_FFI_Error_Code_OK,
  kCancelled = XLA_FFI_Error_Code_CANCELLED,
  kUnknown = XLA_FFI_Error_Code_UNKNOWN,
  kInvalidArgument = XLA_FFI_Error_Code_INVALID_ARGUMENT,
  kDeadlineExceeded = XLA_FFI_Error_Code_DEADLINE_EXCEEDED,
  kNotFound = XLA_FFI_Error_Code_NOT_FOUND,
  kAlreadyExists = XLA_FFI_Error_Code_ALREADY_EXISTS,
  kPermissionDenied = XLA_FFI_Error_Code_PERMISSION_DENIED,
  kResourceExhausted = XLA_FFI_Error_Code_RESOURCE_EXHAUSTED,
  kFailedPrecondition = XLA_FFI_Error_Code_FAILED_PRECONDITION,
  kAborted = XLA_FFI_Error_Code_ABORTED,
  kOutOfRange = XLA_FFI_Error_Code_OUT_OF_RANGE,
  kUnimplemented = XLA_FFI_Error_Code_UNIMPLEMENTED,
  kInternal = XLA_FFI_Error_Code_INTERNAL,
  kUnavailable = XLA_FFI_Error_Code_UNAVAILABLE,
  kDataLoss = XLA_FFI_Error_Code_DATA_LOSS,
  kUnauthenticated = XLA_FFI_Error_Code_UNAUTHENTICATED,
};

class Error {
 public:
  Error() = default;

  Error(ErrorCode errc, std::string message)
      : errc_(errc), message_(std::move(message)) {}

  Error(XLA_FFI_Error_Code errc, std::string message)
      : Error(static_cast<ErrorCode>(errc), std::move(message)) {}

  bool success() const { return errc_ == ErrorCode::kOk; }
  bool failure() const { return !success(); }

  std::optional<ErrorCode> errc() const { return errc_; }
  const std::string& message() const { return message_; }

  static Error Success() { return Error(); }

  static Error Internal(std::string message) {
    return Error(ErrorCode::kInternal, std::move(message));
  }

  static Error InvalidArgument(std::string message) {
    return Error(ErrorCode::kInvalidArgument, std::move(message));
  }

 private:
  ErrorCode errc_ = ErrorCode::kOk;
  std::string message_;
};

//===----------------------------------------------------------------------===//
// Expected<T, E> and ErrorOr<T>
//===----------------------------------------------------------------------===//

// Forward declare.
template <typename E>
class Unexpected;

// TODO(slebedev): Replace with `std::expected` when C++23 is available.
template <typename T, typename E>
class Expected {
 public:
  constexpr Expected(T value) : data_(std::move(value)) {}  // NOLINT
  constexpr Expected(Unexpected<E> u);                      // NOLINT

  constexpr operator bool() const {  // NOLINT
    return has_value();
  }

  constexpr T& operator*() & { return value(); }
  constexpr const T& operator*() const& { return value(); }
  constexpr T&& operator*() && { return std::move(value()); }
  constexpr const T& operator*() const&& { return std::move(value()); }

  constexpr T* operator->() { return &value(); }
  constexpr const T* operator->() const { return &value(); }

  constexpr bool has_value() const { return std::holds_alternative<T>(data_); }
  constexpr bool has_error() const { return std::holds_alternative<E>(data_); }

  constexpr T& value() & { return std::get<T>(data_); }
  constexpr const T& value() const& { return std::get<T>(data_); }
  constexpr T&& value() && { return std::get<T>(std::move(data_)); }
  constexpr const T& value() const&& { return std::get<T>(std::move(data_)); }

  constexpr E& error() & { return std::get<E>(data_); }
  constexpr const E& error() const& { return std::get<E>(data_); }
  constexpr E&& error() && { return std::get<E>(std::move(data_)); }
  constexpr const E&& error() const&& { return std::get<E>(std::move(data_)); }

 private:
  std::variant<T, E> data_;
};

template <typename E>
class Unexpected {
 public:
  constexpr Unexpected(E error) : error_(std::move(error)) {}  // NOLINT

 private:
  template <typename, typename>
  friend class Expected;

  E error_;
};

Unexpected(const char*) -> Unexpected<std::string>;

template <typename T, typename E>
constexpr Expected<T, E>::Expected(Unexpected<E> u)
    : data_(std::move(u.error_)) {}

template <typename T>
class ErrorOr : public Expected<T, Error> {
 public:
  using Expected<T, Error>::Expected;
};

//===----------------------------------------------------------------------===//
// Arguments
//===----------------------------------------------------------------------===//

// Dynamically-typed buffer.
//
// No checks are done at decoding time. Any dtype and rank combination is
// accepted.
class AnyBuffer {
 public:
  using Dimensions = Span<const int64_t>;

  explicit AnyBuffer(const XLA_FFI_Buffer* buf) : buf_(buf) {
    assert(buf != nullptr && "XLA_FFI_Buffer must be non-null");
  }

  DataType element_type() const { return DataType(buf_->dtype); }

  Dimensions dimensions() const { return Dimensions(buf_->dims, buf_->rank); }

  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE size_t size_bytes() const {
    return ByteWidth(element_type()) * element_count();
  }

  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE size_t element_count() const {
    Dimensions dims = dimensions();
    return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                           std::multiplies<>());
  }

  void* untyped_data() const { return buf_->data; }

 private:
  const XLA_FFI_Buffer* buf_;
};

namespace internal {

// A workaround for the fact that a static_assertion can be evaluated
// whether or not the template is instantiated
template <DataType dtype>
struct always_false : std::false_type {};

template <DataType dtype>
struct DataTypeToNative {
  static_assert(always_false<dtype>::value, "unsupported data type");
};

#define XLA_FFI_REGISTER_DATATYPE_MAPPING(data_type_value, actual_type) \
  template <>                                                           \
  struct DataTypeToNative<data_type_value> {                            \
    using type = actual_type;                                           \
  };

XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::PRED, bool);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::U8, uint8_t);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::U16, uint16_t);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::U32, uint32_t);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::U64, uint64_t);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::S8, int8_t);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::S16, int16_t);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::S32, int32_t);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::S64, int64_t);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::F16, uint16_t);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::F32, float);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::F64, double);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::BF16, uint16_t);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::C64, std::complex<float>);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::C128, std::complex<double>);
XLA_FFI_REGISTER_DATATYPE_MAPPING(DataType::TOKEN, void);

#undef XLA_FFI_REGISTER_DATATYPE_MAPPING

inline constexpr size_t kDynamicRank = std::numeric_limits<size_t>::max();

}  // namespace internal

constexpr DataType ToComplex(DataType dtype) {
  switch (dtype) {
    case DataType::F32:
      return DataType::C64;
    case DataType::F64:
      return DataType::C128;
    default:
      return DataType::INVALID;
  }
}

constexpr DataType ToReal(DataType dtype) {
  switch (dtype) {
    case DataType::C64:
      return DataType::F32;
    case DataType::C128:
      return DataType::F64;
    default:
      return dtype;
  }
}

constexpr DataType ToImag(DataType dtype) {
  switch (dtype) {
    case DataType::C64:
      return DataType::F32;
    case DataType::C128:
      return DataType::F64;
    default:
      return dtype;
  }
}

template <DataType dtype>
using NativeType = typename internal::DataTypeToNative<dtype>::type;

template <DataType dtype>
constexpr bool IsComplexType() {
  return std::is_same_v<NativeType<dtype>,
                        std::complex<NativeType<ToReal(dtype)>>>;
}

static_assert(ToReal(DataType::C64) == DataType::F32);
static_assert(ToReal(DataType::C128) == DataType::F64);
static_assert(ToReal(DataType::F32) == DataType::F32);
static_assert(ToComplex(DataType::F32) == DataType::C64);
static_assert(ToComplex(DataType::F64) == DataType::C128);
static_assert(ToComplex(DataType::S32) == DataType::INVALID);
static_assert(ToComplex(ToReal(DataType::C64)) == DataType::C64);
static_assert(ToComplex(ToImag(DataType::C128)) == DataType::C128);
static_assert(IsComplexType<DataType::C64>());
static_assert(IsComplexType<DataType::C128>());
static_assert(!IsComplexType<DataType::F32>());

// Buffer with a statically-known dtype and rank.
//
// The dtype and rank are checked at decoding time. If rank is not specified,
// any rank is accepted.
template <DataType dtype, size_t rank = internal::kDynamicRank>
class Buffer {
 public:
  using Dimensions = AnyBuffer::Dimensions;

  explicit Buffer(const XLA_FFI_Buffer* buf) : buf_(buf) {
    assert(buf_ != nullptr && "XLA_FFI_Buffer must be non-null");
  }

  DataType element_type() const { return dtype; }

  Dimensions dimensions() const {
    return Dimensions(buf_->dims,
                      rank == internal::kDynamicRank ? buf_->rank : rank);
  }

  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE size_t size_bytes() const {
    return ByteWidth(dtype) * element_count();
  }

  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE size_t element_count() const {
    Dimensions dims = dimensions();
    return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                           std::multiplies<>());
  }

  void* untyped_data() const { return buf_->data; }

  NativeType<dtype>* typed_data() const {
    return reinterpret_cast<NativeType<dtype>*>(untyped_data());
  }

 private:
  const XLA_FFI_Buffer* buf_;
};

// clang-format off
template <DataType dtype> using BufferR0 = Buffer<dtype, 0>;
template <DataType dtype> using BufferR1 = Buffer<dtype, 1>;
template <DataType dtype> using BufferR2 = Buffer<dtype, 2>;
template <DataType dtype> using BufferR3 = Buffer<dtype, 3>;
template <DataType dtype> using BufferR4 = Buffer<dtype, 4>;
// clang-format on

using Token = BufferR0<DataType::TOKEN>;  // NOLINT

namespace internal {

template <DataType dtype, size_t rank>
XLA_FFI_ATTRIBUTE_ALWAYS_INLINE std::optional<Buffer<dtype, rank>> DecodeBuffer(
    XLA_FFI_Buffer* buf, DiagnosticEngine& diagnostic) {
  if (auto buf_dtype = static_cast<DataType>(buf->dtype);
      XLA_FFI_PREDICT_FALSE(buf_dtype != dtype)) {
    return diagnostic.Emit("Wrong buffer dtype: expected ")
           << dtype << " but got " << buf_dtype;
  }

  if constexpr (rank != internal::kDynamicRank) {
    if (XLA_FFI_PREDICT_FALSE(buf->rank != rank)) {
      return diagnostic.Emit("Wrong buffer rank: expected ")
             << rank << " but got " << buf->rank;
    }
  }

  return Buffer<dtype, rank>(buf);
}

}  // namespace internal

template <DataType dtype, size_t rank = internal::kDynamicRank>
using ResultBuffer = Result<Buffer<dtype, rank>>;

// clang-format off
template <DataType dtype> using ResultBufferR0 = ResultBuffer<dtype, 0>;
template <DataType dtype> using ResultBufferR1 = ResultBuffer<dtype, 1>;
template <DataType dtype> using ResultBufferR2 = ResultBuffer<dtype, 2>;
template <DataType dtype> using ResultBufferR3 = ResultBuffer<dtype, 3>;
template <DataType dtype> using ResultBufferR4 = ResultBuffer<dtype, 4>;
// clang-format on

//===----------------------------------------------------------------------===//
// Arguments binding
//===----------------------------------------------------------------------===//

template <>
struct ArgBinding<AnyBuffer> {
  using Arg = AnyBuffer;
};

template <DataType dtype, size_t rank>
struct ArgBinding<Buffer<dtype, rank>> {
  using Arg = Buffer<dtype, rank>;
};

//===----------------------------------------------------------------------===//
// Results binding
//===----------------------------------------------------------------------===//

template <>
struct RetBinding<Result<AnyBuffer>> {
  using Ret = AnyBuffer;
};

template <DataType dtype, size_t rank>
struct RetBinding<Result<Buffer<dtype, rank>>> {
  using Ret = Buffer<dtype, rank>;
};

//===----------------------------------------------------------------------===//
// Arguments decoding
//===----------------------------------------------------------------------===//

inline std::ostream& operator<<(std::ostream& os, const XLA_FFI_ArgType type) {
  switch (type) {
    case XLA_FFI_ArgType_BUFFER:
      return os << "buffer";
  }
}

template <>
struct ArgDecoding<AnyBuffer> {
  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<AnyBuffer> Decode(XLA_FFI_ArgType type, void* arg,
                                         DiagnosticEngine& diagnostic) {
    if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_ArgType_BUFFER)) {
      return diagnostic.Emit("Wrong argument type: expected ")
             << XLA_FFI_ArgType_BUFFER << " but got " << type;
    }
    return AnyBuffer(reinterpret_cast<XLA_FFI_Buffer*>(arg));
  }
};

template <DataType dtype, size_t rank>
struct ArgDecoding<Buffer<dtype, rank>> {
  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<Buffer<dtype, rank>> Decode(
      XLA_FFI_ArgType type, void* arg, DiagnosticEngine& diagnostic) {
    if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_ArgType_BUFFER)) {
      return diagnostic.Emit("Wrong argument type: expected ")
             << XLA_FFI_ArgType_BUFFER << " but got " << type;
    }

    return internal::DecodeBuffer<dtype, rank>(
        reinterpret_cast<XLA_FFI_Buffer*>(arg), diagnostic);
  }
};

//===----------------------------------------------------------------------===//
// Type-safe wrapper for accessing a variable number of arguments.
//===----------------------------------------------------------------------===//

class RemainingArgs : public internal::RemainingArgsBase {
 public:
  using internal::RemainingArgsBase::RemainingArgsBase;

  template <typename T>
  ErrorOr<T> get(size_t index) const {
    size_t idx = offset() + index;
    if (XLA_FFI_PREDICT_FALSE(idx >= args()->size)) {
      return Unexpected(
          Error(ErrorCode::kInvalidArgument, "Index out of range"));
    }

    DiagnosticEngine diagnostic;
    std::optional<T> value = ArgDecoding<T>::Decode(
        args()->types[idx], args()->args[idx], diagnostic);
    if (XLA_FFI_PREDICT_FALSE(!value.has_value())) {
      return Unexpected(Error::Internal(diagnostic.Result()));
    }

    return *value;
  }
};

template <>
struct internal::Decode<internal::RemainingArgsTag> {
  static std::optional<RemainingArgs> call(DecodingOffsets& offsets,
                                           DecodingContext& ctx,
                                           DiagnosticEngine& diagnostic) {
    return RemainingArgs(&ctx.call_frame->args, offsets.args);
  }
};

//===----------------------------------------------------------------------===//
// Results decoding
//===----------------------------------------------------------------------===//

inline std::ostream& operator<<(std::ostream& os, const XLA_FFI_RetType type) {
  switch (type) {
    case XLA_FFI_RetType_BUFFER:
      return os << "buffer";
  }
}

template <>
struct RetDecoding<AnyBuffer> {
  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<Result<AnyBuffer>> Decode(XLA_FFI_RetType type,
                                                 void* ret,
                                                 DiagnosticEngine& diagnostic) {
    if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_RetType_BUFFER)) {
      return diagnostic.Emit("Wrong result type: expected ")
             << XLA_FFI_RetType_BUFFER << " but got " << type;
    }
    return AnyBuffer(reinterpret_cast<XLA_FFI_Buffer*>(ret));
  }
};

template <DataType dtype, size_t rank>
struct RetDecoding<Buffer<dtype, rank>> {
  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<Result<Buffer<dtype, rank>>> Decode(
      XLA_FFI_RetType type, void* ret, DiagnosticEngine& diagnostic) {
    if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_RetType_BUFFER)) {
      return diagnostic.Emit("Wrong result type: expected ")
             << XLA_FFI_RetType_BUFFER << " but got " << type;
    }

    return internal::DecodeBuffer<dtype, rank>(
        reinterpret_cast<XLA_FFI_Buffer*>(ret), diagnostic);
  }
};

//===----------------------------------------------------------------------===//
// Type-safe wrapper for accessing a variable number of results.
//===----------------------------------------------------------------------===//

class RemainingRets : public internal::RemainingRetsBase {
 public:
  using internal::RemainingRetsBase::RemainingRetsBase;

  template <typename T>
  ErrorOr<Result<T>> get(size_t index) const {
    size_t idx = offset() + index;
    if (XLA_FFI_PREDICT_FALSE(idx >= rets()->size)) {
      return Unexpected(
          Error(ErrorCode::kInvalidArgument, "Index out of range"));
    }

    DiagnosticEngine diagnostic;
    std::optional<Result<T>> value = RetDecoding<T>::Decode(
        rets()->types[idx], rets()->rets[idx], diagnostic);
    if (XLA_FFI_PREDICT_FALSE(!value.has_value())) {
      return Unexpected(Error::Internal(diagnostic.Result()));
    }

    return *value;
  }
};

template <>
struct internal::Decode<internal::RemainingRetsTag> {
  static std::optional<RemainingRets> call(DecodingOffsets& offsets,
                                           DecodingContext& ctx,
                                           DiagnosticEngine& diagnostic) {
    return RemainingRets(&ctx.call_frame->rets, offsets.rets);
  }
};

//===----------------------------------------------------------------------===//
// Attributes decoding
//===----------------------------------------------------------------------===//

#define XLA_FFI_REGISTER_ARRAY_ATTR_DECODING(T, TYPE)                       \
  template <>                                                               \
  struct AttrDecoding<Span<const T>> {                                      \
    using Type = Span<const T>;                                             \
    static std::optional<Type> Decode(XLA_FFI_AttrType type, void* attr,    \
                                      DiagnosticEngine& diagnostic) {       \
      if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_AttrType_ARRAY)) {          \
        return diagnostic.Emit("Wrong attribute type: expected ")           \
               << XLA_FFI_AttrType_ARRAY << " but got " << type;            \
      }                                                                     \
                                                                            \
      auto* array = reinterpret_cast<XLA_FFI_Array*>(attr);                 \
      if (XLA_FFI_PREDICT_FALSE(array->dtype != TYPE)) {                    \
        return diagnostic.Emit("Wrong array data type: expected ")          \
               << TYPE << " but got " << array->dtype;                      \
      }                                                                     \
                                                                            \
      return Span<const T>(reinterpret_cast<T*>(array->data), array->size); \
    }                                                                       \
  }

XLA_FFI_REGISTER_ARRAY_ATTR_DECODING(int8_t, XLA_FFI_DataType_S8);
XLA_FFI_REGISTER_ARRAY_ATTR_DECODING(int16_t, XLA_FFI_DataType_S16);
XLA_FFI_REGISTER_ARRAY_ATTR_DECODING(int32_t, XLA_FFI_DataType_S32);
XLA_FFI_REGISTER_ARRAY_ATTR_DECODING(int64_t, XLA_FFI_DataType_S64);
XLA_FFI_REGISTER_ARRAY_ATTR_DECODING(float, XLA_FFI_DataType_F32);
XLA_FFI_REGISTER_ARRAY_ATTR_DECODING(double, XLA_FFI_DataType_F64);

#undef XLA_FFI_REGISTER_ARRAY_ATTR_DECODING

// A type tag to mark i64 attributes as pointers to `T`.
template <typename T>
struct Pointer {};

template <typename T>
struct AttrDecoding<Pointer<T>> {
  using Type = T*;

  static std::optional<Type> Decode(XLA_FFI_AttrType type, void* attr,
                                    DiagnosticEngine& diagnostic) {
    auto* scalar = reinterpret_cast<XLA_FFI_Scalar*>(attr);
    if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_AttrType_SCALAR ||
                              scalar->dtype != XLA_FFI_DataType_S64)) {
      return diagnostic.Emit("Wrong attribute type: ")
             << "expected i64 scalar for passing pointer but got " << type;
    }

    static_assert(sizeof(uintptr_t) == sizeof(int64_t));
    uintptr_t ptr = *reinterpret_cast<uintptr_t*>(scalar->value);
    return reinterpret_cast<Type>(ptr);
  }
};

//===----------------------------------------------------------------------===//
// Type-safe wrapper for accessing dictionary attributes.
//===----------------------------------------------------------------------===//

class Dictionary : public internal::DictionaryBase {
 public:
  using internal::DictionaryBase::DictionaryBase;

  template <typename T>
  ErrorOr<T> get(std::string_view name) const {
    DiagnosticEngine diagnostic;
    std::optional<T> value = internal::DictionaryBase::get<T>(name, diagnostic);
    if (!value.has_value()) {
      return Unexpected(Error::Internal(diagnostic.Result()));
    }
    return *value;
  }
};

// Decode `AttrsTag` (all attributes) into a `Dictionary`.
template <>
struct internal::Decode<internal::AttrsTag<Dictionary>> {
  static std::optional<Dictionary> call(DecodingOffsets& offsets,
                                        DecodingContext& ctx,
                                        DiagnosticEngine& diagnostic) {
    return Dictionary(&ctx.call_frame->attrs);
  }
};

// Decode individual attribute into `Dictionary` type.
template <>
struct AttrDecoding<Dictionary> {
  using Type = Dictionary;
  static std::optional<Dictionary> Decode(XLA_FFI_AttrType type, void* attr,
                                          DiagnosticEngine& diagnostic) {
    if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_AttrType_DICTIONARY)) {
      return diagnostic.Emit("Wrong attribute type: expected ")
             << XLA_FFI_AttrType_DICTIONARY << " but got " << type;
    }
    return Dictionary(reinterpret_cast<XLA_FFI_Attrs*>(attr));
  }
};

//===----------------------------------------------------------------------===//
// Error helpers
//===----------------------------------------------------------------------===//

namespace internal {

inline XLA_FFI_Error* CreateError(const XLA_FFI_Api* api, const Error& error) {
  XLA_FFI_Error_Create_Args args;
  args.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.errc = static_cast<XLA_FFI_Error_Code>(*error.errc());
  args.message = error.message().c_str();
  return api->XLA_FFI_Error_Create(&args);
}

inline void DestroyError(const XLA_FFI_Api* api, XLA_FFI_Error* error) {
  XLA_FFI_Error_Destroy_Args args;
  args.struct_size = XLA_FFI_Error_Destroy_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.error = error;
  api->XLA_FFI_Error_Destroy(&args);
}

inline const char* GetErrorMessage(const XLA_FFI_Api* api,
                                   XLA_FFI_Error* error) {
  XLA_FFI_Error_GetMessage_Args args;
  args.struct_size = XLA_FFI_Error_GetMessage_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.error = error;
  api->XLA_FFI_Error_GetMessage(&args);
  return args.message;
}

}  // namespace internal

//===----------------------------------------------------------------------===//
// Result encoding
//===----------------------------------------------------------------------===//

// Encodes `Error` as an FFI error.
template <ExecutionStage stage>
struct ResultEncoding<stage, Error> {
  static XLA_FFI_Error* Encode(const XLA_FFI_Api* api,
                               XLA_FFI_ExecutionContext* ctx, Error error) {
    if (XLA_FFI_PREDICT_TRUE(error.success())) {
      return nullptr;
    }

    return internal::CreateError(api, error);
  }
};

// Encodes `ErrorOr<std::unique_ptr<T>>` as an FFI state.
template <typename T>
struct ResultEncoding<ExecutionStage::kInstantiate,
                      ErrorOr<std::unique_ptr<T>>> {
  static_assert(std::is_same_v<decltype(T::id), TypeId>,
                "State type must have a static `TypeId id` field");

  static XLA_FFI_Error* Encode(const XLA_FFI_Api* api,
                               XLA_FFI_ExecutionContext* ctx,
                               ErrorOr<std::unique_ptr<T>> state) {
    if (XLA_FFI_PREDICT_TRUE(state.has_value())) {
      XLA_FFI_State_Set_Args args;
      args.struct_size = XLA_FFI_State_Set_Args_STRUCT_SIZE;
      args.priv = nullptr;
      args.ctx = ctx;
      args.type_id = &T::id;
      args.state = state.value().release();
      args.deleter = +[](void* state) { delete reinterpret_cast<T*>(state); };
      return api->XLA_FFI_State_Set(&args);
    }

    return internal::CreateError(api, state.error());
  }
};

//===----------------------------------------------------------------------===//
// PlatformStream
//===----------------------------------------------------------------------===//

template <typename T>
struct PlatformStream {};

// Context decoding for platform stream.
//
// Example: Ffi::Bind().Ctx<PlatformStream<cudaStream_t>()
//                     .To([](cudaStream_t stream) { ... });
template <typename T>
struct CtxDecoding<PlatformStream<T>> {
  using Type = T;

  static_assert(std::is_pointer_v<T>, "stream type must be a pointer");

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    XLA_FFI_Stream_Get_Args args;
    args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.ctx = ctx;
    args.stream = nullptr;

    if (XLA_FFI_Error* error = api->XLA_FFI_Stream_Get(&args)) {
      diagnostic.Emit("Failed to get platform stream: ")
          << internal::GetErrorMessage(api, error);
      internal::DestroyError(api, error);
      return std::nullopt;
    }

    return reinterpret_cast<T>(args.stream);
  }
};

//===----------------------------------------------------------------------===//
// ScratchAllocator
//===----------------------------------------------------------------------===//

// Interface for "scratch" allocator for device memory, which deallocates all
// buffers it has allocated at destruction.
//
// WARNING: It is illegal to keep scratch allocator alive after returning from
// the FFI handler as it relies on execution context whose lifetime is bound to
// the particular call to FFI handler.
class ScratchAllocator {
 public:
  ScratchAllocator(const XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx,
                   DiagnosticEngine& diagnostic);
  ~ScratchAllocator();

  ScratchAllocator(ScratchAllocator&&) = default;
  ScratchAllocator& operator=(ScratchAllocator&&) = default;

  std::optional<void*> Allocate(size_t size, size_t alignment = 1);

 private:
  struct Allocation {
    size_t size;
    void* data;
  };

  const XLA_FFI_Api* api_;
  XLA_FFI_ExecutionContext* ctx_;

  DiagnosticEngine& diagnostic_;
  std::vector<Allocation> allocations_;
};

// Context decoding for scratch allocator.
//
// Example: Ffi::Bind().Ctx<ScratchAllocator>()
//                     .To([](ScratchAllocator scratch) { ... });
template <>
struct CtxDecoding<ScratchAllocator> {
  using Type = ScratchAllocator;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return ScratchAllocator(api, ctx, diagnostic);
  }
};

inline std::optional<void*> ScratchAllocator::Allocate(size_t size,
                                                       size_t alignment) {
  XLA_FFI_DeviceMemory_Allocate_Args args;
  args.struct_size = XLA_FFI_DeviceMemory_Allocate_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.ctx = ctx_;
  args.size = size;
  args.alignment = alignment;
  args.data = nullptr;
  if (XLA_FFI_Error* error = api_->XLA_FFI_DeviceMemory_Allocate(&args)) {
    diagnostic_.Emit("Failed to allocate scratch memory: ")
        << internal::GetErrorMessage(api_, error);
    internal::DestroyError(api_, error);
    return std::nullopt;
  }
  allocations_.push_back({size, args.data});
  return args.data;
}

inline ScratchAllocator::ScratchAllocator(const XLA_FFI_Api* api,
                                          XLA_FFI_ExecutionContext* ctx,
                                          DiagnosticEngine& diagnostic)
    : api_(api), ctx_(ctx), diagnostic_(diagnostic) {}

inline ScratchAllocator::~ScratchAllocator() {
  for (Allocation& alloc : allocations_) {
    XLA_FFI_DeviceMemory_Free_Args args;
    args.struct_size = XLA_FFI_DeviceMemory_Free_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.ctx = ctx_;
    args.size = alloc.size;
    args.data = alloc.data;
    if (XLA_FFI_Error* error = api_->XLA_FFI_DeviceMemory_Free(&args)) {
      diagnostic_.Emit("Failed to free scratch memory: ")
          << internal::GetErrorMessage(api_, error);
      internal::DestroyError(api_, error);
    }
  }
}

//===----------------------------------------------------------------------===//
// Type Registration
//===----------------------------------------------------------------------===//

namespace internal {

inline XLA_FFI_Error* RegisterType(const XLA_FFI_Api* api,
                                   std::string_view name,
                                   XLA_FFI_TypeId* type_id) {
  XLA_FFI_TypeId_Register_Args args;
  args.struct_size = XLA_FFI_TypeId_Register_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.name = XLA_FFI_ByteSpan{name.data(), name.size()};
  args.type_id = type_id;
  return api->XLA_FFI_TypeId_Register(&args);
}

}  // namespace internal

#define XLA_FFI_REGISTER_TYPE(API, NAME, TYPE_ID) \
  XLA_FFI_REGISTER_TYPE_(API, NAME, TYPE_ID, __COUNTER__)
#define XLA_FFI_REGISTER_TYPE_(API, NAME, TYPE_ID, N) \
  XLA_FFI_REGISTER_TYPE__(API, NAME, TYPE_ID, N)
#define XLA_FFI_REGISTER_TYPE__(API, NAME, TYPE_ID, N)                 \
  XLA_FFI_ATTRIBUTE_UNUSED static const XLA_FFI_Error*                 \
      xla_ffi_type_##N##_registered_ = [] {                            \
        return ::xla::ffi::internal::RegisterType(API, NAME, TYPE_ID); \
      }()

//===----------------------------------------------------------------------===//
// UserData
//===----------------------------------------------------------------------===//

// A type tag for automatic user data decoding passed via the execution
// context.
template <typename T>
struct UserData {};

// Context decoding for user data of type `T`.
//
// Example: Ffi::Bind().Ctx<UserData<MyData>>()
//                     .To([](MyData* data) { ... });
template <typename T>
struct CtxDecoding<UserData<T>> {
  using Type = T*;

  static_assert(std::is_same_v<decltype(T::id), TypeId>,
                "UserData type must have a static `TypeId id` field");

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    XLA_FFI_ExecutionContext_Get_Args args;
    args.struct_size = XLA_FFI_ExecutionContext_Get_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.ctx = ctx;
    args.type_id = &T::id;
    args.data = nullptr;

    assert(args.type_id->type_id > 0 && "type must be registered with XLA FFI");

    if (XLA_FFI_Error* err = api->XLA_FFI_ExecutionContext_Get(&args); err) {
      diagnostic.Emit("Failed to get user data from execution context: ")
          << internal::GetErrorMessage(api, err);
      internal::DestroyError(api, err);
      return std::nullopt;
    }

    return static_cast<Type>(args.data);
  }
};

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

// A type tag for automatic state decoding passed via the execution
// context.
template <typename T>
struct State {};

// Context decoding for state of type `T`.
//
// Example: Ffi::Bind().Ctx<State<MyState>>()
//                     .To([](MyState* state) { ... });
template <typename T>
struct CtxDecoding<State<T>> {
  using Type = T*;

  static_assert(std::is_same_v<decltype(T::id), TypeId>,
                "State type must have a static `TypeId id` field");

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    XLA_FFI_State_Get_Args args;
    args.struct_size = XLA_FFI_State_Get_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.ctx = ctx;
    args.type_id = &T::id;
    args.state = nullptr;

    assert(args.type_id->type_id > 0 && "type must be registered with XLA FFI");

    if (XLA_FFI_Error* err = api->XLA_FFI_State_Get(&args); err) {
      diagnostic.Emit("Failed to get state from execution context: ")
          << internal::GetErrorMessage(api, err);
      internal::DestroyError(api, err);
      return std::nullopt;
    }

    return static_cast<Type>(args.state);
  }
};

}  // namespace xla::ffi

#endif  // XLA_FFI_API_FFI_H_

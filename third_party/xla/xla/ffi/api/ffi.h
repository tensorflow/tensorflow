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
#error Two different XLA FFI implementations cannot be included together
#endif  // XLA_FFI_FFI_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "xla/ffi/api/c_api.h"

// IWYU pragma: begin_exports
#include "xla/ffi/api/api.h"
// IWYU pragma: end_exports

namespace xla::ffi {

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
};

inline std::ostream& operator<<(std::ostream& os, const DataType dtype) {
  static constexpr const char* kDataTypeNames[] = {
      "INVALID", "PRED", "S8",  "S16", "S32", "S64", "U8",
      "U16",     "U32",  "U64", "F16", "F32", "F64", "BF16",
  };
  return os << kDataTypeNames[static_cast<int>(dtype)];
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

class Error {
 public:
  Error() = default;
  Error(XLA_FFI_Error_Code errc, std::string message)
      : errc_(errc), message_(std::move(message)) {}

  static Error Success() { return Error(); }

  bool success() const { return errc_ == XLA_FFI_Error_Code_OK; }
  bool failure() const { return !success(); }

  std::optional<XLA_FFI_Error_Code> errc() const { return errc_; }
  const std::string& message() const { return message_; }

 private:
  XLA_FFI_Error_Code errc_;
  std::string message_;
};

//===----------------------------------------------------------------------===//
// Arguments
//===----------------------------------------------------------------------===//

struct BufferBase {
  DataType dtype;
  void* data;
  Span<const int64_t> dimensions;
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

// clang-format off
template <> struct DataTypeToNative<DataType::PRED> { using type = bool; };
template <> struct DataTypeToNative<DataType::U8>   { using type = uint8_t; };
template <> struct DataTypeToNative<DataType::U16>  { using type = uint16_t; };
template <> struct DataTypeToNative<DataType::U32>  { using type = uint32_t; };
template <> struct DataTypeToNative<DataType::U64>  { using type = uint64_t; };
template <> struct DataTypeToNative<DataType::S8>   { using type = int8_t; };
template <> struct DataTypeToNative<DataType::S16>  { using type = int16_t; };
template <> struct DataTypeToNative<DataType::S32>  { using type = int32_t; };
template <> struct DataTypeToNative<DataType::S64>  { using type = int64_t; };
template <> struct DataTypeToNative<DataType::F16>  { using type = uint16_t; };
template <> struct DataTypeToNative<DataType::F32>  { using type = float; };
template <> struct DataTypeToNative<DataType::F64>  { using type = double; };
template <> struct DataTypeToNative<DataType::BF16> { using type = uint16_t; };
// clang-format on

inline constexpr size_t kDynamicRank = std::numeric_limits<size_t>::max();

template <DataType dtype>
using NativeType = typename DataTypeToNative<dtype>::type;

}  // namespace internal

template <DataType dtype, size_t rank = internal::kDynamicRank>
struct Buffer {
  internal::NativeType<dtype>* data;
  Span<const int64_t> dimensions;
};

// clang-format off
template <DataType dtype> using BufferR0 = Buffer<dtype, 0>;
template <DataType dtype> using BufferR1 = Buffer<dtype, 1>;
template <DataType dtype> using BufferR2 = Buffer<dtype, 2>;
template <DataType dtype> using BufferR3 = Buffer<dtype, 3>;
template <DataType dtype> using BufferR4 = Buffer<dtype, 4>;
// clang-format on

namespace internal {

inline BufferBase DecodeBuffer(XLA_FFI_Buffer* buf) {
  return BufferBase{static_cast<DataType>(buf->dtype), buf->data,
                    Span<const int64_t>(buf->dims, buf->rank)};
}

template <DataType dtype, size_t rank>
std::optional<Buffer<dtype, rank>> DecodeBuffer(XLA_FFI_Buffer* buf,
                                                DiagnosticEngine& diagnostic) {
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

  Buffer<dtype, rank> buffer;
  buffer.data = static_cast<internal::NativeType<dtype>*>(buf->data);
  buffer.dimensions = Span<const int64_t>(buf->dims, buf->rank);
  return buffer;
}

}  // namespace internal

//===----------------------------------------------------------------------===//
// Arguments binding
//===----------------------------------------------------------------------===//

template <>
struct ArgBinding<BufferBase> {
  using Arg = BufferBase;
};

template <DataType dtype, size_t rank>
struct ArgBinding<Buffer<dtype, rank>> {
  using Arg = Buffer<dtype, rank>;
};

//===----------------------------------------------------------------------===//
// Results binding
//===----------------------------------------------------------------------===//

template <>
struct RetBinding<Result<BufferBase>> {
  using Ret = BufferBase;
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
struct ArgDecoding<BufferBase> {
  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<BufferBase> Decode(XLA_FFI_ArgType type, void* arg,
                                          DiagnosticEngine& diagnostic) {
    if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_ArgType_BUFFER)) {
      return diagnostic.Emit("Wrong argument type: expected ")
             << XLA_FFI_ArgType_BUFFER << " but got " << type;
    }
    return internal::DecodeBuffer(reinterpret_cast<XLA_FFI_Buffer*>(arg));
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
// Results decoding
//===----------------------------------------------------------------------===//

inline std::ostream& operator<<(std::ostream& os, const XLA_FFI_RetType type) {
  switch (type) {
    case XLA_FFI_RetType_BUFFER:
      return os << "buffer";
  }
}

template <>
struct RetDecoding<BufferBase> {
  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<Result<BufferBase>> Decode(
      XLA_FFI_RetType type, void* ret, DiagnosticEngine& diagnostic) {
    if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_RetType_BUFFER)) {
      return diagnostic.Emit("Wrong result type: expected ")
             << XLA_FFI_RetType_BUFFER << " but got " << type;
    }
    return internal::DecodeBuffer(reinterpret_cast<XLA_FFI_Buffer*>(ret));
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
// Attributes decoding
//===----------------------------------------------------------------------===//

#define XLA_FFI_REGISTER_ARRRAY_ATTR_DECODING(T, TYPE)                      \
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

XLA_FFI_REGISTER_ARRRAY_ATTR_DECODING(int32_t, XLA_FFI_DataType_S32);
XLA_FFI_REGISTER_ARRRAY_ATTR_DECODING(int64_t, XLA_FFI_DataType_S64);
XLA_FFI_REGISTER_ARRRAY_ATTR_DECODING(float, XLA_FFI_DataType_F32);

#undef XLA_FFI_REGISTER_SCALAR_ATTR_DECODING

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
// Result encoding
//===----------------------------------------------------------------------===//

template <>
struct ResultEncoding<Error> {
  static XLA_FFI_Error* Encode(const XLA_FFI_Api* api, Error error) {
    if (error.success()) return nullptr;

    XLA_FFI_Error_Create_Args args;
    args.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.errc = *error.errc();
    args.message = error.message().c_str();
    return api->XLA_FFI_Error_Create(&args);
  }
};

//===----------------------------------------------------------------------===//
// PlatformStream
//===----------------------------------------------------------------------===//

template <typename T>
struct PlatformStream {};

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

    if (XLA_FFI_Error* error = api->XLA_FFI_Stream_Get(&args); error) {
      diagnostic.Emit("Failed to get platform stream: ")
          << GetErrorMessage(api, error);
      DestroyError(api, error);
      return std::nullopt;
    }

    return reinterpret_cast<T>(args.stream);
  }

  static const char* GetErrorMessage(const XLA_FFI_Api* api,
                                     XLA_FFI_Error* error) {
    XLA_FFI_Error_GetMessage_Args args;
    args.struct_size = XLA_FFI_Error_GetMessage_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.error = error;
    api->XLA_FFI_Error_GetMessage(&args);
    return args.message;
  }

  static void DestroyError(const XLA_FFI_Api* api, XLA_FFI_Error* error) {
    XLA_FFI_Error_Destroy_Args args;
    args.struct_size = XLA_FFI_Error_Destroy_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.error = error;
    api->XLA_FFI_Error_Destroy(&args);
  }
};

}  // namespace xla::ffi

#endif  // XLA_FFI_API_FFI_H_

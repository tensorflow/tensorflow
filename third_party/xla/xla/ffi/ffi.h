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

#ifndef XLA_FFI_FFI_H_
#define XLA_FFI_FFI_H_

#ifdef XLA_FFI_API_FFI_H_
#error Two different XLA FFI implementations cannot be included together
#endif  // XLA_FFI_API_FFI_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

// IWYU pragma: begin_exports
#include "xla/ffi/api/api.h"
// IWYU pragma: end_exports

#include "absl/types/span.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/primitive_util.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

namespace xla::ffi {

// Type tags to bind parameters passed via execution context to FFI handler.
struct Stream {};             // binds `se::Stream*`
struct ScratchAllocator {};   // binds `se::OwningScratchAllocator`
struct CalledComputation {};  // binds `HloComputation*`

//===----------------------------------------------------------------------===//
// Arguments
//===----------------------------------------------------------------------===//

struct BufferBase {
  PrimitiveType dtype;
  se::DeviceMemoryBase data;
  absl::Span<const int64_t> dimensions;
};

namespace internal {

inline constexpr size_t kDynamicRank = std::numeric_limits<size_t>::max();

template <PrimitiveType dtype>
using NativeType = typename primitive_util::PrimitiveTypeToNative<dtype>::type;

}  // namespace internal

template <PrimitiveType dtype, size_t rank = internal::kDynamicRank>
struct Buffer {
  se::DeviceMemory<internal::NativeType<dtype>> data;
  absl::Span<const int64_t> dimensions;
};

// clang-format off
template <PrimitiveType dtype> using BufferR0 = Buffer<dtype, 0>;
template <PrimitiveType dtype> using BufferR1 = Buffer<dtype, 1>;
template <PrimitiveType dtype> using BufferR2 = Buffer<dtype, 2>;
template <PrimitiveType dtype> using BufferR3 = Buffer<dtype, 3>;
template <PrimitiveType dtype> using BufferR4 = Buffer<dtype, 4>;
// clang-format on

using Token = BufferR0<PrimitiveType::TOKEN>;

namespace internal {

inline BufferBase DecodeBuffer(XLA_FFI_Buffer* buf) {
  size_t size_bytes = 0;
  if (primitive_util::IsArrayType(PrimitiveType(buf->dtype))) {
    size_bytes = primitive_util::ByteWidth(PrimitiveType(buf->dtype));
    for (int64_t i = 0; i < buf->rank; ++i) size_bytes *= buf->dims[i];
  }

  BufferBase buffer;
  buffer.dtype = PrimitiveType(buf->dtype);
  buffer.data = se::DeviceMemoryBase(buf->data, size_bytes);
  buffer.dimensions = absl::MakeConstSpan(buf->dims, buf->rank);
  return buffer;
}

template <PrimitiveType dtype, size_t rank>
std::optional<Buffer<dtype, rank>> DecodeBuffer(XLA_FFI_Buffer* buf,
                                                DiagnosticEngine& diagnostic) {
  if (auto buf_dtype = PrimitiveType(buf->dtype);
      XLA_FFI_PREDICT_FALSE(buf_dtype != dtype)) {
    return diagnostic.Emit("Wrong buffer dtype: expected ")
           << primitive_util::LowercasePrimitiveTypeName(dtype) << " but got "
           << primitive_util::LowercasePrimitiveTypeName(buf_dtype);
  }

  if constexpr (rank != internal::kDynamicRank) {
    if (XLA_FFI_PREDICT_FALSE(buf->rank != rank)) {
      return diagnostic.Emit("Wrong buffer rank: expected ")
             << rank << " but got " << buf->rank;
    }
  }

  size_t size_bytes = 0;
  if (primitive_util::IsArrayType(PrimitiveType(buf->dtype))) {
    size_bytes = primitive_util::ByteWidth(PrimitiveType(buf->dtype));
    for (int64_t i = 0; i < buf->rank; ++i) size_bytes *= buf->dims[i];
  }

  Buffer<dtype, rank> buffer;
  buffer.data = se::DeviceMemory<NativeType<dtype>>(
      se::DeviceMemoryBase(buf->data, size_bytes));
  buffer.dimensions = absl::MakeConstSpan(buf->dims, buf->rank);
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

template <PrimitiveType dtype, size_t rank>
struct ArgBinding<Buffer<dtype, rank>> {
  using Arg = Buffer<dtype, rank>;
};

//===----------------------------------------------------------------------===//
// Arguments decoding
//===----------------------------------------------------------------------===//

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

template <PrimitiveType dtype, size_t rank>
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

template <>
struct RetDecoding<BufferBase> {
  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<Result<BufferBase>> Decode(
      XLA_FFI_RetType type, void* arg, DiagnosticEngine& diagnostic) {
    if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_RetType_BUFFER)) {
      return diagnostic.Emit("Wrong result type: expected ")
             << XLA_FFI_RetType_BUFFER << " but got " << type;
    }
    return internal::DecodeBuffer(reinterpret_cast<XLA_FFI_Buffer*>(arg));
  }
};

template <PrimitiveType dtype, size_t rank>
struct RetDecoding<Buffer<dtype, rank>> {
  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<Result<Buffer<dtype, rank>>> Decode(
      XLA_FFI_RetType type, void* arg, DiagnosticEngine& diagnostic) {
    if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_RetType_BUFFER)) {
      return diagnostic.Emit("Wrong result type: expected ")
             << XLA_FFI_RetType_BUFFER << " but got " << type;
    }

    return internal::DecodeBuffer<dtype, rank>(
        reinterpret_cast<XLA_FFI_Buffer*>(arg), diagnostic);
  }
};

//===----------------------------------------------------------------------===//
// Attributes decoding
//===----------------------------------------------------------------------===//

#define XLA_FFI_REGISTER_ARRRAY_ATTR_DECODING(T, TYPE)                   \
  template <>                                                            \
  struct AttrDecoding<absl::Span<const T>> {                             \
    using Type = absl::Span<const T>;                                    \
    static std::optional<Type> Decode(XLA_FFI_AttrType type, void* attr, \
                                      DiagnosticEngine& diagnostic) {    \
      if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_AttrType_ARRAY)) {       \
        return diagnostic.Emit("Wrong attribute type: expected ")        \
               << XLA_FFI_AttrType_ARRAY << " but got " << type;         \
      }                                                                  \
                                                                         \
      auto* array = reinterpret_cast<XLA_FFI_Array*>(attr);              \
      if (XLA_FFI_PREDICT_FALSE(array->dtype != TYPE)) {                 \
        return diagnostic.Emit("Wrong array data type: expected ")       \
               << TYPE << " but got " << array->dtype;                   \
      }                                                                  \
                                                                         \
      return absl::Span<const T>(reinterpret_cast<T*>(array->data),      \
                                 array->size);                           \
    }                                                                    \
  }

XLA_FFI_REGISTER_ARRRAY_ATTR_DECODING(int8_t, XLA_FFI_DataType_S8);
XLA_FFI_REGISTER_ARRRAY_ATTR_DECODING(int16_t, XLA_FFI_DataType_S16);
XLA_FFI_REGISTER_ARRRAY_ATTR_DECODING(int32_t, XLA_FFI_DataType_S32);
XLA_FFI_REGISTER_ARRRAY_ATTR_DECODING(int64_t, XLA_FFI_DataType_S64);
XLA_FFI_REGISTER_ARRRAY_ATTR_DECODING(float, XLA_FFI_DataType_F32);
XLA_FFI_REGISTER_ARRRAY_ATTR_DECODING(double, XLA_FFI_DataType_F64);

#undef XLA_FFI_REGISTER_ARRRAY_ATTR_DECODING

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
// Context decoding
//===----------------------------------------------------------------------===//

template <>
struct CtxDecoding<Stream> {
  using Type = se::Stream*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine&) {
    void* ptr = api->internal_api->XLA_FFI_INTERNAL_Stream_Get(ctx);
    return reinterpret_cast<Type>(ptr);
  }
};

template <>
struct CtxDecoding<ScratchAllocator> {
  using Type = se::OwningScratchAllocator<>;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine&) {
    int32_t device_ordinal =
        api->internal_api->XLA_FFI_INTERNAL_DeviceOrdinal_Get(ctx);
    void* device_allocator =
        api->internal_api->XLA_FFI_INTERNAL_DeviceMemoryAllocator_Get(ctx);

    return se::OwningScratchAllocator<>(
        device_ordinal,
        reinterpret_cast<se::DeviceMemoryAllocator*>(device_allocator));
  }
};

template <>
struct CtxDecoding<CalledComputation> {
  using Type = const HloComputation*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine&) {
    void* ptr = api->internal_api->XLA_FFI_INTERNAL_CalledComputation_Get(ctx);
    return reinterpret_cast<Type>(ptr);
  }
};

//===----------------------------------------------------------------------===//
// Result encoding
//===----------------------------------------------------------------------===//

template <>
struct ResultEncoding<Status> {
  static XLA_FFI_Error* Encode(const XLA_FFI_Api* api, Status status) {
    return api->internal_api->XLA_FFI_INTERNAL_Error_Forward(&status);
  }
};

}  // namespace xla::ffi

#endif  // XLA_FFI_FFI_H_

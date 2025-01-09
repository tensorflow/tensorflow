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
#error Two different XLA FFI implementations cannot be included together. \
       See README.md for more details.
#endif  // XLA_FFI_API_FFI_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

// IWYU pragma: begin_exports
#include "xla/ffi/api/api.h"
// IWYU pragma: end_exports

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/ffi/execution_context.h"
#include "xla/ffi/execution_state.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"

namespace xla::ffi {

// Type tags to bind parameters passed via execution context to FFI handler.
struct Stream {};             // binds `se::Stream*`
struct DeviceOrdinal {};      // binds `int32_t` with device ordinal
struct Allocator {};          // binds `se::DeviceMemoryAllocator*`
struct ScratchAllocator {};   // binds `se::OwningScratchAllocator`
struct CalledComputation {};  // binds `HloComputation*`
struct IntraOpThreadPool {};  // binds `const Eigen::ThreadPoolDevice*`

template <typename T>
struct PlatformStream {};  // binds a platform stream, e.g. `cudaStream_t`

//===----------------------------------------------------------------------===//
// Arguments
//===----------------------------------------------------------------------===//

namespace internal {

inline constexpr size_t kDynamicRank = std::numeric_limits<size_t>::max();

template <PrimitiveType dtype>
using NativeType = typename primitive_util::PrimitiveTypeToNative<dtype>::type;

}  // namespace internal

// Dynamically-typed buffer.
//
// No checks are done at decoding time. Any dtype and rank combination is
// accepted.
class AnyBuffer {
 public:
  using Dimensions = absl::Span<const int64_t>;

  explicit AnyBuffer(absl::Nonnull<const XLA_FFI_Buffer*> buf) : buf_(buf) {
    DCHECK(buf_ != nullptr) << "XLA_FFI_Buffer must be non-null";
  }

  PrimitiveType element_type() const { return PrimitiveType(buf_->dtype); }

  Dimensions dimensions() const { return Dimensions(buf_->dims, buf_->rank); }

  ABSL_ATTRIBUTE_ALWAYS_INLINE size_t size_bytes() const {
    if (ABSL_PREDICT_TRUE(primitive_util::IsArrayType(element_type()))) {
      return primitive_util::ByteWidth(element_type()) * element_count();
    }
    return 0;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE size_t element_count() const {
    return absl::c_accumulate(dimensions(), int64_t{1}, std::multiplies<>());
  }

  void* untyped_data() const { return buf_->data; }

  template <typename T>
  T* typed_data() const {
    DCHECK(primitive_util::NativeToPrimitiveType<T>() == element_type())
        << "Template type must match the underlying buffer dtype";
    return reinterpret_cast<T*>(buf_->data);
  }

  template <typename T>
  T* reinterpret_data() const {
    DCHECK(primitive_util::IsArrayType(element_type()) &&
           sizeof(T) == primitive_util::ByteWidth(element_type()) &&
           !(reinterpret_cast<std::uintptr_t>(buf_->data) % alignof(T)))
        << "Requested type must have the same byte width and alignment as the "
           "underlying buffer type";
    return reinterpret_cast<T*>(buf_->data);
  }

  se::DeviceMemoryBase device_memory() const {
    return se::DeviceMemoryBase(untyped_data(), size_bytes());
  }

 private:
  const XLA_FFI_Buffer* buf_;
};

// Buffer with a statically-known dtype and rank.
//
// The dtype and rank are checked at decoding time. If rank is not specified,
// any rank is accepted.
template <PrimitiveType dtype, size_t rank = internal::kDynamicRank>
class Buffer {
 public:
  using Dimensions = AnyBuffer::Dimensions;

  explicit Buffer(absl::Nonnull<const XLA_FFI_Buffer*> buf) : buf_(buf) {
    DCHECK(buf_ != nullptr) << "XLA_FFI_Buffer must be non-null";
  }

  PrimitiveType element_type() const { return dtype; }

  Dimensions dimensions() const {
    return Dimensions(buf_->dims,
                      rank == internal::kDynamicRank ? buf_->rank : rank);
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE size_t size_bytes() const {
    if constexpr (primitive_util::IsArrayType(dtype)) {
      return primitive_util::ByteWidth(dtype) * element_count();
    }
    return 0;
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE size_t element_count() const {
    return absl::c_accumulate(dimensions(), int64_t{1}, std::multiplies<>());
  }

  void* untyped_data() const { return buf_->data; }

  internal::NativeType<dtype>* typed_data() const {
    return reinterpret_cast<internal::NativeType<dtype>*>(untyped_data());
  }

  se::DeviceMemory<internal::NativeType<dtype>> device_memory() const {
    return se::DeviceMemory<internal::NativeType<dtype>>(
        se::DeviceMemoryBase(untyped_data(), size_bytes()));
  }

 private:
  const XLA_FFI_Buffer* buf_;
};

// clang-format off
template <PrimitiveType dtype> using BufferR0 = Buffer<dtype, 0>;
template <PrimitiveType dtype> using BufferR1 = Buffer<dtype, 1>;
template <PrimitiveType dtype> using BufferR2 = Buffer<dtype, 2>;
template <PrimitiveType dtype> using BufferR3 = Buffer<dtype, 3>;
template <PrimitiveType dtype> using BufferR4 = Buffer<dtype, 4>;
// clang-format on

using Token = BufferR0<PrimitiveType::TOKEN>;  // NOLINT

namespace internal {

template <PrimitiveType dtype, size_t rank>
ABSL_ATTRIBUTE_ALWAYS_INLINE std::optional<Buffer<dtype, rank>> DecodeBuffer(
    XLA_FFI_Buffer* buf, DiagnosticEngine& diagnostic) {
  if (auto buf_dtype = PrimitiveType(buf->dtype);
      ABSL_PREDICT_FALSE(buf_dtype != dtype)) {
    return diagnostic.Emit("Wrong buffer dtype: expected ")
           << primitive_util::LowercasePrimitiveTypeName(dtype) << " but got "
           << primitive_util::LowercasePrimitiveTypeName(buf_dtype);
  }

  if constexpr (rank != internal::kDynamicRank) {
    if (ABSL_PREDICT_FALSE(buf->rank != rank)) {
      return diagnostic.Emit("Wrong buffer rank: expected ")
             << rank << " but got " << buf->rank;
    }
  }

  return Buffer<dtype, rank>(buf);
}

}  // namespace internal

//===----------------------------------------------------------------------===//
// Arguments binding
//===----------------------------------------------------------------------===//

template <>
struct ArgBinding<AnyBuffer> {
  using Arg = AnyBuffer;
};

template <PrimitiveType dtype, size_t rank>
struct ArgBinding<Buffer<dtype, rank>> {
  using Arg = Buffer<dtype, rank>;
};

//===----------------------------------------------------------------------===//
// Arguments decoding
//===----------------------------------------------------------------------===//

template <>
struct ArgDecoding<AnyBuffer> {
  ABSL_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<AnyBuffer> Decode(XLA_FFI_ArgType type, void* arg,
                                         DiagnosticEngine& diagnostic) {
    if (ABSL_PREDICT_FALSE(type != XLA_FFI_ArgType_BUFFER)) {
      return diagnostic.Emit("Wrong argument type: expected ")
             << XLA_FFI_ArgType_BUFFER << " but got " << type;
    }

    return AnyBuffer(reinterpret_cast<XLA_FFI_Buffer*>(arg));
  }
};

template <PrimitiveType dtype, size_t rank>
struct ArgDecoding<Buffer<dtype, rank>> {
  ABSL_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<Buffer<dtype, rank>> Decode(
      XLA_FFI_ArgType type, void* arg, DiagnosticEngine& diagnostic) {
    if (ABSL_PREDICT_FALSE(type != XLA_FFI_ArgType_BUFFER)) {
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
  absl::StatusOr<T> get(size_t index) const {
    size_t idx = offset() + index;
    if (ABSL_PREDICT_FALSE(idx >= args()->size)) {
      return InvalidArgument("Index out of range.");
    }

    DiagnosticEngine diagnostic;
    std::optional<T> value = ArgDecoding<T>::Decode(
        args()->types[idx], args()->args[idx], diagnostic);
    if (ABSL_PREDICT_FALSE(!value.has_value())) {
      return Internal("%s", diagnostic.Result());
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

template <>
struct RetDecoding<AnyBuffer> {
  ABSL_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<Result<AnyBuffer>> Decode(XLA_FFI_RetType type,
                                                 void* arg,
                                                 DiagnosticEngine& diagnostic) {
    if (ABSL_PREDICT_FALSE(type != XLA_FFI_RetType_BUFFER)) {
      return diagnostic.Emit("Wrong result type: expected ")
             << XLA_FFI_RetType_BUFFER << " but got " << type;
    }
    return AnyBuffer(reinterpret_cast<XLA_FFI_Buffer*>(arg));
  }
};

template <PrimitiveType dtype, size_t rank>
struct RetDecoding<Buffer<dtype, rank>> {
  ABSL_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<Result<Buffer<dtype, rank>>> Decode(
      XLA_FFI_RetType type, void* arg, DiagnosticEngine& diagnostic) {
    if (ABSL_PREDICT_FALSE(type != XLA_FFI_RetType_BUFFER)) {
      return diagnostic.Emit("Wrong result type: expected ")
             << XLA_FFI_RetType_BUFFER << " but got " << type;
    }

    return internal::DecodeBuffer<dtype, rank>(
        reinterpret_cast<XLA_FFI_Buffer*>(arg), diagnostic);
  }
};

//===----------------------------------------------------------------------===//
// Type-safe wrapper for accessing a variable number of results.
//===----------------------------------------------------------------------===//

class RemainingRets : public internal::RemainingRetsBase {
 public:
  using internal::RemainingRetsBase::RemainingRetsBase;

  template <typename T>
  absl::StatusOr<Result<T>> get(size_t index) const {
    size_t idx = offset() + index;
    if (ABSL_PREDICT_FALSE(idx >= rets()->size)) {
      return InvalidArgument("Index out of range.");
    }

    DiagnosticEngine diagnostic;
    std::optional<Result<T>> value = RetDecoding<T>::Decode(
        rets()->types[idx], rets()->rets[idx], diagnostic);
    if (ABSL_PREDICT_FALSE(!value.has_value())) {
      return Internal("%s", diagnostic.Result());
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

#define XLA_FFI_REGISTER_ARRRAY_ATTR_DECODING(T, TYPE)                   \
  template <>                                                            \
  struct AttrDecoding<absl::Span<const T>> {                             \
    using Type = absl::Span<const T>;                                    \
    static std::optional<Type> Decode(XLA_FFI_AttrType type, void* attr, \
                                      DiagnosticEngine& diagnostic) {    \
      if (ABSL_PREDICT_FALSE(type != XLA_FFI_AttrType_ARRAY)) {          \
        return diagnostic.Emit("Wrong attribute type: expected ")        \
               << XLA_FFI_AttrType_ARRAY << " but got " << type;         \
      }                                                                  \
                                                                         \
      auto* array = reinterpret_cast<XLA_FFI_Array*>(attr);              \
      if (ABSL_PREDICT_FALSE(array->dtype != TYPE)) {                    \
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

template <>
struct AttrDecoding<absl::string_view> {
  using Type = absl::string_view;
  static std::optional<std::string_view> Decode(XLA_FFI_AttrType type,
                                                void* attr,
                                                DiagnosticEngine& diagnostic) {
    if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_AttrType_STRING)) {
      return diagnostic.Emit("Wrong attribute type: expected ")
             << XLA_FFI_AttrType_STRING << " but got " << type;
    }

    auto* span = reinterpret_cast<XLA_FFI_ByteSpan*>(attr);
    return std::string_view(span->ptr, span->len);
  }
};

// A type tag to mark i64 attributes as pointers to `T`.
template <typename T>
struct Pointer {};

template <typename T>
struct AttrDecoding<Pointer<T>> {
  using Type = T*;

  static std::optional<Type> Decode(XLA_FFI_AttrType type, void* attr,
                                    DiagnosticEngine& diagnostic) {
    auto* scalar = reinterpret_cast<XLA_FFI_Scalar*>(attr);
    if (ABSL_PREDICT_FALSE(type != XLA_FFI_AttrType_SCALAR ||
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
  absl::StatusOr<T> get(std::string_view name) const {
    DiagnosticEngine diagnostic;
    std::optional<T> value = internal::DictionaryBase::get<T>(name, diagnostic);
    if (!value.has_value()) {
      return Internal("%s", diagnostic.Result());
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
// Context decoding
//===----------------------------------------------------------------------===//

template <>
struct CtxDecoding<Stream> {
  using Type = se::Stream*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    void* ptr = api->internal_api->XLA_FFI_INTERNAL_Stream_Get(ctx);
    if (ABSL_PREDICT_FALSE(ptr == nullptr)) {
      return diagnostic.Emit("Failed to decode stream");
    }
    return reinterpret_cast<Type>(ptr);
  }
};

template <>
struct CtxDecoding<DeviceOrdinal> {
  using Type = int32_t;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine&) {
    return api->internal_api->XLA_FFI_INTERNAL_DeviceOrdinal_Get(ctx);
  }
};

template <>
struct CtxDecoding<Allocator> {
  using Type = se::DeviceMemoryAllocator*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine&) {
    void* device_allocator =
        api->internal_api->XLA_FFI_INTERNAL_DeviceMemoryAllocator_Get(ctx);
    return reinterpret_cast<Type>(device_allocator);
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

template <>
struct CtxDecoding<IntraOpThreadPool> {
  using Type = const Eigen::ThreadPoolDevice*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine&) {
    void* intra_op_thread_pool =
        api->internal_api->XLA_FFI_INTERNAL_IntraOpThreadPool_Get(ctx);
    return reinterpret_cast<Type>(intra_op_thread_pool);
  }
};

template <typename T>
struct CtxDecoding<PlatformStream<T>> {
  using Type = T;
  static_assert(std::is_pointer_v<T>, "platform stream type must be a pointer");

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    if (auto stream = CtxDecoding<Stream>::Decode(api, ctx, diagnostic)) {
      return reinterpret_cast<Type>(
          stream.value()->platform_specific_handle().stream);
    }
    return std::nullopt;
  }
};

//===----------------------------------------------------------------------===//
// UserData
//===----------------------------------------------------------------------===//

// A type tag for automatic user data decoding passed via the execution context.
template <typename T>
struct UserData {};

template <typename T>
struct CtxDecoding<UserData<T>> {
  using Type = T*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    auto* execution_context = reinterpret_cast<const ExecutionContext*>(
        api->internal_api->XLA_FFI_INTERNAL_ExecutionContext_Get(ctx));

    if (execution_context == nullptr) {
      return diagnostic.Emit(
          "Execution context must be not null to fetch UserData parameter");
    }

    auto user_data = execution_context->Lookup<T>();
    if (!user_data.ok()) {
      return diagnostic.Emit("Failed to get user data from execution context: ")
             << user_data.status().message();
    }

    return *std::move(user_data);
  }
};

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

// A type tag for automatic state decoding passed via the execution context.
template <typename T>
struct State {};

template <typename T>
struct CtxDecoding<State<T>> {
  using Type = T*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    auto* execution_state = reinterpret_cast<const ExecutionState*>(
        api->internal_api->XLA_FFI_INTERNAL_ExecutionState_Get(ctx));

    if (execution_state == nullptr) {
      return diagnostic.Emit(
          "Execution state must be not null to fetch State parameter");
    }

    auto state = execution_state->Get<T>();
    if (!state.ok()) {
      return diagnostic.Emit("Failed to get state from execution context: ")
             << state.status().message();
    }

    return *std::move(state);
  }
};

//===----------------------------------------------------------------------===//
// Result encoding
//===----------------------------------------------------------------------===//

template <ExecutionStage stage>
struct ResultEncoding<stage, absl::Status> {
  static XLA_FFI_Error* Encode(const XLA_FFI_Api* api,
                               XLA_FFI_ExecutionContext* ctx,
                               absl::Status status) {
    if (ABSL_PREDICT_TRUE(status.ok())) {
      return nullptr;
    }
    return api->internal_api->XLA_FFI_INTERNAL_Error_Forward(&status);
  }
};

template <typename T>
struct ResultEncoding<ExecutionStage::kInstantiate,
                      absl::StatusOr<std::unique_ptr<T>>> {
  static XLA_FFI_Error* Encode(const XLA_FFI_Api* api,
                               XLA_FFI_ExecutionContext* ctx,
                               absl::StatusOr<std::unique_ptr<T>> state) {
    if (ABSL_PREDICT_TRUE(state.ok())) {
      auto* execution_state = reinterpret_cast<ExecutionState*>(
          api->internal_api->XLA_FFI_INTERNAL_ExecutionState_Get(ctx));
      absl::Status status = execution_state->Set<T>(*std::move(state));
      if (ABSL_PREDICT_TRUE(status.ok())) {
        return nullptr;
      }
      return api->internal_api->XLA_FFI_INTERNAL_Error_Forward(&status);
    }

    absl::Status status = state.status();
    return api->internal_api->XLA_FFI_INTERNAL_Error_Forward(&status);
  }
};

template <ExecutionStage stage>
struct ResultEncoding<stage, tsl::AsyncValueRef<tsl::Chain>> {
  static XLA_FFI_Future* Encode(const XLA_FFI_Api* api,
                                XLA_FFI_ExecutionContext* ctx,
                                tsl::AsyncValueRef<tsl::Chain> async_value) {
    return api->internal_api->XLA_FFI_INTERNAL_Future_Forward(
        async_value.release());
  }
};

}  // namespace xla::ffi

#endif  // XLA_FFI_FFI_H_

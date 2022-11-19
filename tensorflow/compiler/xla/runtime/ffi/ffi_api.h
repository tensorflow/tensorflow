/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_FFI_FFI_API_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_FFI_FFI_API_H_

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/runtime/ffi/ffi_c_api.h"

namespace xla {
namespace runtime {
namespace ffi {

using TypeId = XLA_FFI_TypeId;

using Error = XLA_FFI_Error;
using ErrorCode = XLA_FFI_Error_Code;
using ExecutionContext = XLA_FFI_ExecutionContext;

// Forward declare template defined below.
template <typename... Ts>
class FfiBinding;

// Forward declare template defined below.
template <typename Fn, typename... Ts>
class FfiHandler;

// FFI arguments allocated by the jit-compiled code and we need to mark all
// memory initialized to suppress memory sanitizer errors.
#if defined(MEMORY_SANITIZER)
#define XLA_FFI_ANNOTATE_MEMORY_IS_INITIALIZED(address, size) \
  __msan_unpoison(address, size)
#else
#define XLA_FFI_ANNOTATE_MEMORY_IS_INITIALIZED(address, size)  // empty
#endif  // MEMORY_SANITIZER

//===----------------------------------------------------------------------===//
// XLA FFI status wrapper around error reporting APIs.
//===----------------------------------------------------------------------===//

class FfiStatus {
 public:
  static FfiStatus Ok() { return FfiStatus(); }

  static FfiStatus Internal(std::string message) {
    ErrorCode errc = XLA_FFI_Error_Code_INTERNAL;
    return FfiStatus(errc, message);
  }

  static FfiStatus InvalidArgument(std::string message) {
    ErrorCode errc = XLA_FFI_Error_Code_INVALID_ARGUMENT;
    return FfiStatus(errc, message);
  }

  std::optional<ErrorCode> errc() const { return errc_; }

  std::string_view message() const {
    return message_.has_value() ? *message_ : std::string_view();
  }

  const char* message_c_str() const {
    return message_.has_value() ? message_->c_str() : "";
  }

 private:
  FfiStatus() = default;

  FfiStatus(ErrorCode errc, std::string message)
      : errc_(errc), message_(std::move(message)) {}

  std::optional<ErrorCode> errc_;
  std::optional<std::string> message_;
};

//===----------------------------------------------------------------------===//
// XLA FFI virtual base for implementing FFI handlers.
//===----------------------------------------------------------------------===//

class Ffi {
 public:
  virtual ~Ffi() = default;

  virtual std::string_view name() const = 0;
  virtual XLA_FFI_Error* operator()(ExecutionContext* ctx, void** args,
                                    void** attrs, void** rets) const = 0;

  static FfiBinding<> Bind(std::string name);

  static void Register(const XLA_FFI_Api* api, const std::string& target,
                       XLA_FFI_Function function) {
    XLA_FFI_Register_Args args;
    args.struct_size = XLA_FFI_Register_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.target = target.data();
    args.function = function;

    api->XLA_FFI_Register(&args);
  }

  template <typename T>
  static bool Isa(ExecutionContext* ctx, TypeId type_id);

  template <typename T, typename U, typename... Ts>
  static bool Isa(ExecutionContext* ctx, TypeId type_id) {
    return Isa<T>(ctx, type_id) || Isa<U, Ts...>(ctx, type_id);
  }
};

//===----------------------------------------------------------------------===//
// Arguments supported by the FFI handlers.
//===----------------------------------------------------------------------===//

// This enum corresponds to xla::PrimitiveType enum defined in `hlo.proto`.
enum class PrimitiveType : uint8_t {
  // Invalid primitive type to serve as default.
  PRIMITIVE_TYPE_INVALID = 0,

  // Predicates are two-state booleans.
  PRED = 1,

  // Signed integral values of fixed width.
  S8 = 2,
  S16 = 3,
  S32 = 4,
  S64 = 5,

  // Unsigned integral values of fixed width.
  U8 = 6,
  U16 = 7,
  U32 = 8,
  U64 = 9,

  // Floating-point values of fixed width.
  //
  // Note: if f16s are not natively supported on the device, they will be
  // converted to f16 from f32 at arbitrary points in the computation.
  F16 = 10,
  F32 = 11,

  // Truncated 16 bit floating-point format. This is similar to IEEE's 16 bit
  // floating-point format, but uses 1 bit for the sign, 8 bits for the exponent
  // and 7 bits for the mantissa.
  BF16 = 16,

  F64 = 12,
};

constexpr std::string_view PrimitiveTypeToString(PrimitiveType type) {
  switch (type) {
    case PrimitiveType::PRIMITIVE_TYPE_INVALID:
      return "invalid";
    case PrimitiveType::PRED:
      return "pred";
    case PrimitiveType::S8:
      return "s8";
    case PrimitiveType::S16:
      return "s16";
    case PrimitiveType::S32:
      return "s32";
    case PrimitiveType::S64:
      return "s64";
    case PrimitiveType::U8:
      return "u8";
    case PrimitiveType::U16:
      return "u16";
    case PrimitiveType::U32:
      return "u32";
    case PrimitiveType::U64:
      return "u64";
    case PrimitiveType::F16:
      return "f16";
    case PrimitiveType::F32:
      return "f32";
    case PrimitiveType::BF16:
      return "bf16";
    case PrimitiveType::F64:
      return "f64";
  }
}

// TODO(ezhulenev): Replace with `std::span` when C++20 is available.
template <typename T>
class Span {
 public:
  Span(T* data, size_t size) : data_(data), size_(size) {}
  T& operator[](size_t index) const { return data_[index]; }

  size_t size() const { return size_; }

 private:
  T* data_;
  size_t size_;
};

// A view into the buffer argument. Buffers with non-identity layouts can be
// decoded only as a StridedBufferArg.
struct StridedBufferArg {
  std::string ToString() const;

  PrimitiveType dtype;
  void* data;
  Span<const int64_t> sizes;
  Span<const int64_t> strides;
};

// A view into the buffer argument with an identity (row major) layout.
struct BufferArg {
  std::string ToString() const;

  PrimitiveType dtype;
  void* data;
  Span<const int64_t> sizes;
};

template <typename T>
bool Ffi::Isa(ExecutionContext* ctx, TypeId type_id) {
  if constexpr (std::is_same_v<T, float>)
    return ctx->XLA_FFI_Get_Float_TypeId() == type_id;
  else if constexpr (std::is_same_v<T, int32_t>)
    return ctx->XLA_FFI_Get_Int32_TypeId() == type_id;
  else if constexpr (std::is_same_v<T, StridedBufferArg>)
    return ctx->XLA_FFI_Get_StridedBufferArg_TypeId() == type_id;
  else if constexpr (std::is_same_v<T, BufferArg>)
    return ctx->XLA_FFI_Get_BufferArg_TypeId() == type_id;
  else
    // Static assert has to be type-dependent, and `!sizeof` is just one of the
    // ways to always produce `false`.
    static_assert(!sizeof(T), "Unsupported type");
}

//===----------------------------------------------------------------------===//
// Pretty printing for buffers.
//===----------------------------------------------------------------------===//

static void PrintArray(std::stringstream& ss, Span<const int64_t> arr) {
  ss << "[";
  for (unsigned i = 0; i < arr.size(); ++i)
    (i > 0) ? ss << ", " << arr[i] : ss << arr[i];
  ss << "]";
}

inline std::string StridedBufferArg::ToString() const {
  std::stringstream ss;
  ss << "Buffer: dtype=" << PrimitiveTypeToString(dtype);
  ss << " sizes=";
  PrintArray(ss, sizes);
  ss << " strides=";
  PrintArray(ss, strides);
  return ss.str();
}

inline std::string BufferArg::ToString() const {
  std::stringstream ss;
  ss << "Buffer: dtype=" << PrimitiveTypeToString(dtype);
  ss << " sizes=";
  PrintArray(ss, sizes);
  return ss.str();
}

//===----------------------------------------------------------------------===//
// FFI binding describes the function signature expected by the FFI handler
// using its variadic template parameter.
//
//   FFI binding:
//     FfiBinding<int32_t, float>
//
//   Corresponds to the function signature:
//     FfiStatus MyHandler(int32_t arg0, float arg1);
//
//===----------------------------------------------------------------------===//

namespace internal {

// A type tag to distinguish arguments tied to the attributes in the
// `FfiBinding` variadic template argument.
template <typename T>
struct Attr {};

// A template for checking if type is a wrapped attribute or user data.
// clang-format off
template <typename>   struct IsWrapped : std::false_type {};
template <typename T> struct IsWrapped<Attr<T>> : std::true_type {};
// clang-format on

}  // namespace internal

template <typename... Ts>
class FfiBinding {
 public:
  template <typename T>
  FfiBinding<Ts..., T> Arg() && {
    return {std::move(*this)};
  }

  template <typename T>
  FfiBinding<Ts..., internal::Attr<T>> Attr(std::string attr) && {
    attrs_.push_back(std::move(attr));
    return {std::move(*this)};
  }

  template <typename Fn>
  std::unique_ptr<FfiHandler<Fn, Ts...>> To(Fn fn) {
    return std::unique_ptr<FfiHandler<Fn, Ts...>>(new FfiHandler<Fn, Ts...>(
        std::forward<Fn>(fn), std::move(name_), std::move(attrs_)));
  }

 private:
  template <typename...>
  friend class FfiBinding;
  friend class Ffi;

  explicit FfiBinding(std::string name) : name_(std::move(name)) {
    static_assert(sizeof...(Ts) == 0, "ffi arguments must be empty");
  }

  template <typename... TTs>
  FfiBinding(FfiBinding<TTs...>&& other)  // NOLINT
      : name_(std::move(other.name_)), attrs_(std::move(other.attrs_)) {}

  FfiBinding(FfiBinding&) = delete;

  std::string name_;                // ffi name
  std::vector<std::string> attrs_;  // names of bound attributes
};

inline FfiBinding<> Ffi::Bind(std::string name) {
  return FfiBinding<>(std::move(name));
}

//===----------------------------------------------------------------------===//
// C structures that XLA FFI uses internally to encode arguments and attributes.
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Structures used for encoding must be shared between FFI and
// custom calls because it's our ABI boundary.

namespace internal {

struct EncodedMemref {
  uint8_t dtype;
  uint8_t rank;
  void* data;
  int64_t dims[];
};

template <typename T>
struct EncodedArray {
  int64_t size;
  const T* data;
};

}  // namespace internal

//===----------------------------------------------------------------------===//
// Helpers for decoding opaque arguments and attributes' memory.
//===----------------------------------------------------------------------===//

namespace internal {

// Decoded pair of argument type and opaque value.
struct DecodedArg {
  XLA_FFI_TypeId type_id;
  void* value;
};

// Decoded triple of attribute name, type and opaque value.
struct DecodedAttr {
  std::string_view name;
  XLA_FFI_TypeId type_id;
  void* value;
};

// A convenience wrapper around opaque arguments memory.
class DecodedArgs {
 public:
  explicit DecodedArgs(void** args) {
    XLA_FFI_ANNOTATE_MEMORY_IS_INITIALIZED(args, sizeof(void*));
    size_ = *reinterpret_cast<int64_t*>(args[0]);
    if (size_) {
      XLA_FFI_ANNOTATE_MEMORY_IS_INITIALIZED(args + 1, sizeof(void*));
      types_ = reinterpret_cast<void**>(args[1]);
      values_ = args + 2;
      XLA_FFI_ANNOTATE_MEMORY_IS_INITIALIZED(types_, size_ * sizeof(void*));
      XLA_FFI_ANNOTATE_MEMORY_IS_INITIALIZED(values_, size_ * sizeof(void*));
    }
  }

  int64_t size() const { return size_; }

  DecodedArg operator[](size_t i) const {
    DecodedArg arg;
    arg.type_id = types_[i];
    arg.value = values_[i];
    return arg;
  }

 private:
  int64_t size_;
  void** types_ = nullptr;
  void** values_ = nullptr;
};

// A convenience wrapper around opaque attributes' memory.
class DecodedAttrs {
 public:
  explicit DecodedAttrs(void** attrs) : encoded_(attrs + 1) {
    XLA_FFI_ANNOTATE_MEMORY_IS_INITIALIZED(attrs, sizeof(void*));
    size_ = *reinterpret_cast<int64_t*>(attrs[0]);
    XLA_FFI_ANNOTATE_MEMORY_IS_INITIALIZED(encoded_, 3 * size_ * sizeof(void*));
  }

  int64_t size() const { return size_; }

  DecodedAttr operator[](size_t i) const {
    void** attr_base = encoded_ + i * 3;

    DecodedAttr attr;
    auto* name = reinterpret_cast<internal::EncodedArray<char>*>(attr_base[0]);
    attr.name = std::string_view(name->data, name->size);
    attr.type_id = attr_base[1];
    attr.value = attr_base[2];

    return attr;
  }

 private:
  void** encoded_;
  int64_t size_;
};

}  // namespace internal

//===----------------------------------------------------------------------===//
// XLA FFI arguments decoding implementation.
//===----------------------------------------------------------------------===//

// XLA FFI arguments decoding must be defined by specializing this template.
//
// Example: decoding for the `MyType` arguments
//
//   template <>
//   struct FfiArgDecoding<MyType> {
//    static std::optional<MyType> Decode(ExecutionContext* ctx, TypeId type_id,
//                                        void* value);
//   };
//
template <typename T>
struct FfiArgDecoding;

// XLA FFI attribute decoding must be defined by specializing this template.
//
// Example: decoding for the `MyType` attributes
//
//   template <>
//   struct FfiAttrDecoding<MyType> {
//    static std::optional<MyType> Decode(ExecutionContext* ctx,
//                                        std::string_view name,
//                                        TypeId type_id, void* value);
//   }
//
template <typename T>
struct FfiAttrDecoding;

namespace internal {

// When decoding input data we need to keep track of how many arguments,
// attributes, and returns we decoded so far to index into the correct data
// strucuture.
struct DecodingOffsets {
  int64_t args = 0;
  int64_t attrs = 0;
  int64_t rets = 0;
  int64_t values = 0;
};

template <typename T>
struct Decode {
  static std::optional<T> call(ExecutionContext* ctx, DecodingOffsets& offsets,
                               internal::DecodedArgs args,
                               const std::vector<std::string>& attrs_names,
                               const std::vector<size_t>& attrs_idx,
                               internal::DecodedAttrs attrs) {
    internal::DecodedArg arg = args[offsets.args++];
    return FfiArgDecoding<T>::Decode(ctx, arg.type_id, arg.value);
  }
};

template <typename T>
struct Decode<Attr<T>> {
  static std::optional<T> call(ExecutionContext* ctx, DecodingOffsets& offsets,
                               internal::DecodedArgs args,
                               const std::vector<std::string>& attrs_names,
                               const std::vector<size_t>& attrs_idx,
                               internal::DecodedAttrs attrs) {
    // Find decoded attribute corresponding to the given attribute index.
    int64_t idx = offsets.attrs++;

    // Get mapping from the attribute to its index in the sorted array.
    size_t i = attrs_idx[idx];

    // Attribute name does not match.
    if (attrs[i].name != attrs_names[idx]) return std::nullopt;

    return FfiAttrDecoding<T>::Decode(ctx, attrs[i].name, attrs[i].type_id,
                                      attrs[i].value);
  }
};

}  // namespace internal

//===----------------------------------------------------------------------===//
// Ffi handler binds concrete ffi implementation of type `Fn` to the ffi
// function signature. `Fn` can be a function pointer or a lambda.
//
// Ffi handler uses the variadic template parameter `Ts` to decode the
// opaque pointers passed to the `call` function into the C++ types that are
// forwarded to the ffi implementation.
//===----------------------------------------------------------------------===//

namespace internal {

// A helper template to extract the type of the handler argument.
// clang-format off
template <typename T> struct FnArgType          { using Type = T; };
template <typename T> struct FnArgType<Attr<T>> { using Type = T; };
// clang-format on

// A template for counting regular arguments in the Ts pack.
template <typename... Ts>
struct NumArgs;
template <>
struct NumArgs<> {
  static constexpr int64_t value = 0;
};

template <typename T, typename... Ts>
struct NumArgs<T, Ts...> {
  static constexpr int64_t value = !IsWrapped<T>::value + NumArgs<Ts...>::value;
};

}  // namespace internal

template <typename Fn, typename... Ts>
class FfiHandler : public Ffi {
  static constexpr int64_t kSize = sizeof...(Ts);
  static constexpr int64_t kNumArgs = internal::NumArgs<Ts...>::value;

  template <typename T>
  using FnArgType = typename internal::FnArgType<T>::Type;

  // Check if FFI function returns `FfiStatus`.
  static constexpr bool kIsFfiStatusHandler =
      std::is_invocable_r_v<FfiStatus, Fn, FnArgType<Ts>...>;
  static_assert(kIsFfiStatusHandler, "unsupported FFI handler type");

  static Error* ToError(ExecutionContext* ctx, FfiStatus status) {
    if (!status.errc().has_value()) return nullptr;

    XLA_FFI_Error_Create_Args args;
    args.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.errc = *status.errc();
    args.message = status.message_c_str();

    return ctx->XLA_FFI_Error_Create(&args);
  }

 public:
  std::string_view name() const final { return name_; }

  Error* operator()(ExecutionContext* ctx, void** args, void** attrs,
                    void** rets) const final {
    // Decode arguments and attributes from the opaque pointers.
    internal::DecodedArgs decoded_args(args);
    internal::DecodedAttrs decoded_attrs(attrs);

    int64_t num_args = decoded_args.size();
    int64_t num_attrs = decoded_attrs.size();

    // Check that we have the correct number of arguments passed to the handler.
    if (num_args != kNumArgs) {
      std::ostringstream err;
      err << "Wrong number of arguments: expected " << kNumArgs << " got "
          << num_args;
      return ToError(ctx, FfiStatus::InvalidArgument(err.str()));
    }

    // Check that we have the correct number of attributes passed to the
    // handler. Each individual attribute decoding will check the name and the
    // type of the attribute.
    if (num_attrs != attrs_.size()) {
      std::ostringstream err;
      err << "Wrong number of attributes: expected " << attrs_.size() << " got "
          << num_attrs;
      return ToError(ctx, FfiStatus::InvalidArgument(err.str()));
    }

    // Define index sequence to access ffi handler arguments.
    using Is = std::make_index_sequence<kSize>;
    return call(ctx, decoded_args, decoded_attrs, Is{});
  }

 private:
  template <typename...>
  friend class FfiBinding;

  template <size_t... Is>
  Error* call(ExecutionContext* ctx, internal::DecodedArgs args,
              internal::DecodedAttrs attrs, std::index_sequence<Is...>) const {
    // A helper structure to allow each decoder find the correct offset in the
    // arguments, attributes or results.
    internal::DecodingOffsets offsets;

    // Decode all operands into `std::optional` containers. It is guaranteed
    // that initializer list will be evaluated left-to-right, and we can rely
    // on correct offsets computation.
    std::tuple<std::optional<FnArgType<Ts>>...> fn_args = {
        internal::Decode<Ts>::call(ctx, offsets, args, attrs_, attrs_idx_,
                                   attrs)...};

    // Check if all arguments, attributes and results were decoded;
    bool all_decoded = (std::get<Is>(fn_args).has_value() && ...);
    if (!all_decoded) {
      return ToError(
          ctx, FfiStatus::InvalidArgument("Failed to decode all FFI operands"));
    }

    // Custom call returns `FfiStatus`, we can call it directly.
    if constexpr (kIsFfiStatusHandler) {
      return ToError(ctx, fn_(std::move(*std::get<Is>(fn_args))...));
    }

    return ToError(ctx, FfiStatus::Ok());
  }

  FfiHandler(Fn fn, std::string name, std::vector<std::string> attrs)
      : fn_(std::move(fn)),
        name_(std::move(name)),
        attrs_(std::move(attrs)),
        attrs_idx_(attrs_.size()) {
    // Sort attributes names.
    std::vector<std::string> sorted = attrs_;
    std::sort(sorted.begin(), sorted.end());

    // Find the index of every attribute in the sorted attributes vector.
    for (size_t i = 0; i < attrs_.size(); ++i) {
      const std::string& attr = attrs_[i];
      attrs_idx_[i] = std::distance(
          sorted.begin(), std::find(sorted.begin(), sorted.end(), attr));
    }
  }

  Fn fn_;

  std::string name_;
  std::vector<std::string> attrs_;

  // A mapping from the attribute index to its index in the lexicographically
  // sorted vector of attribute names. Attributes are passed to the ffi handler
  // sorted by the name, we use this index to efficiently find the decoded
  // attribute entry.
  std::vector<size_t> attrs_idx_;
};

//===----------------------------------------------------------------------===//
// XLA FFI arguments decoding.
//===----------------------------------------------------------------------===//

#define XLA_FFI_REGISTER_SCALAR_ARG_DECODING(T)                           \
  template <>                                                             \
  struct FfiArgDecoding<T> {                                              \
    static std::optional<T> Decode(ExecutionContext* ctx, TypeId type_id, \
                                   void* value) {                         \
      if (!Ffi::Isa<T>(ctx, type_id)) {                                   \
        return std::nullopt;                                              \
      }                                                                   \
                                                                          \
      XLA_FFI_ANNOTATE_MEMORY_IS_INITIALIZED(value, sizeof(T));           \
      return *reinterpret_cast<T*>(value);                                \
    }                                                                     \
  }

XLA_FFI_REGISTER_SCALAR_ARG_DECODING(int32_t);

#undef XLA_FFI_REGISTER_SCALAR_ARG_DECODING

template <>
struct FfiArgDecoding<StridedBufferArg> {
  using EncodedMemref = internal::EncodedMemref;

  static std::optional<StridedBufferArg> Decode(ExecutionContext* ctx,
                                                TypeId type_id, void* value) {
    if (!Ffi::Isa<BufferArg, StridedBufferArg>(ctx, type_id)) {
      return std::nullopt;
    }

    auto* encoded = reinterpret_cast<EncodedMemref*>(value);
    XLA_FFI_ANNOTATE_MEMORY_IS_INITIALIZED(encoded, sizeof(EncodedMemref));
    XLA_FFI_ANNOTATE_MEMORY_IS_INITIALIZED(
        encoded, sizeof(EncodedMemref) + encoded->rank * sizeof(int64_t));

    PrimitiveType dtype = static_cast<PrimitiveType>(encoded->dtype);
    return StridedBufferArg{dtype,
                            encoded->data,
                            {encoded->dims, encoded->rank},
                            {encoded->dims + encoded->rank, encoded->rank}};
  }
};

template <>
struct FfiArgDecoding<BufferArg> {
  using EncodedMemref = internal::EncodedMemref;

  static std::optional<BufferArg> Decode(ExecutionContext* ctx, TypeId type_id,
                                         void* value) {
    if (!Ffi::Isa<BufferArg>(ctx, type_id)) {
      return std::nullopt;
    }

    auto* encoded = reinterpret_cast<EncodedMemref*>(value);
    XLA_FFI_ANNOTATE_MEMORY_IS_INITIALIZED(encoded, sizeof(EncodedMemref));
    XLA_FFI_ANNOTATE_MEMORY_IS_INITIALIZED(
        encoded, sizeof(EncodedMemref) + encoded->rank * sizeof(int64_t));

    PrimitiveType dtype = static_cast<PrimitiveType>(encoded->dtype);
    return BufferArg{dtype, encoded->data, {encoded->dims, encoded->rank}};
  }
};

//===----------------------------------------------------------------------===//
// XLA FFI attributes decoding.
//===----------------------------------------------------------------------===//

#define XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(T)                          \
  template <>                                                             \
  struct FfiAttrDecoding<T> {                                             \
    static std::optional<T> Decode(ExecutionContext* ctx,                 \
                                   std::string_view name, TypeId type_id, \
                                   void* value) {                         \
      if (!Ffi::Isa<T>(ctx, type_id)) {                                   \
        return std::nullopt;                                              \
      }                                                                   \
                                                                          \
      return *reinterpret_cast<T*>(value);                                \
    }                                                                     \
  }

XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(float);

#undef XLA_FFI_REGISTER_SCALAR_ATTR_DECODING

//===----------------------------------------------------------------------===//
// XLA FFI helper macro for registering FFI implementations.
//===----------------------------------------------------------------------===//

#define XLA_FFI_DEFINE_FUNCTION(fn, impl, binding)                             \
  static XLA_FFI_Error* fn(XLA_FFI_Function_Args* args) {                      \
    if (args->struct_size != XLA_FFI_Function_Args_STRUCT_SIZE) {              \
      std::cerr << "Unexpected XLA_FFI_Function_Args  size: expected "         \
                << XLA_FFI_Function_Args_STRUCT_SIZE << " << got "             \
                << args->struct_size << ". Check installed software versions." \
                << std::endl;                                                  \
      std::abort();                                                            \
    }                                                                          \
    static auto* handler = binding.To(impl).release();                         \
    return (*handler)(args->ctx, args->args, args->attrs, args->rets);         \
  }

}  // namespace ffi
}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_FFI_FFI_API_H_

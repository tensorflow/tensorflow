/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_FFI_API_API_H_
#define XLA_FFI_API_API_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// This is a header-only base C++ library that defines templates for decoding
// XLA FFI call frames and invoking corresponding C++ functions. This must have
// no dependencies outside of the C++ standard library.
//
// There are two extensions to this base library:
//
//   (1) xla/ffi/api/ffi.h for defining "external" FFI handlers loaded from
//       dynamic libraries potentially built with different toolchains and/or
//       a different XLA commit. It is a header-only library without any
//       dependencies.
//
//   (2) xla/ffi/ffi.h for defining "internal" FFI handlers that must be
//       statically linked into the binary and must be built from the same
//       commit using the same toolchain, as it provides access to XLA
//       implementation details (e.g. ServiceExecutableOptions) and C++ ABI
//       across different libraries is hard.
//
// Extensions define template specializations for argument-decoding hooks
// defined in this file.

#include "xla/ffi/api/c_api.h"

namespace xla::ffi {

// Forward declare template defined below.
template <typename... Ts>
class Binding;

// Forward declare template defined below.
template <typename Fn, typename... Ts>
class Handler;

//===----------------------------------------------------------------------===//
// XLA FFI virtual base for implementing FFI handlers
//===----------------------------------------------------------------------===//

class Ffi {
 public:
  static Binding<> Bind();

  virtual ~Ffi() = default;
  virtual XLA_FFI_Error* Call(const XLA_FFI_CallFrame* call_frame) const = 0;

  // Registers handler with an XLA runtime under the given name.
  static inline XLA_FFI_Error* RegisterStaticHandler(const XLA_FFI_Api* api,
                                                     std::string_view name,
                                                     XLA_FFI_Handler* handler);

 protected:
  template <typename... Args>
  static std::string StrCat(Args... args);

  static inline XLA_FFI_Error* MakeError(const XLA_FFI_Api* api,
                                         XLA_FFI_Error_Code errc,
                                         std::string message);

  static inline XLA_FFI_Error* InvalidArgument(const XLA_FFI_Api* api,
                                               std::string message);

  static inline XLA_FFI_Error* CheckStructSize(const XLA_FFI_Api* api,
                                               std::string_view struct_name,
                                               size_t expected, size_t actual);
};

XLA_FFI_Error* Ffi::RegisterStaticHandler(const XLA_FFI_Api* api,
                                          std::string_view name,
                                          XLA_FFI_Handler* handler) {
  std::string name_str(name);  // make a copy to guarantee it's null terminated

  XLA_FFI_Handler_Register_Args args;
  args.struct_size = XLA_FFI_Handler_Register_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.name = name_str.c_str();
  args.handler = handler;
  return api->XLA_FFI_Handler_Register(&args);
}

template <typename... Args>
std::string Ffi::StrCat(Args... args) {
  std::stringstream ss;
  (ss << ... << args);
  return ss.str();
}

XLA_FFI_Error* Ffi::MakeError(const XLA_FFI_Api* api, XLA_FFI_Error_Code errc,
                              std::string message) {
  XLA_FFI_Error_Create_Args args;
  args.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.errc = errc;
  args.message = message.c_str();
  return api->XLA_FFI_Error_Create(&args);
}

XLA_FFI_Error* Ffi::InvalidArgument(const XLA_FFI_Api* api,
                                    std::string message) {
  return MakeError(api, XLA_FFI_Error_Code_INVALID_ARGUMENT,
                   std::move(message));
}

XLA_FFI_Error* Ffi::CheckStructSize(const XLA_FFI_Api* api,
                                    std::string_view struct_name,
                                    size_t expected, size_t actual) {
  if (expected != actual) {
    return InvalidArgument(
        api, StrCat("Unexpected ", struct_name, " size: expected ", expected,
                    " got ", actual, ". Check installed software versions."));
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Type tags for distinguishing handler argument types
//===----------------------------------------------------------------------===//

namespace internal {

// A type tag to distinguish arguments tied to the attributes in the
// `Binding` variadic template argument.
template <typename T>
struct AttrTag {};

// A type tag to distinguish arguments extracted from an execution context.
template <typename T>
struct CtxTag {};

}  // namespace internal

//===----------------------------------------------------------------------===//
// Binding variadic template defines FFI handler signature
//===----------------------------------------------------------------------===//

template <typename... Ts>
class Binding {
 public:
  template <typename T>
  Binding<Ts..., T> Arg() && {
    return {std::move(*this)};
  }

  template <typename T>
  Binding<Ts..., internal::CtxTag<T>> Ctx() && {
    return {std::move(*this)};
  }

  template <typename T>
  Binding<Ts..., internal::AttrTag<T>> Attr(std::string attr) && {
    attrs_.push_back(std::move(attr));
    return {std::move(*this)};
  }

  template <typename Fn>
  std::unique_ptr<Handler<Fn, Ts...>> To(Fn fn) {
    return std::unique_ptr<Handler<Fn, Ts...>>(
        new Handler<Fn, Ts...>(std::forward<Fn>(fn), std::move(attrs_)));
  }

 private:
  template <typename...>
  friend class Binding;
  friend class Ffi;

  explicit Binding() {
    static_assert(sizeof...(Ts) == 0, "arguments must be empty");
  }

  template <typename... TTs>
  Binding(Binding<TTs...>&& other)  // NOLINT
      : attrs_(std::move(other.attrs_)) {}

  Binding(Binding&) = delete;

  std::vector<std::string> attrs_;  // names of bound attributes
};

inline Binding<> Ffi::Bind() { return xla::ffi::Binding<>(); }

//===----------------------------------------------------------------------===//
// Arguments decoding implementation
//===----------------------------------------------------------------------===//

// XLA FFI arguments decoding must be defined by specializing this template.
//
// Example: decoding for the `MyType` arguments
//
//   template <>
//   struct ArgDecoding<MyType> {
//     static std::optional<MyType> Decode(XLA_FFI_ArgType type, void* arg);
//   };
//
// If argument can't be decoded it should return the empty optional.
template <typename T>
struct ArgDecoding;

//===----------------------------------------------------------------------===//
// Attributes decoding implementation
//===----------------------------------------------------------------------===//

// XLA FFI attribute decoding must be defined by specializing this template.
//
// Example: decoding for the `MyType` attributes
//
//   template <>
//   struct AttrDecoding<MyType> {
//    static std::optional<MyType> Decode(std::string_view name,
//                                        XLA_FFI_AttrType type, void* attr);
//   }
//
template <typename T>
struct AttrDecoding;

//===----------------------------------------------------------------------===//
// Context decoding implementation
//===----------------------------------------------------------------------===//

// XLA FFI execution context decoding must be defined by specializing this
// template.
//
// Example: decoding for the `MyType` context
//
//   template <>
//   struct CtxDecoding<MyType> {
//    using Type = <handler argument type for context type MyType>;
//    static std::optional<Type> Decode(const XLA_FFI_Api* api,
//                                      XLA_FFI_ExecutionContext* ctx);
//   }
//
// TODO(ezhulenev): Add an example for decoding opaque data passed together with
// a handler registration (not yet implemented). Today this is only used as
// internal implementation detail of builtin FFI handlers.
template <typename T>
struct CtxDecoding;

//===----------------------------------------------------------------------===//
// Result encoding implementation
//===----------------------------------------------------------------------===//

// XLA FFI result encoding (conversion from a returned status-like type to FFI
// error type) must be defined by specializing this template.
//
// Example: encoding `absl::Status` result
//
//   template<>
//   struct ResultEncoding<absl::Status> {
//     XLA_FFI_Error* Encode(const XLA_FFI_Api* api, absl::Status status) {...}
//   }
//
template <typename T>
struct ResultEncoding;

//===----------------------------------------------------------------------===//
// Decoding arguments and attributes
//===----------------------------------------------------------------------===//

namespace internal {

// When decoding input data we need to keep track of how many arguments and
// attributes we decoded so far to compute call frame offsets.
struct DecodingOffsets {
  int64_t args = 0;
  int64_t attrs = 0;
};

struct DecodingContext {
  const XLA_FFI_CallFrame* call_frame;

  const std::string* attrs_names;  // not owned
  const std::size_t* attrs_idx;    // not owned
};

template <typename T>
struct Decode {
  static std::optional<T> call(DecodingOffsets& offsets, DecodingContext& ctx) {
    int64_t idx = offsets.args++;
    return ArgDecoding<T>::Decode(ctx.call_frame->args.types[idx],
                                  ctx.call_frame->args.args[idx]);
  }
};

template <typename T>
struct Decode<internal::AttrTag<T>> {
  static std::optional<T> call(DecodingOffsets& offsets, DecodingContext& ctx) {
    // Find decoded attribute corresponding to the given attribute index.
    int64_t idx = offsets.attrs++;

    // Get mapping from the attribute to its index in the sorted array.
    size_t i = ctx.attrs_idx[idx];

    // Load attribute from call frame using index into the sorted array.
    XLA_FFI_AttrType type = ctx.call_frame->attrs.types[i];
    XLA_FFI_ByteSpan* name = ctx.call_frame->attrs.names[i];
    void* attr = ctx.call_frame->attrs.attrs[i];

    // TODO(ezhulenev): Currently we require that attributes passed to the FFI
    // handler must match attributes referenced in a binding, however
    // we could safely ignore extra attributes. Relax this if needed.

    // Attribute name does not match.
    std::string_view name_view = {name->ptr, name->len};
    if (name_view != ctx.attrs_names[idx]) return std::nullopt;

    return AttrDecoding<T>::Decode(name_view, type, attr);
  }
};

template <typename T>
struct Decode<internal::CtxTag<T>> {
  using R = typename CtxDecoding<T>::Type;

  static std::optional<R> call(DecodingOffsets& offsets, DecodingContext& ctx) {
    return CtxDecoding<T>::Decode(ctx.call_frame->api, ctx.call_frame->ctx);
  }
};

}  // namespace internal

//===----------------------------------------------------------------------===//
// Template metaprogramming for decoding handler signature
//===----------------------------------------------------------------------===//

namespace internal {

// A helper struct to extract the type of the handler argument.
template <typename T>
struct FnArgType {
  using Type = T;
};

// Extracts the underlying type from the attribute type tag.
template <typename T>
struct FnArgType<internal::AttrTag<T>> {
  using Type = T;
};

// Extracts the underlying type from the context type tag.
template <typename T>
struct FnArgType<internal::CtxTag<T>> {
  using Type = typename CtxDecoding<T>::Type;
};

// A template for checking if type is a wrapped attribute or user data.
template <typename>
struct IsWrapped : std::false_type {};
template <typename T>
struct IsWrapped<AttrTag<T>> : std::true_type {};
template <typename T>
struct IsWrapped<CtxTag<T>> : std::true_type {};

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

// A template for counting tagged arguments in the Ts pack (i.e. attributes).
template <template <typename> class Tag, typename... Ts>
struct NumTagged;

template <template <typename> class Tag>
struct NumTagged<Tag> {
  static constexpr int64_t value = 0;
};

template <template <typename> class Tag, typename T, typename... Ts>
struct NumTagged<Tag, Tag<T>, Ts...> {
  static constexpr int64_t value = 1 + NumTagged<Tag, Ts...>::value;
};

template <template <typename> class Tag, typename T, typename... Ts>
struct NumTagged<Tag, T, Ts...> {
  static constexpr int64_t value = 0 + NumTagged<Tag, Ts...>::value;
};

}  // namespace internal

//===----------------------------------------------------------------------===//
// Handler decodes FFI call frame and invokes `Fn` with decoded arguments
//===----------------------------------------------------------------------===//

template <typename Fn, typename... Ts>
class Handler : public Ffi {
  static constexpr int64_t kSize = sizeof...(Ts);

  static constexpr int64_t kNumArgs = internal::NumArgs<Ts...>::value;
  static constexpr int64_t kNumAttrs =
      internal::NumTagged<internal::AttrTag, Ts...>::value;

  template <typename T>
  using FnArgType = typename internal::FnArgType<T>::Type;

  static_assert(std::is_invocable_v<Fn, FnArgType<Ts>...>,
                "FFI binding signature is not compatible with a function type");

  using ResultType = std::invoke_result_t<Fn, FnArgType<Ts>...>;

 public:
  XLA_FFI_Error* Call(const XLA_FFI_CallFrame* call_frame) const override {
    // Sanity checking call frame struct size.
    if (auto* err = CheckStructSize(call_frame->api, "XLA_FFI_CallFrame",
                                    XLA_FFI_CallFrame_STRUCT_SIZE,
                                    call_frame->struct_size))
      return err;

    // Check that the number of passed arguments matches the signature. Each
    // individual argument decoding will check the actual type.
    if (call_frame->args.num_args != kNumArgs) {
      return InvalidArgument(
          call_frame->api,
          StrCat("Wrong number of arguments: expected ", kNumArgs, " but got ",
                 call_frame->args.num_args));
    }

    // Check that the number of passed attributes matches the signature. Each
    // individual attribute decoding will check the actual type.
    if (call_frame->attrs.num_attrs != kNumAttrs) {
      return InvalidArgument(
          call_frame->api,
          StrCat("Wrong number of attributes: expected ", kNumAttrs,
                 " but got ", call_frame->attrs.num_attrs));
    }

    // Define index sequences to access custom call operands.
    using Is = std::make_index_sequence<kSize>;

    return Call(call_frame, Is{});
  }

 private:
  template <size_t... Is>
  XLA_FFI_Error* Call(const XLA_FFI_CallFrame* call_frame,
                      std::index_sequence<Is...>) const {
    // A helper structure to allow each decoder find the correct offset.
    internal::DecodingOffsets offsets;

    // Package all the data required for decoding ffi handler operands.
    internal::DecodingContext ctx = {call_frame, attrs_.data(),
                                     attrs_idx_.data()};

    std::tuple<std::optional<FnArgType<Ts>>...> args = {
        internal::Decode<Ts>::call(offsets, ctx)...};

    bool all_decoded = (std::get<Is>(args).has_value() && ...);
    if (!all_decoded) {
      return FailedDecodeError(call_frame, {std::get<Is>(args).has_value()...});
    }

    auto result = fn_(std::move(*std::get<Is>(args))...);
    return ResultEncoding<ResultType>::Encode(call_frame->api,
                                              std::move(result));
  }

  XLA_FFI_Error* FailedDecodeError(const XLA_FFI_CallFrame* call_frame,
                                   std::array<bool, kSize> decoded) const {
    std::string message =
        "Failed to decode all FFI handler operands (bad operands at: ";
    for (size_t cnt = 0, idx = 0; idx < kSize; ++idx) {
      if (!decoded[idx]) {
        if (cnt++) message.append(", ");
        message.append(std::to_string(idx));
      }
    }
    message.append(")");
    return InvalidArgument(call_frame->api, message);
  }

  template <typename...>
  friend class Binding;

  Handler(Fn fn, std::vector<std::string> attrs)
      : fn_(std::move(fn)), attrs_(std::move(attrs)) {
    // Sort attributes' names and remove duplicates. These unique attributes are
    // what we'll be looking for in the call frame attributes.
    std::vector<std::string> sorted = attrs_;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(
        std::unique(sorted.begin(), sorted.end(), std::equal_to<std::string>()),
        sorted.end());

    // Find index of every attribute in the sorted attributes vector.
    for (size_t i = 0; i < attrs_.size(); ++i) {
      attrs_idx_.push_back(std::distance(
          sorted.begin(), std::find(sorted.begin(), sorted.end(), attrs_[i])));
    }
  }

  Fn fn_;

  std::vector<std::string> attrs_;  // names of bound attributes

  // A mapping from the attribute index (index into the `attrs_` member) to its
  // index in the lexicographically sorted vector of attribute names. Call frame
  // passes attributes sorted by name, and with this index we can find the
  // attribute we are looking for using O(1) lookup, assuming if the call frame
  // has exact same attributes as the binding. If not, this allows to do a more
  // efficient binary search by skipping a part of the call frame attributes.
  std::vector<size_t> attrs_idx_;
};

//===----------------------------------------------------------------------===//
// Builtin attributes decoding
//===----------------------------------------------------------------------===//

#define XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(T, TYPE)                  \
  template <>                                                           \
  struct AttrDecoding<T> {                                              \
    static std::optional<T> Decode(std::string_view name,               \
                                   XLA_FFI_AttrType type, void* attr) { \
      if (type != TYPE) {                                               \
        return std::nullopt;                                            \
      }                                                                 \
                                                                        \
      return *reinterpret_cast<T*>(attr);                               \
    }                                                                   \
  }

XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(int32_t, XLA_FFI_AttrType_I32);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(float, XLA_FFI_AttrType_F32);

#undef XLA_FFI_REGISTER_SCALAR_ATTR_DECODING

template <>
struct AttrDecoding<std::string_view> {
  static std::optional<std::string_view> Decode(std::string_view name,
                                                XLA_FFI_AttrType type,
                                                void* attr) {
    if (type != XLA_FFI_AttrType_STRING) {
      return std::nullopt;
    }

    auto* span = reinterpret_cast<XLA_FFI_ByteSpan*>(attr);
    return std::string_view(span->ptr, span->len);
  }
};

//===----------------------------------------------------------------------===//
// Helper macro for registering FFI implementations
//===----------------------------------------------------------------------===//

#if (defined(__GNUC__) || defined(__APPLE__)) && !defined(SWIG)  // GCC-style
#define XLA_FFI_ATTRIBUTE_UNUSED __attribute__((unused))
#else  // Non-GCC equivalents
#define XLA_FFI_ATTRIBUTE_UNUSED
#endif

// Use captureless lambda to function pointer conversion to create a static
// XLA_FFI_Handler function pointer variable.
#define XLA_FFI_DEFINE_HANDLER(fn, impl, binding)                             \
  static constexpr XLA_FFI_Handler* fn = +[](XLA_FFI_CallFrame* call_frame) { \
    static auto* handler = binding.To(impl).release();                        \
    return handler->Call(call_frame);                                         \
  }

// TODO(ezhulenev): Add a callback so that end users can log registration error
// to appropriate logging destination, e.g. LOG(FATAL) for duplicate internal
// FFI handlers.
#define XLA_FFI_REGISTER_HANDLER(API, NAME, FUNC) \
  XLA_FFI_REGISTER_HANDLER_(API, NAME, FUNC, __COUNTER__)
#define XLA_FFI_REGISTER_HANDLER_(API, NAME, FUNC, N) \
  XLA_FFI_REGISTER_HANDLER__(API, NAME, FUNC, N)
#define XLA_FFI_REGISTER_HANDLER__(API, NAME, FUNC, N)                  \
  XLA_FFI_ATTRIBUTE_UNUSED static const XLA_FFI_Error*                  \
      xla_ffi_static_handler_##N##_registered_ = [] {                   \
        return ::xla::ffi::Ffi::RegisterStaticHandler(API, NAME, FUNC); \
      }()

}  // namespace xla::ffi

#endif  // XLA_FFI_API_API_H_

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

#ifndef XLA_FFI_API_API_H_
#define XLA_FFI_API_API_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
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

#ifdef __has_builtin
#define XLA_FFI_HAS_BUILTIN(x) __has_builtin(x)
#else
#define XLA_FFI_HAS_BUILTIN(x) 0
#endif

#if __has_attribute(always_inline)
#define XLA_FFI_ATTRIBUTE_ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define XLA_FFI_ATTRIBUTE_ALWAYS_INLINE __forceinline
#else
#define XLA_FFI_ATTRIBUTE_ALWAYS_INLINE inline
#endif

#if __has_attribute(noinline)
#define XLA_FFI_ATTRIBUTE_NEVER_INLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define XLA_FFI_ATTRIBUTE_NEVER_INLINE __declspec(noinline)
#else
#define XLA_FFI_ATTRIBUTE_NEVER_INLINE
#endif

#if XLA_FFI_HAS_BUILTIN(__builtin_expect)
#define XLA_FFI_PREDICT_FALSE(x) (__builtin_expect(false || (x), false))
#define XLA_FFI_PREDICT_TRUE(x) (__builtin_expect(false || (x), true))
#else
#define XLA_FFI_PREDICT_FALSE(x) (x)
#define XLA_FFI_PREDICT_TRUE(x) (x)
#endif

//===----------------------------------------------------------------------===//
// Builtin enum pretty printing
//===----------------------------------------------------------------------===//

inline std::ostream& operator<<(std::ostream& os,
                                const XLA_FFI_DataType dtype) {
  switch (dtype) {
    case XLA_FFI_DataType_INVALID:
      return os << "INVALID";
    case XLA_FFI_DataType_PRED:
      return os << "PRED";
    case XLA_FFI_DataType_S8:
      return os << "S8";
    case XLA_FFI_DataType_S16:
      return os << "S16";
    case XLA_FFI_DataType_S32:
      return os << "S32";
    case XLA_FFI_DataType_S64:
      return os << "S64";
    case XLA_FFI_DataType_U8:
      return os << "U8";
    case XLA_FFI_DataType_U16:
      return os << "U16";
    case XLA_FFI_DataType_U32:
      return os << "U32";
    case XLA_FFI_DataType_U64:
      return os << "U64";
    case XLA_FFI_DataType_F16:
      return os << "F16";
    case XLA_FFI_DataType_F32:
      return os << "F32";
    case XLA_FFI_DataType_F64:
      return os << "F64";
    case XLA_FFI_DataType_BF16:
      return os << "BF16";
    case XLA_FFI_DataType_C64:
      return os << "C64";
    case XLA_FFI_DataType_C128:
      return os << "C128";
    case XLA_FFI_DataType_TOKEN:
      return os << "TOKEN";
  }
}

inline std::ostream& operator<<(std::ostream& os, const XLA_FFI_AttrType type) {
  switch (type) {
    case XLA_FFI_AttrType_ARRAY:
      return os << "array";
    case XLA_FFI_AttrType_DICTIONARY:
      return os << "dictionary";
    case XLA_FFI_AttrType_SCALAR:
      return os << "scalar";
    case XLA_FFI_AttrType_STRING:
      return os << "string";
  }
}

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
  // Creates and empty binding specification wich allows to define FFI handler
  // signature separately from implementation and rely on compile time type
  // checking to verify that signature matches the provided implementation.
  static Binding<> Bind();

  // Automatic FFI binding that does binding specification inference from the
  // `fn` type signature and binds `fn` to it. This enables a more concise FFI
  // handler registration with fully automatic type inference at the cost of
  // less readable error messages, template metaprogramming "magic" and a risk
  // to accidentally change handler type without noticing it.
  template <typename Fn>
  static auto BindTo(Fn fn);

  virtual ~Ffi() = default;
  virtual XLA_FFI_Error* Call(const XLA_FFI_CallFrame* call_frame) const = 0;

  // Registers FFI handler bundle with an XLA runtime under the given name on a
  // given platform.
  static inline XLA_FFI_Error* RegisterStaticHandler(
      const XLA_FFI_Api* api, std::string_view name, std::string_view platform,
      XLA_FFI_Handler_Bundle bundle, XLA_FFI_Handler_Traits traits = 0);

  // Registers FFI execute handler with an XLA runtime under the given name on a
  // given platform.
  static inline XLA_FFI_Error* RegisterStaticHandler(
      const XLA_FFI_Api* api, std::string_view name, std::string_view platform,
      XLA_FFI_Handler* execute, XLA_FFI_Handler_Traits traits = 0) {
    return RegisterStaticHandler(
        api, name, platform, XLA_FFI_Handler_Bundle{nullptr, nullptr, execute},
        traits);
  }

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
                                          std::string_view platform,
                                          XLA_FFI_Handler_Bundle bundle,
                                          XLA_FFI_Handler_Traits traits) {
  XLA_FFI_Handler_Register_Args args;
  args.struct_size = XLA_FFI_Handler_Register_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.name = XLA_FFI_ByteSpan{name.data(), name.size()};
  args.platform = XLA_FFI_ByteSpan{platform.data(), platform.size()};
  args.bundle = bundle;
  args.traits = traits;
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

// Forward declare.
class Dictionary;

namespace internal {

// WARNING: A lot of template metaprogramming on top of C++ variadic templates
// parameter packs. We need this to be able to pattern match FFI handler
// signature at compile time.

// A type tag to forward all remaining args as `RemainingArgs`.
struct RemainingArgsTag {};

// A type tag to forward all remaining results as `RemainingRets`.
struct RemainingRetsTag {};

// A type tag to distinguish parameters tied to results in the `Binding`
// variadic template. In XLA FFI we use destination passing style APIs and don't
// return anything from the handler, but instead pass a destination where the
// handler should write the result.
template <typename T>
struct RetTag {};

// A type tag to distinguish parameters tied to the attributes in the
// `Binding` variadic template.
template <typename T>
struct AttrTag {};

// A type tag to forward all attributes as `Dictionary` (and optionally decode
// it into a custom struct).
template <typename T = Dictionary>
struct AttrsTag {};

// A type tag to distinguish parameter extracted from an execution context.
template <typename T>
struct CtxTag {};

//----------------------------------------------------------------------------//
// A template for counting tagged arguments in the Ts pack (i.e. attributes).
//----------------------------------------------------------------------------//

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

//----------------------------------------------------------------------------//

// Checks if remaining arguments are in the parameter pack.
template <typename... Ts>
using HasRemainingArgsTag =
    std::disjunction<std::is_same<RemainingArgsTag, Ts>...>;

// Checks if remaining results are in the parameter pack.
template <typename... Ts>
using HasRemainingRetsTag =
    std::disjunction<std::is_same<RemainingRetsTag, Ts>...>;

//----------------------------------------------------------------------------//

template <typename T>
XLA_FFI_DataType NativeTypeToCApiDataType() {
  if constexpr (std::is_same_v<T, bool>) {
    return XLA_FFI_DataType_PRED;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return XLA_FFI_DataType_S8;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return XLA_FFI_DataType_S16;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return XLA_FFI_DataType_S32;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return XLA_FFI_DataType_S64;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return XLA_FFI_DataType_U8;
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return XLA_FFI_DataType_U16;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return XLA_FFI_DataType_U32;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return XLA_FFI_DataType_U64;
  } else if constexpr (std::is_same_v<T, float>) {
    return XLA_FFI_DataType_F32;
  } else if constexpr (std::is_same_v<T, double>) {
    return XLA_FFI_DataType_F64;
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return XLA_FFI_DataType_C64;
  } else {
    static_assert(std::is_same_v<T, std::complex<double>>,
                  "unsupported FFI data type");
    return XLA_FFI_DataType_C128;
  }
}

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
  Binding<Ts..., internal::RetTag<T>> Ret() && {
    return {std::move(*this)};
  }

  Binding<Ts..., internal::RemainingArgsTag> RemainingArgs() && {
    static_assert(!internal::HasRemainingArgsTag<Ts...>::value,
                  "remaining arguments can be passed just once");
    return {std::move(*this)};
  }

  Binding<Ts..., internal::RemainingRetsTag> RemainingResults() && {
    static_assert(!internal::HasRemainingRetsTag<Ts...>::value,
                  "remaining results can be passed just once");
    return {std::move(*this)};
  }

  template <typename T>
  Binding<Ts..., internal::CtxTag<T>> Ctx() && {
    return {std::move(*this)};
  }

  template <typename T>
  Binding<Ts..., internal::AttrTag<T>> Attr(std::string attr) && {
    static_assert(internal::NumTagged<internal::AttrsTag, Ts...>::value == 0,
                  "dictionary attributes can't be mixed with regular ones");
    attrs_.push_back(std::move(attr));
    return {std::move(*this)};
  }

  template <typename T = Dictionary>
  Binding<Ts..., internal::AttrsTag<T>> Attrs() && {
    static_assert(internal::NumTagged<internal::AttrTag, Ts...>::value == 0,
                  "dictionary attributes can't be mixed with regular ones");
    return {std::move(*this)};
  }

  template <typename Fn>
  std::unique_ptr<Handler<Fn, Ts...>> To(Fn fn) {
    return std::unique_ptr<Handler<Fn, Ts...>>(
        new Handler<Fn, Ts...>(std::move(fn), std::move(attrs_)));
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
// Template metaprogramming to automatially infer Binding from invocable object.
//===----------------------------------------------------------------------===//

// A little bit of metaprogramming that automatically infers the binding schema
// from an invocable type signature.

// XLA FFI binding for an argument.
//
// Example: binding for the `MyType` argument
//
//   template <>
//   struct ArgBinding<MyType> {
//     using Arg = MyType;
//   };
//
template <typename T>
struct ArgBinding {
  using Arg = void;
};

// XLA FFI binding for a returned result.
//
// Example: binding for the `MyType` result
//
//   template <>
//   struct RetBinding<MyType> {
//     using Ret = MyType;
//   };
//
template <typename T>
struct RetBinding {
  using Ret = void;
};

// XLA FFI binding for a named attribute.
//
// Example: binding for the `MyType` attribute
//
//   template <>
//   struct AttrBinding<MyAttr> {
//     using Attr = MyAttr;
//     static constexpr std::string_view name() { return "my_attr"; }
//   };
//
template <typename T>
struct AttrBinding {
  using Attr = void;
};

// XLA FFI binding for dictionary attributes: automatic parsing of all
// attributes into user defined struct.
template <typename T>
struct AttrsBinding {
  using Attrs = void;
};

// XLA FFI binding for values passed via context.
//
// Example: binding for the `gpuStream_t` platform stream
//
//   template <>
//   struct CtxBinding<gpuStream_t> {
//     using Ctx = PlatformStream<gpuStream_t>;
//   };
//
template <typename T>
struct CtxBinding {
  using Ctx = void;
};

namespace internal {

template <typename Param>
inline constexpr bool is_arg_binding_v =
    !std::is_void_v<typename ArgBinding<Param>::Arg>;

template <typename Param>
inline constexpr bool is_ret_binding_v =
    !std::is_void_v<typename RetBinding<Param>::Ret>;

template <typename Param>
inline constexpr bool is_attr_binding_v =
    !std::is_void_v<typename AttrBinding<Param>::Attr>;

template <typename Param>
inline constexpr bool is_attrs_binding_v =
    !std::is_void_v<typename AttrsBinding<Param>::Attrs>;

template <typename Param>
inline constexpr bool is_ctx_binding_v =
    !std::is_void_v<typename CtxBinding<Param>::Ctx>;

// A helper template to bind `Params` to `Fn` one by one.
template <typename Fn, typename... Params>
struct BindOne;

// A specialization that binds one parameter.
template <typename Fn, typename Param, typename... Params>
struct BindOne<Fn, Param, Params...> {
  // Binds single parameter and then continues with remaining parameters using
  // recursive template instantiation.
  template <typename InFlightBinding>
  static auto To(Fn fn, InFlightBinding binding) {
    if constexpr (is_arg_binding_v<Param>) {
      // Bind parameter as an FFI handler argument.
      return BindOne<Fn, Params...>::To(
          std::move(fn),
          std::move(binding).template Arg<typename ArgBinding<Param>::Arg>());
    } else if constexpr (is_ret_binding_v<Param>) {
      // Bind parameter as an FFI handler result.
      return BindOne<Fn, Params...>::To(
          std::move(fn),
          std::move(binding).template Ret<typename RetBinding<Param>::Ret>());

    } else if constexpr (is_attr_binding_v<Param>) {
      // Bind parameter as a named FFI handler attribute.
      return BindOne<Fn, Params...>::To(
          std::move(fn),
          std::move(binding).template Attr<typename AttrBinding<Param>::Attr>(
              std::string(AttrBinding<Param>::name())));

    } else if constexpr (is_attrs_binding_v<Param>) {
      // Bind parameter as attributes dictionary.
      return BindOne<Fn, Params...>::To(
          std::move(fn),
          std::move(binding)
              .template Attrs<typename AttrsBinding<Param>::Attrs>());

    } else if constexpr (is_ctx_binding_v<Param>) {
      // Bind parameter as an FFI handler context.
      return BindOne<Fn, Params...>::To(
          std::move(fn),
          std::move(binding).template Ctx<typename CtxBinding<Param>::Ctx>());

    } else {
      // Parameter is not recognized as one of the types that can be bound to
      // FFI handler.
      static_assert(sizeof(Param) == 0,
                    "parameter is not supported for binding");
    }
  }
};

// A specialization that binds `Fn` after consuming all parameters.
template <typename Fn>
struct BindOne<Fn> {
  template <typename InFlightBinding>
  static auto To(Fn fn, InFlightBinding binding) {
    return binding.To(std::move(fn));
  }
};

template <typename Fn>
struct Bind;

// Binding specialization for function pointers (and captureless lambdas that
// can be casted to function pointers).
template <typename ResultType, typename... Params>
struct Bind<ResultType (*)(Params...)> {
  using Fn = ResultType (*)(Params...);

  static auto To(Fn fn) {
    return BindOne<Fn, Params...>::To(std::move(fn), Ffi::Bind());
  }
};

// Binding specialization for callables (lambdas with captures).
template <typename ResultType, typename Fn, typename... Params>
struct Bind<ResultType (Fn::*)(Params...) const> {
  static auto To(Fn fn) {
    return BindOne<Fn, Params...>::To(std::move(fn), Ffi::Bind());
  }
};

}  // namespace internal

template <typename Fn>
auto Ffi::BindTo(Fn fn) {
  if constexpr (std::is_pointer_v<Fn>) {
    return internal::Bind<Fn>::To(fn);
  } else {
    return internal::Bind<decltype(&Fn::operator())>::To(std::move(fn));
  }
}

// A container for defining parameters corresponding to results.
template <typename T>
class Result {
 public:
  Result(T value) : value_(value) {}  // NOLINT
  T& operator*() { return value_; }
  T* operator->() { return &value_; }

 private:
  T value_;
};

// A container for defining parameters corresponding to attributes with an
// attribute name available as compile time value.
template <typename T, char const* attr_name>
class Attr {
 public:
  Attr(T value) : value_(value) {}  // NOLINT
  T& operator*() { return value_; }
  T* operator->() { return &value_; }

 private:
  T value_;
};

//===----------------------------------------------------------------------===//
// Attributes bindings
//===----------------------------------------------------------------------===//

// Default attribute binding for `Attr` parameters.
template <typename T, const char* attr_name>
struct AttrBinding<Attr<T, attr_name>> {
  using Attr = T;
  static constexpr std::string_view name() { return attr_name; }
};

// Default attributes binding for `Dictonary` parameters.
template <>
struct AttrsBinding<Dictionary> {
  using Attrs = Dictionary;
};

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
// Results decoding implementation
//===----------------------------------------------------------------------===//

// XLA FFI results decoding must be defined by specializing this template.
//
// Example: decoding for the `MyType` results
//
//   template <>
//   struct RetDecoding<MyType> {
//     static std::optional<MyType> Decode(XLA_FFI_RetType type, void* ret);
//   };
//
// If argument can't be decoded it should return the empty optional.
template <typename T>
struct RetDecoding;

//===----------------------------------------------------------------------===//
// Attributes decoding implementation
//===----------------------------------------------------------------------===//

// XLA FFI attribute decoding must be defined by specializing this template.
//
// Example: decoding for the `MyType` attributes
//
//   template <>
//   struct AttrDecoding<MyType> {
//    using Type = <handler argument type for attribute type MyType>
//    static std::optional<MyType> Decode(XLA_FFI_AttrType type, void* attr,
//                                        DiagnosticEngine&);
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
// Second template parameter is used to conditionally enable/disable context
// decoding specialization for a given type via SFINAE.
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
// Diagnostics
//===----------------------------------------------------------------------===//

class DiagnosticEngine;

// RAII wrapper around constructed, but but not yet emitted diagnostic. In
// flight diagnostic gives an opportunity to build a diagnostic before reporting
// it to the engine, similar to the builder pattern.
class InFlightDiagnostic {
 public:
  explicit InFlightDiagnostic(DiagnosticEngine* engine, std::string s)
      : engine_(engine) {
    stream_ << s;
  }
  InFlightDiagnostic(const InFlightDiagnostic&) = delete;
  InFlightDiagnostic& operator=(const InFlightDiagnostic&) = delete;

  ~InFlightDiagnostic();

  template <typename Arg>
  InFlightDiagnostic& operator<<(Arg&& arg) {
    stream_ << std::forward<Arg>(arg);
    return *this;
  }

  template <typename T>
  operator std::optional<T>() const {  // NOLINT
    return std::nullopt;
  }

 private:
  DiagnosticEngine* engine_;
  std::stringstream stream_;
};

class DiagnosticEngine {
 public:
  DiagnosticEngine() = default;
  DiagnosticEngine(const DiagnosticEngine&) = delete;
  DiagnosticEngine& operator=(const DiagnosticEngine&) = delete;

  InFlightDiagnostic Emit(std::string message) {
    return InFlightDiagnostic(this, std::move(message));
  }

  std::string Result() const { return acc_; }

 private:
  friend class InFlightDiagnostic;

  void append(std::string s) { acc_.append(std::move(s)); }

  std::string acc_;
};

inline InFlightDiagnostic::~InFlightDiagnostic() {
  engine_->append(stream_.str());
}

//===----------------------------------------------------------------------===//
// Decoding arguments and attributes
//===----------------------------------------------------------------------===//

namespace internal {

// When decoding input data we need to keep track of how many arguments and
// attributes we decoded so far to compute call frame offsets.
struct DecodingOffsets {
  int64_t args = 0;
  int64_t rets = 0;
  int64_t attrs = 0;
};

struct DecodingContext {
  const XLA_FFI_CallFrame* call_frame;

  const std::string* attrs_names;  // not owned
  const std::size_t* attrs_idx;    // not owned
};

template <typename T>
struct Decode {
  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<T> call(DecodingOffsets& offsets, DecodingContext& ctx,
                               DiagnosticEngine& diagnostic) {
    int64_t idx = offsets.args++;
    return ArgDecoding<T>::Decode(ctx.call_frame->args.types[idx],
                                  ctx.call_frame->args.args[idx], diagnostic);
  }
};

}  // namespace internal

template <typename T>
struct internal::Decode<internal::RetTag<T>> {
  static std::optional<Result<T>> call(DecodingOffsets& offsets,
                                       DecodingContext& ctx,
                                       DiagnosticEngine& diagnostic) {
    int64_t idx = offsets.rets++;
    return RetDecoding<T>::Decode(ctx.call_frame->rets.types[idx],
                                  ctx.call_frame->rets.rets[idx], diagnostic);
  }
};

template <typename T>
struct internal::Decode<internal::AttrTag<T>> {
  using R = typename AttrDecoding<T>::Type;

  static std::optional<R> call(DecodingOffsets& offsets, DecodingContext& ctx,
                               DiagnosticEngine& diagnostic) {
    // Find decoded attribute corresponding to the given attribute index.
    int64_t i = offsets.attrs++;

    // Get mapping from the attribute to its index in the sorted array.
    size_t idx = ctx.attrs_idx[i];

    // Load attribute from call frame using index into the sorted array.
    XLA_FFI_AttrType attr_type = ctx.call_frame->attrs.types[idx];
    XLA_FFI_ByteSpan* attr_name = ctx.call_frame->attrs.names[idx];
    void* attr = ctx.call_frame->attrs.attrs[idx];

    // TODO(ezhulenev): Currently we require that attributes passed to the FFI
    // handler must match attributes referenced in a binding, however
    // we could safely ignore extra attributes. Relax this if needed.

    // Attribute name does not match.
    std::string_view attr_name_view = {attr_name->ptr, attr_name->len};
    if (attr_name_view != ctx.attrs_names[i]) {
      return diagnostic.Emit("Attribute name mismatch: ")
             << attr_name_view << " vs " << ctx.attrs_names[i];
    }

    return AttrDecoding<T>::Decode(attr_type, attr, diagnostic);
  }
};

template <typename T>
struct internal::Decode<internal::CtxTag<T>> {
  using R = typename CtxDecoding<T>::Type;

  static std::optional<R> call(DecodingOffsets& offsets, DecodingContext& ctx,
                               DiagnosticEngine& diagnostic) {
    return CtxDecoding<T>::Decode(ctx.call_frame->api, ctx.call_frame->ctx,
                                  diagnostic);
  }
};

//===----------------------------------------------------------------------===//
// Expected
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
  explicit constexpr Unexpected(E error) : error_(std::move(error)) {}

 private:
  template <typename, typename>
  friend class Expected;

  E error_;
};

Unexpected(const char*) -> Unexpected<std::string>;

template <typename T, typename E>
constexpr Expected<T, E>::Expected(Unexpected<E> u)
    : data_(std::move(u.error_)) {}

//===----------------------------------------------------------------------===//
// Type-safe wrapper for accessing a variable number of arguments.
//===----------------------------------------------------------------------===//

class RemainingArgs {
 public:
  RemainingArgs(const XLA_FFI_Args* args, size_t offset)
      : args_(args), offset_(offset) {
    assert(offset <= args_->size && "illegal remaining args offset");
  }

  size_t size() const { return args_->size - offset_; }
  bool empty() const { return size() == 0; }

  template <typename T>
  Expected<T, std::string> get(size_t index) const {
    size_t idx = offset_ + index;
    if (idx >= args_->size) {
      return Unexpected("Index out of range.");
    }

    DiagnosticEngine diagnostic;
    auto value_opt =
        ArgDecoding<T>::Decode(args_->types[idx], args_->args[idx], diagnostic);
    if (!value_opt.has_value()) {
      return Unexpected(diagnostic.Result());
    }
    return *value_opt;
  }

 private:
  const XLA_FFI_Args* args_;  // not owned
  size_t offset_;
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
// Type-safe wrapper for accessing a variable number of results.
//===----------------------------------------------------------------------===//

class RemainingResults {
 public:
  RemainingResults(const XLA_FFI_Rets* rets, size_t offset)
      : rets_(rets), offset_(offset) {
    assert(offset <= rets_->size && "illegal remaining rets offset");
  }

  size_t size() const { return rets_->size - offset_; }
  bool empty() const { return size() == 0; }

  template <typename T>
  Expected<T, std::string> get(size_t index) const {
    size_t idx = offset_ + index;
    if (idx >= rets_->size) {
      return Unexpected("Index out of range.");
    }

    DiagnosticEngine diagnostic;
    auto value_opt =
        RetDecoding<T>::Decode(rets_->types[idx], rets_->rets[idx], diagnostic);
    if (!value_opt.has_value()) {
      return Unexpected(diagnostic.Result());
    }
    return **value_opt;
  }

 private:
  const XLA_FFI_Rets* rets_;  // not owned
  size_t offset_;
};

template <>
struct internal::Decode<internal::RemainingRetsTag> {
  static std::optional<RemainingResults> call(DecodingOffsets& offsets,
                                              DecodingContext& ctx,
                                              DiagnosticEngine& diagnostic) {
    return RemainingResults(&ctx.call_frame->rets, offsets.rets);
  }
};

//===----------------------------------------------------------------------===//
// Type-safe wrapper for accessing dictionary attributes.
//===----------------------------------------------------------------------===//

class Dictionary {
 public:
  explicit Dictionary(const XLA_FFI_Attrs* attrs) : attrs_(attrs) {}

  size_t size() const { return attrs_->size; }

  bool contains(std::string_view name) const {
    return Find(name) < attrs_->size;
  }

  template <typename T>
  Expected<T, std::string> get(std::string_view name) const {
    DiagnosticEngine diagnostic;
    auto value_opt = get<T>(name, diagnostic);
    if (!value_opt.has_value()) {
      return Unexpected(diagnostic.Result());
    }
    return *value_opt;
  }

  template <typename T>
  std::optional<T> get(std::string_view name,
                       DiagnosticEngine& diagnostic) const {
    size_t idx = Find(name);
    if (idx >= attrs_->size) {
      return diagnostic.Emit("Unexpected attribute: ") << name;
    }

    XLA_FFI_AttrType attr_type = attrs_->types[idx];
    void* attr = attrs_->attrs[idx];
    return AttrDecoding<T>::Decode(attr_type, attr, diagnostic);
  }

 private:
  size_t Find(std::string_view name) const {
    XLA_FFI_ByteSpan** begin = attrs_->names;
    XLA_FFI_ByteSpan** end = begin + attrs_->size;

    auto name_eq = [&](XLA_FFI_ByteSpan* attr) {
      std::string_view name_view = {attr->ptr, attr->len};
      return name_view == name;
    };

    // TODO(ezhulenev): Attributes names sorted by name. We can use a binary
    // search here instead of a linear scan.
    return std::distance(begin, std::find_if(begin, end, name_eq));
  }

  const XLA_FFI_Attrs* attrs_;
};

// Decode `AttrsTag` into a generic `Dictionary` attribute.
template <>
struct internal::Decode<internal::AttrsTag<Dictionary>> {
  static std::optional<Dictionary> call(DecodingOffsets& offsets,
                                        DecodingContext& ctx,
                                        DiagnosticEngine& diagnostic) {
    return Dictionary(&ctx.call_frame->attrs);
  }
};

// Decode `AttrsTag` into a type `T` relying on struct decoding defined below.
template <typename T>
struct internal::Decode<internal::AttrsTag<T>> {
  static std::optional<T> call(DecodingOffsets& offsets, DecodingContext& ctx,
                               DiagnosticEngine& diagnostic) {
    return AttrDecoding<T>::Decode(
        XLA_FFI_AttrType_DICTIONARY,
        const_cast<XLA_FFI_Attrs*>(&ctx.call_frame->attrs), diagnostic);
  }
};

//===----------------------------------------------------------------------===//
// Template metaprogramming for decoding handler signature
//===----------------------------------------------------------------------===//

namespace internal {
// A helper struct to extract the type of the handler argument.
template <typename T>
struct FnArgType {
  using Type = T;
};

template <>
struct FnArgType<internal::RemainingArgsTag> {
  using Type = RemainingArgs;
};

template <>
struct FnArgType<internal::RemainingRetsTag> {
  using Type = RemainingResults;
};

// Extracts the underlying type from the returned result type tag.
template <typename T>
struct FnArgType<internal::RetTag<T>> {
  using Type = Result<T>;
};

// Extracts the underlying type from the attribute type tag.
template <typename T>
struct FnArgType<internal::AttrTag<T>> {
  using Type = typename AttrDecoding<T>::Type;
};

template <typename T>
struct FnArgType<internal::AttrsTag<T>> {
  using Type = T;
};

// Extracts the underlying type from the context type tag.
template <typename T>
struct FnArgType<internal::CtxTag<T>> {
  using Type = typename CtxDecoding<T>::Type;
};

// A template for checking if type in a parameter pack is a tagged one and has
// a special decoding rule defined by template specialization.
template <typename>
struct IsTagged : std::false_type {};
template <typename T>
struct IsTagged<RetTag<T>> : std::true_type {};
template <typename T>
struct IsTagged<AttrTag<T>> : std::true_type {};
template <typename T>
struct IsTagged<AttrsTag<T>> : std::true_type {};
template <typename T>
struct IsTagged<CtxTag<T>> : std::true_type {};
template <>
struct IsTagged<RemainingArgsTag> : std::true_type {};
template <>
struct IsTagged<RemainingRetsTag> : std::true_type {};

// A template for counting regular arguments in the Ts pack.
template <typename... Ts>
struct NumArgs;

template <>
struct NumArgs<> {
  static constexpr int64_t value = 0;
};

template <typename T, typename... Ts>
struct NumArgs<T, Ts...> {
  static constexpr int64_t value = !IsTagged<T>::value + NumArgs<Ts...>::value;
};

}  // namespace internal

//===----------------------------------------------------------------------===//
// Handler decodes FFI call frame and invokes `Fn` with decoded arguments
//===----------------------------------------------------------------------===//

template <typename Fn, typename... Ts>
class Handler : public Ffi {
  static constexpr int64_t kSize = sizeof...(Ts);

  static constexpr int64_t kNumArgs = internal::NumArgs<Ts...>::value;

  static constexpr int64_t kNumRets =
      internal::NumTagged<internal::RetTag, Ts...>::value;

  static constexpr int64_t kNumAttrs =
      internal::NumTagged<internal::AttrTag, Ts...>::value;

  static constexpr int64_t kNumDictAttrs =
      internal::NumTagged<internal::AttrsTag, Ts...>::value;

  static_assert(kNumAttrs == 0 || kNumDictAttrs == 0,
                "dictionary attributes can't be mixed with regular ones");

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
    if (internal::HasRemainingArgsTag<Ts...>::value) {
      if (XLA_FFI_PREDICT_FALSE(call_frame->args.size < kNumArgs)) {
        return InvalidArgument(
            call_frame->api,
            StrCat("Wrong number of arguments: expected at least ",
                   kNumArgs - 1, " but got ", call_frame->args.size));
      }
    } else {
      if (XLA_FFI_PREDICT_FALSE(call_frame->args.size != kNumArgs)) {
        return InvalidArgument(
            call_frame->api,
            StrCat("Wrong number of arguments: expected ", kNumArgs,
                   " but got ", call_frame->args.size));
      }
    }

    // Check that the number of results matches the signature. Each individual
    // result decoding will check the actual type.
    if (internal::HasRemainingRetsTag<Ts...>::value) {
      if (XLA_FFI_PREDICT_FALSE(call_frame->rets.size < kNumRets)) {
        return InvalidArgument(
            call_frame->api,
            StrCat("Wrong number of results: expected at least ", kNumRets - 1,
                   " but got ", call_frame->rets.size));
      }
    } else {
      if (XLA_FFI_PREDICT_FALSE(call_frame->rets.size != kNumRets)) {
        return InvalidArgument(
            call_frame->api,
            StrCat("Wrong number of results: expected ", kNumRets, " but got ",
                   call_frame->rets.size));
      }
    }

    // Check that the number of passed attributes matches the signature. Each
    // individual attribute decoding will check the actual type. If we decode
    // attributes into a dictionary (or a custom struct decoded from a
    // dictionary), then there is no need to check attributes, as the FFI
    // handler (or a struct decoding) should be responsible for it.
    if (XLA_FFI_PREDICT_FALSE(kNumDictAttrs == 0 &&
                              call_frame->attrs.size != kNumAttrs)) {
      return InvalidArgument(
          call_frame->api,
          StrCat("Wrong number of attributes: expected ", kNumAttrs,
                 " but got ", call_frame->attrs.size));
    }

    // Define index sequences to access custom call operands.
    using Is = std::make_index_sequence<kSize>;

    return Call(call_frame, Is{});
  }

 private:
  template <size_t... Is>
  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE XLA_FFI_Error* Call(
      const XLA_FFI_CallFrame* call_frame, std::index_sequence<Is...>) const {
    // A helper structure to allow each decoder find the correct offset.
    internal::DecodingOffsets offsets;

    // Package all the data required for decoding ffi handler operands.
    internal::DecodingContext ctx = {call_frame, attrs_.data(),
                                     attrs_idx_.data()};

    DiagnosticEngine diagnostic;

    std::tuple<std::optional<FnArgType<Ts>>...> args = {
        internal::Decode<Ts>::call(offsets, ctx, diagnostic)...};

    bool all_decoded = (std::get<Is>(args).has_value() && ...);
    if (XLA_FFI_PREDICT_FALSE(!all_decoded)) {
      return FailedDecodeError(call_frame, {std::get<Is>(args).has_value()...},
                               diagnostic);
    }

    auto result = fn_(std::move(*std::get<Is>(args))...);
    return ResultEncoding<ResultType>::Encode(call_frame->api,
                                              std::move(result));
  }

  XLA_FFI_Error* FailedDecodeError(const XLA_FFI_CallFrame* call_frame,
                                   std::array<bool, kSize> decoded,
                                   const DiagnosticEngine& diagnostic) const {
    auto stage = [&] {
      switch (call_frame->stage) {
        case XLA_FFI_ExecutionStage_PREPARE:
          return "prepare";
        case XLA_FFI_ExecutionStage_INITIALIZE:
          return "initialize";
        case XLA_FFI_ExecutionStage_EXECUTE:
          return "execute";
      }
    };

    std::stringstream message;
    message << "[" << stage() << "] "
            << "Failed to decode all FFI handler operands (bad operands at: ";
    for (size_t cnt = 0, idx = 0; idx < kSize; ++idx) {
      if (!decoded[idx]) {
        if (cnt++) message << ", ";
        message << std::to_string(idx);
      }
    }
    message << ")";
    if (auto s = std::move(diagnostic).Result(); !s.empty()) {
      message << "\nDiagnostics:\n" << s;
    }
    return InvalidArgument(call_frame->api, message.str());
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

#define XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(T, TYPE)                \
  template <>                                                         \
  struct AttrDecoding<T> {                                            \
    using Type = T;                                                   \
    static std::optional<T> Decode(XLA_FFI_AttrType type, void* attr, \
                                   DiagnosticEngine& diagnostic) {    \
      if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_AttrType_SCALAR)) {   \
        return diagnostic.Emit("Wrong attribute type: expected ")     \
               << XLA_FFI_AttrType_SCALAR << " but got " << type;     \
      }                                                               \
                                                                      \
      auto* scalar = reinterpret_cast<XLA_FFI_Scalar*>(attr);         \
      if (XLA_FFI_PREDICT_FALSE(scalar->dtype != TYPE)) {             \
        return diagnostic.Emit("Wrong scalar data type: expected ")   \
               << TYPE << " but got " << scalar->dtype;               \
      }                                                               \
                                                                      \
      return *reinterpret_cast<T*>(scalar->value);                    \
    }                                                                 \
  }

XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(bool, XLA_FFI_DataType_PRED);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(int8_t, XLA_FFI_DataType_S8);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(int16_t, XLA_FFI_DataType_S16);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(int32_t, XLA_FFI_DataType_S32);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(int64_t, XLA_FFI_DataType_S64);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(uint8_t, XLA_FFI_DataType_U8);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(uint16_t, XLA_FFI_DataType_U16);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(uint32_t, XLA_FFI_DataType_U32);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(uint64_t, XLA_FFI_DataType_U64);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(float, XLA_FFI_DataType_F32);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(double, XLA_FFI_DataType_F64);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(std::complex<float>,
                                      XLA_FFI_DataType_C64);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(std::complex<double>,
                                      XLA_FFI_DataType_C128);

#undef XLA_FFI_REGISTER_SCALAR_ATTR_DECODING

template <>
struct AttrDecoding<std::string_view> {
  using Type = std::string_view;
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

template <>
struct AttrDecoding<Dictionary> {
  using Type = Dictionary;
  static std::optional<Dictionary> Decode(XLA_FFI_AttrType type, void* attr,
                                          DiagnosticEngine& diagnostic) {
    if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_AttrType_DICTIONARY)) {
      return diagnostic.Emit("Wrong attribute type: expected ")
             << XLA_FFI_AttrType_DICTIONARY << " but got " << type;
    }

    auto* attrs = reinterpret_cast<XLA_FFI_Attrs*>(attr);
    return Dictionary(attrs);
  }
};

//===----------------------------------------------------------------------===//
// Automatic dictionary attributes to structs decoding.
//===----------------------------------------------------------------------===//

template <typename T>
struct StructMember {
  using Type = T;

  explicit StructMember(std::string_view name) : name(name) {}
  std::string_view name;
};

namespace internal {

// Decodes dictionary attribute into the object of type `T` that must be
// constructible from the `Ts` types.
template <typename T, typename... Ts>
struct DecodeDictionaryAttr {
  static constexpr size_t kSize = sizeof...(Ts);

  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE
  static std::optional<T> Decode(const XLA_FFI_Attrs* attrs,
                                 std::array<std::string_view, kSize> names,
                                 DiagnosticEngine& diagnostic) {
    return Decode(attrs, names, std::make_index_sequence<kSize>{}, diagnostic);
  }

  template <size_t... Is>
  XLA_FFI_ATTRIBUTE_ALWAYS_INLINE static std::optional<T> Decode(
      const XLA_FFI_Attrs* attrs, std::array<std::string_view, kSize> names,
      std::index_sequence<Is...>, DiagnosticEngine& diagnostic) {
    if (XLA_FFI_PREDICT_FALSE(kSize != attrs->size)) {
      return diagnostic.Emit("Wrong number of attributes: expected ")
             << kSize << " attributes but got " << attrs->size;
    }

    // TODO(ezhulenev): We rely on dictionary to lookup struct members by name
    // at run time, however it can become really expensive. We should
    // pre-compute mapping from `names` to index in the `XLA_FFI_Attrs`
    // (attributes ordered by name) in a static variable, and rely on it
    // to decode attributes with constant run time complexity.
    //
    // Consider using `static auto decoder = ...` below, and compute mapping in
    // constructor. Add benchmarks first to know what to improve!
    Dictionary dict(attrs);

    std::tuple<std::optional<Ts>...> members = {
        dict.get<Ts>(names[Is], diagnostic)...};
    bool all_decoded = (std::get<Is>(members).has_value() && ...);
    if (XLA_FFI_PREDICT_FALSE(!all_decoded)) return std::nullopt;

    return T{std::move(*std::get<Is>(members))...};
  }
};

template <typename... Members>
auto StructMemberNames(Members... m) {
  return std::array<std::string_view, sizeof...(Members)>{m.name...};
}

template <typename T, typename... Members>
auto DictionaryDecoder(Members... m) {
  return DecodeDictionaryAttr<T, typename Members::Type...>();
}

}  // namespace internal

// Example: register decoding for a user-defined struct
//
//   struct PairOfI64 { int64_t a; int64_t b; };
//
//   XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(
//     PairOfI64,
//     StructMember<int64_t>("a"),
//     StructMember<int64_t>("b"));
//
// Automatically registers attributes binding for a struct that allows automatic
// binding specification inference from a callable signature.
//
#define XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(T, ...)                   \
  template <>                                                           \
  struct AttrsBinding<T> {                                              \
    using Attrs = T;                                                    \
  };                                                                    \
                                                                        \
  template <>                                                           \
  struct AttrDecoding<T> {                                              \
    using Type = T;                                                     \
    static std::optional<T> Decode(XLA_FFI_AttrType type, void* attr,   \
                                   DiagnosticEngine& diagnostic) {      \
      if (XLA_FFI_PREDICT_FALSE(type != XLA_FFI_AttrType_DICTIONARY)) { \
        diagnostic.Emit("Wrong attribute type: expected ")              \
            << XLA_FFI_AttrType_DICTIONARY << " but got " << type;      \
        return std::nullopt;                                            \
      }                                                                 \
                                                                        \
      auto decoder = internal::DictionaryDecoder<T>(__VA_ARGS__);       \
      return decltype(decoder)::Decode(                                 \
          reinterpret_cast<const XLA_FFI_Attrs*>(attr),                 \
          internal::StructMemberNames(__VA_ARGS__), diagnostic);        \
    }                                                                   \
  }

// Registers decoding for a user-defined enum class type. Uses enums underlying
// type to decode the attribute as a scalar value and cast it to the enum type.
#define XLA_FFI_REGISTER_ENUM_ATTR_DECODING(T)                                \
  template <>                                                                 \
  struct ::xla::ffi::AttrDecoding<T> {                                        \
    using Type = T;                                                           \
    using U = std::underlying_type_t<Type>;                                   \
    static_assert(std::is_enum<Type>::value, "Expected enum class");          \
                                                                              \
    static std::optional<Type> Decode(XLA_FFI_AttrType attr_type, void* attr, \
                                      DiagnosticEngine& diagnostic) {         \
      if (XLA_FFI_PREDICT_FALSE(attr_type != XLA_FFI_AttrType_SCALAR)) {      \
        return diagnostic.Emit("Wrong attribute type: expected ")             \
               << XLA_FFI_AttrType_SCALAR << " but got " << attr_type;        \
      }                                                                       \
                                                                              \
      auto* scalar = reinterpret_cast<XLA_FFI_Scalar*>(attr);                 \
      auto expected_dtype = internal::NativeTypeToCApiDataType<U>();          \
      if (XLA_FFI_PREDICT_FALSE(scalar->dtype != expected_dtype)) {           \
        return diagnostic.Emit("Wrong scalar data type: expected ")           \
               << expected_dtype << " but got " << scalar->dtype;             \
      }                                                                       \
                                                                              \
      auto underlying = *reinterpret_cast<U*>(scalar->value);                 \
      return static_cast<Type>(underlying);                                   \
    }                                                                         \
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

// Use explicit binding specification to create a handler.
#define XLA_FFI_DEFINE_HANDLER_EXPLICIT(fn, impl, binding)                    \
  static constexpr XLA_FFI_Handler* fn = +[](XLA_FFI_CallFrame* call_frame) { \
    static auto* handler = binding.To(impl).release();                        \
    return handler->Call(call_frame);                                         \
  }

// Automatically infer binding specification from the implementation.
#define XLA_FFI_DEFINE_HANDLER_AUTO(fn, impl)                                 \
  static constexpr XLA_FFI_Handler* fn = +[](XLA_FFI_CallFrame* call_frame) { \
    static auto* handler = ::xla::ffi::Ffi::BindTo(impl).release();           \
    return handler->Call(call_frame);                                         \
  }

#define XLA_FFI_DEFINE_HANDLER_X(x, fn, impl, binding, FUNC, ...) FUNC

// Define XLA FFI handler as a static function pointer variable, which allows
// to define handlers in nested scopes without polluting the global namespace.
//
// This is a trick to define macro with optional parameters.
// Source: https://stackoverflow.com/a/8814003
#define XLA_FFI_DEFINE_HANDLER(fn, impl, ...)                 \
  XLA_FFI_DEFINE_HANDLER_X(                                   \
      , fn, impl, ##__VA_ARGS__,                              \
      XLA_FFI_DEFINE_HANDLER_EXPLICIT(fn, impl, __VA_ARGS__), \
      XLA_FFI_DEFINE_HANDLER_AUTO(fn, impl))

// TODO(ezhulenev): Add a callback so that end users can log registration error
// to appropriate logging destination, e.g. LOG(FATAL) for duplicate internal
// FFI handlers.
#define XLA_FFI_REGISTER_HANDLER(API, NAME, PLATFORM, FUNC, ...)    \
  XLA_FFI_REGISTER_HANDLER_(API, NAME, PLATFORM, FUNC, __COUNTER__, \
                            ##__VA_ARGS__)
#define XLA_FFI_REGISTER_HANDLER_(API, NAME, PLATFORM, FUNC, N, ...) \
  XLA_FFI_REGISTER_HANDLER__(API, NAME, PLATFORM, FUNC, N, ##__VA_ARGS__)
#define XLA_FFI_REGISTER_HANDLER__(API, NAME, PLATFORM, FUNC, N, ...)       \
  XLA_FFI_ATTRIBUTE_UNUSED static const XLA_FFI_Error*                      \
      xla_ffi_static_handler_##N##_registered_ = [] {                       \
        return ::xla::ffi::Ffi::RegisterStaticHandler(API, NAME, PLATFORM,  \
                                                      FUNC, ##__VA_ARGS__); \
      }()

// Following two APIs are intended for users who want to export XLA FFI handler
// from a shared library as a C function symbol.

// Declares C function that implements FFI handler.
#define XLA_FFI_DECLARE_HANDLER_SYMBOL(fn) \
  extern "C" XLA_FFI_Error* fn(XLA_FFI_CallFrame* call_frame)

// Defines C function that implements FFI handler.
#define XLA_FFI_DEFINE_HANDLER_SYMBOL(fn, impl, ...)                           \
  extern "C" XLA_FFI_Error* fn(XLA_FFI_CallFrame* call_frame) {                \
    XLA_FFI_DEFINE_HANDLER(handler, impl, ##__VA_ARGS__);                      \
    return (*handler)(call_frame);                                             \
  }                                                                            \
                                                                               \
  static_assert(                                                               \
      std::is_invocable_r_v<XLA_FFI_Error*, decltype(fn), XLA_FFI_CallFrame*>, \
      "FFI handler must return XLA_FFI_Error* and accept XLA_FFI_CallFrame*")

}  // namespace xla::ffi

#endif  // XLA_FFI_API_API_H_

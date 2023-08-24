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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_CUSTOM_CALL_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_CUSTOM_CALL_H_

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/dynamic_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "third_party/eigen3/Eigen/Core"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/runtime/async_runtime.h"
#include "tensorflow/compiler/xla/runtime/diagnostics.h"
#include "tensorflow/compiler/xla/runtime/errors.h"
#include "tensorflow/compiler/xla/runtime/ffi/ffi_abi.h"
#include "tensorflow/compiler/xla/runtime/logical_result.h"
#include "tensorflow/compiler/xla/runtime/map_by_type.h"
#include "tensorflow/compiler/xla/runtime/memref_view.h"
#include "tensorflow/compiler/xla/runtime/state.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"
#include "tfrt/concurrency/async_value_ref.h"  // from @tf_runtime
#include "tfrt/concurrency/chain.h"  // from @tf_runtime

namespace xla {
namespace runtime {

// Forward declare.
struct ExecutionContext;

// Forward declare template defined below.
template <typename... Ts>
class CustomCallBinding;

// Registers mappings from TypeIDs supported by the custom calls to their unique
// names in the given registry.
void PopulateCustomCallTypeIdNames(TypeIDNameRegistry& registry);

// A type tag to declare MLIR TypeID specializations for types passed to the
// custom calls. We don't want to declare specializations for scalar types
// directly in this translation unit, so we rely on a tag to wrap them.
//
// See explicit TypeID declarations at the end of this file.
template <typename T>
struct Tagged {};

class CustomCall {
 public:
  // Container for passing data between XLA user and the custom call handler.
  using UserData = PtrMapByType<CustomCall>;

  // A type for matching all remaining custom call arguments.
  class RemainingArgs;

  // A type for passing an argument of different types at the same position,
  // and the handler will do the decoding.
  class VariantArg;
  class VariantAttr;

  // A type for representing tensors with shapes.
  template <typename T>
  struct TensorRef {
    absl::Span<const int64_t> shape;
    absl::Span<const T> data;
  };

  // An ordinal of a function exported from executable.
  struct FunctionOrdinal {
    unsigned ordinal;
  };

  // Custom call handler can check arguments and attributes types and names
  // at runtime, however this comes at extra cost and can be optionally
  // disabled. If the version of the compiler that generated the XLA executable
  // doesn't match the custom call handler, it can lead to undefined behavior.
  enum class RuntimeChecks : uint8_t {
    // Check arguments and attributes types, also check attribute names. It is
    // safe to pass extra attributes (if `exact_attrs` is false) to the custom
    // call when name checking is enabled, because it will safely skip
    // irrelevant attributes.
    kDefault = 0,

    // Check only the types of the arguments and attributes. At this check level
    // custom calls never check the names of the attributes because it can be
    // too expensive, however type checking should prevent catastrophic
    // segfaults. This is the recommended checks level in optimized builds.
    kLess = 1,

    // Do not check the number of arguments and attributes and their types, and
    // do not check that the user data was passed to the custom call. This is
    // the most dangerous option, because it blindly reinterprets opaque memory
    // passed to the handler, and can easily lead to segfaults if the data
    // doesn't match the expected custom call signature.
    kNone = 2
  };

  struct Options {
    // Check that attributes passed at run time exactly match the attributes
    // defined by the custom call binding. If `false` then custom call handler
    // will happily ignore any additional attributes passed at run time. It is
    // unsafe to disable run-time checks for custom calls that support non-exact
    // attributes, as the custom call handler uses pre-computed attributes
    // offsets based on the binding specification.
    bool exact_attrs = true;
  };

  static constexpr bool CheckNames(RuntimeChecks checks) {
    return checks == RuntimeChecks::kDefault;
  }

  static constexpr bool CheckTypes(RuntimeChecks checks) {
    return checks != RuntimeChecks::kNone;
  }

  static constexpr bool CheckUserData(RuntimeChecks checks) {
    return checks != RuntimeChecks::kNone;
  }

  template <typename T>
  static bool Isa(RuntimeChecks checks, TypeID type_id) {
    return !CheckTypes(checks) || type_id == TypeID::get<Tagged<T>>();
  }

  template <typename T, typename U, typename... Ts>
  static bool Isa(RuntimeChecks checks, TypeID type_id) {
    return !CheckTypes(checks) || type_id == TypeID::get<Tagged<T>>() ||
           Isa<U, Ts...>(checks, type_id);
  }

  virtual ~CustomCall() = default;

  virtual std::string_view name() const = 0;
  virtual LogicalResult call(void** args, void** attrs, void** rets,
                             const UserData* user_data,
                             const DiagnosticEngine* diagnostic) const = 0;

  static CustomCallBinding<> Bind(std::string callee);
  static CustomCallBinding<> Bind(std::string callee, const Options& opts);

  // This is a helper template that allows to convert functions pointers from
  // the run time values to compile time values (template arguments) with
  // automatic template arguments inference.
  //
  // Example:
  //
  //   static LogicalResult Foo(int32_t arg) {... }
  //
  //   template<typename Callable>
  //   void call(Callable callable) { callable(42); }
  //
  //   call(Foo);                     // `Foo` passed as a runtime value
  //   call(FunctionWrapper<Foo>())   // `Foo` passed as a template argument
  //
  // In the first case compiler will not be able to inline `Foo` into the `call`
  // body. However in the second case it can do that, because function pointer
  // is a statically known value (template non-type argument).
  template <auto fn>
  struct FunctionWrapper;

  template <typename Ret, typename... Args, Ret (*fn)(Args...)>
  struct FunctionWrapper<fn> {
    ABSL_ATTRIBUTE_ALWAYS_INLINE Ret operator()(Args... args) const {
      return fn(args...);
    }
  };
};

// Forward declare template defined below.
template <CustomCall::RuntimeChecks checks, typename Fn, typename... Ts>
class CustomCallHandler;

namespace internal {

// A type tag to distinguish arguments tied to the attributes in the
// `CustomCallBinding` variadic template argument.
template <typename T>
struct Attr {};

// A type tag to distinguish arguments tied to the return in the
// `CustomCallBinding` variadic template argument.
template <typename T>
struct Ret {};

// A type tag to distinguish arguments tied to the user data in the
// `CustomCallBinding` variadic template argument.
template <typename T>
struct UserData {};

// A type tag to distinguish arguments tied to the state in the
// `CustomCallBinding` variadic template argument.
template <typename T>
struct StateTag {};

// A type tag to distinguish arguments tied to the constant values in the
// `CustomCallBinding` variadic template argument.
template <typename T>
struct Value {};

// A template for checking if type is a regular argument or one of the special
// arguments wrapped in a type tag (e.g. attr, user data, etc...).
template <typename>
struct IsWrapped : std::false_type {};

template <typename T>
struct IsWrapped<internal::Attr<T>> : std::true_type {};

template <typename T>
struct IsWrapped<internal::Ret<T>> : std::true_type {};

template <typename T>
struct IsWrapped<internal::UserData<T>> : std::true_type {};

template <typename T>
struct IsWrapped<internal::StateTag<T>> : std::true_type {};

template <typename T>
struct IsWrapped<internal::Value<T>> : std::true_type {};

template <typename T>
struct IsResult : std::false_type {};

template <typename T>
struct IsResult<internal::Ret<T>> : std::true_type {};

// Checks if remaining arguments are in the parameter pack.
template <typename... Ts>
using HasRemainingArgs =
    std::disjunction<std::is_same<CustomCall::RemainingArgs, Ts>...>;

}  // namespace internal

// Custom call binding describes the function signature of the expected custom
// call handler using its variadic template parameter.
//
//   Custom call binding:
//     CustomCallBinding<int32_t, MemrefView>
//
//   Function signature:
//     LogicalResult MyHandle(int32_t algo, MemrefView memref);
//
template <typename... Ts>
class CustomCallBinding {
 public:
  using Options = CustomCall::Options;
  using RuntimeChecks = CustomCall::RuntimeChecks;

  template <typename T>
  CustomCallBinding<Ts..., T> Arg() && {
    return {std::move(*this)};
  }

  CustomCallBinding<Ts..., CustomCall::RemainingArgs> RemainingArgs() && {
    static_assert(!internal::HasRemainingArgs<Ts...>::value,
                  "remaining arguments can be passed just once");
    return {std::move(*this)};
  }

  template <typename T>
  CustomCallBinding<Ts..., internal::Attr<T>> Attr(std::string attr) && {
    attrs_.push_back(std::move(attr));
    return {std::move(*this)};
  }

  template <typename T>
  CustomCallBinding<Ts..., internal::Ret<T>> Ret() && {
    return {std::move(*this)};
  }

  template <typename T>
  CustomCallBinding<Ts..., internal::UserData<T>> UserData() && {
    static_assert(std::is_pointer<T>::value, "user data must be a pointer");
    return {std::move(*this)};
  }

  template <typename T>
  CustomCallBinding<Ts..., internal::StateTag<T>> State(std::string id) && {
    attrs_.push_back(std::move(id));
    return {std::move(*this)};
  }

  template <typename T>
  CustomCallBinding<Ts..., internal::Value<T>> Value(T value) && {
    values_.push_back(std::move(value));
    return {std::move(*this)};
  }

  template <RuntimeChecks checks = RuntimeChecks::kDefault, typename Fn>
  std::unique_ptr<CustomCallHandler<checks, Fn, Ts...>> To(Fn fn) {
    return std::unique_ptr<CustomCallHandler<checks, Fn, Ts...>>(
        new CustomCallHandler<checks, Fn, Ts...>(
            std::forward<Fn>(fn), std::move(callee_), std::move(attrs_),
            std::move(values_), opts_));
  }

 private:
  template <typename...>
  friend class CustomCallBinding;
  friend class CustomCall;

  CustomCallBinding(std::string callee, const Options& opts)
      : callee_(std::move(callee)), opts_(opts) {
    static_assert(sizeof...(Ts) == 0, "custom call arguments must be empty");
  }

  template <typename... TTs>
  CustomCallBinding(CustomCallBinding<TTs...>&& other)  // NOLINT
      : callee_(std::move(other.callee_)),
        attrs_(std::move(other.attrs_)),
        values_(std::move(other.values_)),
        opts_(other.opts_) {}

  CustomCallBinding(CustomCallBinding&) = delete;

  std::string callee_;              // custom call target
  std::vector<std::string> attrs_;  // names of bound attributes
  std::vector<std::any> values_;    // values bound to arguments
  Options opts_;
};

inline CustomCallBinding<> CustomCall::Bind(std::string callee) {
  return Bind(std::move(callee), Options());
}

inline CustomCallBinding<> CustomCall::Bind(std::string callee,
                                            const Options& opts) {
  return CustomCallBinding<>(std::move(callee), opts);
}

// Custom calls return results to the caller through the template
// specializations of the `Result`. Each template specialization is responsible
// for definining the result encoding/decoding to/from opaque memory.
template <typename T>
class Result;

// Custom call arguments decoding must be defined by specializing this template.
//
// Example: decoding for the `MyType` arguments
//
//   template <CustomCall::RuntimeChecks checks>
//   struct CustomCallArgDecoding<MyType, checks> {
//    static FailureOr<MyType> Decode(TypeID type_id, void* value);
//   };
//
template <typename T, CustomCall::RuntimeChecks>
struct CustomCallArgDecoding;

// Custom call attribute decoding must be defined by specializing this template.
//
// Example: decoding for the `MyType` attributes
//
//   template <CustomCall::RuntimeChecks checks>
//   struct CustomCallAttrDecoding<MyType, checks> {
//    static FailureOr<MyType> Decode(std::string_view name,
//                                    TypeID type_id, void* value);
//   }
//
template <typename T, CustomCall::RuntimeChecks>
struct CustomCallAttrDecoding;

// Custom call returns decoding must be defined by specializing this template.
//
// Example: decoding for the `MyType` arguments
//
//   template <CustomCall::RuntimeChecks checks>
//   struct CustomCallRetDecoding<MyType, checks> {
//    static FailureOr<Result<MyType>> Decode(TypeID type_id, void* value);
//   };
//
template <typename T, CustomCall::RuntimeChecks>
struct CustomCallRetDecoding;

//===----------------------------------------------------------------------===//
// Helpers for decoding opaque arguments and attributes memory.
//===----------------------------------------------------------------------===//

namespace internal {

// Decoded pair of an argument type and opaque value.
struct DecodedArg {
  TypeID type_id;
  void* value;
};

// Decoded triple of an attribute name, type and opaque value.
struct DecodedAttr {
  std::string_view name;
  TypeID type_id;
  void* value;
};

// A convenience wrapper around opaque arguments memory.
class DecodedArgs {
 public:
  explicit DecodedArgs(void** args) {
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(args, sizeof(void*));
    size_ = *reinterpret_cast<int64_t*>(args[0]);
    if (size_) {
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(args + 1, sizeof(void*));
      type_table_ = reinterpret_cast<void**>(args[1]);
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(type_table_, size_ * sizeof(void*));
      values_ = args + 2;
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(values_, size_ * sizeof(void*));
    }
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE int64_t size() const { return size_; }

  ABSL_ATTRIBUTE_ALWAYS_INLINE DecodedArg operator[](size_t i) const {
    DecodedArg arg;
    arg.type_id = TypeID::getFromOpaquePointer(type_table_[i]);
    arg.value = values_[i];
    return arg;
  }

 private:
  int64_t size_;
  void** type_table_ = nullptr;
  void** values_ = nullptr;
};

// A convenience wrapper around opaque attributes memory.
class DecodedAttrs {
 public:
  explicit DecodedAttrs(void** attrs) : encoded_(attrs + 1) {
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(attrs, sizeof(void*));
    size_ = *reinterpret_cast<int64_t*>(attrs[0]);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(encoded_, 3 * size_ * sizeof(void*));
  }

  ABSL_ATTRIBUTE_ALWAYS_INLINE int64_t size() const { return size_; }

  ABSL_ATTRIBUTE_ALWAYS_INLINE DecodedAttr operator[](size_t i) const {
    void** attr_base = encoded_ + i * 3;

    DecodedAttr attr;
    auto* name = reinterpret_cast<internal::EncodedArray<char>*>(attr_base[0]);
    attr.name = std::string_view(name->data, name->size);
    attr.type_id = TypeID::getFromOpaquePointer(attr_base[1]);
    attr.value = attr_base[2];

    return attr;
  }

 private:
  void** encoded_;
  int64_t size_;
};

// Using the same class for decoded returns
using DecodedRet = DecodedArg;
using DecodedRets = DecodedArgs;

}  // namespace internal

//===----------------------------------------------------------------------===//
// CustomCall remaining arguments wraps the type-erased `DecodedArg` container,
// and provides a type-safe API for accessing individual arguments.
//===----------------------------------------------------------------------===//

class CustomCall::RemainingArgs {
 public:
  using RuntimeChecks = CustomCall::RuntimeChecks;

  RemainingArgs(internal::DecodedArgs args, size_t offset)
      : args_(args), offset_(offset) {
    assert(offset <= args_.size() && "illegal remaining args offset");
  }

  size_t size() const { return args_.size() - offset_; }
  bool empty() const { return size() == 0; }

  template <typename T>
  bool isa(size_t index) const {
    return args_[index + offset_].type_id == TypeID::get<Tagged<T>>();
  }

  template <typename T, RuntimeChecks checks = RuntimeChecks::kDefault>
  FailureOr<T> get(size_t index) const {
    internal::DecodedArg arg = args_[index + offset_];
    return CustomCallArgDecoding<T, checks>::Decode(arg.type_id, arg.value);
  }

 private:
  internal::DecodedArgs args_;
  size_t offset_;
};

class CustomCall::VariantArg {
 public:
  using RuntimeChecks = CustomCall::RuntimeChecks;

  VariantArg(internal::DecodedArgs args, size_t offset)
      : args_(args), offset_(offset) {
    assert(offset <= args_.size() && "illegal remaining args offset");
  }

  template <typename T>
  bool isa() const {
    return args_[offset_].type_id == TypeID::get<Tagged<T>>();
  }

  template <typename T, RuntimeChecks checks = RuntimeChecks::kDefault>
  FailureOr<T> get() const {
    internal::DecodedArg arg = args_[offset_];
    return CustomCallArgDecoding<T, checks>::Decode(arg.type_id, arg.value);
  }

 private:
  internal::DecodedArgs args_;
  size_t offset_;
};

class CustomCall::VariantAttr {
 public:
  using RuntimeChecks = CustomCall::RuntimeChecks;

  VariantAttr(std::string_view name, TypeID type_id, void* value)
      : name_(name), type_id_(type_id), value_(value) {}

  template <typename T>
  bool isa() const {
    return type_id_ == TypeID::get<Tagged<T>>();
  }

  template <typename T, RuntimeChecks checks = RuntimeChecks::kDefault>
  FailureOr<T> get() const {
    return CustomCallAttrDecoding<T, checks>::Decode(name_, type_id_, value_);
  }

 private:
  std::string_view name_;
  TypeID type_id_;
  void* value_;
};

//===----------------------------------------------------------------------===//
// A little bit of template metaprogramming to implement type safe binding
// of custom calls to C++ functions. This is internal implementation details,
// and must not be relied on in any of the client code.
//===----------------------------------------------------------------------===//

namespace internal {

// A helper struct to extract the type of the handler argument.
template <typename T>
struct FnArgType {
  using Type = T;
};

// Extracts the underlying type from the attribute type tag.
template <typename T>
struct FnArgType<internal::Attr<T>> {
  using Type = T;
};

// Extracts the underlying type from the return type tag.
template <typename T>
struct FnArgType<internal::Ret<T>> {
  using Type = Result<T>;
};

// Extracts the underlying type from the user data type tag.
template <typename T>
struct FnArgType<internal::UserData<T>> {
  using Type = T;
};

// Extracts the underlying type from the state type tag.
template <typename T>
struct FnArgType<internal::StateTag<T>> {
  using Type = State<T>;
};

// Extracts the underlying type from the value type tag.
template <typename T>
struct FnArgType<internal::Value<T>> {
  using Type = T;
};

// A template for counting regular arguments in the Ts pack.
template <typename... Ts>
struct NumArgs;

template <typename T, typename... Ts>
struct NumArgs<T, Ts...> {
  static constexpr int64_t value = !IsWrapped<T>::value + NumArgs<Ts...>::value;
};

template <>
struct NumArgs<> {
  static constexpr int64_t value = 0;
};

// A template for counting returns in the Ts pack.
template <typename... Ts>
struct NumRets;

template <typename T, typename... Ts>
struct NumRets<T, Ts...> {
  static constexpr int64_t value = IsResult<T>::value + NumRets<Ts...>::value;
};

template <>
struct NumRets<> {
  static constexpr int64_t value = 0;
};

// Unwrap return type to get the type expected by result `Set` method.
//
// TODO(ezhulenev): Result template itself should define what type `T` it
// expects to see in the `Set` method, because it's not necessery the same as
// the template type of the result.
template <typename T>
struct UnwrapRet;

template <typename T>
struct UnwrapRet<Result<T>> {
  using Type = T;
};

// A helper template to concatenate index + index sequence.
template <size_t, typename>
struct ConsIdx;

template <size_t idx, size_t... Is>
struct ConsIdx<idx, std::index_sequence<Is...>> {
  using Type = std::index_sequence<idx, Is...>;
};

// Get indices of the variadic template type parameters corresponding to
// results. This template will produce an `std::index_sequence` type with
// indices of custom call result arguments.
template <size_t idx, typename... Ts>
struct IndexRets;

template <size_t idx>
struct IndexRets<idx> {
  using Is = std::index_sequence<>;
};

template <size_t idx, typename T, typename... Ts>
struct IndexRets<idx, T, Ts...> {
  using Is = std::conditional_t<
      IsResult<T>::value,
      typename ConsIdx<idx, typename IndexRets<idx + 1, Ts...>::Is>::Type,
      typename IndexRets<idx + 1, Ts...>::Is>;
};

// Get indices of the variadic template type parameters corresponding to
// all arguments excluding results.
template <size_t idx, typename... Ts>
struct IndexArgs;

template <size_t idx>
struct IndexArgs<idx> {
  using Is = std::index_sequence<>;
};

template <size_t idx, typename T, typename... Ts>
struct IndexArgs<idx, T, Ts...> {
  using Is = std::conditional_t<
      !IsResult<T>::value,
      typename ConsIdx<idx, typename IndexArgs<idx + 1, Ts...>::Is>::Type,
      typename IndexArgs<idx + 1, Ts...>::Is>;
};

template <size_t... FnArgsIs, size_t... ResultIs, typename FnArgs,
          typename Result>
void SetResultsFromTuple(std::index_sequence<ResultIs...>, FnArgs fn_args,
                         Result tuple) {
  ((*std::get<FnArgsIs>(fn_args)).Set(std::get<ResultIs>(tuple)), ...);
}

// When decoding input data we need to keep track of how many arguments,
// attributes, and returns we decoded so far to index into the correct data
// structure.
struct DecodingOffsets {
  int64_t args = 0;
  int64_t attrs = 0;
  int64_t rets = 0;
  int64_t values = 0;
};

struct DecodingContext {
  internal::DecodedArgs args;
  internal::DecodedRets rets;
  internal::DecodedAttrs attrs;

  // Attributes' names and mapping from attrs' offsets to indices in `attrs`.
  absl::Span<const std::string> attrs_names;
  absl::Span<const size_t> attrs_idx;

  // Values bound to arguments at handler construction time.
  absl::Span<const std::any> values;

  // User-provided auxiliary data.
  const CustomCall::UserData* user_data;

  // User-provided diagnostic engine for reporting detailed errors.
  const DiagnosticEngine* diagnostic;
};

template <typename T, CustomCall::RuntimeChecks checks>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline FailureOr<T*> DecodeUserData(
    const CustomCall::UserData* user_data) {
  if (!CustomCall::CheckUserData(checks)) return user_data->get<T>();

  // TODO(ezhulenev): Add an option to request nullable user data, because
  // right now we do not distinguish between a user data pointer that doesn't
  // exist, and a null pointer passed by the user.

  // Get the requested value if user data was passed to the custom call.
  auto* ptr = user_data ? user_data->getIfExists<T>() : nullptr;
  if (LLVM_UNLIKELY(!ptr)) return failure();
  return ptr;
}

template <typename T, CustomCall::RuntimeChecks checks>
ABSL_ATTRIBUTE_ALWAYS_INLINE inline FailureOr<T> DecodeAttr(
    DecodingOffsets& offsets, absl::Span<const std::string> attrs_names,
    absl::Span<const size_t> attrs_idx, internal::DecodedAttrs attrs) {
  // Find decoded attribute corresponding for the given attribute index.
  int64_t idx = offsets.attrs++;

  // Do not check the attribute name, and decode attribute at the given index.
  if (!CustomCall::CheckNames(checks)) {
    size_t i = attrs_idx[idx];
    return CustomCallAttrDecoding<T, checks>::Decode(
        attrs[i].name, attrs[i].type_id, attrs[i].value);
  }

  std::string_view attr_name = attrs_names[idx];

  // Given that attributes are passed to the custom call handler
  // lexicographically sorted by name, we can find the attribute we are
  // looking for only between the `attrs_idx` offset and the end of the
  // attributes array.
  for (size_t i = attrs_idx[idx]; i < attrs.size(); ++i) {
    if (LLVM_LIKELY(attrs[i].name == attr_name))
      return CustomCallAttrDecoding<T, checks>::Decode(
          attrs[i].name, attrs[i].type_id, attrs[i].value);
  }

  // Attribute we were looking for was not passed as an argument.
  return failure();
}

template <typename T, CustomCall::RuntimeChecks checks>
struct Decode {
  ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> call(
      DecodingOffsets& offsets, DecodingContext& ctx) {
    internal::DecodedArg arg = ctx.args[offsets.args++];
    return CustomCallArgDecoding<T, checks>::Decode(arg.type_id, arg.value);
  }
};

template <typename T, CustomCall::RuntimeChecks checks>
struct Decode<internal::Ret<T>, checks> {
  ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<Result<T>> call(
      DecodingOffsets& offsets, DecodingContext& ctx) {
    internal::DecodedRet ret = ctx.rets[offsets.rets++];
    return CustomCallRetDecoding<T, checks>::Decode(ret.type_id, ret.value);
  }
};

template <typename T, CustomCall::RuntimeChecks checks>
struct Decode<internal::Attr<T>, checks> {
  ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> call(
      DecodingOffsets& offsets, DecodingContext& ctx) {
    return DecodeAttr<T, checks>(offsets, ctx.attrs_names, ctx.attrs_idx,
                                 ctx.attrs);
  }
};

template <typename T, CustomCall::RuntimeChecks checks>
struct Decode<internal::UserData<T>, checks> {
  ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> call(
      DecodingOffsets& offsets, DecodingContext& ctx) {
    using UserDataT = std::remove_pointer_t<T>;
    if (auto decoded = DecodeUserData<UserDataT, checks>(ctx.user_data);
        LLVM_LIKELY(succeeded(decoded)))
      return decoded;
    return ctx.diagnostic->EmitError(InternalError(
        "failed to decode UserData of type %s", typeid(T).name()));
  }
};

template <typename T, CustomCall::RuntimeChecks checks>
struct Decode<internal::StateTag<T>, checks> {
  using Snapshot = typename StateVector<T>::Snapshot;

  ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<runtime::State<T>> call(
      DecodingOffsets& offsets, DecodingContext& ctx) {
    // Get the state snapshot and state id from user data and attributes.
    FailureOr<Snapshot*> snapshot =
        DecodeUserData<Snapshot, checks>(ctx.user_data);
    FailureOr<int64_t> id = DecodeAttr<int64_t, checks>(
        offsets, ctx.attrs_names, ctx.attrs_idx, ctx.attrs);
    if (LLVM_UNLIKELY(failed(snapshot) || failed(id))) return failure();

    return (*snapshot)->state(*id);
  }
};

template <typename T, CustomCall::RuntimeChecks checks>
struct Decode<internal::Value<T>, checks> {
  ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> call(
      DecodingOffsets& offsets, DecodingContext& ctx) {
    return std::any_cast<T>(ctx.values[offsets.values++]);
  }
};

template <CustomCall::RuntimeChecks checks>
struct Decode<CustomCall::RemainingArgs, checks> {
  ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<CustomCall::RemainingArgs> call(
      DecodingOffsets& offsets, DecodingContext& ctx) {
    return CustomCall::RemainingArgs(ctx.args, offsets.args);
  }
};

template <CustomCall::RuntimeChecks checks>
struct Decode<CustomCall::VariantArg, checks> {
  ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<CustomCall::VariantArg> call(
      DecodingOffsets& offsets, DecodingContext& ctx) {
    return CustomCall::VariantArg(ctx.args, offsets.args++);
  }
};

}  // namespace internal

// Custom call handler binds concrete custom call implementation of type `Fn` to
// the custom call function signature. `Fn` can be a function pointer, or a
// lambda.
//
// Custom call handler uses the variadic template parameter `Ts` to decode the
// opaque pointers passed to the `call` function into the C++ types that are
// forwarded to the custom call implementation.
template <CustomCall::RuntimeChecks checks, typename Fn, typename... Ts>
class CustomCallHandler : public CustomCall {
  static constexpr int64_t kSize = sizeof...(Ts);
  static constexpr int64_t kNumArgs = internal::NumArgs<Ts...>::value;
  static constexpr int64_t kNumRets = internal::NumRets<Ts...>::value;

  template <typename T>
  using FnArgType = typename internal::FnArgType<T>::Type;

  template <typename T>
  using UnwrapRet = typename internal::UnwrapRet<T>::Type;

  // Custom call can signal error using a LogicalError result.
  static constexpr bool kIsLogicalErr =
      std::is_invocable_r_v<LogicalResult, Fn, FnArgType<Ts>...>;

  // Custom call can signal error together with a detailed error message.
  static constexpr bool kIsStatusErr =
      std::is_invocable_r_v<absl::Status, Fn, FnArgType<Ts>...>;

  // Custom call returns results as `absl::StatusOr<std::tuple<Ts...>>`
  // (multiple results) or `absl::StatusOr<T>` (single result).
  template <size_t... RetsIs, size_t... ArgsIs>
  static constexpr bool IsStatusOrInvocable(std::index_sequence<RetsIs...>,
                                            std::index_sequence<ArgsIs...>) {
    // Define a tuple to help extracting type by index.
    using ArgsTuple = std::tuple<FnArgType<Ts>...>;

    // Custom call doesn't have any results.
    if constexpr (sizeof...(RetsIs) == 0) return false;

    // Custom call returns a single result.
    if constexpr (sizeof...(RetsIs) == 1) {
      using StatusOr =
          absl::StatusOr<UnwrapRet<std::tuple_element_t<RetsIs, ArgsTuple>>...>;
      return std::is_invocable_r_v<StatusOr, Fn,
                                   std::tuple_element_t<ArgsIs, ArgsTuple>...>;
    }

    // Custom call returns multiple results as a tuple.
    if constexpr (sizeof...(RetsIs) > 1) {
      using StatusOr = absl::StatusOr<
          std::tuple<UnwrapRet<std::tuple_element_t<RetsIs, ArgsTuple>>...>>;
      return std::is_invocable_r_v<StatusOr, Fn,
                                   std::tuple_element_t<ArgsIs, ArgsTuple>...>;
    }

    llvm_unreachable("unsupported result rank");
  }

  static constexpr bool kIsStatusOrResult =
      IsStatusOrInvocable(typename internal::IndexRets<0, Ts...>::Is{},
                          typename internal::IndexArgs<0, Ts...>::Is{});

  static_assert(kIsLogicalErr || kIsStatusErr || kIsStatusOrResult,
                "incompatible custom call handler types");

 public:
  std::string_view name() const final { return callee_; }

  ABSL_ATTRIBUTE_ALWAYS_INLINE LogicalResult
  call(void** args, void** attrs, void** rets, const UserData* user_data,
       const DiagnosticEngine* diagnostic) const final {
    // Decode arguments and attributes from the opaque pointers.
    internal::DecodedArgs decoded_args(args);
    internal::DecodedAttrs decoded_attrs(attrs);
    internal::DecodedRets decoded_rets(rets);

    int64_t num_args = decoded_args.size();
    int64_t num_attrs = decoded_attrs.size();
    int64_t num_rets = decoded_rets.size();

    if (LLVM_UNLIKELY(diagnostic == nullptr))
      diagnostic = DiagnosticEngine::DefaultDiagnosticEngine();

    // If all runtime checks are disabled we are just reinterpreting opaque
    // `args`, `attrs` and `rets` memory according to the custom call handler
    // signature and skip all checks (these checks will be optimized out).
    auto eval = [](bool condition) {
      return checks == RuntimeChecks::kNone ? false : condition;
    };

    // Check that the number of passed arguments matches the signature. Each
    // individual argument decoding will check the actual type.
    if (internal::HasRemainingArgs<Ts...>::value) {
      if (LLVM_UNLIKELY(eval(num_args < kNumArgs - 1)))
        return diagnostic->EmitError(InvalidArgument(
            "Wrong number of arguments: expected at least %d got %d",
            kNumArgs - 1, num_args));
    } else {
      if (LLVM_UNLIKELY(eval(num_args != kNumArgs)))
        return diagnostic->EmitError(
            InvalidArgument("Wrong number of arguments: expected %d got %d",
                            kNumArgs, num_args));
    }

    // Check that the number of returns matches the signature. The return
    // decoding will check the actual type.
    if (LLVM_UNLIKELY(eval(num_rets != kNumRets)))
      return diagnostic->EmitError(InvalidArgument(
          "Wrong number of returns: expected %d got %d", kNumRets, num_rets));

    // Check that we have a correct number of attributes passed to the custom
    // call. Each individual attribute decoding will check the name and the
    // type of the attribute.
    if (LLVM_UNLIKELY(eval(opts_.exact_attrs ? num_attrs != num_encoded_attrs_
                                             : num_attrs < num_encoded_attrs_)))
      return diagnostic->EmitError(InvalidArgument(
          "Wrong number of attributes: expected %s%d got %d",
          opts_.exact_attrs ? "" : "at least ", num_encoded_attrs_, num_attrs));

    // Define index sequences to access custom call operands.
    using Is = std::make_index_sequence<kSize>;
    using ArgsIs = typename internal::IndexArgs<0, Ts...>::Is;
    using RetsIs = typename internal::IndexRets<0, Ts...>::Is;

    return call(decoded_args, decoded_attrs, decoded_rets, user_data,
                diagnostic, Is{}, ArgsIs{}, RetsIs{});
  }

  template <size_t... Is, size_t... ArgsIs, size_t... RetsIs>
  ABSL_ATTRIBUTE_ALWAYS_INLINE LogicalResult
  call(internal::DecodedArgs args, internal::DecodedAttrs attrs,
       internal::DecodedRets rets, const UserData* user_data,
       const DiagnosticEngine* diagnostic, std::index_sequence<Is...>,
       std::index_sequence<ArgsIs...>, std::index_sequence<RetsIs...>) const {
    // A helper structure to allow each decoder find the correct offset in the
    // arguments, attributes or results.
    internal::DecodingOffsets offsets;

    // Package all the data required for decoding custom call operands.
    internal::DecodingContext ctx{args,       rets,    attrs,     attrs_,
                                  attrs_idx_, values_, user_data, diagnostic};

    // Decode all operands into FailureOr containers. It is guaranteed
    // that initializer list will be evaluated left-to-right, and we can rely
    // on correct offsets computation.
    std::tuple<FailureOr<FnArgType<Ts>>...> fn_args = {
        internal::Decode<Ts, checks>::call(offsets, ctx)...};

    // Check if all operands and results were decoded.
    bool all_decoded = (succeeded(std::get<Is>(fn_args)) && ...);

    if (LLVM_UNLIKELY(!all_decoded)) {
      std::array<bool, kSize> decoded = {succeeded(std::get<Is>(fn_args))...};
      auto bad_args = llvm::make_filter_range(
          llvm::enumerate(decoded), [](auto pair) { return !pair.value(); });
      auto to_str = [](auto pair) { return std::to_string(pair.index()); };

      return diagnostic->EmitError(InvalidArgument(
          "Failed to decode all custom call operands (bad operads at: %s)",
          llvm::join(llvm::map_range(bad_args, to_str), ", ")));
    }

    // Custom call returns logical result to signal failures.
    if constexpr (kIsLogicalErr) {
      return fn_(std::move(*std::get<Is>(fn_args))...);
    }

    // Custom call returns detailed error to signal failures.
    if constexpr (kIsStatusErr) {
      if (auto st = fn_(std::move(*std::get<Is>(fn_args))...); !st.ok()) {
        return diagnostic->EmitError(std::move(st));
      }
      return success();
    }

    // Custom call returns result(s) as `absl::StatusOr`.
    if constexpr (kIsStatusOrResult) {
      auto status_or = fn_(std::move(*std::get<ArgsIs>(fn_args))...);
      if (!status_or.ok()) {
        return diagnostic->EmitError(status_or.status());
      }

      static_assert(sizeof...(RetsIs) >= 1, "unsupported number or results");

      if constexpr (sizeof...(RetsIs) == 1) {
        (*std::get<RetsIs...>(fn_args)).Set(status_or.value());
        return success();
      }

      if constexpr (sizeof...(RetsIs) > 1) {
        using ResultIs = std::make_index_sequence<kNumRets>;
        internal::SetResultsFromTuple<RetsIs...>(ResultIs{}, std::move(fn_args),
                                                 std::move(status_or.value()));
        return success();
      }
    }

    llvm_unreachable("unexpected custom call type");
  }

 private:
  template <typename...>
  friend class CustomCallBinding;

  CustomCallHandler(Fn fn, std::string callee, std::vector<std::string> attrs,
                    std::vector<std::any> values, const Options& opts)
      : fn_(std::move(fn)),
        callee_(std::move(callee)),
        attrs_(std::move(attrs)),
        values_(std::move(values)),
        opts_(opts),
        attrs_idx_(attrs_.size()) {
    // Sort attributes names and remove duplicates. These unique attributes are
    // what we'll be looking for in the encoded custom call attributes.
    std::vector<std::string> sorted = attrs_;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(
        std::unique(sorted.begin(), sorted.end(), std::equal_to<std::string>()),
        sorted.end());
    num_encoded_attrs_ = sorted.size();

    // Find index or every attribute in the sorted attributes vector.
    for (size_t i = 0; i < attrs_.size(); ++i) {
      std::string_view attr = attrs_[i];
      attrs_idx_[i] = std::distance(sorted.begin(), llvm::find(sorted, attr));
    }
  }

  Fn fn_;
  std::string callee_;
  std::vector<std::string> attrs_;
  std::vector<std::any> values_;
  Options opts_;

  // A mapping from the attribute index to its index in the lexicographically
  // sorter vector of attribute names. Attributes passed in the custom call
  // handler sorted by the name, we use this index to efficiently find the
  // decoded attribute entry.
  std::vector<size_t> attrs_idx_;

  // The number of attributes we expect in the encoded custom call arguments.
  // This is not the same as `attrs_.size()` because of potential duplicates,
  // e.g. attribute corresponding to state id might be used multiple times.
  size_t num_encoded_attrs_;
};

template <CustomCall::RuntimeChecks checks, typename Fn, typename... Ts>
constexpr int64_t CustomCallHandler<checks, Fn, Ts...>::kSize;

template <CustomCall::RuntimeChecks checks, typename Fn, typename... Ts>
constexpr int64_t CustomCallHandler<checks, Fn, Ts...>::kNumArgs;

template <CustomCall::RuntimeChecks checks, typename Fn, typename... Ts>
constexpr int64_t CustomCallHandler<checks, Fn, Ts...>::kNumRets;

//===----------------------------------------------------------------------===//
// Custom arguments decoding.
//===----------------------------------------------------------------------===//

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const StridedMemrefView&);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const MemrefView&);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const FlatMemrefView&);

template <CustomCall::RuntimeChecks checks>
struct CustomCallArgDecoding<StridedMemrefView, checks> {
  using EncodedMemref = internal::EncodedMemref;

  ABSL_ATTRIBUTE_ALWAYS_INLINE
  static FailureOr<StridedMemrefView> Decode(TypeID type_id, void* value) {
    if (!CustomCall::Isa<MemrefView, StridedMemrefView>(checks, type_id)) {
      return failure();
    }

    auto* encoded = reinterpret_cast<EncodedMemref*>(value);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(encoded, sizeof(EncodedMemref));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
        encoded, sizeof(EncodedMemref) + encoded->rank * sizeof(int64_t));

    PrimitiveType dtype = static_cast<PrimitiveType>(encoded->dtype);
    return StridedMemrefView{dtype,
                             encoded->data,
                             {encoded->dims, encoded->rank},
                             {encoded->dims + encoded->rank, encoded->rank}};
  }
};

template <CustomCall::RuntimeChecks checks>
struct CustomCallArgDecoding<MemrefView, checks> {
  using EncodedMemref = internal::EncodedMemref;

  ABSL_ATTRIBUTE_ALWAYS_INLINE
  static FailureOr<MemrefView> Decode(TypeID type_id, void* value) {
    if (!CustomCall::Isa<MemrefView>(checks, type_id)) {
      return failure();
    }

    auto* encoded = reinterpret_cast<EncodedMemref*>(value);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(encoded, sizeof(EncodedMemref));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
        encoded, sizeof(EncodedMemref) + encoded->rank * sizeof(int64_t));

    PrimitiveType dtype = static_cast<PrimitiveType>(encoded->dtype);
    return MemrefView{dtype, encoded->data, {encoded->dims, encoded->rank}};
  }
};

template <CustomCall::RuntimeChecks checks>
struct CustomCallArgDecoding<FlatMemrefView, checks> {
  using EncodedMemref = internal::EncodedMemref;

  ABSL_ATTRIBUTE_ALWAYS_INLINE
  static FailureOr<FlatMemrefView> Decode(TypeID type_id, void* value) {
    if (!CustomCall::Isa<MemrefView>(checks, type_id)) {
      return failure();
    }

    auto* encoded = reinterpret_cast<EncodedMemref*>(value);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(encoded, sizeof(EncodedMemref));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
        encoded, sizeof(EncodedMemref) + encoded->rank * sizeof(int64_t));

    PrimitiveType dtype = static_cast<PrimitiveType>(encoded->dtype);
    int64_t size_in_bytes = primitive_util::ByteWidth(dtype);
    for (int d = 0; d < encoded->rank; ++d) size_in_bytes *= encoded->dims[d];
    return FlatMemrefView{dtype, encoded->data, size_in_bytes};
  }
};

#define XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING(T)                         \
  template <CustomCall::RuntimeChecks checks>                               \
  struct CustomCallArgDecoding<T, checks> {                                 \
    ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> Decode(TypeID type_id, \
                                                            void* value) {  \
      if (!CustomCall::Isa<T>(checks, type_id)) {                           \
        return failure();                                                   \
      }                                                                     \
                                                                            \
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(value, sizeof(T));                \
      return *reinterpret_cast<T*>(value);                                  \
    }                                                                       \
  }

XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING(bool);
XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING(int8_t);
XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING(int16_t);
XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING(int32_t);
XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING(int64_t);
XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING(float);
XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING(double);

#undef XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING

// Register decoding for special floating point types defined in Eigen.
#define XLA_RUNTIME_REGISTER_EIGEN_FP_ARG_DECODING(T, STORAGE)              \
  template <CustomCall::RuntimeChecks checks>                               \
  struct CustomCallArgDecoding<T, checks> {                                 \
    ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> Decode(TypeID type_id, \
                                                            void* value) {  \
      if (!CustomCall::Isa<T>(checks, type_id)) {                           \
        return failure();                                                   \
      }                                                                     \
                                                                            \
      auto* src = reinterpret_cast<STORAGE*>(value);                        \
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(value, sizeof(STORAGE));          \
      return Eigen::numext::bit_cast<T>(*src);                              \
    }                                                                       \
  }

XLA_RUNTIME_REGISTER_EIGEN_FP_ARG_DECODING(Eigen::bfloat16, uint16_t);
XLA_RUNTIME_REGISTER_EIGEN_FP_ARG_DECODING(Eigen::half, uint16_t);

#undef XLA_RUNTIME_REGISTER_EIGEN_FP_ARG_DECODING

//===----------------------------------------------------------------------===//
// Opaque arguments at run time passed as pointers and decoded by wrapping them
// into a reference type, for example `AsyncValue *` pointer can be wrapped into
// a typed `AsyncValuePtr<T>` pointer wrapper.
//===----------------------------------------------------------------------===//

#define XLA_RUNTIME_REGISTER_OPAQUE_ARG_DECODING(T, PTR)                    \
  template <CustomCall::RuntimeChecks checks>                               \
  struct CustomCallArgDecoding<T, checks> {                                 \
    static_assert(std::is_pointer_v<PTR>, "must be a pointer");             \
    static_assert(std::is_trivially_destructible_v<T>,                      \
                  "must be a trivially destructible reference type");       \
                                                                            \
    ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> Decode(TypeID type_id, \
                                                            void* value) {  \
      if (!CustomCall::Isa<T>(checks, type_id)) {                           \
        return failure();                                                   \
      }                                                                     \
                                                                            \
      auto* src = reinterpret_cast<PTR*>(value);                            \
      ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(value, sizeof(PTR));              \
      T ref{*src};                                                          \
      return std::move(ref);                                                \
    }                                                                       \
  }

XLA_RUNTIME_REGISTER_OPAQUE_ARG_DECODING(void*, void*);

//===----------------------------------------------------------------------===//
// Custom call results decoding.
//===----------------------------------------------------------------------===//

#define XLA_RUNTIME_REGISTER_SCALAR_RET_DECODING(T)                  \
  template <>                                                        \
  class Result<T> {                                                  \
   public:                                                           \
    explicit Result(T* storage) : storage_(storage) {}               \
    void Set(T value) { *storage_ = value; }                         \
                                                                     \
   private:                                                          \
    T* storage_;                                                     \
  };                                                                 \
                                                                     \
  template <CustomCall::RuntimeChecks checks>                        \
  struct CustomCallRetDecoding<T, checks> {                          \
    ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<Result<T>> Decode( \
        TypeID type_id, void* value) {                               \
      if (!CustomCall::Isa<T>(checks, type_id)) {                    \
        return failure();                                            \
      }                                                              \
                                                                     \
      return Result<T>(reinterpret_cast<T*>(value));                 \
    }                                                                \
  };

XLA_RUNTIME_REGISTER_SCALAR_RET_DECODING(bool);
XLA_RUNTIME_REGISTER_SCALAR_RET_DECODING(int8_t);
XLA_RUNTIME_REGISTER_SCALAR_RET_DECODING(int16_t);
XLA_RUNTIME_REGISTER_SCALAR_RET_DECODING(int32_t);
XLA_RUNTIME_REGISTER_SCALAR_RET_DECODING(int64_t);
XLA_RUNTIME_REGISTER_SCALAR_RET_DECODING(float);
XLA_RUNTIME_REGISTER_SCALAR_RET_DECODING(double);

#undef XLA_RUNTIME_REGISTER_SCALAR_RET_DECODING

//===----------------------------------------------------------------------===//
// Opaque results at run time passed as pointers, and a typed wrapper binds
// together the reference type and the underlying pointer type.
//===----------------------------------------------------------------------===//

#define XLA_RUNTIME_REGISTER_OPAQUE_RET_DECODING(T, PTR)             \
  template <>                                                        \
  class Result<T> {                                                  \
    static_assert(std::is_pointer_v<PTR>, "must be a pointer type"); \
                                                                     \
   public:                                                           \
    explicit Result(PTR* storage) : storage_(storage) {}             \
    void Set(PTR value) { *storage_ = value; }                       \
                                                                     \
   private:                                                          \
    PTR* storage_;                                                   \
  };                                                                 \
                                                                     \
  template <CustomCall::RuntimeChecks checks>                        \
  struct CustomCallRetDecoding<T, checks> {                          \
    static_assert(std::is_pointer_v<PTR>, "must be a pointer type"); \
                                                                     \
    ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<Result<T>> Decode( \
        TypeID type_id, void* value) {                               \
      if (!CustomCall::Isa<T>(checks, type_id)) {                    \
        return failure();                                            \
      }                                                              \
                                                                     \
      return Result<T>(reinterpret_cast<PTR*>(value));               \
    }                                                                \
  };

XLA_RUNTIME_REGISTER_OPAQUE_RET_DECODING(void*, void*);

//===----------------------------------------------------------------------===//

// Custom call memref result decoding
template <>
class Result<MemrefView> {
  using EncodedMemref = internal::EncodedMemref;

 public:
  explicit Result(EncodedMemref* storage) : storage_(storage) {}
  void Set(MemrefView value) {
    assert(IsCompatible(value) &&
           "Custom call return types is not compatible with types in MLIR");
    storage_->data = value.data;
    for (unsigned i = 0; i < storage_->rank; ++i) {
      storage_->dims[i] = value.sizes[i];
    }
  }

  PrimitiveType GetDType() { return PrimitiveType{storage_->dtype}; }
  absl::Span<const int64_t> GetDims() {
    return absl::Span<const int64_t>(storage_->dims, storage_->rank);
  }

 private:
  bool IsCompatible(MemrefView value) {
    bool is_compatible =
        storage_->dtype == value.dtype && storage_->rank == value.sizes.size();
    if (!is_compatible) return false;

    for (unsigned i = 0; i < storage_->rank; ++i) {
      is_compatible = (storage_->dims[i] == value.sizes[i]) ||
                      (storage_->dims[i] == /*MemrefType::kDynamic=*/-1);
    }
    return is_compatible;
  }

  EncodedMemref* storage_;
};

template <CustomCall::RuntimeChecks checks>
struct CustomCallRetDecoding<MemrefView, checks> {
  using EncodedMemref = internal::EncodedMemref;

  ABSL_ATTRIBUTE_ALWAYS_INLINE
  static FailureOr<Result<MemrefView>> Decode(TypeID type_id, void* value) {
    if (!CustomCall::Isa<MemrefView>(checks, type_id)) return failure();

    auto* encoded = reinterpret_cast<EncodedMemref*>(value);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(encoded, sizeof(EncodedMemref));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
        encoded, sizeof(EncodedMemref) + encoded->rank * sizeof(int64_t));
    return Result<MemrefView>(encoded);
  }
};

//===----------------------------------------------------------------------===//

// Custom call AsyncValueRef result decoding
#define XLA_RUNTIME_REGISTER_ASYNC_SCALAR_VALUE_RET_DECODING(T)              \
  template <>                                                                \
  class Result<tsl::AsyncValueRef<T>> {                                      \
   public:                                                                   \
    explicit Result(void** storage) : storage_(storage) {}                   \
    void Set(tsl::AsyncValueRef<T> value) {                                  \
      auto write = [](const T* v, std::byte* store) {                        \
        T* store_t = reinterpret_cast<T*>(store);                            \
        *store_t = *v;                                                       \
      };                                                                     \
      *storage_ = runtime::AsyncRuntime::AsValue<T>(                         \
          value, sizeof(T), alignof(std::max_align_t), write);               \
    }                                                                        \
                                                                             \
   private:                                                                  \
    void** storage_;                                                         \
  };                                                                         \
                                                                             \
  template <CustomCall::RuntimeChecks checks>                                \
  struct CustomCallRetDecoding<tsl::AsyncValueRef<T>, checks> {              \
    LLVM_ATTRIBUTE_ALWAYS_INLINE                                             \
    static FailureOr<Result<tsl::AsyncValueRef<T>>> Decode(TypeID type_id,   \
                                                           void* value) {    \
      if (!CustomCall::Isa<tsl::AsyncValueRef<T>>(checks, type_id))          \
        return failure();                                                    \
      return Result<tsl::AsyncValueRef<T>>(reinterpret_cast<void**>(value)); \
    }                                                                        \
  };

XLA_RUNTIME_REGISTER_ASYNC_SCALAR_VALUE_RET_DECODING(bool);
XLA_RUNTIME_REGISTER_ASYNC_SCALAR_VALUE_RET_DECODING(int8_t);
XLA_RUNTIME_REGISTER_ASYNC_SCALAR_VALUE_RET_DECODING(int16_t);
XLA_RUNTIME_REGISTER_ASYNC_SCALAR_VALUE_RET_DECODING(int32_t);
XLA_RUNTIME_REGISTER_ASYNC_SCALAR_VALUE_RET_DECODING(int64_t);
XLA_RUNTIME_REGISTER_ASYNC_SCALAR_VALUE_RET_DECODING(float);
XLA_RUNTIME_REGISTER_ASYNC_SCALAR_VALUE_RET_DECODING(double);

#undef XLA_RUNTIME_REGISTER_ASYNC_SCALAR_VALUE_RET_DECODING

template <>
class Result<tsl::AsyncValueRef<tsl::Chain>> {
 public:
  explicit Result(void** storage) : storage_(storage) {}
  void Set(tsl::AsyncValueRef<tsl::Chain> value) {
    *storage_ = runtime::AsyncRuntime::AsToken(value);
  }

 private:
  void** storage_;
};

template <CustomCall::RuntimeChecks checks>
struct CustomCallRetDecoding<tsl::AsyncValueRef<tsl::Chain>, checks> {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  static FailureOr<Result<tsl::AsyncValueRef<tsl::Chain>>> Decode(
      TypeID type_id, void* value) {
    if (!CustomCall::Isa<tsl::AsyncValueRef<tsl::Chain>>(checks, type_id))
      return failure();

    return Result<tsl::AsyncValueRef<tsl::Chain>>(
        reinterpret_cast<void**>(value));
  }
};

template <>
class Result<tsl::AsyncValueRef<MemrefView>> {
  using EncodedMemref = internal::EncodedMemref;

  struct MemrefDescriptor {
    void* allocated_ptr;
    void* aligned_ptr;
    int64_t offset;
    int64_t dims[];
  };

 public:
  explicit Result(EncodedMemref* storage) : storage_(storage) {}
  void Set(tsl::AsyncValueRef<MemrefView> value) {
    auto write = [this](const MemrefView* view, std::byte* store) {
      assert(IsCompatible(*view) &&
             "Custom call return types is not compatible with types in MLIR");
      MemrefDescriptor* store_t = reinterpret_cast<MemrefDescriptor*>(store);
      store_t->allocated_ptr = view->data;
      store_t->aligned_ptr = view->data;
      store_t->offset = 0;
      for (unsigned i = 0; i < storage_->rank; ++i) {
        store_t->dims[i] = view->sizes[i];
      }
    };
    storage_->data = runtime::AsyncRuntime::AsValue<MemrefView>(
        value, 3 * sizeof(int64_t) + 2 * storage_->rank * sizeof(int64_t),
        alignof(std::max_align_t), write);
  }

  PrimitiveType GetDType() { return PrimitiveType{storage_->dtype}; }
  absl::Span<const int64_t> GetDims() {
    return absl::Span<const int64_t>(storage_->dims, storage_->rank);
  }

 private:
  bool IsCompatible(MemrefView value) {
    bool is_compatible =
        storage_->dtype == value.dtype && storage_->rank == value.sizes.size();
    if (!is_compatible) return false;

    for (unsigned i = 0; i < storage_->rank; ++i) {
      is_compatible = (storage_->dims[i] == value.sizes[i]) ||
                      (storage_->dims[i] == /*MemrefType::kDynamic=*/-1);
    }

    return is_compatible;
  }

  EncodedMemref* storage_;
};

template <CustomCall::RuntimeChecks checks>
struct CustomCallRetDecoding<tsl::AsyncValueRef<MemrefView>, checks> {
  using EncodedMemref = internal::EncodedMemref;

  LLVM_ATTRIBUTE_ALWAYS_INLINE
  static FailureOr<Result<tsl::AsyncValueRef<MemrefView>>> Decode(
      TypeID type_id, void* value) {
    if (!CustomCall::Isa<tsl::AsyncValueRef<MemrefView>>(checks, type_id))
      return failure();

    auto* encoded = reinterpret_cast<EncodedMemref*>(value);
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(encoded, sizeof(EncodedMemref));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
        encoded, sizeof(EncodedMemref) + encoded->rank * sizeof(int64_t));

    return Result<tsl::AsyncValueRef<MemrefView>>(encoded);
  }
};

// XLA_RUNTIME_REGISTER_ASYNC_VALUE_RET_DECODING(MemrefView);

//===----------------------------------------------------------------------===//
// Custom call attributes decoding.
//===----------------------------------------------------------------------===//

template <CustomCall::RuntimeChecks checks>
struct CustomCallAttrDecoding<std::string_view, checks> {
  ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<std::string_view> Decode(
      std::string_view name, TypeID type_id, void* value) {
    if (!CustomCall::Isa<std::string_view>(checks, type_id)) {
      return failure();
    }

    auto* encoded = reinterpret_cast<internal::EncodedArray<char>*>(value);
    return std::string_view(encoded->data, encoded->size);
  }
};

template <CustomCall::RuntimeChecks checks>
struct CustomCallAttrDecoding<CustomCall::FunctionOrdinal, checks> {
  using FunctionOrdinal = CustomCall::FunctionOrdinal;

  ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<FunctionOrdinal> Decode(
      std::string_view name, TypeID type_id, void* value) {
    if (!CustomCall::Isa<FunctionOrdinal>(checks, type_id)) {
      return failure();
    }

    unsigned ordinal = *reinterpret_cast<int32_t*>(value);
    return FunctionOrdinal{ordinal};
  }
};

template <typename T, CustomCall::RuntimeChecks checks>
struct CustomCallAttrDecoding<std::optional<T>, checks> {
  using ValueDecoding = CustomCallAttrDecoding<T, checks>;

  ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<std::optional<T>> Decode(
      std::string_view name, TypeID type_id, void* value) {
    // Convert nullptr to empty optional.
    bool is_nullopt = CustomCall::Isa<std::nullopt_t>(checks, type_id);
    if (is_nullopt && value == nullptr) return std::optional<T>();

    // Try to decode the underlying value if it is present.
    if (auto decoded = ValueDecoding::Decode(name, type_id, value);
        succeeded(decoded)) {
      return std::optional<T>(std::move(*decoded));
    }
    return failure();
  }
};

template <CustomCall::RuntimeChecks checks>
struct CustomCallAttrDecoding<CustomCall::VariantAttr, checks> {
  ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<CustomCall::VariantAttr> Decode(
      std::string_view name, TypeID type_id, void* value) {
    return CustomCall::VariantAttr(name, type_id, value);
  }
};

#define XLA_RUNTIME_REGISTER_SCALAR_ATTR_DECODING(T)          \
  template <CustomCall::RuntimeChecks checks>                 \
  struct CustomCallAttrDecoding<T, checks> {                  \
    ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> Decode(  \
        std::string_view name, TypeID type_id, void* value) { \
      if (!CustomCall::Isa<T>(checks, type_id)) {             \
        return failure();                                     \
      }                                                       \
                                                              \
      return *reinterpret_cast<T*>(value);                    \
    }                                                         \
  }

XLA_RUNTIME_REGISTER_SCALAR_ATTR_DECODING(bool);
XLA_RUNTIME_REGISTER_SCALAR_ATTR_DECODING(int32_t);
XLA_RUNTIME_REGISTER_SCALAR_ATTR_DECODING(int64_t);
XLA_RUNTIME_REGISTER_SCALAR_ATTR_DECODING(float);
XLA_RUNTIME_REGISTER_SCALAR_ATTR_DECODING(double);

#undef XLA_RUNTIME_REGISTER_SCALAR_ATTR_DECODING

// A type tag to represent empty arrays of unknown element type.
struct EmptyArray {};

// Both EncodedArray and 1-D EncodedDenseElements can be decoded as an
// absl::Span. Pointers to both EncodedArray and 1-D EncodedDenseElements
// can be dereferenced as a pointer to EncodedArray.
#define XLA_RUNTIME_REGISTER_ARRAY_ATTR_DECODING(T)                            \
  template <CustomCall::RuntimeChecks checks>                                  \
  struct CustomCallAttrDecoding<absl::Span<const T>, checks> {                 \
    ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<absl::Span<const T>> Decode( \
        std::string_view name, TypeID type_id, void* value) {                  \
      if (!CustomCall::Isa<absl::Span<const T>, CustomCall::TensorRef<T>,      \
                           EmptyArray>(checks, type_id)) {                     \
        return failure();                                                      \
      }                                                                        \
                                                                               \
      auto* encoded = reinterpret_cast<internal::EncodedArray<T>*>(value);     \
      return absl::Span<const T>(encoded->data, encoded->size);                \
    }                                                                          \
  }

XLA_RUNTIME_REGISTER_ARRAY_ATTR_DECODING(int32_t);
XLA_RUNTIME_REGISTER_ARRAY_ATTR_DECODING(int64_t);
XLA_RUNTIME_REGISTER_ARRAY_ATTR_DECODING(float);
XLA_RUNTIME_REGISTER_ARRAY_ATTR_DECODING(double);

#undef XLA_RUNTIME_REGISTER_ARRAY_ATTR_DECODING

#define XLA_RUNTIME_REGISTER_DENSE_ELEMENTS_ATTR_DECODING(T)                \
  template <CustomCall::RuntimeChecks checks>                               \
  struct CustomCallAttrDecoding<CustomCall::TensorRef<T>, checks> {         \
    ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<CustomCall::TensorRef<T>> \
    Decode(std::string_view name, TypeID type_id, void* value) {            \
      if (!CustomCall::Isa<CustomCall::TensorRef<T>>(checks, type_id)) {    \
        return failure();                                                   \
      }                                                                     \
                                                                            \
      auto* encoded =                                                       \
          reinterpret_cast<internal::EncodedDenseElements<T>*>(value);      \
      auto payload = encoded->payload;                                      \
      absl::Span<const T> data(payload.data, payload.size);                 \
      absl::Span<const int64_t> shape(encoded->shape, encoded->rank);       \
      return CustomCall::TensorRef<T>({shape, data});                       \
    }                                                                       \
  }

XLA_RUNTIME_REGISTER_DENSE_ELEMENTS_ATTR_DECODING(int32_t);
XLA_RUNTIME_REGISTER_DENSE_ELEMENTS_ATTR_DECODING(int64_t);
XLA_RUNTIME_REGISTER_DENSE_ELEMENTS_ATTR_DECODING(float);
XLA_RUNTIME_REGISTER_DENSE_ELEMENTS_ATTR_DECODING(double);

#undef XLA_RUNTIME_REGISTER_DENSE_ELEMENTS_ATTR_DECODING

//===----------------------------------------------------------------------===//
// Register an XLA custom call attribute decoding for enum class. At runtime the
// value should be passed as the underlying enum type.
//===----------------------------------------------------------------------===//

// Example: register decoding for a user-defined enum class
//
//   enum class MyEnumType { kFoo, kBar, kBaz };
//
//   XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(MyEnumType);
//
#define XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(T)                \
  template <CustomCall::RuntimeChecks checks>                     \
  struct CustomCallAttrDecoding<T, checks> {                      \
    static_assert(std::is_enum<T>::value, "expected enum class"); \
    using U = std::underlying_type_t<T>;                          \
                                                                  \
    ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> Decode(      \
        std::string_view name, TypeID type_id, void* value) {     \
      if (!CustomCall::Isa<T>(checks, type_id)) {                 \
        return failure();                                         \
      }                                                           \
                                                                  \
      return static_cast<T>(*reinterpret_cast<U*>(value));        \
    }                                                             \
  }

//===----------------------------------------------------------------------===//
// Register an XLA custom call attribute decoding for aggregate attributes.
//===----------------------------------------------------------------------===//

template <typename T>
struct AggregateMember {
  using Type = T;

  explicit AggregateMember(std::string_view name) : name(name) {}
  std::string_view name;
};

// Example: register decoding for a user-defined struct
//
//   struct PairOfI64 { int64_t a; int64_t b; };
//
//   XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(
//     PairOfI64,
//     AggregateMember<int64_t>("a"),
//     AggregateMember<int64_t>("b"));
//
#define XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(T, ...)                   \
  template <CustomCall::RuntimeChecks checks>                                  \
  struct CustomCallAttrDecoding<T, checks> {                                   \
    ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> Decode(                   \
        std::string_view name, TypeID type_id, void* value) {                  \
      if (!CustomCall::Isa<T>(checks, type_id)) {                              \
        return failure();                                                      \
      }                                                                        \
                                                                               \
      auto decoder = internal::AggregateDecoder<T, checks>(__VA_ARGS__);       \
      return decltype(decoder)::Decode(reinterpret_cast<void**>(value),        \
                                       internal::AggregateNames(__VA_ARGS__)); \
    }                                                                          \
  }

namespace internal {
// Decodes aggregate attribute into the object of type `T` that must be
// constructible from the `Ts` types.
template <typename T, CustomCall::RuntimeChecks checks, typename... Ts>
struct DecodeAggregateAttr {
  static constexpr size_t kSize = sizeof...(Ts);

  using RuntimeChecks = CustomCall::RuntimeChecks;

  ABSL_ATTRIBUTE_ALWAYS_INLINE
  static FailureOr<T> Decode(void** value,
                             std::array<std::string_view, kSize> names) {
    internal::DecodedAttrs attrs(value);
    return Decode(attrs, names, std::make_index_sequence<kSize>{});
  }

  template <size_t... Is>
  ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> Decode(
      internal::DecodedAttrs attrs, std::array<std::string_view, kSize> names,
      std::index_sequence<Is...>) {
    // Check that the number of encoded attributes matches the signature.
    if (checks != RuntimeChecks::kNone && kSize != attrs.size())
      return failure();

    // Check that aggregate member names match the expected names.
    if (CustomCall::CheckNames(checks)) {
      for (unsigned i = 0; i < kSize; ++i)
        if (attrs[i].name != names[i]) return failure();
    }

    // Decode all arguments into FailureOr containers. It is guaranteed
    // that initializer list will be evaluated left-to-right, and we can rely
    // on correct offsets computation.
    std::tuple<FailureOr<Ts>...> members = {
        CustomCallAttrDecoding<Ts, checks>::Decode(
            attrs[Is].name, attrs[Is].type_id, attrs[Is].value)...};

    bool all_decoded = (succeeded(std::get<Is>(members)) && ...);
    if (LLVM_UNLIKELY(!all_decoded)) return failure();

    // Forward unpacked members to the type constructor.
    return T{std::move(*std::get<Is>(members))...};
  }
};

template <typename... Members>
auto AggregateNames(Members... m) {
  return std::array<std::string_view, sizeof...(Members)>{m.name...};
}

template <typename T, CustomCall::RuntimeChecks checks, typename... Members>
auto AggregateDecoder(Members... m) {
  return DecodeAggregateAttr<T, checks, typename Members::Type...>();
}

}  // namespace internal

//===----------------------------------------------------------------------===//
// Register an XLA custom call attribute decoding for dictionary attributes.
//===----------------------------------------------------------------------===//

// Dictionary attributes are encoded using the same scheme as aggregate
// attributes and as custom call attributes: <type_id, name, data> x length.
class Dictionary {
  using RuntimeChecks = CustomCall::RuntimeChecks;

 public:
  explicit Dictionary(internal::DecodedAttrs attrs) : attrs_(attrs) {}

  int64_t size() { return attrs_.size(); }

  std::vector<std::string_view> keys() {
    std::vector<std::string_view> attr_keys(attrs_.size());
    for (int64_t i = 0; i < attrs_.size(); ++i) {
      attr_keys[i] = attrs_[i].name;
    }
    return attr_keys;
  }

  template <typename T, RuntimeChecks checks = RuntimeChecks::kDefault>
  ABSL_ATTRIBUTE_ALWAYS_INLINE FailureOr<T> get(std::string_view name) const {
    // TODO(ezhulenev): Use `std::binary_search` because it's guaranteed that
    // encoded attributes are sorted by name.
    for (int64_t i = 0; i < attrs_.size(); ++i) {
      if (auto attr = attrs_[i]; attr.name == name)
        return CustomCallAttrDecoding<T, checks>::Decode(
            attr.name, attr.type_id, attr.value);
    }
    return failure();
  }

 private:
  internal::DecodedAttrs attrs_;
};

template <CustomCall::RuntimeChecks checks>
struct CustomCallAttrDecoding<Dictionary, checks> {
  ABSL_ATTRIBUTE_ALWAYS_INLINE static FailureOr<Dictionary> Decode(
      std::string_view name, TypeID type_id, void* value) {
    if (!CustomCall::Isa<Dictionary>(checks, type_id)) return failure();
    return Dictionary(internal::DecodedAttrs(reinterpret_cast<void**>(value)));
  }
};

//===----------------------------------------------------------------------===//
// XLA Custom Call helper macro for registering custom call handlers.
//===----------------------------------------------------------------------===//

#define XLA_RUNTIME_DEFINE_CUSTOM_CALL(fn, impl, checks, bind)             \
  static bool fn(::xla::runtime::ExecutionContext* ctx, void** args,       \
                 void** attrs, void** rets) {                              \
    static auto* handler = bind.To<checks>(impl).release();                \
    return ::xla::runtime::succeeded(                                      \
        xla::runtime::Executable::Call(ctx, *handler, args, attrs, rets)); \
  }

#define XLA_RUNTIME_DEFINE_CUSTOM_CALL_TEMPLATE(param, fn, impl, checks, bind) \
  template <param>                                                             \
  static bool fn(::xla::runtime::ExecutionContext* ctx, void** args,           \
                 void** attrs, void** rets) {                                  \
    static auto* handler = bind.To<checks>(impl).release();                    \
    return ::xla::runtime::succeeded(                                          \
        xla::runtime::Executable::Call(ctx, *handler, args, attrs, rets));     \
  }

//===----------------------------------------------------------------------===//
// Declare/define an explicit specialization for TypeID for types used
// by the custom calls. This forces the compiler to emit a strong definition for
// a class and controls which translation unit and shared object will actually
// have it.
//
// See TypeID for more documentation.
//
// Because custom calls do not "own" the types passed across the function
// boundary, we declare/define specializations for tagged types to avoid
// potential conflicts with other libraries.
//===----------------------------------------------------------------------===//

#define XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(T) \
  MLIR_DECLARE_EXPLICIT_TYPE_ID(::xla::runtime::Tagged<T>)

#define XLA_RUNTIME_DEFINE_EXPLICIT_TYPE_ID(T) \
  MLIR_DEFINE_EXPLICIT_TYPE_ID(::xla::runtime::Tagged<T>)

}  // namespace runtime
}  // namespace xla

XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(std::string_view);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(xla::runtime::StridedMemrefView);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(xla::runtime::MemrefView);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(xla::runtime::FlatMemrefView);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(xla::runtime::EmptyArray);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(xla::runtime::Dictionary);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(int32_t);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(int64_t);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(float);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(double);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(absl::Span<const int32_t>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(absl::Span<const int64_t>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(absl::Span<const float>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(absl::Span<const double>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(tsl::AsyncValueRef<bool>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(tsl::AsyncValueRef<int8_t>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(tsl::AsyncValueRef<int16_t>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(tsl::AsyncValueRef<int32_t>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(tsl::AsyncValueRef<int64_t>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(tsl::AsyncValueRef<float>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(tsl::AsyncValueRef<double>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(
    tsl::AsyncValueRef<xla::runtime::MemrefView>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(tsl::AsyncValueRef<tsl::Chain>);

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_CUSTOM_CALL_H_

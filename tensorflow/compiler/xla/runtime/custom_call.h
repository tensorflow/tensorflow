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

#include <any>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/dynamic_annotations.h"
#include "third_party/eigen3/Eigen/Core"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/runtime/diagnostics.h"
#include "tensorflow/compiler/xla/runtime/logical_result.h"
#include "tensorflow/compiler/xla/runtime/map_by_type.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"

namespace xla {
namespace runtime {

// Forward declare.
struct KernelContext;

// Forward declare template defined below.
template <typename... Ts>
class CustomCallBinding;

// Registers mappings from TypeIDs supported by the custom calls to their unique
// names in the given registry.
void PopulateCustomCallTypeIdNames(TypeIDNameRegistry& registry);

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
    llvm::ArrayRef<int64_t> shape;
    llvm::ArrayRef<T> data;
  };

  // Custom call handler can check arguments and attributes types and names
  // at runtime, however this comes at extra cost and can be optionally
  // disabled. If the version of the compiler that generated the XLA executable
  // doesn't match the custom call handler, it can lead to undefined behavior.
  enum class RuntimeChecks : uint8_t {
    // Check arguments and attributes types, also check attribute names. It is
    // safe to pass extra arguments to the custom call handler when name
    // checking is enabled, because it will safely skip irrelevant attributes.
    kDefault = 0,

    // Check only the types of the arguments and attributes. If an attribute
    // with the same type but different name is passed to the custom call
    // handler,
    // it will happily proceed ignoring the name mismatch.
    kTypes = 1,

    // Do not check the number of arguments and attributes and their types, and
    // do not check that the user data was passed to the custom call. This is
    // the most dangerous option, because it blindly reinterprets opaque memory
    // passed to the handler, and can easily lead to segfaults if the data
    // doesn't match the expected custom call signature.
    kNone = 2
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
  static bool CheckType(RuntimeChecks checks, TypeID type_id) {
    return !CheckTypes(checks) || type_id == TypeID::get<T>();
  }

  virtual ~CustomCall() = default;

  virtual std::string_view name() const = 0;
  virtual LogicalResult call(void** args, void** attrs,
                             const UserData* user_data,
                             const DiagnosticEngine* diagnostic) const = 0;

  static CustomCallBinding<> Bind(std::string callee);
};

// Direct custom call is a custom call that can be linked directly with the
// compiled executable, and doesn't have to go through the custom call look up
// by name at run time (see CustomCallRegistry).
//
// Direct custom call is a preffered way of implemenenting custom calls with
// low run time overheads, as they will become just an indirect function calls
// once LLVM ORC links them with the executable.
//
// See `GetSymbolsBinding` to convert custom call library to symbols binding.
class DirectCustomCallLibrary {
 public:
  // Function type corresponding to the direct custom call (custom calls
  // linked directly with the compiled executable).
  using DirectCustomCall = bool (*)(KernelContext* kernel_context, void** args,
                                    void** attrs);

  void Insert(std::string_view name, DirectCustomCall custom_call) {
    lib_.try_emplace(name, custom_call);
  }

  void ForEach(
      std::function<void(std::string_view, DirectCustomCall)> f) const {
    for (auto& kv : lib_) f(kv.first(), kv.second);
  }

 private:
  llvm::StringMap<DirectCustomCall> lib_;
};

// Forward declare template defined below.
template <CustomCall::RuntimeChecks checks, typename Fn, typename... Ts>
class CustomCallHandler;

namespace internal {

// A type tag to distinguish arguments tied to the attributes in the
// `CustomCallBinding` variadic template argument.
template <typename T>
struct Attr {};

// A type tag to distinguish arguments tied to the user data in the
// `CustomCallBinding` variadic template argument.
template <typename T>
struct UserData {};

// A type tag to distinguish arguments tied to the constant values in the
// `CustomCallBinding` variadic template argument.
template <typename T>
struct Value {};

// A template for checking if type is a wrapped attribute or user data.
template <typename>
struct IsWrapped : std::false_type {};

template <typename T>
struct IsWrapped<internal::Attr<T>> : std::true_type {};

template <typename T>
struct IsWrapped<internal::UserData<T>> : std::true_type {};

template <typename T>
struct IsWrapped<internal::Value<T>> : std::true_type {};

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
  CustomCallBinding<Ts..., internal::UserData<T>> UserData() && {
    static_assert(std::is_pointer<T>::value, "user data must be a pointer");
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
            std::move(values_)));
  }

 private:
  template <typename...>
  friend class CustomCallBinding;
  friend class CustomCall;

  explicit CustomCallBinding(std::string callee) : callee_(std::move(callee)) {
    static_assert(sizeof...(Ts) == 0, "custom call arguments must be empty");
  }

  template <typename... TTs>
  CustomCallBinding(CustomCallBinding<TTs...>&& other)  // NOLINT
      : callee_(std::move(other.callee_)),
        attrs_(std::move(other.attrs_)),
        values_(std::move(other.values_)) {}

  CustomCallBinding(CustomCallBinding&) = delete;

  std::string callee_;              // custom call target
  std::vector<std::string> attrs_;  // names of bound attributes
  std::vector<std::any> values_;    // values bound to arguments
};

inline CustomCallBinding<> CustomCall::Bind(std::string callee) {
  return CustomCallBinding<>(std::move(callee));
}

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

// A type tag to declare MLIR TypeID specializations for types passed to the
// custom calls. We don't want to declare specializations for scalar types
// directly in this translation unit, so we rely on a tag to wrap them.
//
// See explicit TypeID declarations at the end of this file.
template <typename T>
struct Tagged {};

// A type tag to represent empty arrays of unknown element type.
struct EmptyArrayRef {};

//===----------------------------------------------------------------------===//
// C structures corresponding to the `rt-to-llvm` pass LLVM structs encoding
// various types of arguments/attributes.

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

template <typename T>
struct EncodedDenseElements {
  struct EncodedArray<T> payload;
  int64_t rank;
  int64_t shape[];
};

}  // namespace internal

//===----------------------------------------------------------------------===//
// Helpers for decoding opaque arguments and attributes memory.

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
  explicit DecodedArgs(void** args)
      : args_(args), num_args_(*reinterpret_cast<int64_t*>(args_[0])) {}

  LLVM_ATTRIBUTE_ALWAYS_INLINE int64_t size() const { return num_args_; }

  LLVM_ATTRIBUTE_ALWAYS_INLINE DecodedArg operator[](size_t i) const {
    void** arg_base = args_ + 1 + i * 2;

    DecodedArg arg;
    arg.type_id = TypeID::getFromOpaquePointer(arg_base[0]);
    arg.value = arg_base[1];

    return arg;
  }

 private:
  void** args_;
  int64_t num_args_;
};

// A convenience wrapper around opaque attributes memory.
class DecodedAttrs {
 public:
  explicit DecodedAttrs(void** attrs)
      : attrs_(attrs), num_attrs_(*reinterpret_cast<int64_t*>(attrs_[0])) {}

  LLVM_ATTRIBUTE_ALWAYS_INLINE int64_t size() const { return num_attrs_; }

  LLVM_ATTRIBUTE_ALWAYS_INLINE DecodedAttr operator[](size_t i) const {
    void** attr_base = attrs_ + 1 + i * 3;

    DecodedAttr attr;
    auto* name = reinterpret_cast<internal::EncodedArray<char>*>(attr_base[0]);
    attr.name = std::string_view(name->data, name->size);
    attr.type_id = TypeID::getFromOpaquePointer(attr_base[1]);
    attr.value = attr_base[2];

    return attr;
  }

 private:
  void** attrs_;
  int64_t num_attrs_;
};

}  // namespace internal

//===----------------------------------------------------------------------===//
// CustomCall remaining arguments wraps the type-erased `DecodedArg` container,
// and provides a type-safe API for accessing individual arguments.

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

// Extracts the underlying type from the user data type tag.
template <typename T>
struct FnArgType<internal::UserData<T>> {
  using Type = T;
};

// Extracts the underlying type from the value type tag.
template <typename T>
struct FnArgType<internal::Value<T>> {
  using Type = T;
};

// A template for counting regular arguments in the Ts pack.
template <typename T, typename... Ts>
struct NumArgs {
  static constexpr int64_t value = !IsWrapped<T>::value + NumArgs<Ts...>::value;
};

template <typename T>
struct NumArgs<T> {
  static constexpr int64_t value = !IsWrapped<T>::value;
};

// When decoding input data we need to keep track of how many arguments and
// attributes we decoded so far to index into the correct data strucuture.
struct DecodingOffsets {
  int64_t args = 0;
  int64_t attrs = 0;
  int64_t values = 0;
};

template <typename T, CustomCall::RuntimeChecks checks>
struct Decode {
  LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> call(
      DecodingOffsets& offsets, internal::DecodedArgs args,
      llvm::ArrayRef<std::string> attrs_names, llvm::ArrayRef<size_t> attrs_idx,
      internal::DecodedAttrs attrs, llvm::ArrayRef<std::any> values,
      const CustomCall::UserData* user_data) {
    internal::DecodedArg arg = args[offsets.args++];
    return CustomCallArgDecoding<T, checks>::Decode(arg.type_id, arg.value);
  }
};

template <typename T, CustomCall::RuntimeChecks checks>
struct Decode<internal::Attr<T>, checks> {
  LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> call(
      DecodingOffsets& offsets, internal::DecodedArgs args,
      llvm::ArrayRef<std::string> attrs_names, llvm::ArrayRef<size_t> attrs_idx,
      internal::DecodedAttrs attrs, llvm::ArrayRef<std::any> values,
      const CustomCall::UserData* user_data) {
    // Find decoded attribute corresponding for the given attribute index.
    int64_t idx = offsets.attrs++;

    // Do not check the attribute name, and decode attribute at the given index.
    if (!CustomCall::CheckNames(checks)) {
      size_t i = attrs_idx[idx];
      return CustomCallAttrDecoding<T, checks>::Decode(
          attrs[i].name, attrs[i].type_id, attrs[i].value);
    }

    std::string_view attr = attrs_names[idx];

    // Given that attributes are passed to the custom call handler
    // lexicographically sorted by name, we can find the attribute we are
    // looking for only between the `attrs_idx` offset and the end of the
    // attributes array.
    for (size_t i = attrs_idx[idx]; i < attrs.size(); ++i) {
      if (LLVM_LIKELY(attrs[i].name == attr))
        return CustomCallAttrDecoding<T, checks>::Decode(
            attrs[i].name, attrs[i].type_id, attrs[i].value);
    }

    // Attribute we were looking for was not passed as an argument.
    return failure();
  }
};

template <typename T, CustomCall::RuntimeChecks checks>
struct Decode<internal::UserData<T>, checks> {
  LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> call(
      DecodingOffsets& offsets, internal::DecodedArgs args,
      llvm::ArrayRef<std::string> attrs_names, llvm::ArrayRef<size_t> attrs_idx,
      internal::DecodedAttrs attrs, llvm::ArrayRef<std::any> values,
      const CustomCall::UserData* user_data) {
    using UserDataT = std::remove_pointer_t<T>;

    if (!CustomCall::CheckUserData(checks)) return user_data->get<UserDataT>();

    // TODO(ezhulenev): Add an option to request nullable user data, because
    // right now we do not distinguish between a user data pointer that doesn't
    // exist, and a null pointer passed by the user.

    // Get the requested value if user data was passed to the custom call.
    auto* ptr = user_data ? user_data->getIfExists<UserDataT>() : nullptr;
    if (LLVM_UNLIKELY(!ptr)) return failure();
    return ptr;
  }
};

template <typename T, CustomCall::RuntimeChecks checks>
struct Decode<internal::Value<T>, checks> {
  LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> call(
      DecodingOffsets& offsets, internal::DecodedArgs args,
      llvm::ArrayRef<std::string> attrs_names, llvm::ArrayRef<size_t> attrs_idx,
      internal::DecodedAttrs attrs, llvm::ArrayRef<std::any> values,
      const CustomCall::UserData* user_data) {
    return std::any_cast<T>(values[offsets.values++]);
  }
};

template <CustomCall::RuntimeChecks checks>
struct Decode<CustomCall::RemainingArgs, checks> {
  LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<CustomCall::RemainingArgs> call(
      DecodingOffsets& offsets, internal::DecodedArgs args,
      llvm::ArrayRef<std::string> attr_names, llvm::ArrayRef<size_t> attrs_idx,
      internal::DecodedAttrs attrs, llvm::ArrayRef<std::any> values,
      const CustomCall::UserData* user_data) {
    return CustomCall::RemainingArgs(args, offsets.args);
  }
};

template <CustomCall::RuntimeChecks checks>
struct Decode<CustomCall::VariantArg, checks> {
  LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<CustomCall::VariantArg> call(
      DecodingOffsets& offsets, internal::DecodedArgs args,
      llvm::ArrayRef<std::string> attr_names, llvm::ArrayRef<size_t> attrs_idx,
      internal::DecodedAttrs attrs, llvm::ArrayRef<std::any> values,
      const CustomCall::UserData* user_data) {
    return CustomCall::VariantArg(args, offsets.args++);
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

  template <typename T>
  using FnArgType = typename internal::FnArgType<T>::Type;

  // Custom call can signal error using a LogicalError result.
  static constexpr bool kIsLogicalErr =
      std::is_invocable_r_v<LogicalResult, Fn, FnArgType<Ts>...>;

  // Custom call can signal error together with a detailed error message.
  static constexpr bool kIsDetailedErr =
      std::is_invocable_r_v<llvm::Error, Fn, FnArgType<Ts>...>;

  static_assert(kIsLogicalErr || kIsDetailedErr,
                "incompatible custom call handler types");

 public:
  std::string_view name() const final { return callee_; }

  LLVM_ATTRIBUTE_ALWAYS_INLINE LogicalResult
  call(void** args, void** attrs, const UserData* user_data,
       const DiagnosticEngine* diagnostic) const final {
    // Unpoison the first pointer to get the args and attrs sizes.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(args, sizeof(void*));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(attrs, sizeof(void*));

    // Decode arguments and attributes from the opaque pointers.
    internal::DecodedArgs decoded_args(args);
    internal::DecodedAttrs decoded_attrs(attrs);

    int64_t num_args = decoded_args.size();
    int64_t num_attrs = decoded_attrs.size();

    // Unpoison the rest of the of args and attrs data.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(args,
                                        (1 + 2 * num_args) * sizeof(void*));
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(attrs,
                                        (1 + 3 * num_attrs) * sizeof(void*));

    if (LLVM_UNLIKELY(diagnostic == nullptr))
      diagnostic = DiagnosticEngine::DefaultDiagnosticEngine();

    // If all runtime checks are disabled we are just reinterpreting opaque
    // `args` and `attrs` memory acording to the requested handler signature.
    if (checks != RuntimeChecks::kNone) {
      // Check that the number of passed arguments matches the signature. Each
      // individual argument decoding will check the actual type.
      if (internal::HasRemainingArgs<Ts...>::value) {
        if (LLVM_UNLIKELY(num_args < kNumArgs - 1))
          return diagnostic->EmitError()
                 << "Wrong number of arguments: expected at least "
                 << (kNumArgs - 1) << " got " << num_args;
      } else {
        if (LLVM_UNLIKELY(num_args != kNumArgs))
          return diagnostic->EmitError()
                 << "Wrong number of arguments: expected " << kNumArgs
                 << " got " << num_args;
      }

      // Check that we have enough attributes passed to the custom call. Each
      // individual attribute decoding will check the name and the type.
      if (LLVM_UNLIKELY(num_attrs < attrs_.size()))
        return diagnostic->EmitError()
               << "Wrong number of attributes: expected at least "
               << attrs_.size() << " got " << num_attrs;
    }

    return call(decoded_args, decoded_attrs, user_data, diagnostic,
                std::make_index_sequence<kSize>{});
  }

  template <size_t... Is>
  LLVM_ATTRIBUTE_ALWAYS_INLINE LogicalResult
  call(internal::DecodedArgs args, internal::DecodedAttrs attrs,
       const UserData* user_data, const DiagnosticEngine* diagnostic,
       std::index_sequence<Is...>) const {
    // A helper structure to allow each decoder find the correct offset in the
    // arguments or attributes.
    internal::DecodingOffsets offsets;

    // Check if all arguments and attributes were decoded.
    bool all_decoded = true;
    auto check_all_decoded = [&](auto result) {
      all_decoded &= succeeded(result);
      return std::move(result);
    };

    // Decode all arguments into FailureOr containers. It is guaranteed
    // that initializer list will be evaluated left-to-right, and we can rely
    // on correct offsets computation.
    std::tuple<FailureOr<FnArgType<Ts>>...> fn_args = {
        check_all_decoded(internal::Decode<Ts, checks>::call(
            offsets, args, attrs_, attrs_idx_, attrs, values_, user_data))...};
    if (LLVM_UNLIKELY(!all_decoded))
      return diagnostic->EmitError()
             << "Failed to decode all custom call arguments and attributes";

    // Custom call returns logical result to signal failures.
    if constexpr (kIsLogicalErr)
      return fn_(std::move(*std::get<Is>(fn_args))...);

    // Custom call returns detailed error to signal failures.
    if constexpr (kIsDetailedErr) {
      if (auto err = fn_(std::move(*std::get<Is>(fn_args))...))
        return diagnostic->EmitError() << std::move(err);
      return success();
    }

    llvm_unreachable("unexpected custom call type");
  }

 private:
  template <typename...>
  friend class CustomCallBinding;

  CustomCallHandler(Fn fn, std::string callee, std::vector<std::string> attrs,
                    std::vector<std::any> values)
      : fn_(std::move(fn)),
        callee_(std::move(callee)),
        attrs_(std::move(attrs)),
        values_(std::move(values)),
        attrs_idx_(attrs_.size()) {
    // Sort attributes names.
    std::vector<std::string> sorted = attrs_;
    llvm::sort(sorted);

    // Find index or every attribute in the sorted attributes vector.
    for (size_t i = 0; i < attrs_.size(); ++i) {
      const std::string& attr = attrs_[i];
      attrs_idx_[i] = std::distance(sorted.begin(), llvm::find(sorted, attr));
    }
  }

  Fn fn_;
  std::string callee_;
  std::vector<std::string> attrs_;
  std::vector<std::any> values_;
  // A mapping from the attribute index to its index in the lexicographically
  // sorter vector of attribute names. Attributes passed in the custom call
  // handler sorted by the name, we use this index to efficiently find the
  // decoded attribute entry.
  std::vector<size_t> attrs_idx_;
};

template <CustomCall::RuntimeChecks checks, typename Fn, typename... Ts>
constexpr int64_t CustomCallHandler<checks, Fn, Ts...>::kSize;

template <CustomCall::RuntimeChecks checks, typename Fn, typename... Ts>
constexpr int64_t CustomCallHandler<checks, Fn, Ts...>::kNumArgs;

//===----------------------------------------------------------------------===//
// Custom arguments attributes decoding.

// A view into the memref argument. Corresponds to the MemrefDesc, however it
// doesn't own the sizes/strides vectors, and cheap to pass around. Memrefs with
// non-identity layouts can be decoded only as a StridedMemrefView.
struct StridedMemrefView {
  PrimitiveType dtype;
  void* data;
  llvm::ArrayRef<int64_t> sizes;
  llvm::ArrayRef<int64_t> strides;
};

// A view into the memref argument with an identity (row major) layout.
struct MemrefView {
  PrimitiveType dtype;
  void* data;
  llvm::ArrayRef<int64_t> sizes;
};

// A flat view into memref argument with an identity (row major) layout. If the
// memref shape and strides are not required for the custom call, it's cheaper
// to pass the flat view.
struct FlatMemrefView {
  PrimitiveType dtype;
  void* data;
  int64_t size_in_bytes;
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const StridedMemrefView&);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const MemrefView&);
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const FlatMemrefView&);

template <CustomCall::RuntimeChecks checks>
struct CustomCallArgDecoding<StridedMemrefView, checks> {
  using EncodedMemref = internal::EncodedMemref;

  LLVM_ATTRIBUTE_ALWAYS_INLINE
  static FailureOr<StridedMemrefView> Decode(TypeID type_id, void* value) {
    if (!(CustomCall::CheckType<Tagged<MemrefView>>(checks, type_id) ||
          CustomCall::CheckType<Tagged<StridedMemrefView>>(checks, type_id)))
      return failure();

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

  LLVM_ATTRIBUTE_ALWAYS_INLINE
  static FailureOr<MemrefView> Decode(TypeID type_id, void* value) {
    if (!CustomCall::CheckType<Tagged<MemrefView>>(checks, type_id))
      return failure();

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

  LLVM_ATTRIBUTE_ALWAYS_INLINE
  static FailureOr<FlatMemrefView> Decode(TypeID type_id, void* value) {
    if (!CustomCall::CheckType<Tagged<MemrefView>>(checks, type_id))
      return failure();

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
    LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> Decode(TypeID type_id, \
                                                            void* value) {  \
      if (!CustomCall::CheckType<Tagged<T>>(checks, type_id))               \
        return failure();                                                   \
                                                                            \
      return *reinterpret_cast<T*>(value);                                  \
    }                                                                       \
  }

XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING(bool);
XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING(int32_t);
XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING(int64_t);
XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING(float);
XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING(double);

#undef XLA_RUNTIME_REGISTER_SCALAR_ARG_DECODING

template <CustomCall::RuntimeChecks checks>
struct CustomCallArgDecoding<Eigen::half, checks> {
  LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<Eigen::half> Decode(
      TypeID type_id, void* value) {
    if (!CustomCall::CheckType<Tagged<Eigen::half>>(checks, type_id))
      return failure();

    auto* src = reinterpret_cast<uint16_t*>(value);
    return Eigen::numext::bit_cast<Eigen::half>(*src);
  }
};

//===----------------------------------------------------------------------===//
// Custom call attributes decoding.

template <CustomCall::RuntimeChecks checks>
struct CustomCallAttrDecoding<std::string_view, checks> {
  LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<std::string_view> Decode(
      std::string_view name, TypeID type_id, void* value) {
    if (!CustomCall::CheckType<Tagged<std::string_view>>(checks, type_id)) {
      return failure();
    }

    auto* encoded = reinterpret_cast<internal::EncodedArray<char>*>(value);
    return std::string_view(encoded->data, encoded->size);
  }
};

template <CustomCall::RuntimeChecks checks>
struct CustomCallAttrDecoding<CustomCall::VariantAttr, checks> {
  LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<CustomCall::VariantAttr> Decode(
      std::string_view name, TypeID type_id, void* value) {
    return CustomCall::VariantAttr(name, type_id, value);
  }
};

#define XLA_RUNTIME_REGISTER_SCALAR_ATTR_DECODING(T)          \
  template <CustomCall::RuntimeChecks checks>                 \
  struct CustomCallAttrDecoding<T, checks> {                  \
    LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> Decode(  \
        std::string_view name, TypeID type_id, void* value) { \
      if (!CustomCall::CheckType<Tagged<T>>(checks, type_id)) \
        return failure();                                     \
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

// Both EncodedArray and 1-D EncodedDenseElements can be decoded as an
// llvm::ArrayRef. Pointers to both EncodedArray and 1-D EncodedDenseElements
// can be dereferenced as a pointer to EncodedArray.
#define XLA_RUNTIME_REGISTER_ARRAY_ATTR_DECODING(T)                           \
  template <CustomCall::RuntimeChecks checks>                                 \
  struct CustomCallAttrDecoding<llvm::ArrayRef<T>, checks> {                  \
    LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<llvm::ArrayRef<T>> Decode(  \
        std::string_view name, TypeID type_id, void* value) {                 \
      if ((!CustomCall::CheckType<Tagged<llvm::ArrayRef<T>>>(checks,          \
                                                             type_id)) &&     \
          (!CustomCall::CheckType<Tagged<CustomCall::TensorRef<T>>>(          \
              checks, type_id)) &&                                            \
          (!CustomCall::CheckType<Tagged<EmptyArrayRef>>(checks, type_id))) { \
        return failure();                                                     \
      }                                                                       \
                                                                              \
      auto* encoded = reinterpret_cast<internal::EncodedArray<T>*>(value);    \
      return llvm::ArrayRef<T>(encoded->data, encoded->size);                 \
    }                                                                         \
  }

XLA_RUNTIME_REGISTER_ARRAY_ATTR_DECODING(int32_t);
XLA_RUNTIME_REGISTER_ARRAY_ATTR_DECODING(int64_t);
XLA_RUNTIME_REGISTER_ARRAY_ATTR_DECODING(float);
XLA_RUNTIME_REGISTER_ARRAY_ATTR_DECODING(double);

#undef XLA_RUNTIME_REGISTER_ARRAY_ATTR_DECODING

#define XLA_RUNTIME_REGISTER_DENSE_ELEMENTS_ATTR_DECODING(T)                 \
  template <CustomCall::RuntimeChecks checks>                                \
  struct CustomCallAttrDecoding<CustomCall::TensorRef<T>, checks> {          \
    LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<CustomCall::TensorRef<T>>  \
    Decode(std::string_view name, TypeID type_id, void* value) {             \
      if (!CustomCall::CheckType<Tagged<CustomCall::TensorRef<T>>>(checks,   \
                                                                   type_id)) \
        return failure();                                                    \
                                                                             \
      auto* encoded =                                                        \
          reinterpret_cast<internal::EncodedDenseElements<T>*>(value);       \
      auto payload = encoded->payload;                                       \
      llvm::ArrayRef<T> data(payload.data, payload.size);                    \
      llvm::ArrayRef<int64_t> shape(encoded->shape, encoded->rank);          \
      return CustomCall::TensorRef<T>({shape, data});                        \
    }                                                                        \
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
    LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> Decode(      \
        std::string_view name, TypeID type_id, void* value) {     \
      if (!CustomCall::CheckType<Tagged<T>>(checks, type_id))     \
        return failure();                                         \
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
    LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> Decode(                   \
        std::string_view name, TypeID type_id, void* value) {                  \
      if (!CustomCall::CheckType<Tagged<T>>(checks, type_id)) {                \
        return failure();                                                      \
      }                                                                        \
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

  LLVM_ATTRIBUTE_ALWAYS_INLINE
  static FailureOr<T> Decode(void** value,
                             std::array<std::string_view, kSize> names) {
    internal::DecodedAttrs attrs(value);
    return Decode(attrs, names, std::make_index_sequence<kSize>{});
  }

  template <size_t... Is>
  LLVM_ATTRIBUTE_ALWAYS_INLINE static FailureOr<T> Decode(
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

    // Check if all members were decoded.
    bool all_decoded = true;
    auto check_all_decoded = [&](auto result) {
      all_decoded &= succeeded(result);
      return std::move(result);
    };

    // Decode all arguments into FailureOr containers. It is guaranteed
    // that initializer list will be evaluated left-to-right, and we can rely
    // on correct offsets computation.
    std::tuple<FailureOr<Ts>...> members = {
        check_all_decoded(CustomCallAttrDecoding<Ts, checks>::Decode(
            attrs[Is].name, attrs[Is].type_id, attrs[Is].value))...};
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

// Declare/define an explicit specialialization for TypeID for types used
// by the custom calls. This forces the compiler to emit a strong definition for
// a class and controls which translation unit and shared object will actually
// have it.
//
// See TypeID for more documentation.
//
// Because custom calls do not "own" the types passed across the function
// boundary, we declare/define specializations for tagged types to avoid
// potential conflicts with other libraries.
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
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(int32_t);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(int64_t);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(float);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(double);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(llvm::ArrayRef<int32_t>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(llvm::ArrayRef<int64_t>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(llvm::ArrayRef<float>);
XLA_RUNTIME_DECLARE_EXPLICIT_TYPE_ID(llvm::ArrayRef<double>);

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_CUSTOM_CALL_H_

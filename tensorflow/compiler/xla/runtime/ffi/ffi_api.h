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
#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/runtime/ffi/ffi_abi.h"
#include "tensorflow/compiler/xla/runtime/ffi/ffi_c_api.h"

namespace xla {
namespace runtime {
namespace ffi {

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
// Check struct sizes passed across the C API to detect mismatched versions.
//===----------------------------------------------------------------------===//

namespace internal {
inline void CheckStructSize(std::string_view struct_name, size_t expected_size,
                            size_t actual_size) {
  if (expected_size != actual_size) {
    std::cerr << "Unexpected " << struct_name << " size: expected "
              << expected_size << " << got " << actual_size
              << ". Check installed software versions." << std::endl;
    std::abort();
  }
}
}  // namespace internal

#define CHECK_ARGS_SIZE(name, args)                            \
  internal::CheckStructSize("XLA_FFI_##name##_Args",           \
                            XLA_FFI_##name##_Args_STRUCT_SIZE, \
                            args->struct_size)

//===----------------------------------------------------------------------===//
// Span is non-owning view into contiguous values ot type `T`.
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): Replace with `std::span` when C++20 is available.
template <typename T>
class Span {
 public:
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
// XLA FFI status wrapper around error reporting APIs.
//===----------------------------------------------------------------------===//

class FfiStatus {
 public:
  static FfiStatus Ok() { return FfiStatus(); }

  static FfiStatus Internal(std::string message) {
    XLA_FFI_Error_Code errc = XLA_FFI_Error_Code_INTERNAL;
    return FfiStatus(errc, message);
  }

  static FfiStatus InvalidArgument(std::string message) {
    XLA_FFI_Error_Code errc = XLA_FFI_Error_Code_INVALID_ARGUMENT;
    return FfiStatus(errc, message);
  }

  std::optional<XLA_FFI_Error_Code> errc() const { return errc_; }

  std::string_view message() const {
    return message_.has_value() ? *message_ : std::string_view();
  }

  const char* message_c_str() const {
    return message_.has_value() ? message_->c_str() : "";
  }

 private:
  FfiStatus() = default;

  FfiStatus(XLA_FFI_Error_Code errc, std::string message)
      : errc_(errc), message_(std::move(message)) {}

  std::optional<XLA_FFI_Error_Code> errc_;
  std::optional<std::string> message_;
};

//===----------------------------------------------------------------------===//
// XLA FFI virtual base for implementing FFI functions.
//===----------------------------------------------------------------------===//

class Ffi {
 public:
  virtual ~Ffi() = default;

  virtual XLA_FFI_Error* operator()(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx, void** args,
                                    void** attrs, void** rets) const = 0;

  static FfiBinding<> Binding();

  template <typename T>
  static bool Isa(const XLA_FFI_Api* api, XLA_FFI_TypeId type_id);

  template <typename T, typename U, typename... Ts>
  static bool Isa(const XLA_FFI_Api* api, XLA_FFI_TypeId type_id) {
    return Isa<T>(api, type_id) || Isa<U, Ts...>(api, type_id);
  }
};

//===----------------------------------------------------------------------===//
// XLA FFI module is a base class for stateful and stateless FFI modules.
//===----------------------------------------------------------------------===//

class Module {
 public:
  virtual ~Module() = default;

  struct ExportedFunction {
    std::string target;
    XLA_FFI_Function* function;
  };

 protected:
  Module(const XLA_FFI_Api* api, std::string module_name,
         std::vector<ExportedFunction> exported_functions,
         XLA_FFI_Module_StateType state_type,
         XLA_FFI_Module_CreateState* create_state,
         XLA_FFI_Module_DestroyState* destroy_state)
      : api_(api),
        module_name_(std::move(module_name)),
        exported_functions_(std::move(exported_functions)) {
    Register(state_type, create_state, destroy_state);
  }

 private:
  // Register `this` module with the XLA runtime.
  void Register(XLA_FFI_Module_StateType state_type,
                XLA_FFI_Module_CreateState* create_state,
                XLA_FFI_Module_DestroyState* destroy_state) {
    XLA_FFI_Module_Register_Args args;
    args.struct_size = XLA_FFI_Module_Register_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.name = module_name_.c_str();
    args.module = reinterpret_cast<XLA_FFI_Module*>(this);
    args.state_type = state_type;
    args.create_state = create_state;
    args.destroy_state = destroy_state;

    std::vector<const char*> exported_names;
    std::vector<XLA_FFI_Function*> exported_functions;
    for (auto& fn : exported_functions_) {
      exported_names.push_back(fn.target.c_str());
      exported_functions.push_back(fn.function);
    }

    args.num_exported_functions = exported_functions_.size();
    args.exported_names = exported_names.data();
    args.exported_functions = exported_functions.data();

    api_->XLA_FFI_Module_Register(&args);
  }

  // Module is registered with the XLA runtime behind this API instance, and any
  // module manipulation (e.g. export functions) must be done through it.
  const XLA_FFI_Api* api_;

  std::string module_name_;
  std::vector<ExportedFunction> exported_functions_;
};

//===----------------------------------------------------------------------===//
// XLA FFI stateful module is a collection of FFI functions and a state.
//===----------------------------------------------------------------------===//

template <typename State>
class StatefulModule : public Module {
 public:
  // TODO(ezhulenev): To gracefully fail if state can't be created, this has to
  // return `FfiStatusOr<std::unique_ptr<State>>`, but we do not have a
  // `StatusOr` implementation yet and we can't depend on absl.
  virtual std::unique_ptr<State> CreateState() = 0;

 protected:
  StatefulModule(const XLA_FFI_Api* api, std::string module_name,
                 std::vector<ExportedFunction> exported_functions,
                 bool per_execution_state = false)
      : Module(api, std::move(module_name), std::move(exported_functions),
               per_execution_state ? XLA_FFI_Module_State_PER_EXECUTION
                                   : XLA_FFI_Module_State_PER_EXECUTABLE,
               CreateState, DestroyState) {}

 private:
  // Implements `XLA_FFI_Module_CreateState` API function.
  static XLA_FFI_Error* CreateState(XLA_FFI_Module_CreateState_Args* args);

  // Implements `XLA_FFI_Module_DestroyState` API function.
  static void DestroyState(XLA_FFI_Module_DestroyState_Args* args);
};

template <typename State>
XLA_FFI_Error* StatefulModule<State>::CreateState(
    XLA_FFI_Module_CreateState_Args* args) {
  CHECK_ARGS_SIZE(Module_CreateState, args);

  auto* module = reinterpret_cast<StatefulModule*>(args->module);
  auto* state = module->CreateState().release();
  args->state = reinterpret_cast<XLA_FFI_Module_State*>(state);
  return nullptr;  // success
}

template <typename State>
void StatefulModule<State>::DestroyState(
    XLA_FFI_Module_DestroyState_Args* args) {
  CHECK_ARGS_SIZE(Module_DestroyState, args);

  delete reinterpret_cast<State*>(args->state);
}

//===----------------------------------------------------------------------===//
// XLA FFI stateless module is a collection of FFI functions without a state.
//===----------------------------------------------------------------------===//

class StatelessModule : public Module {
 protected:
  StatelessModule(const XLA_FFI_Api* api, std::string module_name,
                  std::vector<ExportedFunction> exported_functions)
      : Module(api, std::move(module_name), std::move(exported_functions),
               /*state_type=*/XLA_FFI_Module_State_PER_EXECUTABLE,
               /*create_state=*/nullptr, /*destroy_state=*/nullptr) {}
};

//===----------------------------------------------------------------------===//
// Helper macro to define a static module registration.
//===----------------------------------------------------------------------===//

#define XLA_REGISTER_FFI_MODULE(FUNC) \
  XLA_REGISTER_FFI_MODULE_IMPL(FUNC, __COUNTER__)

#define XLA_REGISTER_FFI_MODULE_IMPL(FUNC, N)           \
  static bool xla_ffi_module_##N##_registered_ = []() { \
    static auto* module = FUNC.release();               \
    return module != nullptr;                           \
  }()

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

// A type tag to represent dictionary attributes that can be decoded into
// structs using aggregate attribute decoding.
struct Dictionary {};

template <typename T>
bool Ffi::Isa(const XLA_FFI_Api* api, XLA_FFI_TypeId type_id) {
#define ISA(type, name)                  \
  if constexpr (std::is_same_v<T, type>) \
    return api->XLA_FFI_Get_##name##_TypeId() == type_id;

  ISA(std::string_view, String);

  ISA(float, Float);
  ISA(double, Double);
  ISA(bool, Int1);
  ISA(int32_t, Int32);
  ISA(int64_t, Int64);

  ISA(Span<const float>, FloatArray);
  ISA(Span<const double>, DoubleArray);
  ISA(Span<const int32_t>, Int32Array);
  ISA(Span<const int64_t>, Int64Array);

  ISA(StridedBufferArg, StridedBufferArg);
  ISA(BufferArg, BufferArg);
  ISA(Dictionary, Dictionary);

  assert(false && "Unsupported type");
  return false;

#undef ISA
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
struct AttrTag {};

// A type tag to distinguish argument tied to FFI module state.
template <typename T>
struct StateTag {};

// A type tag to distinguish argument tied to XLA runtime stream.
template <typename T>
struct StreamTag {};

// A template for checking if type is a wrapped attribute or user data.
// clang-format off
template <typename>   struct IsWrapped               : std::false_type {};
template <typename T> struct IsWrapped<AttrTag<T>>   : std::true_type {};
template <typename T> struct IsWrapped<StateTag<T>>  : std::true_type {};
template <typename T> struct IsWrapped<StreamTag<T>> : std::true_type {};
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
  FfiBinding<Ts..., internal::AttrTag<T>> Attr(std::string attr) && {
    attrs_.push_back(std::move(attr));
    return {std::move(*this)};
  }

  template <typename T>
  FfiBinding<Ts..., internal::StateTag<T>> State() && {
    return {std::move(*this)};
  }

  template <typename T>
  FfiBinding<Ts..., internal::StreamTag<T>> Stream() && {
    static_assert(std::is_pointer_v<T>,
                  "T must be a pointer type, e.g. for GPU platform it must be "
                  "se::gpu::GpuStreamHandle");
    return {std::move(*this)};
  }

  template <typename Fn>
  std::unique_ptr<FfiHandler<Fn, Ts...>> To(Fn fn) {
    return std::unique_ptr<FfiHandler<Fn, Ts...>>(
        new FfiHandler<Fn, Ts...>(std::forward<Fn>(fn), std::move(attrs_)));
  }

 private:
  template <typename...>
  friend class FfiBinding;
  friend class Ffi;

  explicit FfiBinding() {
    static_assert(sizeof...(Ts) == 0, "ffi arguments must be empty");
  }

  template <typename... TTs>
  FfiBinding(FfiBinding<TTs...>&& other)  // NOLINT
      : attrs_(std::move(other.attrs_)) {}

  FfiBinding(FfiBinding&) = delete;

  std::vector<std::string> attrs_;  // names of bound attributes
};

inline FfiBinding<> Ffi::Binding() { return FfiBinding<>(); }

//===----------------------------------------------------------------------===//
// Helpers for decoding opaque arguments and attributes' memory.
//===----------------------------------------------------------------------===//

namespace internal {

using runtime::internal::EncodedArray;
using runtime::internal::EncodedMemref;

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
  static std::optional<T> call(const XLA_FFI_Api* api,
                               XLA_FFI_ExecutionContext* ctx,
                               DecodingOffsets& offsets,
                               internal::DecodedArgs args,
                               const std::vector<std::string>& attrs_names,
                               const std::vector<size_t>& attrs_idx,
                               internal::DecodedAttrs attrs) {
    internal::DecodedArg arg = args[offsets.args++];
    return FfiArgDecoding<T>::Decode(api, arg.type_id, arg.value);
  }
};

template <typename T>
struct Decode<AttrTag<T>> {
  static std::optional<T> call(const XLA_FFI_Api* api,
                               XLA_FFI_ExecutionContext* ctx,
                               DecodingOffsets& offsets,
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

    return FfiAttrDecoding<T>::Decode(api, attrs[i].name, attrs[i].type_id,
                                      attrs[i].value);
  }
};

template <typename T>
struct Decode<StateTag<T>> {
  static std::optional<T*> call(const XLA_FFI_Api* api,
                                XLA_FFI_ExecutionContext* ctx,
                                DecodingOffsets& offsets, internal::DecodedArgs,
                                const std::vector<std::string>& attrs_names,
                                const std::vector<size_t>& attrs_idx,
                                internal::DecodedAttrs attrs) {
    XLA_FFI_ExecutionContext_GetModuleState_Args args;
    args.struct_size = XLA_FFI_ExecutionContext_GetModuleState_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.ctx = ctx;

    XLA_FFI_Module_State* state =
        api->XLA_FFI_ExecutionContext_GetModuleState(&args);
    return reinterpret_cast<T*>(state);
  }
};

template <typename T>
struct Decode<StreamTag<T>> {
  static std::optional<T> call(const XLA_FFI_Api* api,
                               XLA_FFI_ExecutionContext* ctx,
                               DecodingOffsets& offsets, internal::DecodedArgs,
                               const std::vector<std::string>& attrs_names,
                               const std::vector<size_t>& attrs_idx,
                               internal::DecodedAttrs attrs) {
    XLA_FFI_ExecutionContext_GetStream_Args args;
    args.struct_size = XLA_FFI_ExecutionContext_GetStream_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.ctx = ctx;

    XLA_FFI_Stream* stream = api->XLA_FFI_ExecutionContext_GetStream(&args);
    return reinterpret_cast<T>(stream);
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
template <typename T> struct FnArgType               { using Type = T;  };
template <typename T> struct FnArgType<AttrTag<T>>   { using Type = T;  };
template <typename T> struct FnArgType<StateTag<T>>  { using Type = T*; };
template <typename T> struct FnArgType<StreamTag<T>> { using Type = T;  };
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

  static XLA_FFI_Error* ToError(const XLA_FFI_Api* api, FfiStatus status) {
    if (!status.errc().has_value()) return nullptr;

    XLA_FFI_Error_Create_Args args;
    args.struct_size = XLA_FFI_Error_Create_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.errc = *status.errc();
    args.message = status.message_c_str();

    return api->XLA_FFI_Error_Create(&args);
  }

 public:
  XLA_FFI_Error* operator()(const XLA_FFI_Api* api,
                            XLA_FFI_ExecutionContext* ctx, void** args,
                            void** attrs, void** rets) const final {
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
      return ToError(api, FfiStatus::InvalidArgument(err.str()));
    }

    // Check that we have the correct number of attributes passed to the
    // handler. Each individual attribute decoding will check the name and the
    // type of the attribute.
    if (num_attrs != attrs_.size()) {
      std::ostringstream err;
      err << "Wrong number of attributes: expected " << attrs_.size() << " got "
          << num_attrs;
      return ToError(api, FfiStatus::InvalidArgument(err.str()));
    }

    // Define index sequence to access ffi handler arguments.
    using Is = std::make_index_sequence<kSize>;
    return call(api, ctx, decoded_args, decoded_attrs, Is{});
  }

 private:
  template <typename...>
  friend class FfiBinding;

  template <size_t... Is>
  XLA_FFI_Error* call(const XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx,
                      internal::DecodedArgs args, internal::DecodedAttrs attrs,
                      std::index_sequence<Is...>) const {
    // A helper structure to allow each decoder find the correct offset in the
    // arguments, attributes or results.
    internal::DecodingOffsets offsets;

    // Decode all operands into `std::optional` containers. It is guaranteed
    // that initializer list will be evaluated left-to-right, and we can rely
    // on correct offsets computation.
    std::tuple<std::optional<FnArgType<Ts>>...> fn_args = {
        internal::Decode<Ts>::call(api, ctx, offsets, args, attrs_, attrs_idx_,
                                   attrs)...};

    // Check if all arguments, attributes and results were decoded;
    bool all_decoded = (std::get<Is>(fn_args).has_value() && ...);
    if (!all_decoded) {
      return ToError(
          api, FfiStatus::InvalidArgument("Failed to decode all FFI operands"));
    }

    // Custom call returns `FfiStatus`, we can call it directly.
    if constexpr (kIsFfiStatusHandler) {
      return ToError(api, fn_(std::move(*std::get<Is>(fn_args))...));
    }

    return ToError(api, FfiStatus::Ok());
  }

  FfiHandler(Fn fn, std::vector<std::string> attrs)
      : fn_(std::move(fn)),
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
    static std::optional<T> Decode(const XLA_FFI_Api* api,                \
                                   XLA_FFI_TypeId type_id, void* value) { \
      if (!Ffi::Isa<T>(api, type_id)) {                                   \
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

  static std::optional<StridedBufferArg> Decode(const XLA_FFI_Api* api,
                                                XLA_FFI_TypeId type_id,
                                                void* value) {
    if (!Ffi::Isa<BufferArg, StridedBufferArg>(api, type_id)) {
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

  static std::optional<BufferArg> Decode(const XLA_FFI_Api* api,
                                         XLA_FFI_TypeId type_id, void* value) {
    if (!Ffi::Isa<BufferArg>(api, type_id)) {
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
    static std::optional<T> Decode(const XLA_FFI_Api* api,                \
                                   std::string_view name,                 \
                                   XLA_FFI_TypeId type_id, void* value) { \
      if (!Ffi::Isa<T>(api, type_id)) {                                   \
        return std::nullopt;                                              \
      }                                                                   \
                                                                          \
      return *reinterpret_cast<T*>(value);                                \
    }                                                                     \
  }

XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(float);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(double);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(bool);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(int32_t);
XLA_FFI_REGISTER_SCALAR_ATTR_DECODING(int64_t);

#undef XLA_FFI_REGISTER_SCALAR_ATTR_DECODING

#define XLA_FFI_REGISTER_ARRAY_ATTR_DECODING(T)                            \
  template <>                                                              \
  struct FfiAttrDecoding<Span<const T>> {                                  \
    static std::optional<Span<const T>> Decode(const XLA_FFI_Api* api,     \
                                               std::string_view name,      \
                                               XLA_FFI_TypeId type_id,     \
                                               void* value) {              \
      if (!Ffi::Isa<Span<const T>>(api, type_id)) {                        \
        return std::nullopt;                                               \
      }                                                                    \
                                                                           \
      auto* encoded = reinterpret_cast<internal::EncodedArray<T>*>(value); \
      return Span<const T>(encoded->data, encoded->size);                  \
    }                                                                      \
  }

XLA_FFI_REGISTER_ARRAY_ATTR_DECODING(float);
XLA_FFI_REGISTER_ARRAY_ATTR_DECODING(double);
XLA_FFI_REGISTER_ARRAY_ATTR_DECODING(bool);
XLA_FFI_REGISTER_ARRAY_ATTR_DECODING(int32_t);
XLA_FFI_REGISTER_ARRAY_ATTR_DECODING(int64_t);

#undef XLA_FFI_REGISTER_ARRAY_ATTR_DECODING

template <>
struct FfiAttrDecoding<std::string_view> {
  static std::optional<std::string_view> Decode(const XLA_FFI_Api* api,
                                                std::string_view name,
                                                XLA_FFI_TypeId type_id,
                                                void* value) {
    if (!Ffi::Isa<std::string_view>(api, type_id)) {
      return std::nullopt;
    }

    auto* encoded = reinterpret_cast<internal::EncodedArray<char>*>(value);
    return std::string_view(encoded->data, encoded->size);
  }
};

//===----------------------------------------------------------------------===//
// Register an XLA FFI attribute decoding from dictionaries to structs.
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
//   XLA_FFI_REGISTER_AGGREGATE_ATTR_DECODING(
//     PairOfI64,
//     AggregateMember<int64_t>("a"),
//     AggregateMember<int64_t>("b"));
//
#define XLA_FFI_REGISTER_AGGREGATE_ATTR_DECODING(T, ...)                       \
  template <>                                                                  \
  struct FfiAttrDecoding<T> {                                                  \
    static std::optional<T> Decode(const XLA_FFI_Api* api,                     \
                                   std::string_view name,                      \
                                   XLA_FFI_TypeId type_id, void* value) {      \
      if (!Ffi::Isa<Dictionary>(api, type_id)) {                               \
        return std::nullopt;                                                   \
      }                                                                        \
                                                                               \
      auto decoder = internal::AggregateDecoder<T>(__VA_ARGS__);               \
      return decltype(decoder)::Decode(api, reinterpret_cast<void**>(value),   \
                                       internal::AggregateNames(__VA_ARGS__)); \
    }                                                                          \
  }

namespace internal {
// Decodes aggregate attribute into the object of type `T` that must be
// constructible from the `Ts` types.
template <typename T, typename... Ts>
struct DecodeAggregateAttr {
  static constexpr size_t kSize = sizeof...(Ts);

  static std::optional<T> Decode(const XLA_FFI_Api* api, void** value,
                                 std::array<std::string_view, kSize> names) {
    internal::DecodedAttrs attrs(value);
    return Decode(api, attrs, names, std::make_index_sequence<kSize>{});
  }

  template <size_t... Is>
  static std::optional<T> Decode(const XLA_FFI_Api* api,
                                 internal::DecodedAttrs attrs,
                                 std::array<std::string_view, kSize> names,
                                 std::index_sequence<Is...>) {
    // Check that the number of encoded attributes matches the signature.
    if (kSize != attrs.size()) return std::nullopt;

    // Check that aggregate member names match the expected names.
    for (unsigned i = 0; i < kSize; ++i)
      if (attrs[i].name != names[i]) return std::nullopt;

    // Decode all arguments into std::optional containers. It is guaranteed
    // that initializer list will be evaluated left-to-right, and we can rely
    // on correct offsets computation.
    std::tuple<std::optional<Ts>...> members = {FfiAttrDecoding<Ts>::Decode(
        api, attrs[Is].name, attrs[Is].type_id, attrs[Is].value)...};

    bool all_decoded = (std::get<Is>(members).has_value() && ...);
    if (!all_decoded) return std::nullopt;

    // Forward unpacked members to the type constructor.
    return T{std::move(*std::get<Is>(members))...};
  }
};

template <typename... Members>
auto AggregateNames(Members... m) {
  return std::array<std::string_view, sizeof...(Members)>{m.name...};
}

template <typename T, typename... Members>
auto AggregateDecoder(Members... m) {
  return DecodeAggregateAttr<T, typename Members::Type...>();
}

}  // namespace internal

//===----------------------------------------------------------------------===//
// XLA FFI helper macro for registering FFI implementations.
//===----------------------------------------------------------------------===//

#define XLA_FFI_DEFINE_FUNCTION(fn, impl, binding)                   \
  static XLA_FFI_Error* fn(XLA_FFI_Function_Args* args) {            \
    ::xla::runtime::ffi::internal::CheckStructSize(                  \
        "XLA_FFI_Function_Args", XLA_FFI_Function_Args_STRUCT_SIZE,  \
        args->struct_size);                                          \
    static auto* handler = binding.To(impl).release();               \
    return (*handler)(args->api, args->ctx, args->args, args->attrs, \
                      args->rets);                                   \
  }

#undef CHECK_ARGS_SIZE

}  // namespace ffi
}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_FFI_FFI_API_H_

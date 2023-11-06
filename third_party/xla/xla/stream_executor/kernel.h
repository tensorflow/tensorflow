/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Suite of datatypes to represent data-parallel kernel objects (code entities).
// Kernel is the untyped variant, whereas TypedKernel takes a type signature
// to do some template-based helper generation and give compile-time type
// checking for kernel launch parameters.
//
// Users typically don't see KernelBase, they see typed kernels, analogous to a
// typed function pointer. TypedKernels express their argument types via
// template parameters like so:
//
//  TypedKernel<DeviceMemory<int>*, int>
//
// Which expresses a data parallel kernel signature for:
//
//  void(int*, int);
//
// And for a const memory region:
//
//  TypedKernel<const DeviceMemory<int>&, int>
//
// Corresponds to a data parallel kernel signature for:
//
//  void(const int*, int)
//
// Note that kernels always have a void return type, so results typically must
// be memcpy'ied from device memory to the host.
//
// Also note that a scalar integer residing in device memory and an array of
// integers residing in device memory have the same signature: DeviceMemory<T>.
// However, in the future, checks may be added for additional safety that arrays
// of minimum sizes are passed when those minimum sizes are contractually
// expected by the kernel.
//
// For user-defined types whose definitions are appropriately shared between the
// host code doing the launching and the kernel code being launched, the user
// defined types are similarly permitted to be expressed as residing in device
// memory:
//
//  TypedKernel<DeviceMemory<MyUserDefinedStructure>>
//
// And, when the alignment and padding are agreed upon, POD types will also be
// able to be passed by value; for example, it is a common idiom to specify a
// bunch of options simultaneously with a structure:
//
//  TypedKernel<MyOptionsStructurePassedByValue, DeviceMemory<float>>
//
// Which corresponds to a data parallel kernel signature like:
//
//  void(MyOptionsStructurePassedByValue value, float *result);
//
// Users typically won't need to type out the TypedKernel signature in full, it
// will be typedef'd by automatically generated code; for example, see
// stream_executor::executor_sample::VecReduceAddKernel.

#ifndef XLA_STREAM_EXECUTOR_KERNEL_H_
#define XLA_STREAM_EXECUTOR_KERNEL_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_memory.h"

namespace stream_executor {

class DeviceMemoryBase;
template <typename ElemT>
class DeviceMemory;
class StreamExecutor;

namespace internal {
class KernelInterface;
}  // namespace internal

//===----------------------------------------------------------------------===//
// Kernel cache config
//===----------------------------------------------------------------------===//

// This enum represents potential configurations of L1/shared memory when
// running a particular kernel. These values represent user preference, and
// the runtime is not required to respect these choices.
enum class KernelCacheConfig {
  // Indicates no preference for device L1/shared memory configuration.
  kNoPreference,

  // Indicates a preference for more shared memory than L1 cache.
  kPreferShared,

  // Indicates a preference for more L1 cache than shared memory.
  kPreferL1,

  // Indicates a preference for equal amounts of L1 cache and shared memory.
  kPreferEqual,
};

//===----------------------------------------------------------------------===//
// Kernel metadata
//===----------------------------------------------------------------------===//

// KernelMetadata holds runtime-queryable attributes of a loaded kernel, such as
// registers allocated, shared memory used, etc.
// Not all platforms support reporting of all information, so each accessor
// returns false if the associated field is not populated in the underlying
// platform.
class KernelMetadata {
 public:
  KernelMetadata() = default;

  // Returns the number of registers used per thread executing this kernel.
  std::optional<int64_t> registers_per_thread() const;

  // Returns the amount of [static] shared memory used per block executing this
  // kernel. Note that dynamic shared memory allocations are not (and can not)
  // be reported here (since they're not specified until kernel launch time).
  std::optional<int64_t> shared_memory_bytes() const;

  void set_registers_per_thread(int registers_per_thread);
  void set_shared_memory_bytes(int shared_memory_bytes);

 private:
  std::optional<int64_t> registers_per_thread_;
  std::optional<int64_t> shared_memory_bytes_;
};

//===----------------------------------------------------------------------===//
// Kernel
//===----------------------------------------------------------------------===//

// A data-parallel kernel (code entity) for launching via the StreamExecutor,
// analogous to a void* device function pointer. See TypedKernel for the typed
// variant.
//
// Thread-compatible.
class KernelBase {
 public:
  KernelBase(KernelBase &&from);

  // Constructs an "empty" (not-yet-loaded) kernel instance.
  //
  // parent is the StreamExecutor that will be responsible for loading the
  // implementation of this kernel. It must not be null.
  explicit KernelBase(StreamExecutor *parent);

  // Test-only constructor that can take a mock KernelInterface implementation.
  KernelBase(StreamExecutor *parent, internal::KernelInterface *implementation);

  // Releases resources associated with the kernel instance (i.e.
  // platform-specific implementation).
  ~KernelBase();

  // Returns the number of parameters that this kernel accepts. (Arity refers to
  // nullary, unary, ...).
  unsigned Arity() const;

  // Returns the StreamExecutor that represents the platform this kernel
  // executes upon.
  StreamExecutor *parent() const { return parent_; }

  // Returns a const pointer to the (opaque) platform-dependent implementation.
  const internal::KernelInterface *implementation() const {
    return implementation_.get();
  }

  // Returns a non-const pointer to the (opaque) platform-dependent
  // implementation.
  internal::KernelInterface *implementation() { return implementation_.get(); }

  void set_metadata(const KernelMetadata &metadata) { metadata_ = metadata; }

  const KernelMetadata &metadata() const { return metadata_; }

  // Sets the preferred cache configuration for a kernel. This is just a
  // suggestion to the runtime, and may not be honored during execution.
  void SetPreferredCacheConfig(KernelCacheConfig config);

  // Gets the preferred cache configuration for a kernel.
  KernelCacheConfig GetPreferredCacheConfig() const;

  void set_name(absl::string_view name);
  const std::string &name() const { return name_; }
  const std::string &demangled_name() const { return demangled_name_; }

 private:
  // The StreamExecutor that loads this kernel object.
  StreamExecutor *parent_;

  // Implementation delegated to for platform-specific functionality.
  std::unique_ptr<internal::KernelInterface> implementation_;

  std::string name_;
  std::string demangled_name_;

  KernelMetadata metadata_;

  KernelBase(const KernelBase &) = delete;
  void operator=(const KernelBase &) = delete;
};

//===----------------------------------------------------------------------===//
// Device memory pointer traits
//===----------------------------------------------------------------------===//

namespace internal {
template <typename T>
struct IsDeviceMemory : public std::false_type {};
template <typename U>
struct IsDeviceMemory<DeviceMemory<U>> : public std::true_type {};
template <>
struct IsDeviceMemory<DeviceMemoryBase> : public std::true_type {};

template <typename U>
struct IsSharedDeviceMemory : public std::false_type {};
template <typename U>
struct IsSharedDeviceMemory<SharedDeviceMemory<U>> : public std::true_type {};
}  // namespace internal

template <typename T>
static constexpr bool is_device_memory_pointer_v =
    std::is_pointer_v<T> &&
    internal::IsDeviceMemory<std::remove_pointer_t<T>>::value;

template <typename T>
static constexpr bool is_device_memory_value_like_v =
    !std::is_pointer_v<T> &&
    internal::IsDeviceMemory<std::remove_reference_t<T>>::value;

template <typename T>
static constexpr bool is_shared_device_memory_v =
    internal::IsSharedDeviceMemory<std::remove_reference_t<T>>::value;

//===----------------------------------------------------------------------===//
// Kernel arguments
//===----------------------------------------------------------------------===//

// Basic data about a kernel argument.
struct KernelArg {
  bool is_shared;
  const void *address;
  size_t size;
};

// Base class for KernelArgsArray.
//
// Supports all the getter methods that do not depend on the compile-time number
// of arguments template parameter.
//
// This class exists as a way to pass kernel arguments to
// StreamExecutorInterface::Launch. That Launch method is virtual, so it can't
// be templated to accept any KernelArgsArray type, therefore a reference to
// this base type is passed instead.
//
// Performance is not a concern here because each of these methods will be
// called at most once per kernel launch. Past performance concerns with
// KernelArgsArray have been in reference to the argument packing routines which
// are called once per kernel argument. Those packing routines are now handled
// by the templated KernelArgsArray subclass of this class where they can take
// advantage of compile-time knowledge of the number of arguments in order to be
// very efficient.
class KernelArgsArrayBase {
 public:
  virtual ~KernelArgsArrayBase() = default;

  // Gets the number of arguments added so far, including shared memory
  // arguments.
  virtual size_t number_of_arguments() const = 0;

  // Gets the total number of shared memory bytes added so far.
  virtual uint64_t number_of_shared_bytes() const = 0;

  // Gets the list of argument addresses.
  virtual absl::Span<const void *const> argument_addresses() const = 0;
};

// A list of arguments for a kernel call.
//
// The template parameter kNumArgs is the maximum number of arguments which can
// be stored in the list.
//
// Contains a list of addresses for non-shared-memory arguments and a list of
// sizes for shared-memory arguments. Since the shared-memory arguments may be
// interspersed with the non-shared-memory arguments, it also stores a list of
// the indices at which the shared-memory arguments appeared.
//
// For example, if the argument address list contains {a, b, c, d, e}, the
// shared-memory arguments list contains the sizes of {A, B, C}, and the
// shared-memory indices list contains {0, 3, 5}, then the original list of
// arguments was {A, a, b, B, c, C, d, e}.
//
// This way of storing the arguments makes CUDA kernel calls efficient because
// they only require the argument address list and the total number of shared
// bytes, but it also makes it possible for OpenCL kernel calls because they
// depend on the location of each shared-memory argument and its size.
//
// Note that the code for adding arguments has been identified as a performance
// hotspot in some real-world applications so this structure has been optimized
// for the performance of argument adding.
template <size_t kNumArgs>
class KernelArgsArray : public KernelArgsArrayBase {
 public:
  static constexpr int kMaxGenericArgSize = 8;

  // Adds an argument to the list.
  template <typename T>
  void add_argument(const T &arg) {
    static_assert(sizeof(T) <= kMaxGenericArgSize,
                  "Please adjust kMaxGenericArgSize");
    static_assert(std::is_pod_v<T>, "Only pod types supported!");
    char *generic_arg_storage =
        &generic_arguments_[number_of_generic_arguments_++ *
                            kMaxGenericArgSize];

    CHECK_EQ(reinterpret_cast<uintptr_t>(generic_arg_storage) % alignof(T), 0);
    std::memcpy(generic_arg_storage, &arg, sizeof(T));

    argument_addresses_[number_of_argument_addresses_] = generic_arg_storage;
    argument_sizes_[number_of_argument_addresses_] = sizeof(arg);
    ++number_of_argument_addresses_;
  }

  // Adds a device memory argument to the list.
  void add_device_memory_argument(const DeviceMemoryBase &arg) {
    const void **copy_ptr =
        &device_memory_opaque_pointers_[number_of_argument_addresses_];
    *copy_ptr = arg.opaque();
    argument_addresses_[number_of_argument_addresses_] = copy_ptr;
    argument_sizes_[number_of_argument_addresses_] = sizeof(void *);
    ++number_of_argument_addresses_;
  }

  // Adds a shared memory argument to the list.
  //
  // The only significant information about a shared argument is its size, so
  // that is the only parameter in this function.
  void add_shared_bytes(size_t number_of_bytes) {
    shared_memory_indices_[number_of_shared_memory_arguments_] =
        number_of_argument_addresses_ + number_of_shared_memory_arguments_;
    shared_memory_bytes_[number_of_shared_memory_arguments_] = number_of_bytes;
    ++number_of_shared_memory_arguments_;
    total_shared_memory_bytes_ += number_of_bytes;
  }

  // Gets the number of arguments added so far, including shared memory
  // arguments.
  size_t number_of_arguments() const override {
    return number_of_argument_addresses_ + number_of_shared_memory_arguments_;
  }

  // Gets the total number of shared memory bytes added so far.
  uint64_t number_of_shared_bytes() const override {
    return total_shared_memory_bytes_;
  }

  // Gets the list of argument addresses.
  absl::Span<const void *const> argument_addresses() const override {
    return absl::Span<const void *const>(argument_addresses_.data(),
                                         number_of_argument_addresses_);
  }

 private:
  // A place to store copies of opaque pointers from device memory arguments.
  std::array<const void *, kNumArgs> device_memory_opaque_pointers_;

  // Addresses for non-shared-memory arguments.
  std::array<const void *, kNumArgs> argument_addresses_;

  // Storage for arguments of templated type.
  alignas(kMaxGenericArgSize)
      std::array<char, kNumArgs * kMaxGenericArgSize> generic_arguments_;

  // Sizes for non-shared-memory arguments.
  std::array<size_t, kNumArgs> argument_sizes_;

  // Size in bytes for each shared memory argument.
  std::array<size_t, kNumArgs> shared_memory_bytes_;

  // Indices in the arguments array for shared memory arguments.
  std::array<size_t, kNumArgs> shared_memory_indices_;

  // Total of all shared memory sizes.
  size_t total_shared_memory_bytes_ = 0;

  // Number of significant entries in argument_addresses_ and argument_sizes_.
  size_t number_of_argument_addresses_ = 0;

  // Number of significant entries in shared_memory_bytes_ and
  // shared_memory_indices_.
  size_t number_of_shared_memory_arguments_ = 0;

  // The number of generic arguments that have been added to generic_arguments_.
  size_t number_of_generic_arguments_ = 0;
};

template <int n>
std::unique_ptr<KernelArgsArrayBase> MakeKernelArgs(
    absl::Span<const DeviceMemoryBase> args, uint32_t shared_mem_bytes) {
  auto kernel_args = std::make_unique<KernelArgsArray<n>>();
  for (const DeviceMemoryBase &buf : args) {
    kernel_args->add_device_memory_argument(buf);
  }
  if (shared_mem_bytes > 0) {
    kernel_args->add_shared_bytes(shared_mem_bytes);
  }
  return kernel_args;
}

//===----------------------------------------------------------------------===//
// Typed kernel
//===----------------------------------------------------------------------===//

// Typed variant of KernelBase, like a typed device function pointer. See the
// file comment for details and example usage.
//
// This class contains template metaprogramming magic to type check the
// parameters passed to a kernel launch are acceptable, and subsequently pack
// them into a form which can be used by the StreamExecutorInterface
// implementation. (i.e.  CUDA and OpenCL both bind void*s with associated
// sizes as kernel arguments.)
//
// Thread-compatible.
template <typename... Params>
class TypedKernel : public KernelBase {
 public:
  static constexpr size_t kNumberOfParameters = sizeof...(Params);

  // Delegates to KernelBase::KernelBase(), see that constructor.
  explicit TypedKernel(StreamExecutor *parent) : KernelBase(parent) {}

  // Test-only constructor that can take a mock KernelInterface implementation.
  // Takes ownership of implementation, it should not be null.
  TypedKernel(StreamExecutor *parent, internal::KernelInterface *implementation)
      : KernelBase(parent, implementation) {}

  // This is the main entry point into the magic. Packs the parameters (which
  // must type check against the class template) into the args and sizes
  // arrays.
  //
  // Const refs are taken as parameters on all of the handlers to avoid
  // implicit type promotion of integers.
  //
  // WARNING: as a performance optimization this method may store pointers to
  // some of the input parameters in the kernel args structure, so any params
  // passed into this method must live at least as long as the kernel args
  // structure.
  void PackParams(KernelArgsArray<kNumberOfParameters> *args,
                  const Params &...params) const {
    PackOneParamFromList(args, params...);
  }

 private:
  // Stream needs access to the specific parameter-packing functionality that
  // the TypedKernel provides for its corresponding type signature (and no other
  // type signatures).
  friend class Stream;

  template <typename T, typename... RestOfParams>
  void PackOneParamFromList(KernelArgsArray<kNumberOfParameters> *args,
                            const T &arg, const RestOfParams &...rest) const {
    PackOneParam(args, arg);
    PackOneParamFromList(args, rest...);
  }

  // Base case for variadic template expansion - nothing to do!
  void PackOneParamFromList(KernelArgsArray<kNumberOfParameters> *args) const {}

  // Packs one (non-DeviceMemoryBase) parameter into the arg and sizes array.
  // The enable_if<> is for excluding DeviceMemoryBase args, which have a
  // separate implementation below.
  template <typename T>
  void PackOneParam(
      KernelArgsArray<kNumberOfParameters> *args, const T &arg,
      typename std::enable_if_t<
          !is_device_memory_value_like_v<T> && !is_device_memory_pointer_v<T> &&
          !is_shared_device_memory_v<T>> * = nullptr) const {
    static_assert(!std::is_pointer_v<T>,
                  "cannot pass raw pointer to the device");
    static_assert(!std::is_convertible_v<T, DeviceMemoryBase>,
                  "cannot pass device memory as a normal value");
    args->add_argument(arg);
  }

  // DeviceMemoryBase family reference override.
  template <typename T>
  void PackOneParam(KernelArgsArray<kNumberOfParameters> *args, const T &arg,
                    typename std::enable_if_t<is_device_memory_value_like_v<T>>
                        * = nullptr) const {
    args->add_device_memory_argument(arg);
  }

  // DeviceMemoryBase family pointer override.
  template <typename T>
  void PackOneParam(KernelArgsArray<kNumberOfParameters> *args, T arg,
                    typename std::enable_if_t<is_device_memory_pointer_v<T>> * =
                        nullptr) const {
    DeviceMemoryBase *ptr = static_cast<DeviceMemoryBase *>(arg);
    args->add_device_memory_argument(*ptr);
  }

  // Dynamic shared device memory has a size, but no associated allocation on
  // the host; internally, the device will allocate storage.
  template <typename T>
  void PackOneParam(KernelArgsArray<kNumberOfParameters> *args, T arg,
                    typename std::enable_if_t<is_shared_device_memory_v<T>> * =
                        nullptr) const {
    args->add_shared_bytes(arg.size());
  }

  TypedKernel(const TypedKernel &) = delete;
  void operator=(const TypedKernel &) = delete;
};

// Template metaprogramming helper type that helps us produce better error
// messages at compile time when the are mismatches between the parameter
// type list and the argument type list.
template <typename ParamTuple, typename ArgTuple>
struct KernelInvocationChecker {
  // Whether the parameter tuple and argument tuple match in length.
  static constexpr bool kLengthMatches =
      std::tuple_size_v<ParamTuple> == std::tuple_size_v<ArgTuple>;

  // The (matching) length of the parameters and arguments type lists.
  static constexpr int kTupleLength =
      static_cast<int>(std::tuple_size_v<ArgTuple>);

  // Helper trait to say whether the parameter wants a DeviceMemory-reference
  // compatible type. This is for inexact type matches, so that it doesn't have
  // to be precisely a const DeviceMemory<T>&, but can also be a value that
  // represents the same.
  template <typename ParamType, typename ArgType>
  struct IsCompatibleDeviceMemoryRef : public std::false_type {};

  // See type trait definition above.
  template <typename U>
  struct IsCompatibleDeviceMemoryRef<const DeviceMemory<U> &, DeviceMemory<U>>
      : public std::true_type {};

  // See type trait definition above.
  template <typename U>
  struct IsCompatibleDeviceMemoryRef<const SharedDeviceMemory<U> &,
                                     SharedDeviceMemory<U>>
      : public std::true_type {};

  // Returns whether ParamT and ArgT are compatible for data parallel kernel
  // parameter packing without any assert functionality.
  template <typename ParamT, typename ArgT>
  static constexpr bool CompatibleNoAssert() {
    return std::is_same_v<typename std::remove_const_t<ParamT>, ArgT> ||
           IsCompatibleDeviceMemoryRef<ParamT, ArgT>::value;
  }

  // Checks whether ParamT and ArgT are compatible for data parallel kernel
  // parameter packing. kArgumentNumber is unused, it just for error display.
  //
  // NOTE: if you encounter an error here, you can see the mismatch by looking
  // at the end of the last error message, which will be of the form:
  //
  //    ...::Compatible<const stream_executor::DeviceMemory<OneThing> &,
  //                    stream_executor::DeviceMemory<AnotherThing>, true,
  //                    0>'
  //    requested here
  //
  // This means that the 0th argument you passed to the kernel invocation should
  // have been DeviceMemory<OneThing> but was observed to be
  // DeviceMemory<AnotherThing>.
  template <typename ParamT, typename ArgT, bool kShouldStaticAssert,
            int kArgumentNumber>
  static constexpr bool Compatible() {
    static_assert(
        kShouldStaticAssert ? CompatibleNoAssert<ParamT, ArgT>() : true,
        "parameter type (LHS) is not compatible with argument type (RHS)");
    return CompatibleNoAssert<ParamT, ArgT>();
  }

  // Checks the parameter/argument match at kArgumentNumber for an out of bounds
  // argument number.
  //
  // This is the base case: we've run out of argument to check, so we're all
  // good.
  template <int kArgumentNumber, bool kShouldStaticAssert>
  static constexpr bool CheckParam(
      typename std::enable_if_t<(kArgumentNumber < 0)> *dummy = nullptr) {
    return true;
  }

  // Checks the parameter/argument match at kArgumentNumber.
  // kShouldStaticAssert determines whether to assert out on a mismatch, or just
  // yield the constexpr boolean value.
  template <int kArgumentNumber, bool kShouldStaticAssert>
  static constexpr bool CheckParam(
      typename std::enable_if_t<kArgumentNumber >= 0> *dummy = nullptr) {
    typedef typename std::tuple_element_t<kArgumentNumber, ParamTuple> ParamT;
    typedef typename std::tuple_element_t<kArgumentNumber, ArgTuple> ArgT;
    return Compatible<ParamT, ArgT, kShouldStaticAssert, kArgumentNumber>() &&
           CheckParam<kArgumentNumber - 1, kShouldStaticAssert>();
  }

  // Checks the parameters/arguments for match, but doesn't static assert out.
  // This is useful for testing/inspecting whether a set of parameters match in
  // things like tests.
  static constexpr bool CheckAllNoStaticAssert() {
    return kLengthMatches && CheckParam<kTupleLength - 1, false>();
  }

  // Checks the parameters and static asserts out with a helpful error message
  // (and useful template parameters in the instantiation stack) if there is an
  // error.
  static constexpr bool CheckAllStaticAssert() {
    static_assert(kLengthMatches,
                  "argument length mismatched against typed kernel parameters");
    return kLengthMatches && CheckParam<kTupleLength - 1, true>();
  }
};

// This is a convenience type for checking whether a typed kernel matches
// against a type list.
template <typename KernelT, typename... Params>
struct KernelParamsOk {
  static constexpr bool kResult = false;
};

// See above.
template <typename... Params, typename... Args>
struct KernelParamsOk<TypedKernel<Params...>, Args...> {
  static constexpr bool kResult =
      KernelInvocationChecker<std::tuple<Params...>,
                              std::tuple<Args...>>::CheckAllNoStaticAssert();
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_KERNEL_H_

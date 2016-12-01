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
// perftools::gputools::executor_sample::VecReduceAddKernel.

#ifndef TENSORFLOW_STREAM_EXECUTOR_KERNEL_H_
#define TENSORFLOW_STREAM_EXECUTOR_KERNEL_H_

#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/kernel_cache_config.h"
#include "tensorflow/stream_executor/lib/stringpiece.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/lib/inlined_vector.h"

namespace perftools {
namespace gputools {

class DeviceMemoryBase;
template <typename ElemT>
class DeviceMemory;
class StreamExecutor;

namespace internal {
class KernelInterface;
}  // namespace internal

// KernelMetadata holds runtime-queryable attributes of a loaded kernel, such as
// registers allocated, shared memory used, etc.
// Not all platforms support reporting of all information, so each accessor
// returns false if the associated field is not populated in the underlying
// platform.
class KernelMetadata {
 public:
  KernelMetadata()
      : has_registers_per_thread_(false), has_shared_memory_bytes_(false) {}

  // Returns the number of registers used per thread executing this kernel.
  bool registers_per_thread(int *registers_per_thread) const;

  // Sets the number of registers used per thread executing this kernel.
  void set_registers_per_thread(int registers_per_thread);

  // Returns the amount of [static] shared memory used per block executing this
  // kernel. Note that dynamic shared memory allocations are not (and can not)
  // be reported here (since they're not specified until kernel launch time).
  bool shared_memory_bytes(int *shared_memory_bytes) const;

  // Sets the amount of [static] shared memory used per block executing this
  // kernel.
  void set_shared_memory_bytes(int shared_memory_bytes);

 private:
  // Holds the value returned by registers_per_thread above.
  bool has_registers_per_thread_;
  int registers_per_thread_;

  // Holds the value returned by shared_memory_bytes above.
  bool has_shared_memory_bytes_;
  int64 shared_memory_bytes_;
};

// A data-parallel kernel (code entity) for launching via the StreamExecutor,
// analogous to a void* device function pointer. See TypedKernel for the typed
// variant.
//
// Thread-compatible.
class KernelBase {
 public:
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

  void set_name(port::StringPiece name);
  const string &name() const { return name_; }
  const string &demangled_name() const { return demangled_name_; }

 private:
  // The StreamExecutor that loads this kernel object.
  StreamExecutor *parent_;

  // Implementation delegated to for platform-specific functionality.
  std::unique_ptr<internal::KernelInterface> implementation_;

  string name_;
  string demangled_name_;

  KernelMetadata metadata_;

  SE_DISALLOW_COPY_AND_ASSIGN(KernelBase);
};

// Whether T is a DeviceMemory-family pointer.
template <typename T>
struct IsDeviceMemoryPointer {
  static constexpr bool value = false;
};

template <typename U>
struct IsDeviceMemoryPointer<DeviceMemory<U> *> {
  static constexpr bool value = true;
};

template <>
struct IsDeviceMemoryPointer<DeviceMemoryBase *> {
  static constexpr bool value = true;
};

// Whether T is a DeviceMemory-family value-like thing (which includes a
// reference). This trait is useful because we pack values in the same manner as
// references.
template <typename T>
struct IsDeviceMemoryValueLike {
  static constexpr bool value = false;
};

template <typename U>
struct IsDeviceMemoryValueLike<DeviceMemory<U> &> {
  static constexpr bool value = true;
};

// We need to treat SharedDeviceMemory types differently than other DeviceMemory
// types (since they maintain no allocations), hence these specializations.
template <typename U>
struct IsDeviceMemoryValueLike<SharedDeviceMemory<U> &> {
  static constexpr bool value = false;
};

template <>
struct IsDeviceMemoryValueLike<DeviceMemoryBase &> {
  static constexpr bool value = true;
};

template <typename U>
struct IsDeviceMemoryValueLike<DeviceMemory<U>> {
  static constexpr bool value = true;
};

template <typename U>
struct IsDeviceMemoryValueLike<SharedDeviceMemory<U>> {
  static constexpr bool value = false;
};

template <>
struct IsDeviceMemoryValueLike<DeviceMemoryBase> {
  static constexpr bool value = true;
};

template <typename U>
struct IsSharedDeviceMemory {
  static constexpr bool value = false;
};

template <typename U>
struct IsSharedDeviceMemory<SharedDeviceMemory<U> &> {
  static constexpr bool value = true;
};

template <typename U>
struct IsSharedDeviceMemory<SharedDeviceMemory<U>> {
  static constexpr bool value = true;
};

// KernelArg encapsulates the information necessary for a back-end executor to
// configure a kernel to launch using the given argument.
struct KernelArg {
  // Indicates the type of an argument: normal, to be passed to the kernel
  // in the standard manner, or shared memory, which has distinct
  // rules for specification per backend.
  enum Type {
    kNormal,
    kSharedMemory,
  } type;

  // The data to pass to the kernel - either a pointer to device memory, or the
  // argument value. compact_array is used to prevent smaller args (ex. u8, u64)
  // from requiring heap allocation.
  port::InlinedVector<uint8, 4> data;

  // The size of this argument in bytes.
  uint64 bytes;
};

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
  // Delegates to KernelBase::KernelBase(), see that constructor.
  explicit TypedKernel(StreamExecutor *parent) : KernelBase(parent) {}

  // Test-only constructor that can take a mock KernelInterface implementation.
  // Takes ownership of implementation, it should not be null.
  TypedKernel(StreamExecutor *parent, internal::KernelInterface *implementation)
      : KernelBase(parent, implementation) {}

 private:
  // Stream needs access to the specific parameter-packing functionality that
  // the TypedKernel provides for its corresponding type signature (and no other
  // type signatures).
  friend class Stream;

  // This is the main entry point into the magic. Packs the parameters (which
  // must type check against the class template) into the args and sizes
  // arrays.
  //
  // Const refs are taken as parameters on all of the handlers to avoid
  // implicit type promotion of integers.
  void PackParams(std::vector<KernelArg> *args, Params... params) const {
    PackOneParam(args, params...);
  }

  template <typename T, typename... RestOfParams>
  void PackOneParam(std::vector<KernelArg> *args, const T &arg,
                    const RestOfParams... rest) const {
    PackOneParam(args, arg);
    PackOneParam(args, rest...);
  }

  // Packs one (non-DeviceMemoryBase) parameter into the arg and sizes array.
  // The enable_if<> is for excluding DeviceMemoryBase args, which have a
  // separate implementation below.
  template <typename T>
  void PackOneParam(
      std::vector<KernelArg> *args, const T &arg,
      typename std::enable_if<!IsDeviceMemoryValueLike<T>::value &&
                              !IsDeviceMemoryPointer<T>::value &&
                              !IsSharedDeviceMemory<T>::value>::type * =
          nullptr) const {
    static_assert(!std::is_pointer<T>::value,
                  "cannot pass raw pointer to the device");
    static_assert(!std::is_convertible<T, DeviceMemoryBase>::value,
                  "cannot pass device memory as a normal value");
    const uint8 *arg_ptr = reinterpret_cast<const uint8 *>(&arg);
    args->emplace_back(KernelArg{
        KernelArg::kNormal,
        port::InlinedVector<uint8, 4>{arg_ptr, arg_ptr + sizeof(arg)}, sizeof(arg)});
  }

  // DeviceMemoryBase family reference override.
  template <typename T>
  void PackOneParam(
      std::vector<KernelArg> *args, const T &arg,
      typename std::enable_if<IsDeviceMemoryValueLike<T>::value>::type * =
          nullptr) const {
    args->emplace_back(parent()->DeviceMemoryToKernelArg(arg));
  }

  // DeviceMemoryBase family pointer override.
  template <typename T>
  void PackOneParam(
      std::vector<KernelArg> *args, T arg,
      typename std::enable_if<IsDeviceMemoryPointer<T>::value>::type * =
          nullptr) const {
    DeviceMemoryBase *ptr = static_cast<DeviceMemoryBase *>(arg);
    args->emplace_back(parent()->DeviceMemoryToKernelArg(*ptr));
  }

  // Dynamic shared device memory has a size, but no associated allocation on
  // the host; internally, the device will allocate storage.
  template <typename T>
  void PackOneParam(
      std::vector<KernelArg> *args, T arg,
      typename std::enable_if<IsSharedDeviceMemory<T>::value>::type * =
          nullptr) const {
    args->emplace_back(KernelArg{KernelArg::kSharedMemory,
                                 port::InlinedVector<uint8, 4>(), arg.size()});
  }

  // Base case for variadic template expansion - nothing to do!
  void PackOneParam(std::vector<KernelArg> *args) const {}

  SE_DISALLOW_COPY_AND_ASSIGN(TypedKernel);
};

// Template metaprogramming helper type that helps us produce better error
// messages at compile time when the are mismatches between the parameter
// type list and the argument type list.
template <typename ParamTuple, typename ArgTuple>
struct KernelInvocationChecker {
  // Whether the parameter tuple and argument tuple match in length.
  static constexpr bool kLengthMatches =
      std::tuple_size<ParamTuple>::value == std::tuple_size<ArgTuple>::value;

  // The (matching) length of the parameters and arguments type lists.
  static constexpr int kTupleLength =
      static_cast<int>(std::tuple_size<ArgTuple>::value);

  // Helper trait to say whether the parameter wants a DeviceMemory-reference
  // compatible type. This is for inexact type matches, so that it doesn't have
  // to be precisely a const DeviceMemory<T>&, but can also be a value that
  // represents the same.
  template <typename ParamType, typename ArgType>
  struct IsCompatibleDeviceMemoryRef {
    static constexpr bool value = false;
  };

  // See type trait definition above.
  template <typename U>
  struct IsCompatibleDeviceMemoryRef<const DeviceMemory<U> &, DeviceMemory<U>> {
    static constexpr bool value = true;
  };

  // See type trait definition above.
  template <typename U>
  struct IsCompatibleDeviceMemoryRef<const SharedDeviceMemory<U> &,
                                     SharedDeviceMemory<U>> {
    static constexpr bool value = true;
  };

  // Returns whether ParamT and ArgT are compatible for data parallel kernel
  // parameter packing without any assert functionality.
  template <typename ParamT, typename ArgT>
  static constexpr bool CompatibleNoAssert() {
    return std::is_same<typename std::remove_const<ParamT>::type,
                        ArgT>::value ||
           IsCompatibleDeviceMemoryRef<ParamT, ArgT>::value;
  }

  // Checks whether ParamT and ArgT are compatible for data parallel kernel
  // parameter packing. kArgumentNumber is unused, it just for error display.
  //
  // NOTE: if you encounter an error here, you can see the mismatch by looking
  // at the end of the last error message, which will be of the form:
  //
  //    ...::Compatible<const perftools::gputools::DeviceMemory<OneThing> &,
  //                    perftools::gputools::DeviceMemory<AnotherThing>, true,
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
      typename std::enable_if<(kArgumentNumber < 0)>::type *dummy = nullptr) {
    return true;
  }

  // Checks the parameter/argument match at kArgumentNumber.
  // kShouldStaticAssert determines whether to assert out on a mismatch, or just
  // yield the constexpr boolean value.
  template <int kArgumentNumber, bool kShouldStaticAssert>
  static constexpr bool CheckParam(
      typename std::enable_if<kArgumentNumber >= 0>::type *dummy = nullptr) {
    typedef typename std::tuple_element<kArgumentNumber, ParamTuple>::type
        ParamT;
    typedef typename std::tuple_element<kArgumentNumber, ArgTuple>::type ArgT;
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
  static constexpr bool kResult = KernelInvocationChecker<
      std::tuple<Params...>, std::tuple<Args...>>::CheckAllNoStaticAssert();
};

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_KERNEL_H_

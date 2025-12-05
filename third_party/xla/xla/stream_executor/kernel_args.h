/* Copyright 2015 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_KERNEL_ARGS_H_
#define XLA_STREAM_EXECUTOR_KERNEL_ARGS_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/base/macros.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/overload.h"
#include "absl/log/check.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_metadata.h"
#include "xla/stream_executor/tensor_map.h"

namespace stream_executor {

//===----------------------------------------------------------------------===//
// Kernel arguments
//===----------------------------------------------------------------------===//

// A virtual base class for passing kernel arguments to a stream executor APIs.
class KernelArgs {
 public:
  template <typename T>
  using IsKernelArgs = std::enable_if_t<std::is_base_of<KernelArgs, T>::value>;

  enum class Kind {
    // A list of type-erased DeviceAddressBase pointers to on-device memory.
    // This
    // type of kernel arguments used only when the kernel has to do its own
    // custom packing, e.g. wrap all device pointers into a custom
    // structure, but can't be implemented as a TypedKernel because it has to be
    // passed around as a generic Kernel.
    kDeviceAddressArray,

    // A list of kernel arguments packed into a storage that can be passed
    // directly to device kernel as void** kernel parameters.
    kPackedArray
  };

  virtual ~KernelArgs() = default;

  // Gets the number of arguments added so far, including shared memory
  // arguments.
  virtual size_t number_of_arguments() const = 0;

  // Gets the total number of shared memory bytes added so far.
  virtual uint64_t number_of_shared_bytes() const = 0;

  virtual Kind kind() const = 0;
};

//===----------------------------------------------------------------------===//
// Kernel arguments packed array
//===----------------------------------------------------------------------===//

// A virtual base class for passing kernel arguments packed into a storage so
// that we have stable addresses for all arguments. This is a low level API for
// passing arguments in a platform-specific way that relies on the knowledge of
// the ABI of the underlying platform.
//
// For example `cuLaunchKernel` accepts arguments as `void** kernelParams`, and
// packed array base guarantees that `argument_addresses` are compatible with
// the CUDA APIs.
//
// See: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
class KernelArgsPackedArrayBase : public KernelArgs {
 public:
  // Gets the list of argument addresses.
  virtual absl::Span<const void* const> argument_addresses() const = 0;

  static bool classof(const KernelArgs* args) {
    return args->kind() == Kind::kPackedArray;
  }

  Kind kind() const final { return Kind::kPackedArray; }
};

//===----------------------------------------------------------------------===//
// Kernel arguments LLVM-style RTTI library
//===----------------------------------------------------------------------===//

template <class T, KernelArgs::IsKernelArgs<T>* = nullptr>
T* Cast(KernelArgs* args) {
  CHECK(T::classof(args)) << "Invalid arguments casting to a destination type: "
                          << typeid(T).name();
  CHECK(args != nullptr) << "Casted arguments must be not null";
  return static_cast<const T*>(args);
}

template <class T, KernelArgs::IsKernelArgs<T>* = nullptr>
const T* Cast(const KernelArgs* args) {
  CHECK(T::classof(args)) << "Invalid arguments casting to a destination type: "
                          << typeid(T).name();
  CHECK(args != nullptr) << "Casted arguments must be not null";
  return static_cast<const T*>(args);
}

template <class T, KernelArgs::IsKernelArgs<T>* = nullptr>
const T* DynCast(const KernelArgs* args) {
  CHECK(args != nullptr) << "Casted arguments must be not null";
  return T::classof(args) ? static_cast<const T*>(args) : nullptr;
}

template <class T, KernelArgs::IsKernelArgs<T>* = nullptr>
const T* DynCastOrNull(const KernelArgs* args) {
  return args && T::classof(args) ? static_cast<const T*>(args) : nullptr;
}

//===----------------------------------------------------------------------===//
// Kernel arguments device address array
//===----------------------------------------------------------------------===//

class KernelArgsDeviceAddressArray : public KernelArgs {
 public:
  KernelArgsDeviceAddressArray(absl::Span<const DeviceAddressBase> args,
                               size_t shared_memory_bytes)
      : device_memory_args_(args.begin(), args.end()),
        shared_memory_bytes_(shared_memory_bytes) {}

  static bool classof(const KernelArgs* args) {
    return args->kind() == Kind::kDeviceAddressArray;
  }

  Kind kind() const final { return Kind::kDeviceAddressArray; }

  size_t number_of_arguments() const final {
    return device_memory_args_.size() + (shared_memory_bytes_ > 0);
  }

  uint64_t number_of_shared_bytes() const final { return shared_memory_bytes_; }

  absl::Span<const DeviceAddressBase> device_memory_args() const {
    return device_memory_args_;
  }

  const void* device_memory_ptr(size_t index) const {
    return device_memory_args_[index].opaque();
  }

  size_t device_memory_size(size_t index) const {
    return device_memory_args_[index].size();
  }

 private:
  absl::InlinedVector<DeviceAddressBase, 4> device_memory_args_;
  size_t shared_memory_bytes_ = 0;
};

// TODO(ezhulenev): Remove this alias once all users are migrated.
using KernelArgsDeviceMemoryArray ABSL_DEPRECATE_AND_INLINE() =
    KernelArgsDeviceAddressArray;

//===----------------------------------------------------------------------===//
// Kernel arguments packing for device memory and POD args
//===----------------------------------------------------------------------===//

// KernelArgsPackedArray is optimized for packing DeviceAddressBase pointers
// and POD arguments (i.e. scalars) when the number and type of arguments are
// not known at compile time.

namespace internal {

// An empty storage for packing just the device memory arguments, that are
// stored directly in the `KernelArgsPackedArray`.
struct EmptyArgs {
  static constexpr size_t kSize = 0;
};

// A storage for POD generic arguments that are smaller than `size` and require
// alignment smaller or equal to `alignment`.
template <size_t capacity, size_t size = 8,
          size_t alignment = alignof(std::max_align_t)>
class PodArgs {
 public:
  static constexpr size_t kSize = size;

 protected:
  template <typename T>
  const std::byte* add_pod_argument(const T& arg) {
    static_assert(std::is_trivially_copyable_v<T> &&
                      sizeof(T) <= size & alignof(T) <= alignment,
                  "Type is not compatible with POD arguments storage");

    assert(num_args_ < capacity && "pod args overflow");
    std::byte* arg_storage = args_storage_[num_args_++].storage;
    std::memcpy(arg_storage, &arg, sizeof(T));

    return arg_storage;
  }

 private:
  struct Arg {
    alignas(alignment) std::byte storage[size];
  };

  size_t num_args_ = 0;
  std::array<Arg, capacity> args_storage_;
};

template <typename ArgsStorage>
static constexpr bool is_pod_args_v = false;

template <size_t capacity, size_t size, size_t alignment>
static constexpr bool is_pod_args_v<PodArgs<capacity, size, alignment>> = true;

}  // namespace internal

// An array of arguments for a kernel call.
//
// The template parameter `num_args` is the maximum number of arguments which
// can be stored in the array.
template <size_t num_args, typename ArgsStorage = internal::PodArgs<num_args>>
class KernelArgsPackedArray : public KernelArgsPackedArrayBase, ArgsStorage {
 public:
  KernelArgsPackedArray() = default;

  // KernelArgsPackedArray is not copyable or movable because argument addresses
  // point to inline storage that can't be moved.
  KernelArgsPackedArray(const KernelArgsPackedArray&) = delete;
  KernelArgsPackedArray& operator=(const KernelArgsPackedArray&) = delete;

  // Adds an argument to the list.
  template <typename T>
  void add_argument(const T& arg) {
    if constexpr (internal::is_pod_args_v<ArgsStorage>) {
      argument_addresses_[number_of_argument_addresses_++] =
          ArgsStorage::add_pod_argument(arg);
    } else {
      // https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html
      static_assert(sizeof(T) == 0, "Arguments storage is not supported");
    }
  }

  // Adds a device memory argument to the list.
  void add_device_memory_argument(const DeviceAddressBase& arg) {
    const void** copy_ptr =
        &device_memory_opaque_pointers_[number_of_argument_addresses_];
    *copy_ptr = arg.opaque();
    argument_addresses_[number_of_argument_addresses_] = copy_ptr;
    ++number_of_argument_addresses_;
  }

  // Adds a shared memory argument to the list.
  //
  // The only significant information about a shared argument is its size, so
  // that is the only parameter in this function.
  void add_shared_bytes(size_t number_of_bytes) {
    shared_memory_bytes_ += number_of_bytes;
  }

  // Gets the number of arguments added so far, including shared memory
  // arguments.
  size_t number_of_arguments() const final {
    return number_of_argument_addresses_ + (shared_memory_bytes_ > 0);
  }

  // Gets the total number of shared memory bytes added so far.
  uint64_t number_of_shared_bytes() const final { return shared_memory_bytes_; }

  // Gets the list of argument addresses.
  absl::Span<const void* const> argument_addresses() const final {
    return absl::Span<const void* const>(argument_addresses_.data(),
                                         number_of_argument_addresses_);
  }

 private:
  // A place to store copies of opaque pointers from device memory arguments.
  std::array<const void*, num_args> device_memory_opaque_pointers_;

  // Addresses for non-shared-memory arguments.
  std::array<const void*, num_args> argument_addresses_;

  // Shared memory required by a kernel.
  size_t shared_memory_bytes_ = 0;

  // Number of significant entries in argument_addresses_.
  size_t number_of_argument_addresses_ = 0;
};

using KernelArgument = std::variant<DeviceAddressBase, TensorMap, int64_t>;

namespace internal {
template <int n>
std::unique_ptr<KernelArgsPackedArrayBase> PackKernelArgs(
    absl::Span<const DeviceAddressBase> args, uint32_t shared_mem_bytes) {
  auto packed = std::make_unique<KernelArgsPackedArray<n, EmptyArgs>>();
  for (const DeviceAddressBase& buf : args) {
    packed->add_device_memory_argument(buf);
  }
  if (shared_mem_bytes > 0) {
    packed->add_shared_bytes(shared_mem_bytes);
  }
  return packed;
}

template <int n, typename ArgsStorage>
std::unique_ptr<KernelArgsPackedArray<n, ArgsStorage>> PackKernelArgsImpl(
    absl::Span<const KernelArgument> args, uint32_t shared_mem_bytes) {
  auto packed = std::make_unique<KernelArgsPackedArray<n, ArgsStorage>>();
  for (const auto& arg : args) {
    std::visit(
        absl::Overload{
            [&](const DeviceAddressBase& device_memory) {
              packed->add_device_memory_argument(device_memory);
            },
            [&](int64_t int_arg) {
              if constexpr (ArgsStorage::kSize >= sizeof(int64_t)) {
                packed->add_argument(int_arg);
              }
            },
            [&](const TensorMap& tensor_map) {
              if constexpr (ArgsStorage::kSize >= sizeof(tensor_map.storage)) {
                packed->add_argument(tensor_map.storage);
              }
            },
        },
        arg);
  }
  if (shared_mem_bytes > 0) {
    packed->add_shared_bytes(shared_mem_bytes);
  }
  return packed;
}

template <int n>
std::unique_ptr<KernelArgsPackedArrayBase> PackKernelArgs(
    absl::Span<const KernelArgument> args, uint32_t shared_mem_bytes) {
  const int32_t pod_size = [](absl::Span<const KernelArgument> args) {
    bool has_int = false;
    for (const auto& arg : args) {
      if (std::holds_alternative<TensorMap>(arg)) {
        return 128;
      }
      if (std::holds_alternative<int64_t>(arg)) {
        has_int = true;
      }
    }
    return has_int ? 64 : 0;
  }(args);

  switch (pod_size) {
    case 128:
      return PackKernelArgsImpl<n, PodArgs<n, 128, 64>>(args, shared_mem_bytes);
    case 64:
      return PackKernelArgsImpl<n, PodArgs<n, 64, 64>>(args, shared_mem_bytes);
    case 0:
      return PackKernelArgsImpl<n, EmptyArgs>(args, shared_mem_bytes);
    default:
      ABSL_UNREACHABLE();
  }
}
}  // namespace internal

template <typename ArgType>
inline absl::StatusOr<std::unique_ptr<KernelArgsPackedArrayBase>>
PackKernelArgs(absl::Span<const ArgType> args, uint32_t shared_mem_bytes) {
  static constexpr int kKernelArgsLimit = 1024;

  if (args.size() > kKernelArgsLimit) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Can't pack device memory arguments array of size ", args.size(),
        " which is larger than the maximum supported size of ",
        kKernelArgsLimit));
  }

  // Specialize kernel arguments array for small sizes to allocate a smaller
  // chunk of memory and hopefully hit a small allocations cache.
  if (args.size() <= 4) {
    return internal::PackKernelArgs<4>(args, shared_mem_bytes);
  }
  if (args.size() <= 8) {
    return internal::PackKernelArgs<8>(args, shared_mem_bytes);
  }
  if (args.size() <= 16) {
    return internal::PackKernelArgs<16>(args, shared_mem_bytes);
  }
  if (args.size() <= 32) {
    return internal::PackKernelArgs<32>(args, shared_mem_bytes);
  }
  if (args.size() <= 64) {
    return internal::PackKernelArgs<64>(args, shared_mem_bytes);
  }
  if (args.size() <= 256) {
    return internal::PackKernelArgs<256>(args, shared_mem_bytes);
  }
  if (args.size() <= 512) {
    return internal::PackKernelArgs<512>(args, shared_mem_bytes);
  }

  return internal::PackKernelArgs<kKernelArgsLimit>(args, shared_mem_bytes);
}

template <typename ArgType>
inline absl::StatusOr<std::unique_ptr<KernelArgsPackedArrayBase>>
PackKernelArgs(absl::Span<const ArgType> args, const KernelMetadata& metadata) {
  return PackKernelArgs(args, metadata.shared_memory_bytes().value_or(0));
}

//===----------------------------------------------------------------------===//
// Kernel arguments packing for statically know argument types
//===----------------------------------------------------------------------===//

// KernelArgsPackedTuple is optimized for packing arguments when their types are
// known at compile time, and somewhat similar to `std::tuple` but with a few
// special rules for passing device memory arguments.

namespace internal {

// PackedArgType template specialization defines what storage type we'll be
// using for each kernel argument type:
//
//   (1) We always strip references and store a copy of an argument.
//   (2) We do not support pointer arguments, as we should not be passing a
//       pointers to host memory to device kernels.
//   (3) DeviceAddress passed as an opaque `void*` pointer.
//   (4) We have a special case for passing pointers to DeviceAddress where we
//       also pass it as an opaque device pointer.
template <typename T>
struct PackedArgType {
  static_assert(!std::is_pointer_v<T>, "cannot pass raw pointer to the device");
  using Type = T;
};

template <>
struct PackedArgType<DeviceAddressBase> {
  using Type = const void*;
};

template <typename T>
struct PackedArgType<DeviceAddress<T>> {
  using Type = typename PackedArgType<DeviceAddressBase>::Type;
};

template <>
struct PackedArgType<DeviceAddressBase*> {
  using Type = typename PackedArgType<DeviceAddressBase>::Type;
};

template <>
struct PackedArgType<const DeviceAddressBase*> {
  using Type = typename PackedArgType<DeviceAddressBase>::Type;
};

template <typename T>
struct PackedArgType<DeviceAddress<T>*> {
  using Type = typename PackedArgType<DeviceAddressBase>::Type;
};

template <typename T>
struct PackedArgType<const DeviceAddress<T>*> {
  using Type = typename PackedArgType<DeviceAddressBase>::Type;
};

// Overload set for packing kernel arguments. This overload set matches
// supported kernel arguments types defined by `PackedArgType`.
template <typename T, std::enable_if_t<!std::is_pointer_v<T>>* = nullptr>
T PackArg(const T& arg) {
  return arg;
}

inline const void* PackArg(const DeviceAddressBase& arg) {
  return arg.opaque();
}
inline const void* PackArg(const DeviceAddressBase* arg) {
  return PackArg(*arg);
}

template <typename T>
const void* PackArg(const DeviceAddress<T>& arg) {
  return arg.opaque();
}

template <typename T>
const void* PackArg(const DeviceAddress<T>* arg) {
  return PackArg(*arg);
}

}  // namespace internal

template <typename... Args>
class KernelArgsPackedTuple : public KernelArgsPackedArrayBase {
 public:
  static constexpr size_t kSize = sizeof...(Args);

  using Storage = std::tuple<
      typename internal::PackedArgType<absl::remove_cvref_t<Args>>::Type...>;

  explicit KernelArgsPackedTuple(Args... args, size_t shared_memory_bytes)
      : storage_(internal::PackArg(std::forward<Args>(args))...),
        shared_memory_bytes_(shared_memory_bytes) {
    InitializeArgumentAddresses(std::make_index_sequence<kSize>{});
  }

  // KernelArgsPackedTuple is not copyable or movable because argument addresses
  // point to inline storage that can't be moved.
  KernelArgsPackedTuple(const KernelArgsPackedTuple&) = delete;
  KernelArgsPackedTuple& operator=(const KernelArgsPackedTuple&) = delete;

  size_t number_of_arguments() const final {
    return kSize + (shared_memory_bytes_ > 0);
  }

  uint64_t number_of_shared_bytes() const final { return shared_memory_bytes_; }

  absl::Span<const void* const> argument_addresses() const final {
    return absl::Span<const void* const>(argument_addresses_.data(), kSize);
  }

  // Compile time check that KernelArgsPackedTuple is compatible with
  // `OtherArgs`: after stripping const and reference all types match.
  template <typename... OtherArgs>
  static void CheckCompatibleStaticAssert() {
    static constexpr size_t kOtherSize = sizeof...(OtherArgs);
    static_assert(kSize == kOtherSize, "length of arguments packs must match");

    using StrippedArgs = std::tuple<absl::remove_cvref_t<Args>...>;
    using StrippedOtherArgs = std::tuple<absl::remove_cvref_t<OtherArgs>...>;
    static_assert(std::is_same_v<StrippedArgs, StrippedOtherArgs>,
                  "arguments types do not match");
  }

 private:
  template <size_t... Is>
  void InitializeArgumentAddresses(std::index_sequence<Is...>) {
    ((argument_addresses_[Is] = &std::get<Is>(storage_)), ...);
  }

  // Storage for packed kernel arguments.
  Storage storage_;

  // Shared memory required by a kernel.
  size_t shared_memory_bytes_ = 0;

  // Pointers into `storage_`.
  std::array<const void*, kSize> argument_addresses_;
};

// Packs the given arguments into a KernelArgsPackedTuple.
template <typename... Args>
std::unique_ptr<KernelArgsPackedArrayBase> PackKernelArgs(int64_t shmem_bytes,
                                                          Args... args) {
  using PackedArgs = KernelArgsPackedTuple<Args...>;
  return std::make_unique<PackedArgs>(std::forward<Args>(args)..., shmem_bytes);
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_KERNEL_ARGS_H_

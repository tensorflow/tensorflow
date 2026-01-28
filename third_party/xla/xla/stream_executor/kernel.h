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

// Suite of datatypes to represent data-parallel kernel objects (code entities).
//
// Kernel is the untyped variant, whereas TypedKernel takes a type signature
// to do some template-based helper generation and give compile-time type
// checking for kernel launch parameters.
//
// Users encouraged to use typed kernels when they know the type signature at
// compile time. TypedKernels express their argument types via template
// parameters like so:
//
//  TypedKernel<DeviceAddress<int>*, int>
//
// Which expresses a data parallel kernel signature for:
//
//  void(int*, int);
//
// And for a const memory region:
//
//  TypedKernel<const DeviceAddress<int>&, int>
//
// Corresponds to a data parallel kernel signature for:
//
//  void(const int*, int)
//
// Note that kernels always have a void return type, so results typically must
// be memcpy'ied from device memory to the host.
//
// Also note that a scalar integer residing in device memory and an array of
// integers residing in device memory have the same signature: DeviceAddress<T>.
// However, in the future, checks may be added for additional safety that arrays
// of minimum sizes are passed when those minimum sizes are contractually
// expected by the kernel.
//
// For user-defined types whose definitions are appropriately shared between the
// host code doing the launching and the kernel code being launched, the user
// defined types are similarly permitted to be expressed as residing in device
// memory:
//
//  TypedKernel<DeviceAddress<MyUserDefinedStructure>>
//
// And, when the alignment and padding are agreed upon, POD types will also be
// able to be passed by value; for example, it is a common idiom to specify a
// bunch of options simultaneously with a structure:
//
//  TypedKernel<MyOptionsStructurePassedByValue, DeviceAddress<float>>
//
// Which corresponds to a data parallel kernel signature like:
//
//  void(MyOptionsStructurePassedByValue value, float *result);
//

#ifndef XLA_STREAM_EXECUTOR_KERNEL_H_
#define XLA_STREAM_EXECUTOR_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {

//===----------------------------------------------------------------------===//
// Kernel
//===----------------------------------------------------------------------===//

// A data-parallel kernel (code entity) for launching via the StreamExecutor,
// analogous to a void* device function pointer. See TypedKernel for the typed
// variant.
//
// Thread-compatible.
class Kernel {
 public:
  // A function for converting kernel arguments into a packed kernels arguments
  // that can be directly passed to a device kernel. This indirection allows
  // registering custom CUDA C++ kernels with non-trivial C++ API with a
  // StreamExecutor as a generic `Kernel`.
  using KernelArgsPacking =
      std::function<absl::StatusOr<std::unique_ptr<KernelArgsPackedArrayBase>>(
          const Kernel& kernel, const KernelArgs& args)>;

  Kernel() = default;
  virtual ~Kernel() = default;

  Kernel(const Kernel&) = delete;
  void operator=(const Kernel&) = delete;

  // Returns the number of parameters that this kernel accepts. (Arity refers to
  // nullary, unary, ...).
  virtual unsigned Arity() const = 0;

  // Returns the maximum number of blocks (per multiprocessor) occupied by the
  // kernel given the number of threads per block and shared memory size.
  virtual absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(
      ThreadDim threads, size_t dynamic_shared_memory_bytes) const = 0;

  const KernelMetadata& metadata() const { return metadata_; }
  void set_metadata(KernelMetadata metadata) { metadata_ = metadata; }

  const KernelArgsPacking& args_packing() const { return args_packing_; }
  void set_args_packing(KernelArgsPacking args_packing) {
    args_packing_ = std::move(args_packing);
  }

  absl::string_view name() const { return name_; }
  void set_name(std::string name) { name_ = std::move(name); }

  // Launches a data parallel kernel with the given thread/block
  // dimensionality and already-packed args/sizes to pass to the underlying
  // platform driver.
  absl::Status Launch(const ThreadDim& thread_dims, const BlockDim& block_dims,
                      Stream* stream, const KernelArgs& args);

  // Helper method to launch a kernel with optional cluster dimensions.
  virtual absl::Status Launch(const ThreadDim& thread_dims,
                              const BlockDim& block_dims,
                              const std::optional<ClusterDim>& cluster_dims,
                              Stream* stream, const KernelArgs& args) = 0;

 private:
  std::string name_;

  KernelMetadata metadata_;
  KernelArgsPacking args_packing_;
};

inline absl::Status Kernel::Launch(const ThreadDim& thread_dims,
                                   const BlockDim& block_dims, Stream* stream,
                                   const KernelArgs& args) {
  return Launch(thread_dims, block_dims, std::nullopt, stream, args);
}

//===----------------------------------------------------------------------===//
// Typed kernel
//===----------------------------------------------------------------------===//
template <typename... Params>
class TypedKernelFactory;

// Typed kernel is a typed smart-pointer-like wrapper around untyped Kernel.
template <typename... Params>
class TypedKernel {
 public:
  static constexpr size_t kNumberOfParameters = sizeof...(Params);

  TypedKernel() = default;

  Kernel& operator*() { return *kernel_; }
  const Kernel& operator*() const { return *kernel_; }

  Kernel* operator->() { return kernel_.get(); }
  const Kernel* operator->() const { return kernel_.get(); }

  operator bool() const { return static_cast<bool>(kernel_); }  // NOLINT

  // Type of factory used to create a TypedKernel.
  using FactoryType = TypedKernelFactory<Params...>;

  // Launches a kernel with the given (variadic) parameters for the invocation
  // onto the specified stream. These arguments can be things
  // like DeviceAddress or primitive types such as int. What arguments you may
  // pass to a given kernel are noted as the template parameters to the
  // TypedKernel type that the compiler generates.
  //
  //  Template parameters:
  //   Params...   The type list of formal parameters that the typed kernel
  //               expects, which is matched against Args...
  //   Args...     The deduced type list for passed actual arguments
  //
  // Implementation: A compile-time compatibility check is performed that has
  // some leniency versus an exact parameter pack match -- for example,
  // `const DeviceAddress<T>` is considered "pack compatible" with a
  // `const DeviceAddress<T>&` formal parameter; in part, because we don't have
  // perfect forwarding support without rvalue references. It also attempts to
  // spit out helpful static_assert error traces with information as to the
  // argument number and types that were mismatched.
  template <typename... Args>
  inline absl::Status Launch(ThreadDim thread_dims, BlockDim block_dims,
                             Stream* stream, Args... args) {
    auto kernel_args = PackKernelArgs(*this, args...);
    return kernel_->Launch(thread_dims, block_dims, stream, *kernel_args);
  }

  template <typename... Args>
  inline absl::Status Launch(ThreadDim thread_dims, BlockDim block_dims,
                             int32_t shmem_bytes, Stream* stream,
                             Args... args) {
    auto kernel_args = PackKernelArgs(shmem_bytes, args...);
    return kernel_->Launch(thread_dims, block_dims, stream, *kernel_args);
  }

 private:
  friend class TypedKernelFactory<Params...>;
  explicit TypedKernel(std::unique_ptr<Kernel> kernel)
      : kernel_(std::move(kernel)) {}

  std::unique_ptr<Kernel> kernel_;
};

// Packs the given arguments into a KernelArgsPackedTuple with compile-time type
// checks that arguments are compatible with TypedKernel signature.
template <typename... Params, typename... Args>
std::unique_ptr<KernelArgsPackedArrayBase> PackKernelArgs(
    const TypedKernel<Params...>& kernel, Args... args) {
  using PackedParams = KernelArgsPackedTuple<Params...>;
  using PackedArgs = KernelArgsPackedTuple<Args...>;

  PackedParams::template CheckCompatibleStaticAssert<Args...>();

  int64_t shmem_bytes = kernel->metadata().shared_memory_bytes().value_or(0);
  return std::make_unique<PackedArgs>(std::forward<Args>(args)..., shmem_bytes);
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_KERNEL_H_

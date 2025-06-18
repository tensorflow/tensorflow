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

// Kernel-loader specs are structures that describe how to load a data-parallel
// kernel on a given platform for subsequent launching.
//
// A kernel with the same exact functionality and type signature may be
// implemented on several different platforms. Typical usage is to register a
// kernel with the GpuKernelRegistry that describes how to load a kernel for
// for each supported platform.
//
//  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
//      RepeatBufferKernelCuda, stream_executor::gpu::RepeatBufferKernel,
//      se::cuda::kCudaPlatformId, ([](size_t arity) {
//        return se::KernelLoaderSpec::CreateInProcessSymbolSpec(
//            absl::bit_cast<void*>(&se::gpu::RepeatBufferKernelImpl),

//            "repeat_buffer_kernel", arity);
//      }));
//
// This lazily instantiates an object that describes how to load CUDA in process
// kernel.

#ifndef XLA_STREAM_EXECUTOR_KERNEL_SPEC_H_
#define XLA_STREAM_EXECUTOR_KERNEL_SPEC_H_

#include <stddef.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/kernel.h"

namespace stream_executor {

// Loads kernel from in process symbol pointer (e.g. pointer to C++ device
// function).
struct InProcessSymbol {
  void *symbol;
};

// Kernel loader specification for PTX text that resides in memory.
struct CudaPtxInMemory {
  absl::string_view ptx;
};

// Kernel loader specification for a CUBIN blob that resides in memory.
struct CudaCubinInMemory {
  absl::Span<const uint8_t> cubin_bytes;
};

// Describes how to load a kernel on any subset of a number of target platforms.
class KernelLoaderSpec {
 public:
  // A function for converting kernel arguments into a packed kernels arguments
  // that can be directly passed to a device kernel. This indirection allows
  // registering custom CUDA C++ kernels with non-trivial C++ API with a
  // StreamExecutor as a generic `Kernel`.
  using KernelArgsPacking =
      std::function<absl::StatusOr<std::unique_ptr<KernelArgsPackedArrayBase>>(
          const Kernel &kernel, const KernelArgs &args)>;

  // Returns the number of arguments that this kernel accepts.
  size_t arity() const { return arity_; }

  // Convenience getters for testing whether these platform variants have
  // kernel loader specifications available.
  bool has_in_process_symbol() const {
    return std::holds_alternative<InProcessSymbol>(payload_);
  }
  bool has_cuda_cubin_in_memory() const {
    return std::holds_alternative<CudaCubinInMemory>(payload_);
  }
  bool has_cuda_ptx_in_memory() const {
    return std::holds_alternative<CudaPtxInMemory>(payload_);
  }

  // Accessors for platform variant kernel load specifications.
  std::optional<InProcessSymbol> in_process_symbol() const {
    if (!has_in_process_symbol()) {
      return std::nullopt;
    }
    return std::get<InProcessSymbol>(payload_);
  }

  std::optional<CudaCubinInMemory> cuda_cubin_in_memory() const {
    if (!has_cuda_cubin_in_memory()) {
      return std::nullopt;
    }
    return std::get<CudaCubinInMemory>(payload_);
  }

  std::optional<CudaPtxInMemory> cuda_ptx_in_memory() const {
    if (!has_cuda_ptx_in_memory()) {
      return std::nullopt;
    }
    return std::get<CudaPtxInMemory>(payload_);
  }

  // Use these factory functions to create a spec of any supported type.
  //
  // Note that the kernel_name parameter must be consistent with the kernel in
  // the PTX being loaded. Also be aware that in CUDA C++ the kernel name may be
  // mangled by the compiler if it is not declared in an extern "C" scope.
  static KernelLoaderSpec CreateInProcessSymbolSpec(
      void *symbol, std::string kernel_name, size_t arity,
      KernelArgsPacking kernel_args_packing = nullptr);
  static KernelLoaderSpec CreateCudaCubinInMemorySpec(
      absl::Span<const uint8_t> cubin_bytes, std::string kernel_name,
      size_t arity, KernelArgsPacking kernel_args_packing = nullptr);
  static KernelLoaderSpec CreateCudaPtxInMemorySpec(
      absl::string_view ptx, std::string kernel_name, size_t arity,
      KernelArgsPacking kernel_args_packing = nullptr);

  void set_kernel_args_packing(KernelArgsPacking kernel_args_packing) {
    kernel_args_packing_ = std::move(kernel_args_packing);
  }

  const KernelArgsPacking &kernel_args_packing() const {
    return kernel_args_packing_;
  }

  const std::string &kernel_name() const { return kernel_name_; }

 private:
  using Payload =
      std::variant<InProcessSymbol, CudaCubinInMemory, CudaPtxInMemory>;

  explicit KernelLoaderSpec(Payload payload, std::string kernel_name,
                            size_t arity,
                            KernelArgsPacking kernel_args_packing = nullptr)
      : payload_(std::move(payload)),
        kernel_name_(std::move(kernel_name)),
        arity_(arity),
        kernel_args_packing_(std::move(kernel_args_packing)) {}

  Payload payload_;
  std::string kernel_name_;

  // Number of parameters that the kernel takes. (This is nicer to have in a
  // constexpr than having to determine it from the types via template
  // metaprogramming).
  size_t arity_;

  // Custom kernel arguments packing.
  KernelArgsPacking kernel_args_packing_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_KERNEL_SPEC_H_

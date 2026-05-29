/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_KERNEL_COMPILER_H_
#define XLA_BACKENDS_GPU_CODEGEN_KERNEL_COMPILER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

using BorrowedMlirContext =
    ObjectPool<std::unique_ptr<mlir::MLIRContext>>::BorrowedObject;

// Abstract base class for asynchronous kernel compilation.
//
// Defines an interface for compiling KernelSource. Implementations of this
// interface handle the specifics of transforming an LlvmKernelSource into an
// executable Thunk.
//
// The compilation process is asynchronous, returning a xla::Future that will
// eventually hold the compiled std::unique_ptr<Thunk> object. This design
// allows for non-blocking compilation and potential parallelism in compilation
// workflows.
class KernelCompiler {
 public:
  using ModuleHook = absl::AnyInvocable<void(const llvm::Module&)>;

  KernelCompiler() = default;
  KernelCompiler(KernelCompiler&& other) = default;
  KernelCompiler& operator=(KernelCompiler&& other) = default;

  KernelCompiler(const KernelCompiler&) = delete;
  KernelCompiler& operator=(const KernelCompiler&) = delete;

  virtual ~KernelCompiler() = default;

  // Compiles the given kernel source asynchronously.
  virtual xla::Future<std::unique_ptr<Thunk>> Compile(
      Thunk::ThunkInfo thunk_info, LlvmKernelSource kernel_source,
      const std::string& sanitized_kernel_name,
      const emitters::KernelArguments& kernel_arguments,
      const LaunchDimensions& launch_dimensions) = 0;

  virtual xla::Future<LlvmKernelSource> CompileMlirToLlvm(
      const se::DeviceDescription& device, const HloModule& hlo_module,
      const std::string& entry_function_name, int unroll_factor,
      MlirKernelSource source, BorrowedMlirContext borrowed_context) = 0;

  virtual xla::Future<std::vector<uint8_t>> CompileToPtx(
      LlvmKernelSource kernel_source) = 0;

  // Sets a callback to be called prior to llvm::Module compilation.
  void SetPreOptimizationHook(ModuleHook hook) {
    pre_optimization_hook_ = std::move(hook);
  }
  ModuleHook& pre_optimization_hook() {
    return pre_optimization_hook_;
  }

 private:
  ModuleHook pre_optimization_hook_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_KERNEL_COMPILER_H_

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

#include "xla/backends/gpu/codegen/cubin_custom_kernel_compiler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Module.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

xla::Future<std::unique_ptr<Thunk>> CubinCustomKernelCompiler::Compile(
    Thunk::ThunkInfo thunk_info, LlvmKernelSource kernel_source,
    const std::string& sanitized_kernel_name,
    const emitters::KernelArguments& kernel_arguments,
    const LaunchDimensions& launch_dimensions) {
  if (!thread_pool_) {
    return CompileImpl(std::move(thunk_info), std::move(kernel_source),
                       sanitized_kernel_name, kernel_arguments,
                       launch_dimensions);
  }
  return tsl::MakeFutureOn(
      *thread_pool_->AsExecutor(),
      [this, thunk_info = std::move(thunk_info),
       kernel_source = std::move(kernel_source), sanitized_kernel_name,
       kernel_arguments, launch_dimensions]() mutable {
        return CompileImpl(std::move(thunk_info), std::move(kernel_source),
                           sanitized_kernel_name, kernel_arguments,
                           launch_dimensions);
      });
}

xla::Future<LlvmKernelSource> CubinCustomKernelCompiler::CompileMlirToLlvm(
    const se::DeviceDescription& device, const HloModule& hlo_module,
    const std::string& entry_function_name, int unroll_factor,
    MlirKernelSource source, BorrowedMlirContext borrowed_context) {
  if (!thread_pool_) {
    return gpu::CompileMlirToLlvm(device, hlo_module, entry_function_name,
                                  unroll_factor, **borrowed_context,
                                  std::move(source));
  }
  return xla::MakeFutureOn(
      *thread_pool_->AsExecutor(),
      [source = std::move(source), device, &hlo_module, entry_function_name,
       unroll_factor,
       borrowed_context = std::move(borrowed_context)]() mutable {
        return gpu::CompileMlirToLlvm(device, hlo_module, entry_function_name,
                                      unroll_factor, **borrowed_context,
                                      std::move(source));
      });
}

xla::Future<std::vector<uint8_t>> CubinCustomKernelCompiler::CompileToPtx(
    LlvmKernelSource kernel_source) {
  if (!thread_pool_) {
    return CompileToPtxImpl(std::move(kernel_source));
  }
  return xla::MakeFutureOn(
      *thread_pool_->AsExecutor(),
      [this, kernel_source = std::move(kernel_source)]() mutable {
        return CompileToPtxImpl(std::move(kernel_source));
      });
}

absl::StatusOr<std::vector<uint8_t>>
CubinCustomKernelCompiler::CompileToPtxImpl(LlvmKernelSource kernel_source) {
  llvm::orc::ThreadSafeModule thread_safe_module =
      std::move(kernel_source).thread_safe_module();
  llvm::Module* llvm_module = thread_safe_module.getModuleUnlocked();

  if (pre_optimization_hook()) {
    pre_optimization_hook()(*llvm_module);
  }

  ASSIGN_OR_RETURN(std::vector<uint8_t> cubin,
                   compiler_(*llvm_module, device_info_, debug_options_));
  return cubin;
}

absl::StatusOr<std::unique_ptr<Thunk>> CubinCustomKernelCompiler::CompileImpl(
    Thunk::ThunkInfo thunk_info, LlvmKernelSource kernel_source,
    const std::string& sanitized_kernel_name,
    const emitters::KernelArguments& kernel_arguments,
    const LaunchDimensions& launch_dimensions) {
  ASSIGN_OR_RETURN(std::vector<uint8_t> cubin,
                   CompileToPtxImpl(std::move(kernel_source)));

  ASSIGN_OR_RETURN(
      CustomKernel custom_kernel,
      kernel::CreateOwnedCubinCustomKernel(
          sanitized_kernel_name, std::move(cubin),
          kernel_arguments.args().size(), launch_dimensions.block_counts(),
          launch_dimensions.thread_counts_per_block(), 0));

  return std::make_unique<CustomKernelThunk>(
      thunk_info, std::move(custom_kernel), kernel_arguments);
}

}  // namespace xla::gpu

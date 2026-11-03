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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/codegen/triton/triton_kernel_source.h"
#include "xla/backends/gpu/codegen/triton/triton_wrapper_result.h"
#include "xla/backends/gpu/codegen/triton/xtile_compiler.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/util.h"

namespace xla::gpu {
using ::xla::xtile::BlockLevelParameters;

namespace {
absl::StatusOr<TritonWrapperResult> CompileTritonToLlvmImpl(
    absl::string_view kernel_name, const HloModule& hlo_module,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    const llvm::Triple& target_triple, const std::string& data_layout,
    TritonKernelSource triton_source, BorrowedMlirContext borrowed_context,
    bool is_xla_fusion) {
  XLA_SCOPED_LOGGING_TIMER_IF(
      absl::StrCat("CubinCustomKernelCompiler::CompileTritonToLlvmImpl for ",
                   hlo_module.name()),
      hlo_module.config().debug_options().xla_enable_scoped_logging_timers());
  return gpu::CompileTritonToLLVM(kernel_name, hlo_module, device_info,
                                  block_level_parameters, target_triple,
                                  data_layout, std::move(triton_source),
                                  **borrowed_context, is_xla_fusion);
}
}  // namespace

xla::Future<std::unique_ptr<Thunk>> CubinCustomKernelCompiler::Compile(
    Thunk::ThunkInfo thunk_info, MlirKernelDefinition kernel_def,
    BorrowedMlirContext borrowed_context, const HloModule& hlo_module,
    int unroll_factor) {
  if (!thread_pool_) {
    return CompileImpl(std::move(thunk_info), std::move(kernel_def),
                       std::move(borrowed_context), hlo_module, unroll_factor);
  }
  return xla::MakeFutureOn(*thread_pool_->AsExecutor(),
                           [this, thunk_info = std::move(thunk_info),
                            borrowed_context = std::move(borrowed_context),
                            kernel_def = std::move(kernel_def), &hlo_module,
                            unroll_factor]() mutable {
                             return CompileImpl(std::move(thunk_info),
                                                std::move(kernel_def),
                                                std::move(borrowed_context),
                                                hlo_module, unroll_factor);
                           });
}

xla::Future<std::vector<uint8_t>>
CubinCustomKernelCompiler::CompileToTargetBinary(
    MlirKernelDefinition kernel_def, BorrowedMlirContext borrowed_context,
    const HloModule& hlo_module, int unroll_factor) {
  if (!thread_pool_) {
    const GpuKernelSpec spec = kernel_def.spec();
    return CompileMlirToCubinImpl(spec, std::move(kernel_def).TakeSource(),
                                  std::move(borrowed_context), hlo_module,
                                  unroll_factor);
  }
  return xla::MakeFutureOn(
      *thread_pool_->AsExecutor(),
      [this, borrowed_context = std::move(borrowed_context),
       kernel_def = std::move(kernel_def), &hlo_module,
       unroll_factor]() mutable {
        const GpuKernelSpec spec = kernel_def.spec();
        return CompileMlirToCubinImpl(spec, std::move(kernel_def).TakeSource(),
                                      std::move(borrowed_context), hlo_module,
                                      unroll_factor);
      });
}

xla::Future<std::vector<uint8_t>>
CubinCustomKernelCompiler::CompileToTargetBinary(
    LlvmKernelSource kernel_source, std::optional<GpuKernelSpec> kernel_spec) {
  if (!thread_pool_) {
    return CompileToCubinImpl(std::move(kernel_source));
  }
  return xla::MakeFutureOn(
      *thread_pool_->AsExecutor(),
      [this, kernel_source = std::move(kernel_source)]() mutable {
        return CompileToCubinImpl(std::move(kernel_source));
      });
}

absl::StatusOr<std::vector<uint8_t>>
CubinCustomKernelCompiler::CompileToCubinImpl(LlvmKernelSource kernel_source) {
  llvm::orc::ThreadSafeModule thread_safe_module =
      std::move(kernel_source).thread_safe_module();
  llvm::Module* llvm_module = thread_safe_module.getModuleUnlocked();

  if (pre_optimization_hook()) {
    pre_optimization_hook()(*llvm_module);
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<uint8_t> cubin,
                   compiler_(*llvm_module, device_info_, debug_options_));
  return cubin;
}

absl::StatusOr<std::vector<uint8_t>>
CubinCustomKernelCompiler::CompileMlirToCubinImpl(
    const GpuKernelSpec& spec, MlirKernelSource mlir_source,
    BorrowedMlirContext borrowed_context, const HloModule& hlo_module,
    int unroll_factor) {
  mlir::MLIRContext* mlir_context =
      borrowed_context ? borrowed_context->get() : mlir_source.mlir_context();
  TF_RET_CHECK(mlir_context != nullptr);
  XLA_SCOPED_LOGGING_TIMER_IF(
      absl::StrCat("CubinCustomKernelCompiler::CompileMlirToCubinImpl for ",
                   hlo_module.name()),
      hlo_module.config().debug_options().xla_enable_scoped_logging_timers());
  ABSL_ASSIGN_OR_RETURN(LlvmKernelSource llvm_source,
                   gpu::CompileMlirToLlvm(
                       device_info_, hlo_module, std::string(spec.name()),
                       unroll_factor, *mlir_context, std::move(mlir_source)));
  // Return `borrowed_context` to the pool before LLVM/PTX compilation so
  // concurrent emitters can reuse it without allocating fresh contexts.
  borrowed_context.reset();
  ABSL_RETURN_IF_ERROR(AnnotateMlirKernelLlvmModule(
      device_info_, spec.launch_dimensions(), spec.name(), data_layout_,
      target_triple_, llvm_source.module()));
  return CompileToCubinImpl(std::move(llvm_source));
}

absl::StatusOr<std::unique_ptr<Thunk>> CubinCustomKernelCompiler::CompileImpl(
    Thunk::ThunkInfo thunk_info, MlirKernelDefinition kernel_def,
    BorrowedMlirContext borrowed_context, const HloModule& hlo_module,
    int unroll_factor) {
  const GpuKernelSpec& spec = kernel_def.spec();
  ABSL_ASSIGN_OR_RETURN(std::vector<uint8_t> cubin,
                   CompileMlirToCubinImpl(
                       spec, std::move(kernel_def).TakeSource(),
                       std::move(borrowed_context), hlo_module, unroll_factor));

  ABSL_ASSIGN_OR_RETURN(
      CustomKernel custom_kernel,
      kernel::CreateOwnedCubinCustomKernel(spec, std::move(cubin)));

  return std::make_unique<CustomKernelThunk>(
      std::move(thunk_info), std::move(custom_kernel), spec.kernel_arguments(),
      gpu_topology_.num_devices_per_process(), spec.use_pdl(),
      std::vector<int64_t>{}, spec.tma_metadata());
}

xla::Future<TritonWrapperResult> CubinCustomKernelCompiler::CompileTritonToLlvm(
    const absl::string_view kernel_name, const HloModule& hlo_module,
    const se::DeviceDescription& device_info,
    const BlockLevelParameters& block_level_parameters,
    const llvm::Triple& target_triple, const std::string& data_layout,
    TritonKernelSource triton_source, BorrowedMlirContext borrowed_context,
    bool is_xla_fusion) {
  if (!thread_pool_) {
    return CompileTritonToLlvmImpl(kernel_name, hlo_module, device_info,
                                   block_level_parameters, target_triple,
                                   data_layout, std::move(triton_source),
                                   std::move(borrowed_context), is_xla_fusion);
  }
  return xla::MakeFutureOn(
      *thread_pool_->AsExecutor(),
      [kernel_name = std::string(kernel_name), hlo_module = &hlo_module,
       device_info, block_level_parameters, target_triple, is_xla_fusion,
       data_layout, borrowed_context = std::move(borrowed_context),
       triton_source = std::move(triton_source)]() mutable {
        return CompileTritonToLlvmImpl(
            kernel_name, *hlo_module, device_info, block_level_parameters,
            target_triple, data_layout, std::move(triton_source),
            std::move(borrowed_context), is_xla_fusion);
      });
}

}  // namespace xla::gpu

/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/compile_module_to_llvm_ir.h"

#include <stdlib.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/execution_stream_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_unnested.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla::gpu {

namespace {

// Prints mlir diagnostic messages to VLOG level 2.
static mlir::LogicalResult DiagnosticHandler(mlir::Diagnostic& diag) {
  VLOG(2) << diag.str();
  return mlir::failure();
}

// Removes all globals from the given module that are both uninitialized and
// have no uses within that module.
void RemoveUnusedAndUninitializedGlobals(
    llvm::Module* llvm_module,
    const std::vector<GpuExecutable::ConstantInfo>& constants) {
  for (const auto& info : constants) {
    // Empty content means the constant is initialized in the LLVM IR, so we
    // must not remove it.
    if (!info.content.span().empty()) {
      llvm::GlobalVariable* global =
          llvm_module->getGlobalVariable(info.symbol_name);
      CHECK(global != nullptr);
      if (global->use_empty()) {
        global->eraseFromParent();
      }
    }
  }
}

}  // namespace

absl::Status LoadCache(IrEmitterContext& ir_emitter_context,
                       absl::string_view cache_file_path) {
  std::string resolved_path;
  if (!tsl::io::ResolveTestPrefixes(cache_file_path, resolved_path)) {
    return FailedPrecondition("File path can not be resolved: %s",
                              cache_file_path);
  }
  if (tsl::Env::Default()->FileExists(resolved_path).ok()) {
    std::string serialized;
    TF_RETURN_IF_ERROR(
        tsl::ReadFileToString(tsl::Env::Default(), resolved_path, &serialized));
    CompilationCacheProto proto;
    if (!proto.ParseFromString(serialized)) {
      return Internal("Failed to parse serialized CompilationCacheProto.");
    }
    // Register all cached kernel names with the name uniquer to avoid
    // naming conflicts.
    for (const auto& [name, _] : proto.entries()) {
      TF_RET_CHECK(ir_emitter_context.name_uniquer()->GetUniqueName(name) ==
                   name)
          << "Failed registering " << name << "in NameUniquer.";
    }
    TF_RETURN_IF_ERROR(ir_emitter_context.kernel_cache().Load(proto));
  } else {
    VLOG(1) << "Compilation cache file does not exist: " << resolved_path;
  }
  return absl::OkStatus();
}

absl::StatusOr<CompileModuleResults> CompileModuleToLlvmIr(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, se::Platform::Id platform_id,
    const se::DeviceDescription& gpu_device_info,
    const HloDataflowAnalysis::CanShareBuffer& can_share_buffer_function,
    const BufferValue::SizeFunction& buffer_size_bytes_function,
    bool split_constants_module) {
  CompileModuleResults results;
  results.llvm_module =
      std::make_unique<llvm::Module>(hlo_module->name(), *llvm_context);
  results.llvm_module->setTargetTriple(target_triple);
  results.llvm_module->setDataLayout(data_layout);

  absl::string_view cache_file_path =
      hlo_module->config().debug_options().xla_gpu_kernel_cache_file();
  const bool use_cache =
      !cache_file_path.empty() && split_constants_module &&
      hlo_module->config()
          .debug_options()
          .xla_gpu_enable_llvm_module_compilation_parallelism();

  if (split_constants_module) {
    // Constants are emitted into a separate module to avoid caching them.
    results.llvm_module_constants = std::make_unique<llvm::Module>(
        absl::StrCat(hlo_module->name(), "_consts"), *llvm_context);
    results.llvm_module_constants->setTargetTriple(target_triple);
    results.llvm_module_constants->setDataLayout(data_layout);
  }

  {
    tsl::profiler::ScopedAnnotation annotation([&] {
      return absl::StrFormat("XlaBufferAssignment:#module=%s,program_id=%d#",
                             hlo_module->name(), hlo_module->unique_id());
    });
    TF_ASSIGN_OR_RETURN(
        results.buffer_assignment,
        BufferAssigner::Run(
            hlo_module,
            std::make_unique<SequentialHloOrdering>(hlo_module->schedule()),
            buffer_size_bytes_function,
            /*color_alignment=*/
            [](LogicalBuffer::Color) { return kXlaAllocatedBufferAlignBytes; },
            /*allocate_buffers_for_constants=*/true,
            /*colorer=*/
            hlo_module->config()
                    .debug_options()
                    .xla_gpu_enable_nccl_user_buffers()
                ? CollectiveColorer()
                : BufferAssigner::DefaultColorer(),
            /*must_not_live_out=*/{},
            /*can_share_buffer*/ can_share_buffer_function,
            /*preset_assignments*/ {},
            /*private_stack*/ {}, /*heap_buffer_interval_compare*/ nullptr,
            /*isolation_options*/ std::nullopt,
            hlo_module->config()
                    .debug_options()
                    .xla_gpu_temp_buffer_use_separate_color()
                ? std::optional<BufferValue::Color>(kTempBufferMemorySpaceColor)
                : std::nullopt));
  }
  VLOG(1) << "Buffer Assignment Stats for " << hlo_module->name() << "\n"
          << results.buffer_assignment->StatsString(
                 /*report_total_fragmentation=*/true);

  results.execution_stream_assignment =
      std::make_unique<ExecutionStreamAssignment>(hlo_module);

  struct GetCcStr {
    std::string operator()(const se::CudaComputeCapability& cc) const {
      return absl::StrCat("sm_", cc.ToString());
    }
    std::string operator()(const se::RocmComputeCapability& cc) const {
      return cc.gfx_version();
    }
  };
  DumpHloModuleIfEnabled(
      *hlo_module, *results.buffer_assignment,
      absl::StrCat(
          std::visit(GetCcStr(), gpu_device_info.gpu_compute_capability()),
          "_gpu_", kAfterOptimizationsDumpName));

  VLOG(1) << "After optimization module fingerprint for " << hlo_module->name()
          << ": " << hlo_module->GetFingerprint128();

  uint64_t start_usecs = tsl::Env::Default()->NowMicros();

  mlir::DialectRegistry registry;
  // Disable MLIR multi-threading to prevent creating too many threads when
  // compiling XLA executables concurrently (e.g. during auto-tuning).
  auto mlir_context = std::make_unique<mlir::MLIRContext>(
      registry, mlir::MLIRContext::Threading::DISABLED);
  mlir_context->getDiagEngine().registerHandler(DiagnosticHandler);

  results.module_name = hlo_module->name();

  tsl::profiler::ScopedAnnotation annotation([&] {
    return absl::StrFormat("XlaEmitLlvmIr:#module=%s,program_id=%d#",
                           hlo_module->name(), hlo_module->unique_id());
  });
  IrEmitterContext ir_emitter_context(
      hlo_module, results.buffer_assignment.get(),
      results.execution_stream_assignment.get(), platform_name, gpu_device_info,
      mlir_context.get(), results.llvm_module.get(),
      results.llvm_module_constants.get(), /*emit_kernels=*/true);

  if (use_cache) {
    TF_RETURN_IF_ERROR(LoadCache(ir_emitter_context, cache_file_path));
  }

  std::vector<BufferAllocation*> allocations;
  results.output_shape = hlo_module->result_shape();
  TF_ASSIGN_OR_RETURN(results.output_info,
                      GetOutputInfo(*hlo_module, *results.buffer_assignment));
  results.use_original_allocations = true;

  auto ir_emitter = IrEmitterUnnested::Create(&ir_emitter_context);

  {
    XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
        "GpuCompiler::RunBackend - IR emission for ", hlo_module->name()));

    TF_RETURN_IF_ERROR(
        ir_emitter->EmitHloComputation(hlo_module->entry_computation()));

    bool supports_runtime_managed_constants =
        // TODO(b/218907125): Implement this feature for ROCm as well.
        platform_id != se::rocm::kROCmPlatformId &&
        hlo_module->config().debug_options().xla_gpu_enable_shared_constants();
    if (supports_runtime_managed_constants) {
      // Remove these globals from the generated code to indicate that XLA is
      // responsible for allocating and initializing them.
      RemoveUnusedAndUninitializedGlobals(
          ir_emitter_context.llvm_module_constants(),
          ir_emitter_context.constants());
    }

    results.constants = std::move(ir_emitter_context.constants());
    uint64_t end_usecs = tsl::Env::Default()->NowMicros();

    // This won't record values for calls that error out (because if they error
    // out we have no way of telling how far through the process we got).
    RecordHloToLlvmDuration(end_usecs - start_usecs);
  }

  results.executable = ir_emitter->ConsumeThunkSequence();
  if (use_cache) {
    results.kernel_compilation_cache =
        ir_emitter_context.kernel_cache().Export();
  }

  return results;
}

}  // namespace xla::gpu

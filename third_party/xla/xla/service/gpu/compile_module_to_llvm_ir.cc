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
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/mlir/backends/gpu/transforms/passes.h"
#include "xla/mlir/runtime/transforms/compilation_pipeline_gpu.h"
#include "xla/mlir_hlo/transforms/gpu_passes.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_unnested.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/gpu/runtime/executable.h"
#include "xla/service/gpu/runtime3/conditional_thunk.h"
#include "xla/service/gpu/runtime3/for_thunk.h"
#include "xla/service/gpu/runtime3/sequential_thunk.h"
#include "xla/service/gpu/runtime3/while_thunk.h"
#include "xla/service/gpu/thunk.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/hlo_ordering.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/translate/mhlo_to_hlo/location_exporter.h"
#include "xla/translate/mhlo_to_lhlo_with_xla/mhlo_to_lhlo_with_xla.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

// Prints mlir diagnostic messages to VLOG level 2.
static mlir::LogicalResult DiagnosticHandler(mlir::Diagnostic& diag) {
  VLOG(2) << diag.str();
  return mlir::failure();
}

static bool HasFp8(const HloModule& hlo_module) {
  for (const HloComputation* computation : hlo_module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (ShapeUtil::HasPrimitiveType(instruction->shape(), F8E5M2) ||
          ShapeUtil::HasPrimitiveType(instruction->shape(), F8E5M2FNUZ) ||
          ShapeUtil::HasPrimitiveType(instruction->shape(), F8E4M3FN) ||
          ShapeUtil::HasPrimitiveType(instruction->shape(), F8E4M3B11FNUZ) ||
          ShapeUtil::HasPrimitiveType(instruction->shape(), F8E4M3FNUZ)) {
        return true;
      }
    }
  }
  return false;
}

class DumpAfterPassIfEnabled : public mlir::PassInstrumentation {
 public:
  DumpAfterPassIfEnabled(const HloModule* hlo_module,
                         const mlir::ModuleOp* mlir_module)
      : hlo_module_{hlo_module}, mlir_module_{mlir_module} {}
  void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override {
    std::string pass_name = pass->getName().str();
    bool should_dump_pass = DumpingEnabledForHloPass(
        pass_name, hlo_module_->config().debug_options());
    if (!should_dump_pass) return;
    std::string module_str = llvm_ir::DumpToString(*mlir_module_);
    auto prefix = "lower_to_xla_gpu_runtime";
    auto suffix =
        absl::StrCat("pass_", absl::StrFormat("%02d", pass_counter_++), ".",
                     "after", ".", pass_name, ".mlir");
    DumpToFileInDirOrStdout(*hlo_module_, prefix, suffix, module_str);
  }

 private:
  const HloModule* hlo_module_;
  const mlir::ModuleOp* mlir_module_;
  int pass_counter_ = 0;
};

// Lowers MLIR module to the XLA Gpu runtime custom calls.
static absl::Status LowerToXlaGpuRuntime(
    mlir::ModuleOp module, llvm::StringRef entry_function_name,
    llvm::ArrayRef<int64_t> buffer_sizes, ThunkSequence* thunk_sequence,
    const HloModule* hlo_module, se::GpuComputeCapability compute_capability) {
  if (!module) {
    return InternalError("No MLIR module to lower.");
  }

  const DebugOptions& debug_options = hlo_module->config().debug_options();
  bool should_verify = debug_options.xla_gpu_llvm_verification_level() >= 1;
#ifndef NDEBUG
  should_verify = true;
#endif

  mlir::PassManager pm(module->getName(), mlir::PassManager::Nesting::Implicit);
  pm.enableVerifier(should_verify);
  if (hlo_module != nullptr && DumpingEnabledForHloModule(*hlo_module)) {
    pm.addInstrumentation(
        std::make_unique<DumpAfterPassIfEnabled>(hlo_module, &module));
  }

  absl::flat_hash_set<DebugOptions::CommandBufferCmdType> command_types;
  for (int command_type_num : debug_options.xla_gpu_enable_command_buffer()) {
    if (!DebugOptions::CommandBufferCmdType_IsValid(command_type_num)) {
      return InternalError("Invalid command buffer command type");
    }
    DebugOptions::CommandBufferCmdType command_type =
        static_cast<DebugOptions::CommandBufferCmdType>(command_type_num);
    command_types.insert(command_type);
  }

  GpuPipelineOpts opts;
  opts.command_types = command_types;
  opts.min_graph_size = debug_options.xla_gpu_graph_min_graph_size();
  opts.enable_concurrent_region =
      debug_options.xla_gpu_graph_enable_concurrent_region();
  opts.compute_capability = compute_capability;
  populateXlaGpuRuntimePasses(pm, thunk_sequence, opts);

  if (pm.run(module).failed()) {
    return InternalError("Failed to lower LMHLO to Gpu runtime custom calls.");
  }

  return absl::OkStatus();
}

}  // namespace

void ForAllThunks(const std::function<void(Thunk*)>& fn,
                  ThunkSequence* thunk_sequence) {
  for (std::unique_ptr<Thunk>& thunk : *thunk_sequence) {
    if (thunk->kind() == Thunk::kConditional) {
      auto* cond_thunk = tensorflow::down_cast<ConditionalThunk*>(thunk.get());
      for (const std::unique_ptr<SequentialThunk>& branch_thunks :
           cond_thunk->branch_thunks()) {
        ForAllThunks(fn, &branch_thunks->thunks());
      }
    } else if (thunk->kind() == Thunk::kFor) {
      auto* for_thunk = tensorflow::down_cast<ForThunk*>(thunk.get());
      ForAllThunks(fn, &for_thunk->body_thunk_sequence()->thunks());
    } else if (thunk->kind() == Thunk::kSequential) {
      auto* sequential_thunk =
          tensorflow::down_cast<SequentialThunk*>(thunk.get());
      ForAllThunks(fn, &sequential_thunk->thunks());
    } else if (thunk->kind() == Thunk::kWhile) {
      auto* while_thunk = tensorflow::down_cast<WhileThunk*>(thunk.get());
      ForAllThunks(fn, &while_thunk->condition_thunk_sequence()->thunks());
      ForAllThunks(fn, &while_thunk->body_thunk_sequence()->thunks());
    } else {
      fn(thunk.get());
    }
  }
}

static void ForwardCollectiveAttrs(mlir::ModuleOp module,
                                   llvm::StringRef entry_function_name,
                                   const HloModuleConfig& config) {
  mlir::OpBuilder b(module.getContext());
  auto func = module.lookupSymbol<mlir::func::FuncOp>(entry_function_name);
  func->setAttr("replica_count", b.getI64IntegerAttr(config.replica_count()));
  func->setAttr("num_partitions", b.getI64IntegerAttr(config.num_partitions()));
}

absl::StatusOr<GpuExecutable::OwnedGpuRuntimeProgram> LowerToJitRt(
    mlir::ModuleOp mlir_module, llvm::StringRef entry_function_name,
    llvm::ArrayRef<int64_t> buffer_sizes,
    std::unique_ptr<ThunkSequence> thunk_sequence, const HloModule* hlo_module,
    se::GpuComputeCapability compute_capability) {
  const auto& module_config = hlo_module->config();
  // Forward collective (NCCL) attributes for use by the lowering pipeline.
  ForwardCollectiveAttrs(mlir_module, entry_function_name, module_config);

  // Lower LMHLO operations to the XLA:GPU runtime custom calls.
  TF_RETURN_IF_ERROR(LowerToXlaGpuRuntime(
      mlir_module, {entry_function_name.data(), entry_function_name.size()},
      buffer_sizes, thunk_sequence.get(), hlo_module, compute_capability));

  // TODO(b/232033540): Pass MLIR module directly to Gpu runtime executable
  // without forcing serialization.
  std::string module_str = llvm_ir::DumpToString(mlir_module);

  if (hlo_module != nullptr) {
    DumpToFileInDirOrStdout(*hlo_module, "gpu_rt_host", "mlir", module_str);
  }

  // Collect allocation indices for handling graph capture functions.
  auto allocation_indices = GetAllocationIndices(mlir_module);

  return std::make_unique<GpuRuntimeProgram>(
      entry_function_name.str(), std::move(module_str), buffer_sizes.vec(),
      std::move(allocation_indices), module_config.debug_options());
}

// Analyze the function signature to reconstruct a vector of BufferAllocation
// objects, as well as other output information.
//
// This function also serves as a half-baked verifier for function arg
// attributes, since a full verifier doesn't exist yet.
static absl::Status GetMlirAllocationInfo(
    mlir::func::FuncOp func, std::vector<BufferAllocation>* allocations,
    absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>* output_info,
    Shape* output_shape) {
  CHECK(allocations->empty());
  allocations->reserve(func.getNumArguments());

  std::vector<int64_t> buffer_sizes;
  for (int i = 0; i < func.getNumArguments(); i++) {
    mlir::BlockArgument arg = func.getArgument(i);

    TF_RET_CHECK(arg.getType().isa<mlir::ShapedType>());
    mlir::ShapedType type = arg.getType().cast<mlir::ShapedType>();
    TF_ASSIGN_OR_RETURN(auto element_type_bytes,
                        GetElementTypeBytes(type.getElementType()));
    size_t size = type.getNumElements() * element_type_bytes;
    buffer_sizes.push_back(size);
  }

  for (int i = 0; i < func.getNumArguments(); i++) {
    llvm::ArrayRef<mlir::NamedAttribute> attrs =
        mlir::function_interface_impl::getArgAttrs(func, i);
    for (const mlir::NamedAttribute& attr : attrs) {
      TF_RET_CHECK(attr.getName() == "lmhlo.params" ||
                   attr.getName() == "lmhlo.param_shape_index" ||
                   attr.getName() == "lmhlo.constant_name" ||
                   attr.getName() == "lmhlo.must_alias" ||
                   attr.getName() == "lmhlo.output_index");
    }
  }

  return GpuExecutable::SetUpMlirAllocation(func, buffer_sizes, allocations,
                                            output_info, output_shape);
}

// The order of `thunk_sequence` corresponds to
// `hlo_schedule->ThunkLaunchOrder()`.
absl::StatusOr<CompileModuleResults> CompileModuleToLlvmIr(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, se::Platform::Id platform_id,
    const se::DeviceDescription& gpu_device_info,
    const HloDataflowAnalysis::CanShareBuffer& can_share_buffer_function,
    const BufferValue::SizeFunction& buffer_size_bytes_function) {
  CompileModuleResults results;
  results.llvm_module = std::make_unique<llvm::Module>("", *llvm_context);
  results.llvm_module->setTargetTriple(target_triple);
  results.llvm_module->setDataLayout(data_layout);

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
          /*must_not_live_out=*/{}, can_share_buffer_function));

  VLOG(1) << "Buffer Assignment Stats for " << hlo_module->name() << "\n"
          << results.buffer_assignment->GetStats().ToString();
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
  IrEmitterUnnested::GetDependentDialects(registry);

  // Disable MLIR multi-threading to prevent creating too many threads when
  // compiling XLA executables concurrently (e.g. during auto-tuning).
  auto mlir_context = std::make_unique<mlir::MLIRContext>(
      registry, mlir::MLIRContext::Threading::DISABLED);

  mlir_context->getDiagEngine().registerHandler(DiagnosticHandler);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module = llvm_ir::CreateMlirModuleOp(
      mlir::Builder(mlir_context.get()).getUnknownLoc(), hlo_module->name());

  absl::flat_hash_map<const mlir::Operation*, const xla::HloInstruction*>
      operation_map;

  // Store the allocations in the order of the LMHLO buffer arguments.
  std::vector<const BufferAllocation*> ordered_allocations;
  TF_RETURN_IF_ERROR(HloToLhloModule(*results.buffer_assignment, *hlo_module,
                                     *mlir_module, &ordered_allocations,
                                     &operation_map));

  results.module_name =
      mlir::mhlo::GetDebugNameFromLocation(mlir_module->getLoc());

  if (DumpingEnabledForHloModule(*hlo_module)) {
    DumpToFileInDirOrStdout(*hlo_module, "lmhlo", mlir_module.get());
  }

  auto entry_function = mlir::cast<mlir::func::FuncOp>(
      mlir_module->lookupSymbol(hlo_module->entry_computation()->name()));

  bool emit_from_hlo = !IsXlaRuntimeExecutableEnabled(hlo_module->config());

  std::vector<BufferAllocation> mlir_allocations;
  absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo> mlir_output_info;
  Shape mlir_output_shape;
  TF_RETURN_IF_ERROR(GetMlirAllocationInfo(entry_function, &mlir_allocations,
                                           &mlir_output_info,
                                           &mlir_output_shape));

  IrEmitterContext ir_emitter_context(
      hlo_module, results.buffer_assignment.get(), platform_name,
      gpu_device_info, mlir_context.get(), results.llvm_module.get(),
      emit_from_hlo, /*emit_kernels=*/true);

  std::vector<BufferAllocation*> allocations;
  if (emit_from_hlo) {
    results.output_shape = hlo_module->result_shape();
    TF_ASSIGN_OR_RETURN(results.output_info,
                        GetOutputInfo(*hlo_module, *results.buffer_assignment));
    TF_RET_CHECK(mlir_allocations.size() == ordered_allocations.size());
    ir_emitter_context.set_allocations(ordered_allocations);
    results.use_original_allocations = true;
  } else {
    results.allocations = std::move(mlir_allocations);
    results.output_shape = mlir_output_shape;
    results.output_info = mlir_output_info;
    allocations.reserve(results.allocations.size());
    for (auto& allocation : results.allocations) {
      allocations.push_back(&allocation);
    }
    ir_emitter_context.set_allocations(allocations);
    results.use_original_allocations = false;
  }

  auto ir_emitter = IrEmitterUnnested::Create(&ir_emitter_context);

  {
    XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
        "GpuCompiler::RunBackend - IR emission for ", hlo_module->name()));

    TF_RETURN_IF_ERROR(
        ir_emitter->EmitLmhloRegion(&entry_function.getBody(), operation_map));

    bool supports_runtime_managed_constants =
        // TODO(b/218907125): Implement this feature for ROCm as well.
        platform_id != se::rocm::kROCmPlatformId &&
        hlo_module->config().debug_options().xla_gpu_enable_shared_constants();
    if (supports_runtime_managed_constants) {
      // Remove these globals from the generated code to indicate that XLA is
      // responsible for allocating and initializing them.
      RemoveUnusedAndUninitializedGlobals(ir_emitter_context.llvm_module(),
                                          ir_emitter_context.constants());
    }

    results.constants = std::move(ir_emitter_context.constants());
    uint64_t end_usecs = tsl::Env::Default()->NowMicros();

    // This won't record values for calls that error out (because if they error
    // out we have no way of telling how far through the process we got).
    RecordHloToLlvmDuration(end_usecs - start_usecs);
  }

  // TODO(ezhulenev): Remove the FP8 check once https://reviews.llvm.org/D140088
  // is submitted. Currently we can't emit LLVM IR with fp8 types.
  if (IsXlaRuntimeExecutableEnabled(hlo_module->config()) &&
      !HasFp8(*hlo_module)) {
    // Sizes of all buffers required for running XLA module.
    std::vector<int64_t> buffer_sizes;
    llvm::transform(
        results.allocations, std::back_inserter(buffer_sizes),
        [](const BufferAllocation& allocation) { return allocation.size(); });

    TF_ASSIGN_OR_RETURN(
        results.executable,
        LowerToJitRt(*mlir_module, entry_function.getName(), buffer_sizes,
                     ir_emitter->ConsumeThunkSequence(), hlo_module,
                     gpu_device_info.gpu_compute_capability()));
  } else {
    auto thunk_sequence = ir_emitter->ConsumeThunkSequence();
    ForAllThunks([](Thunk* thunk) { thunk->ClearCompileTimeInfo(); },
                 thunk_sequence.get());
    results.executable = std::move(thunk_sequence);
  }
  return results;
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

}  // namespace gpu
}  // namespace xla

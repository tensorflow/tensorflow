/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/mlir/backends/gpu/transforms/passes.h"
#include "xla/mlir/runtime/transforms/compilation_pipeline_gpu.h"
#include "xla/mlir_hlo/transforms/gpu_passes.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/buffer_sharing.h"
#include "xla/service/gpu/conditional_thunk.h"
#include "xla/service/gpu/for_thunk.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_unnested.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/gpu/sequential_thunk.h"
#include "xla/service/gpu/while_thunk.h"
#include "xla/service/hlo_dataflow_analysis.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/status.h"
#include "xla/status_macros.h"
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

// Lowers MLIR module to the XLA Gpu runtime custom calls.
static Status LowerToXlaGpuRuntime(
    mlir::ModuleOp module, llvm::StringRef entry_function_name,
    llvm::ArrayRef<int64_t> buffer_sizes, ThunkSequence* thunk_sequence,
    const DebugOptions& debug_options,
    se::GpuComputeCapability compute_capability) {
  if (!module) {
    return InternalError("No MLIR module to lower.");
  }

  bool should_verify = debug_options.xla_gpu_llvm_verification_level() >= 1;
#ifndef NDEBUG
  should_verify = true;
#endif

  mlir::PassManager pm(module->getName(), mlir::PassManager::Nesting::Implicit);
  pm.enableVerifier(should_verify);

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

  return OkStatus();
}

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

}  // namespace

static void ForwardCollectiveAttrs(mlir::ModuleOp module,
                                   llvm::StringRef entry_function_name,
                                   const HloModuleConfig& config) {
  mlir::OpBuilder b(module.getContext());
  auto func = module.lookupSymbol<mlir::func::FuncOp>(entry_function_name);
  func->setAttr("replica_count", b.getI64IntegerAttr(config.replica_count()));
  func->setAttr("num_partitions", b.getI64IntegerAttr(config.num_partitions()));
}

StatusOr<GpuExecutable::OwnedGpuRuntimeProgram> LowerToJitRt(
    mlir::ModuleOp mlir_module, llvm::StringRef entry_function_name,
    llvm::ArrayRef<int64_t> buffer_sizes, const HloModuleConfig& module_config,
    std::unique_ptr<ThunkSequence> thunk_sequence,
    const HloModule* hlo_module_for_dump,
    se::GpuComputeCapability compute_capability) {
  // Forward collective (NCCL) attributes for use by the lowering pipeline.
  ForwardCollectiveAttrs(mlir_module, entry_function_name, module_config);

  // Lower LMHLO operations to the XLA:GPU runtime custom calls.
  TF_RETURN_IF_ERROR(LowerToXlaGpuRuntime(
      mlir_module, {entry_function_name.data(), entry_function_name.size()},
      buffer_sizes, thunk_sequence.get(), module_config.debug_options(),
      compute_capability));

  // TODO(b/232033540): Pass MLIR module directly to Gpu runtime executable
  // without forcing serialization.
  std::string module_str = llvm_ir::DumpToString(mlir_module);

  if (hlo_module_for_dump != nullptr) {
    DumpToFileInDirOrStdout(*hlo_module_for_dump, "gpu_rt_host", "mlir",
                            module_str);
  }

  // Collect allocation indices for handling graph capture functions.
  auto allocation_indices = GetAllocationIndices(mlir_module);

  return std::make_unique<GpuRuntimeProgram>(
      entry_function_name.str(), std::move(module_str), buffer_sizes.vec(),
      std::move(allocation_indices), module_config.debug_options());
}

StatusOr<std::unique_ptr<llvm::Module>> CompileModuleToLlvmIr(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, const se::Platform::Id platform_id,
    const se::DeviceDescription& gpu_device_info,
    const BufferValue::SizeFunction& buffer_size_bytes_function) {
  CompileModuleResults results;
  TF_RETURN_IF_ERROR(CompileModuleToLlvmIrImpl(
      hlo_module, llvm_context, target_triple, data_layout, platform_name,
      platform_id, gpu_device_info, &CanShareBufferHint,
      buffer_size_bytes_function, &results));
  return std::move(results.llvm_module);
}

// Analyze the function signature to reconstruct a vector of BufferAllocation
// objects, as well as other output information.
//
// This function also serves as a half-baked verifier for function arg
// attributes, since a full verifier doesn't exist yet.
static Status GetMlirAllocationInfo(
    mlir::func::FuncOp func, std::vector<BufferAllocation>* allocations,
    absl::flat_hash_map<ShapeIndex, GpuExecutable::OutputInfo>* output_info,
    Shape* output_shape, EntryFunctionAttributes* entry_func_attrs) {
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

  // Encode buffer parameter metadata in a proto for persisting, because BEF
  // doesn't persist function attributes.
  for (int i = 0; i < func.getNumArguments(); i++) {
    auto buffer = entry_func_attrs->add_buffers();
    if (auto param_attr = func.getArgAttr(i, "lmhlo.params")) {
      buffer->set_lmhlo_params_present(true);
      buffer->set_lmhlo_params(param_attr.cast<mlir::IntegerAttr>().getInt());
    }
    if (auto shape_index_attr = func.getArgAttr(i, "lmhlo.param_shape_index")) {
      auto param_shape_index = buffer->mutable_lmhlo_param_shape_index();
      for (const llvm::APInt& element :
           shape_index_attr.cast<mlir::DenseIntElementsAttr>()) {
        param_shape_index->add_indices(element.getSExtValue());
      }
    }
    if (auto constant_name_attr = func.getArgAttr(i, "lmhlo.constant_name")) {
      buffer->set_lmhlo_constant_name(
          constant_name_attr.cast<mlir::StringAttr>().str());
    }
    if (func.getArgAttr(i, "lmhlo.must_alias")) {
      buffer->set_lmhlo_must_alias(true);
    }
    if (auto output_index_attr = func.getArgAttr(i, "lmhlo.output_index")) {
      auto output_index = buffer->mutable_lmhlo_output_index();
      for (const llvm::APInt& element :
           output_index_attr.cast<mlir::DenseIntElementsAttr>()) {
        output_index->add_indices(element.getSExtValue());
      }
    }
  }
  entry_func_attrs->set_result_xla_shape(
      func->getAttrOfType<mlir::StringAttr>("result_xla_shape")
          .getValue()
          .str());

  return GpuExecutable::SetUpMlirAllocation(func, buffer_sizes, allocations,
                                            output_info, output_shape);
}

// The order of `thunk_sequence` corresponds to
// `hlo_schedule->ThunkLaunchOrder()`.
Status CompileModuleToLlvmIrImpl(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, se::Platform::Id platform_id,
    const se::DeviceDescription& gpu_device_info,
    const HloDataflowAnalysis::CanShareBuffer& can_share_buffer_function,
    const BufferValue::SizeFunction& buffer_size_bytes_function,
    CompileModuleResults* results, se::StreamExecutor* stream_exec) {
  results->llvm_module = std::make_unique<llvm::Module>("", *llvm_context);
  results->llvm_module->setTargetTriple(target_triple);
  results->llvm_module->setDataLayout(data_layout);

  TF_ASSIGN_OR_RETURN(
      results->buffer_assignment,
      BufferAssigner::Run(
          hlo_module,
          std::make_unique<SequentialHloOrdering>(hlo_module->schedule()),
          buffer_size_bytes_function,
          /*color_alignment=*/
          [](LogicalBuffer::Color) { return kXlaAllocatedBufferAlignBytes; },
          /*allocate_buffers_for_constants=*/true,
          /*colorer=*/BufferAssigner::DefaultColorer(),
          /*must_not_live_out=*/{}, can_share_buffer_function));

  VLOG(1) << "Buffer Assignment Stats for " << hlo_module->name() << "\n"
          << results->buffer_assignment->GetStats().ToString();
  struct GetCcStr {
    std::string operator()(const se::CudaComputeCapability& cc) const {
      return absl::StrCat("sm_", cc.ToString());
    }
    std::string operator()(const se::RocmComputeCapability& cc) const {
      return cc.gfx_version();
    }
  };
  DumpHloModuleIfEnabled(
      *hlo_module, *results->buffer_assignment,
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
  TF_RETURN_IF_ERROR(HloToLhloModule(*results->buffer_assignment, *hlo_module,
                                     *mlir_module, &operation_map));

  results->module_name =
      mlir::mhlo::GetDebugNameFromLocation(mlir_module->getLoc());

  if (DumpingEnabledForHloModule(*hlo_module)) {
    DumpToFileInDirOrStdout(*hlo_module, "lmhlo", mlir_module.get());
  }

  auto entry_function = mlir::cast<mlir::func::FuncOp>(
      mlir_module->lookupSymbol(hlo_module->entry_computation()->name()));

  // TODO(b/304613751): Add this flag to xla flags.
  constexpr bool emit_ir_from_hlo = false;

  IrEmitterContext ir_emitter_context(
      hlo_module, emit_ir_from_hlo ? results->buffer_assignment.get() : nullptr,
      platform_name, gpu_device_info, mlir_context.get(),
      results->llvm_module.get(), emit_ir_from_hlo);

  if (emit_ir_from_hlo) {
    TF_RET_CHECK(!IsXlaRuntimeExecutableEnabled(hlo_module->config()));
    results->allocations = results->buffer_assignment->Allocations();
    results->output_shape = hlo_module->result_shape();
    TF_ASSIGN_OR_RETURN(
        results->output_info,
        GetOutputInfo(*hlo_module, *results->buffer_assignment));
  } else {
    TF_RETURN_IF_ERROR(GetMlirAllocationInfo(
        entry_function, &results->allocations, &results->output_info,
        &results->output_shape, &results->entry_func_attrs));
    ir_emitter_context.set_allocations(results->allocations);
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

    results->constants = std::move(ir_emitter_context.constants());
    uint64_t end_usecs = tsl::Env::Default()->NowMicros();

    // This won't record values for calls that error out (because if they error
    // out we have no way of telling how far through the process we got).
    RecordHloToLlvmDuration(end_usecs - start_usecs);
  }

  // Sizes of all buffers required for running XLA module.
  std::vector<int64_t> buffer_sizes;
  llvm::transform(
      results->allocations, std::back_inserter(buffer_sizes),
      [](const BufferAllocation& allocation) { return allocation.size(); });

  // TODO(ezhulenev): Remove the FP8 check once https://reviews.llvm.org/D140088
  // is submitted. Currently we can't emit LLVM IR with fp8 types.
  if (IsXlaRuntimeExecutableEnabled(hlo_module->config()) &&
      !HasFp8(*hlo_module)) {
    TF_ASSIGN_OR_RETURN(
        results->executable,
        LowerToJitRt(*mlir_module, entry_function.getName(), buffer_sizes,
                     hlo_module->config(), ir_emitter->ConsumeThunkSequence(),
                     /*hlo_module_for_dump=*/hlo_module,
                     gpu_device_info.gpu_compute_capability()));
    return OkStatus();
  }

  auto thunk_sequence = ir_emitter->ConsumeThunkSequence();
  ForAllThunks([](Thunk* thunk) { thunk->ClearCompileTimeInfo(); },
               thunk_sequence.get());
  results->executable = std::move(thunk_sequence);
  return OkStatus();
}

// Removes all globals from the given module that are both uninitialized and
// have no uses within that module.
void RemoveUnusedAndUninitializedGlobals(
    llvm::Module* llvm_module,
    const std::vector<GpuExecutable::ConstantInfo>& constants) {
  for (const auto& info : constants) {
    // Empty content means the constant is initialized in the LLVM IR, so we
    // must not remove it.
    if (!info.content.empty()) {
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

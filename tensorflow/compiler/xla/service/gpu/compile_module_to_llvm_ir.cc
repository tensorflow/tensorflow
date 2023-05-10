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

#include "tensorflow/compiler/xla/service/gpu/compile_module_to_llvm_ir.h"

#include <stdlib.h>

#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
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
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/compilation_pipeline_gpu.h"
#include "tensorflow/compiler/xla/mlir_hlo/transforms/gpu_passes.h"
#include "tensorflow/compiler/xla/service/bitcast_dtypes_expander.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/gpu/conditional_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/for_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_convert_async_collectives_to_sync.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_unnested.h"
#include "tensorflow/compiler/xla/service/gpu/metrics.h"
#include "tensorflow/compiler/xla/service/gpu/sequential_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/while_thunk.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_rematerialization.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/optimization_barrier_expander.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/location_exporter.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_lhlo_with_xla/mhlo_to_lhlo_with_xla.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"

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
          ShapeUtil::HasPrimitiveType(instruction->shape(), F8E4M3FN) ||
          ShapeUtil::HasPrimitiveType(instruction->shape(), F8E4M3B11FNUZ)) {
        return true;
      }
    }
  }
  return false;
}

// Lowers MLIR module to the XLA Gpu runtime custom calls.
static Status LowerToXlaGpuRuntime(mlir::ModuleOp module,
                                   llvm::StringRef entry_function_name,
                                   llvm::ArrayRef<int64_t> buffer_sizes,
                                   ThunkSequence* thunk_sequence,
                                   const DebugOptions& debug_options) {
  if (!module) {
    return InternalError("No MLIR module to lower.");
  }

  mlir::PassManager pm(module->getName(), mlir::PassManager::Nesting::Implicit);

  GpuPipelineOpts opts;
  opts.cuda_graph_level = debug_options.xla_gpu_cuda_graph_level();
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
      auto* cond_thunk = static_cast<ConditionalThunk*>(thunk.get());
      for (const std::unique_ptr<SequentialThunk>& branch_thunks :
           cond_thunk->branch_thunks()) {
        ForAllThunks(fn, &branch_thunks->thunks());
      }
    } else if (thunk->kind() == Thunk::kFor) {
      auto* for_thunk = static_cast<ForThunk*>(thunk.get());
      ForAllThunks(fn, &for_thunk->body_thunk_sequence()->thunks());
    } else if (thunk->kind() == Thunk::kSequential) {
      auto* sequential_thunk = static_cast<SequentialThunk*>(thunk.get());
      ForAllThunks(fn, &sequential_thunk->thunks());
    } else if (thunk->kind() == Thunk::kWhile) {
      auto* while_thunk = static_cast<WhileThunk*>(thunk.get());
      ForAllThunks(fn, &while_thunk->condition_thunk_sequence()->thunks());
      ForAllThunks(fn, &while_thunk->body_thunk_sequence()->thunks());
    } else {
      fn(thunk.get());
    }
  }
}

}  // namespace

std::optional<bool> DummyCanShareBufferFunction(const HloInstruction*,
                                                const HloInstruction*,
                                                const ShapeIndex&) {
  return std::nullopt;
}

StatusOr<GpuExecutable::OwnedGpuRuntimeProgram> LowerToJitRt(
    mlir::ModuleOp mlir_module, llvm::StringRef entry_function_name,
    llvm::ArrayRef<int64_t> buffer_sizes, const HloModuleConfig& module_config,
    std::unique_ptr<ThunkSequence> thunk_sequence,
    const HloModule* hlo_module_for_dump) {
  // Forward collective (NCCL) attributes for use by the lowering pipeline.
  mlir::OpBuilder builder(mlir_module.getContext());
  mlir::IntegerAttr replica_count_attr =
      builder.getI64IntegerAttr(module_config.replica_count());
  mlir::IntegerAttr num_partitions_attr =
      builder.getI64IntegerAttr(module_config.num_partitions());
  mlir::func::FuncOp func =
      mlir_module.lookupSymbol<mlir::func::FuncOp>(entry_function_name);
  func->setAttr("replica_count", replica_count_attr);
  func->setAttr("num_partitions", num_partitions_attr);

  // Lower LMHLO operations to the JitRt compatible custom calls.
  TF_RETURN_IF_ERROR(LowerToXlaGpuRuntime(
      mlir_module, {entry_function_name.data(), entry_function_name.size()},
      buffer_sizes, thunk_sequence.get(), module_config.debug_options()));

  // TODO(b/232033540): Pass MLIR module directly to Gpu runtime executable
  // without forcing serialization.
  std::string module_str = llvm_ir::DumpToString(mlir_module);

  if (hlo_module_for_dump != nullptr) {
    DumpToFileInDirOrStdout(*hlo_module_for_dump, "gpu_rt_host", "mlir",
                            module_str);
  }

  return std::make_unique<GpuRuntimeProgram>(
      entry_function_name.str(), std::move(module_str), buffer_sizes.vec(),
      module_config.debug_options());
}

StatusOr<std::unique_ptr<llvm::Module>> CompileModuleToLlvmIr(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, const se::Platform::Id platform_id,
    GpuDeviceInfo gpu_device_info,
    se::CudaComputeCapability cuda_compute_capability,
    se::RocmComputeCapability rocm_compute_capability, int pointer_size) {
  CompileModuleResults results;
  TF_RETURN_IF_ERROR(CompileModuleToLlvmIrImpl(
      hlo_module, llvm_context, target_triple, data_layout, platform_name,
      platform_id, gpu_device_info, cuda_compute_capability,
      rocm_compute_capability, DummyCanShareBufferFunction, pointer_size,
      &results));
  return std::move(results.llvm_module);
}

// The order of `thunk_sequence` corresponds to
// `hlo_schedule->ThunkLaunchOrder()`.
Status CompileModuleToLlvmIrImpl(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, se::Platform::Id platform_id,
    GpuDeviceInfo gpu_device_info,
    se::CudaComputeCapability cuda_compute_capability,
    se::RocmComputeCapability rocm_compute_capability,
    const HloDataflowAnalysis::CanShareBuffer& can_share_buffer_function,
    int pointer_size, CompileModuleResults* results,
    se::StreamExecutor* stream_exec) {
  results->llvm_module = std::make_unique<llvm::Module>("", *llvm_context);
  results->llvm_module->setTargetTriple(target_triple);
  results->llvm_module->setDataLayout(data_layout);

  TF_RETURN_IF_ERROR(
      ScheduleGpuModule(hlo_module, pointer_size, gpu_device_info));
  {
    HloPassPipeline pipeline("post-scheduling-passes");

    HloPredicate is_nop =
        HloPredicateIsOp<HloOpcode::kParameter, HloOpcode::kConstant,
                         HloOpcode::kBitcast, HloOpcode::kGetTupleElement>;
    pipeline.AddPass<GpuConvertAsyncCollectivesToSync>(is_nop);
    pipeline.AddPass<OptimizationBarrierExpander>();

    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  auto buffer_size_bytes_function =
      [pointer_size](const BufferValue& buffer_value) -> int64_t {
    return GetSizeOfShape(buffer_value.shape(), pointer_size);
  };

  HloRematerialization::RematerializationSizes sizes;
  HloRematerialization remat(
      [pointer_size](const Shape& shape) {
        return GetSizeOfShape(shape, pointer_size);
      },
      // Assume 75% of the total device memory is available for XLA.
      /*memory_limit_bytes=*/gpu_device_info.device_memory_size * 0.75,
      /*sizes=*/&sizes,
      HloRematerialization::RematerializationPass::kPostFusion,
      /*block_size_limit=*/1, /*block_rematerialization_factor=*/1,
      /*compact_shape_function=*/nullptr,
      HloRematerialization::RematerializationMode::kRecomputeAndCompress);
  TF_ASSIGN_OR_RETURN(bool changed, remat.Run(hlo_module));
  if (changed) {
    VLOG(1) << "HloRematerialization saved "
            << sizes.before_bytes - sizes.after_bytes << " bytes";
  }

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
  DumpHloModuleIfEnabled(*hlo_module, *results->buffer_assignment,
                         absl::StrCat("sm_", cuda_compute_capability.ToString(),
                                      "_gpu_", kAfterOptimizationsDumpName));

  VLOG(1) << "After optimization module fingerprint for " << hlo_module->name()
          << ": " << hlo_module->GetFingerprint128();

  uint64_t start_usecs = tsl::Env::Default()->NowMicros();
  mlir::DialectRegistry registry;
  IrEmitterUnnested::GetDependentDialects(registry);
  mlir::MLIRContext mlir_context(registry);
  mlir_context.getDiagEngine().registerHandler(DiagnosticHandler);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::ModuleOp::create(mlir::Builder(&mlir_context).getUnknownLoc());

  TF_RETURN_IF_ERROR(
      HloToLhloModule(*results->buffer_assignment, *hlo_module, *mlir_module));

  results->module_name =
      mlir::mhlo::GetDebugNameFromLocation(mlir_module->getLoc());

  if (DumpingEnabledForHloModule(*hlo_module)) {
    DumpToFileInDirOrStdout(*hlo_module, "lmhlo", mlir_module.get());
  }

  auto entry_function = mlir::cast<mlir::func::FuncOp>(
      mlir_module->lookupSymbol(hlo_module->entry_computation()->name()));

  TF_RETURN_IF_ERROR(GetMlirAllocationInfo(
      entry_function, &results->allocations, &results->output_info,
      &results->output_shape, &results->entry_func_attrs));

  IrEmitterContext ir_emitter_context(
      /*hlo_module=*/nullptr, /*buffer_assignment=*/nullptr, platform_name,
      gpu_device_info, cuda_compute_capability, rocm_compute_capability,
      &mlir_context, results->llvm_module.get());

  ir_emitter_context.set_allocations(results->allocations);

  TF_ASSIGN_OR_RETURN(
      auto ir_emitter,
      IrEmitterUnnested::Create(hlo_module->config(), &ir_emitter_context));

  {
    XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
        "GpuCompiler::RunBackend - IR emission for ", hlo_module->name()));

    TF_RETURN_IF_ERROR(ir_emitter->EmitLmhloRegion(&entry_function.getBody()));

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

  // TODO(ezhulenev): Remove the FP8 check once https://reviews.llvm.org/D140088
  // is submitted. Currently we can't emit LLVM IR with fp8 types.
  if (IsXlaRuntimeExecutableEnabled(hlo_module->config()) &&
      !HasFp8(*hlo_module)) {
    std::vector<int64_t> buffer_sizes;
    llvm::transform(
        results->allocations, std::back_inserter(buffer_sizes),
        [](const BufferAllocation& allocation) { return allocation.size(); });
    TF_ASSIGN_OR_RETURN(
        results->executable,
        LowerToJitRt(*mlir_module, entry_function.getName(), buffer_sizes,
                     hlo_module->config(), ir_emitter->ConsumeThunkSequence(),
                     /*hlo_module_for_dump=*/hlo_module));
    return OkStatus();
  }

  auto thunk_sequence = ir_emitter->ConsumeThunkSequence();
  ForAllThunks([](Thunk* thunk) { thunk->ClearCompileTimeInfo(); },
               thunk_sequence.get());
  results->executable = std::move(thunk_sequence);
  return OkStatus();
}

// Analyze the function signature to reconstruct a vector of BufferAllocation
// objects, as well as other output information.
//
// This function also serves as a half-baked verifier for function arg
// attributes, since a full verifier doesn't exist yet.
Status GetMlirAllocationInfo(
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

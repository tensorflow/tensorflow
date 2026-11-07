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
#include "xla/backends/gpu/codegen/phased_block_cluster.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialectDecl.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialectDecl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "xla/backends/gpu/codegen/emitters/loop.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/codegen/nested_kernel_utils.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/emitters/transforms/lower_to_llvm_gpu.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "tsl/platform/platform.h"

namespace xla::gpu {

PhasedBlockClusterFusion::PhasedBlockClusterFusion(
    const HloFusionAnalysis& analysis,
    std::optional<LaunchDimensions> outer_launch_dims)
    : device_info_(analysis.device_info()) {
  uint64_t max_blocks = 0;
  uint64_t max_threads = 0;
  nested_kernels_.reserve(analysis.fusion_root_count());
  for (const HloInstructionAdaptor& root : analysis.fusion_roots()) {
    const auto* nested = DynCast<HloFusionInstruction>(&root.instruction());
    LaunchDimensions dims;
    if (nested != nullptr && !nested->shape().IsTuple()) {
      dims = LoopFusion(HloFusionAnalysis::Create(*nested, device_info_))
                 .launch_dimensions();
    }
    max_blocks = std::max(max_blocks, dims.num_blocks());
    max_threads = std::max(max_threads, dims.num_threads_per_block());
    nested_kernels_.push_back({nested, dims});
  }
  if (outer_launch_dims.has_value()) {
    launch_dimensions_ = *outer_launch_dims;
    cluster_dim_ = se::ClusterDim(outer_launch_dims->num_blocks(), 1, 1);
  } else {
    // CUDA `barrier.cluster` only synchronizes CTAs within one cluster, so
    // launch a single portable cluster of power-of-2 size in {1, 2, 4, 8}.
    uint64_t cluster_blocks =
        absl::bit_ceil(std::min<uint64_t>(max_blocks, kMaxBlocksPerCluster));
    launch_dimensions_ = LaunchDimensions(cluster_blocks, max_threads);
    cluster_dim_ = se::ClusterDim(cluster_blocks, 1, 1);
  }
}

absl::Status PhasedBlockClusterFusion::EmitLauncherBody(
    mlir::ModuleOp launcher_module, mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion,
    absl::Span<const mlir::OwningOpRef<mlir::ModuleOp>> nested_modules) const {
  mlir::MLIRContext* ctx = launcher_module.getContext();
  mlir::ImplicitLocOpBuilder b(entry_function.getLoc(), ctx);
  auto i32_ty = b.getI32Type();
  auto ptr_ty = mlir::LLVM::LLVMPointerType::get(ctx);

  llvm::SmallVector<mlir::LLVM::LLVMFuncOp> nested_decls;
  nested_decls.reserve(nested_kernels_.size());
  b.setInsertionPointToStart(launcher_module.getBody());
  for (size_t idx = 0; idx < nested_kernels_.size(); ++idx) {
    std::string fn_name =
        GetNestedKernelFunctionName(entry_function.getName(), idx);
    auto fn =
        nested_modules[idx].get().lookupSymbol<mlir::LLVM::LLVMFuncOp>(fn_name);
    TF_RET_CHECK(fn != nullptr) << "Missing " << fn_name;
    nested_decls.push_back(mlir::LLVM::LLVMFuncOp::create(
        b, fn_name, fn.getFunctionType(), mlir::LLVM::Linkage::External));
  }

  mlir::Block* entry_block = entry_function.addEntryBlock();
  b.setInsertionPointToStart(entry_block);

  // Add unrealized casts to LLVM pointers to be used when calling nested
  // kernels as llvm functions.
  llvm::SmallVector<mlir::Value> arg_ptrs;
  arg_ptrs.reserve(entry_function.getNumArguments());
  for (mlir::Value arg : entry_function.getArguments()) {
    arg_ptrs.push_back(
        mlir::UnrealizedConversionCastOp::create(b, ptr_ty, arg).getResult(0));
  }

  // Emit block and thread IDs.
  mlir::Value bid = mlir::arith::IndexCastUIOp::create(
      b, i32_ty, mlir::gpu::BlockIdOp::create(b, mlir::gpu::Dimension::x));
  mlir::Value tid = mlir::arith::IndexCastUIOp::create(
      b, i32_ty, mlir::gpu::ThreadIdOp::create(b, mlir::gpu::Dimension::x));
  int64_t max_blocks = launch_dimensions_.num_blocks();
  int64_t max_threads = launch_dimensions_.num_threads_per_block();

  // Emit phases separated by cluster barriers.
  for (size_t idx = 0; idx < nested_kernels_.size(); ++idx) {
    const NestedKernelSpec& spec = nested_kernels_[idx];
    b.setInsertionPointToEnd(entry_block);

    // If the nested kernel has fewer blocks than the launcher cluster, we only
    // need to launch the blocks that are in range; remaining CTAs skip
    // directly to the cluster barrier.
    int64_t nested_blocks = spec.launch_dims.num_blocks();
    if (nested_blocks < max_blocks) {
      mlir::Value block_bound =
          mlir::arith::ConstantIntOp::create(b, nested_blocks, 32);
      auto block_if = mlir::scf::IfOp::create(
          b,
          mlir::arith::CmpIOp::create(b, mlir::arith::CmpIPredicate::ult, bid,
                                      block_bound),
          /*withElseRegion=*/false);
      b.setInsertionPointToStart(&block_if.getThenRegion().front());
    }

    // If the nested kernel has fewer threads per block than the launcher
    // kernel, we only need to launch the threads that are in range.
    int64_t nested_threads = spec.launch_dims.num_threads_per_block();
    if (nested_threads < max_threads) {
      mlir::Value tid_bound =
          mlir::arith::ConstantIntOp::create(b, nested_threads, 32);
      auto tid_if = mlir::scf::IfOp::create(
          b,
          mlir::arith::CmpIOp::create(b, mlir::arith::CmpIPredicate::ult, tid,
                                      tid_bound),
          /*withElseRegion=*/false);
      b.setInsertionPointToStart(&tid_if.getThenRegion().front());
    }

    llvm::SmallVector<mlir::Value> call_args;
    call_args.reserve(spec.fusion->operand_count() + 2);
    // Input arguments.
    for (const HloInstruction* operand : spec.fusion->operands()) {
      call_args.push_back(operand->opcode() == HloOpcode::kParameter
                              ? arg_ptrs[operand->parameter_number()]
                              : arg_ptrs[fusion.operand_count() + idx - 1]);
    }
    // Output argument.
    call_args.push_back(arg_ptrs[fusion.operand_count() + idx]);
    // Virtual block ID argument appended by `VirtualizeSubKernelPass`.
    call_args.push_back(bid);
    mlir::LLVM::CallOp::create(b, nested_decls[idx], call_args);

    if (idx + 1 < nested_kernels_.size()) {
      // Emit `.aligned` cluster barriers at `entry_block` top level outside all
      // CTA/thread `if` guards so every thread of every CTA in the cluster
      // participates and phase `idx` writes become visible across the cluster.
      b.setInsertionPointToEnd(entry_block);
      mlir::NVVM::ClusterArriveOp::create(b, b.getUnitAttr());
      mlir::NVVM::ClusterWaitOp::create(b, b.getUnitAttr());
    }
  }

  b.setInsertionPointToEnd(entry_block);
  mlir::func::ReturnOp::create(
      b, entry_function.getArguments().drop_front(fusion.operand_count()));
  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
PhasedBlockClusterFusion::CreateAndLowerMLIRModule(
    mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
    absl::string_view entry_function_name,
    const BufferAssignment* buffer_assignment) const {
  if (fusion.fusion_kind() != HloInstruction::FusionKind::kCustom ||
      !IsGpuFusionKind(fusion, kPhasedBlockClusterFusionKind)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected ", kPhasedBlockClusterFusionKind, " fusion."));
  }
  if (fusion.fused_expression_root()->opcode() != HloOpcode::kTuple) {
    return absl::UnimplementedError("ROOT must be a tuple of nested fusions.");
  }
  for (const HloInstruction* operand : fusion.operands()) {
    if (operand->shape().IsTuple()) {
      return absl::UnimplementedError("Tuple operands are not supported.");
    }
  }
  // `buffer_assignment` is non-null in `Emit()`, and null only in unit tests
  // that lower standalone HLO without running `BufferAssigner`.
  if (buffer_assignment != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        emitters::KernelArguments kernel_args,
        emitters::KernelArguments::Create(
            *buffer_assignment, GetDefaultBufferAlignment(), &fusion));
    // `KernelArguments` lists outer operands followed by `ROOT` tuple outputs;
    // `inserted` is false when an output shares a buffer slice with an operand
    // or an earlier output.
    absl::flat_hash_set<int64_t> seen_slices;
    for (size_t i = 0; i < kernel_args.args().size(); ++i) {
      bool inserted =
          seen_slices.insert(kernel_args.args()[i].slice_index()).second;
      if (i >= fusion.operand_count() && !inserted) {
        return absl::UnimplementedError(
            "Buffer aliasing with PhasedBlockClusterFusion outputs is not "
            "supported yet.");
      }
    }
  }

  mlir_context.appendDialectRegistry(MlirKernelEmitter::GetDialectRegistry());
  mlir_context.loadAllAvailableDialects();
  bool should_verify = fusion.GetModule()
                           ->config()
                           .debug_options()
                           .xla_gpu_llvm_verification_level() >= 1;
  if constexpr (tsl::kIsDebugBuild) {
    should_verify = true;
  }

  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> nested_modules;
  nested_modules.reserve(nested_kernels_.size());
  for (size_t i = 0; i < nested_kernels_.size(); ++i) {
    const NestedKernelSpec& spec = nested_kernels_[i];
    if (spec.fusion == nullptr || spec.fusion->shape().IsTuple()) {
      return absl::UnimplementedError(
          "Nested fusions must have a single array output.");
    }
    HloFusionAnalysis nested_analysis =
        HloFusionAnalysis::Create(*spec.fusion, device_info_);
    if (spec.fusion->fusion_kind() != HloInstruction::FusionKind::kLoop ||
        nested_analysis.emitter_fusion_kind() !=
            HloFusionAnalysis::EmitterFusionKind::kLoop) {
      return absl::UnimplementedError(
          "Only kLoop nested fusions are supported.");
    }
    const se::BlockDim& blocks = spec.launch_dims.block_counts();
    const se::ThreadDim& threads = spec.launch_dims.thread_counts_per_block();
    if (blocks.x > kMaxBlocksPerCluster || blocks.y != 1 || blocks.z != 1 ||
        threads.y != 1 || threads.z != 1) {
      return absl::UnimplementedError(
          "Nested fusion launch dimensions exceed a single 1D cluster.");
    }
    bool uses_prev_phase = false;
    for (const HloInstruction* operand : spec.fusion->operands()) {
      if (operand->opcode() == HloOpcode::kParameter) {
        continue;
      }
      if (i > 0 && operand == nested_kernels_[i - 1].fusion) {
        uses_prev_phase = true;
        continue;
      }
      return absl::UnimplementedError(
          "Only straight-line fusion chains in ROOT tuple order are "
          "supported.");
    }
    if (i > 0 && !uses_prev_phase) {
      return absl::UnimplementedError(
          "Each phase must be a consumer of the immediately preceding phase.");
    }

    std::string fn_name = GetNestedKernelFunctionName(entry_function_name, i);
    LoopFusion emitter(nested_analysis);
    ABSL_ASSIGN_OR_RETURN(MlirKernelSource src,
                     emitter.Emit(&mlir_context, *spec.fusion, fn_name,
                                  /*buffer_assignment=*/nullptr));
    ABSL_ASSIGN_OR_RETURN(src,
                     CompileMlirToLlvmDialect(device_info_, *fusion.GetModule(),
                                              fn_name, emitter.unroll_factor(),
                                              mlir_context, std::move(src)));
    mlir::OwningOpRef<mlir::ModuleOp> nested_module =
        std::move(src).TakeModule();

    VirtualizeSubKernelPassOptions opts;
    opts.entry_function_name_ = fn_name;
    opts.block_dim_x_ = blocks.x;
    opts.thread_dim_x_ = threads.x;
    opts.max_threads_ = launch_dimensions_.num_threads_per_block();
    opts.warp_size_ = device_info_.threads_per_warp();
    mlir::PassManager pm(&mlir_context);
    pm.enableVerifier(should_verify);
    pm.addPass(createVirtualizeSubKernelPass(opts));
    tsl::StatusScopedDiagnosticHandler diagnostic_handler(
        nested_module->getContext());
    ABSL_RETURN_IF_ERROR(diagnostic_handler.consumeStatus(pm.run(*nested_module)));
    nested_modules.push_back(std::move(nested_module));
  }

  mlir::OpBuilder builder(&mlir_context);
  mlir::OwningOpRef<mlir::ModuleOp> launcher_module =
      llvm_ir::CreateMlirModuleOp(
          mlir::NameLoc::get(builder.getStringAttr(fusion.name())));
  ABSL_ASSIGN_OR_RETURN(mlir::func::FuncOp entry_func,
                   emitters::EmitKernelApi(
                       *launcher_module, fusion, buffer_assignment,
                       GetDefaultBufferAlignment(), entry_function_name));
  // Intermediate output buffers are written by sibling CTAs before
  // `barrier.cluster` and read in the next phase; marking them `xla.aliased`
  // tells `MergePointersToSameSlicePass` to omit `llvm.noalias` so LLVM
  // `BasicAA` cannot reorder accesses across the barrier.
  for (size_t i = 0; i + 1 < nested_kernels_.size(); ++i) {
    entry_func.setArgAttr(fusion.operand_count() + i, emitters::kXlaAliasedAttr,
                          builder.getUnitAttr());
  }
  SetBackendKind(&mlir_context, entry_func, BackendKind::kGpu);
  emitters::SetIndexDataLayout(*launcher_module, fusion);
  ABSL_RETURN_IF_ERROR(
      EmitLauncherBody(*launcher_module, entry_func, fusion, nested_modules));
  ABSL_RETURN_IF_ERROR(LowerLauncherModuleToLlvmDialect(
      *launcher_module, device_info_, should_verify));

  auto launcher_entry = launcher_module->lookupSymbol<mlir::LLVM::LLVMFuncOp>(
      entry_function_name);
  TF_RET_CHECK(launcher_entry != nullptr);
  launcher_entry->setAttr(
      mlir::NVVM::NVVMDialect::getClusterDimAttrName(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(cluster_dim_.x),
                                    static_cast<int32_t>(cluster_dim_.y),
                                    static_cast<int32_t>(cluster_dim_.z)}));
  int64_t max_shmem_bytes = 0;
  for (size_t i = 0; i < nested_kernels_.size(); ++i) {
    if (auto shmem_attr =
            (*nested_modules[i])
                ->getAttrOfType<mlir::IntegerAttr>("xla.shared_memory_bytes")) {
      max_shmem_bytes = std::max(max_shmem_bytes, shmem_attr.getInt());
    }
    ABSL_RETURN_IF_ERROR(SpliceNestedKernelModuleIntoLauncher(
        *launcher_module, *nested_modules[i],
        GetNestedKernelFunctionName(entry_function_name, i)));
  }
  (*launcher_module)
      ->setAttr("xla.shared_memory_bytes",
                builder.getI64IntegerAttr(max_shmem_bytes));
  return launcher_module;
}

AsyncThunkSequence PhasedBlockClusterFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  VLOG(4) << "PhasedBlockClusterFusion: "
          << fusion.fused_instructions_computation()->ToString();
  const auto* cuda_cc =
      ir_emitter_context.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc == nullptr || !cuda_cc->IsAtLeastHopper()) {
    return absl::UnimplementedError(absl::StrCat(
        "PhasedBlockClusterFusion requires CUDA Hopper (SM90+), got: ",
        ir_emitter_context.gpu_compute_capability().ToString()));
  }

  ABSL_ASSIGN_OR_RETURN(
      emitters::KernelArguments args,
      emitters::KernelArguments::Create(ir_emitter_context.buffer_assignment(),
                                        GetDefaultBufferAlignment(), &fusion));

  auto [future_entry, cached] = ir_emitter_context.kernel_cache().GetWithStatus(
      fusion.fused_instructions_computation(), args.args(),
      /*discriminator=*/"PhasedBlockClusterFusion",
      [&]() -> xla::Future<KernelReuseCache::Entry> {
        std::string kernel_name = ir_emitter_context.GetSanitizedUniqueName(
            std::string(fusion.name()));
        BorrowedMlirContext borrowed_context =
            ir_emitter_context.BorrowMlirContext();
        ABSL_ASSIGN_OR_RETURN(
            mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
            CreateAndLowerMLIRModule(**borrowed_context, fusion, kernel_name,
                                     &ir_emitter_context.buffer_assignment()));
        auto llvm_ctx = std::make_unique<llvm::LLVMContext>();
        std::unique_ptr<llvm::Module> llvm_mod =
            mlir::translateModuleToLLVMIR(*mlir_module, *llvm_ctx);
        TF_RET_CHECK(llvm_mod != nullptr);
        llvm::orc::ThreadSafeModule safe_module(std::move(llvm_mod),
                                                std::move(llvm_ctx));
        LaunchDimensions launch_dims = launch_dimensions();
        ABSL_RETURN_IF_ERROR(AnnotateMlirKernelLlvmModule(
            ir_emitter_context.gpu_device_info(), launch_dims, kernel_name,
            ir_emitter_context.data_layout(),
            ir_emitter_context.target_triple(),
            safe_module.getModuleUnlocked()));
        bool use_pdl =
            IsPdlEnabled(fusion.GetModule()->config().debug_options(),
                         ir_emitter_context.gpu_compute_capability());
        GpuKernelSpec spec{kernel_name, launch_dims, args,
                           /*scratch_bytes=*/std::nullopt, use_pdl};
        return ir_emitter_context.kernel_compiler()
            ->CompileToTargetBinary(LlvmKernelSource(std::move(safe_module)),
                                    spec)
            .Map([spec = std::move(spec), cluster_dim = cluster_dim_](
                     const std::vector<uint8_t>& cubin) mutable {
              KernelReuseCache::Entry entry{
                  std::string{spec.name()}, spec.launch_dimensions(),
                  cluster_dim, /*shmem_bytes=*/0,
                  std::make_shared<const std::vector<uint8_t>>(cubin)};
              entry.use_pdl = spec.use_pdl();
              return entry;
            });
      });

  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      &fusion, ir_emitter_context.GetNextThunkId());
  return future_entry.Map(
      [&fusion, thunk_info = std::move(thunk_info), args = std::move(args),
       kernel_cached = cached,
       devices_in_process =
           ir_emitter_context.gpu_topology().num_devices_per_process()](
          const KernelReuseCache::Entry& entry) mutable
          -> absl::StatusOr<ThunkSequence> {
        if (kernel_cached) {
          VLOG(3) << "Reuse: " << fusion.name() << " -> " << entry.kernel_name;
        }
        ABSL_ASSIGN_OR_RETURN(
            CustomKernel custom_kernel,
            kernel::CreateSharedCubinCustomKernel(
                entry.kernel_name, entry.binary, args.args().size(),
                entry.launch_dimensions.block_counts(),
                entry.launch_dimensions.thread_counts_per_block(),
                entry.shmem_bytes, entry.cluster_dim));
        return ThunkSequence::Of<CustomKernelThunk>(
            thunk_info, std::move(custom_kernel), args, devices_in_process,
            entry.use_pdl);
      });
}

}  // namespace xla::gpu

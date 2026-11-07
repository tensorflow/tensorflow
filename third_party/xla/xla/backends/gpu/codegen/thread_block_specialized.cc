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
#include "xla/backends/gpu/codegen/thread_block_specialized.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialectDecl.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "xla/backends/gpu/codegen/emitters/concatenate.h"
#include "xla/backends/gpu/codegen/emitters/loop.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/emitters/reduction.h"
#include "xla/backends/gpu/codegen/emitters/scatter.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/gpu/codegen/emitters/transpose.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/ptx_custom_kernel.h"
#include "xla/backends/gpu/codegen/nested_kernel_utils.h"
#include "xla/backends/gpu/codegen/phased_block_cluster.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
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
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/util.h"
#include "tsl/platform/platform.h"

namespace xla::gpu {

/*static*/
std::unique_ptr<MlirKernelEmitter>
ThreadBlockSpecializedFusion::CreateNativeNestedEmitter(
    const HloFusionAnalysis& nested_analysis) {
  switch (nested_analysis.emitter_fusion_kind()) {
    case HloFusionAnalysis::EmitterFusionKind::kLoop:
      return std::make_unique<LoopFusion>(nested_analysis);
    case HloFusionAnalysis::EmitterFusionKind::kReduction:
      return CreateReductionFusion(nested_analysis);
    case HloFusionAnalysis::EmitterFusionKind::kTranspose:
      return CreateTransposeFusion(nested_analysis);
    case HloFusionAnalysis::EmitterFusionKind::kScatter:
      return CreateScatterFusion(nested_analysis);
    case HloFusionAnalysis::EmitterFusionKind::kConcatenate:
      return std::make_unique<ConcatenateFusion>(nested_analysis);
    default:
      return nullptr;
  }
}

ThreadBlockSpecializedFusion::NestedKernelSpec
ThreadBlockSpecializedFusion::NestedKernelSpec::Create(
    const HloFusionInstruction& fusion,
    const se::DeviceDescription& device_info) {
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(fusion, device_info);
  LaunchDimensions launch_dims;
  se::ClusterDim cluster_dim{1, 1, 1};
  if (std::unique_ptr<MlirKernelEmitter> emitter =
          CreateNativeNestedEmitter(analysis)) {
    launch_dims = emitter->launch_dimensions();
  } else if (analysis.emitter_fusion_kind() ==
             HloFusionAnalysis::EmitterFusionKind::kPhasedBlockCluster) {
    PhasedBlockClusterFusion phased_block_cluster_fusion(analysis);
    launch_dims = phased_block_cluster_fusion.launch_dimensions();
    cluster_dim = phased_block_cluster_fusion.cluster_dim();
  } else {
    LOG(FATAL) << "Failed to create nested kernel emitter for "
               << fusion.name();
  }
  int64_t num_outputs = ShapeUtil::GetLeafCount(fusion.shape());
  return NestedKernelSpec{&fusion, launch_dims,
                          std::vector<int64_t>(num_outputs, -1),
                          analysis.emitter_fusion_kind(), cluster_dim};
}

ThreadBlockSpecializedFusion::ThreadBlockSpecializedFusion(
    const HloFusionAnalysis& analysis)
    : device_info_(analysis.device_info()) {
  uint64_t max_threads = 0;
  uint64_t max_cluster_dim_x = 1;
  bool has_cluster_fusion = false;
  absl::flat_hash_map<const HloInstruction*, size_t> fusion_to_kernel_idx;
  for (int64_t root_idx = 0; root_idx < analysis.fusion_root_count();
       ++root_idx) {
    const HloInstruction* nested_fusion =
        &analysis.fusion_root(root_idx).instruction();
    int64_t tuple_idx = 0;
    if (nested_fusion->opcode() == HloOpcode::kGetTupleElement) {
      tuple_idx = nested_fusion->tuple_index();
      nested_fusion = nested_fusion->operand(0);
    }

    auto [it, inserted] =
        fusion_to_kernel_idx.try_emplace(nested_fusion, nested_kernels_.size());
    if (inserted) {
      NestedKernelSpec spec = NestedKernelSpec::Create(
          *Cast<HloFusionInstruction>(nested_fusion), device_info_);
      max_threads =
          std::max(max_threads, spec.launch_dims.num_threads_per_block());
      max_cluster_dim_x = std::max(max_cluster_dim_x, spec.cluster_dim.x);
      if (spec.emitter_fusion_kind ==
          HloFusionAnalysis::EmitterFusionKind::kPhasedBlockCluster) {
        has_cluster_fusion = true;
      }
      nested_kernels_.push_back(std::move(spec));
    }

    NestedKernelSpec& spec = nested_kernels_[it->second];
    CHECK_LT(tuple_idx, static_cast<int64_t>(spec.outer_root_indices.size()));
    spec.outer_root_indices[tuple_idx] = root_idx;
  }

  // Place `PhasedBlockClusterFusion` sub-kernels first so each starts on a
  // cluster boundary and occupies an entire cluster of `max_cluster_dim_x`
  // CTAs, while non-clustered sub-kernels pack contiguously afterward without
  // per-kernel padding.
  absl::c_stable_partition(nested_kernels_, [](const NestedKernelSpec& spec) {
    return spec.emitter_fusion_kind ==
           HloFusionAnalysis::EmitterFusionKind::kPhasedBlockCluster;
  });

  uint64_t total_blocks = 0;
  for (const NestedKernelSpec& spec : nested_kernels_) {
    for (int64_t outer_root_idx : spec.outer_root_indices) {
      CHECK_GE(outer_root_idx, 0)
          << "Multi-output nested fusion has unreferenced output in "
          << spec.fusion->ToString();
    }
    if (spec.emitter_fusion_kind ==
        HloFusionAnalysis::EmitterFusionKind::kPhasedBlockCluster) {
      total_blocks +=
          RoundUpTo<uint64_t>(spec.launch_dims.num_blocks(), max_cluster_dim_x);
    } else {
      total_blocks += spec.launch_dims.num_blocks();
    }
  }
  total_blocks = RoundUpTo<uint64_t>(total_blocks, max_cluster_dim_x);

  launch_dimensions_ = LaunchDimensions(total_blocks, max_threads);
  if (has_cluster_fusion) {
    cluster_dim_ = se::ClusterDim(max_cluster_dim_x, 1, 1);
  }
}

absl::Status ThreadBlockSpecializedFusion::EmitLauncherBody(
    mlir::ModuleOp launcher_module, mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion,
    absl::Span<const mlir::OwningOpRef<mlir::ModuleOp>> nested_modules) const {
  mlir::MLIRContext* ctx = launcher_module.getContext();
  mlir::ImplicitLocOpBuilder b(entry_function.getLoc(), ctx);
  mlir::Type i32_ty = b.getI32Type();
  auto ptr_ty = mlir::LLVM::LLVMPointerType::get(ctx);

  // Declare external llvm.func symbols for each lowered nested kernel.
  llvm::SmallVector<mlir::LLVM::LLVMFuncOp> nested_decls;
  nested_decls.reserve(nested_kernels_.size());
  b.setInsertionPointToStart(launcher_module.getBody());
  for (size_t idx = 0; idx < nested_kernels_.size(); ++idx) {
    std::string nested_fn_name =
        GetNestedKernelFunctionName(entry_function.getName(), idx);
    auto nested_func =
        (*nested_modules[idx])
            .lookupSymbol<mlir::LLVM::LLVMFuncOp>(nested_fn_name);
    TF_RET_CHECK(nested_func != nullptr)
        << "Nested kernel function " << nested_fn_name << " not found.";
    auto decl = mlir::LLVM::LLVMFuncOp::create(b, nested_fn_name,
                                               nested_func.getFunctionType(),
                                               mlir::LLVM::Linkage::External);
    nested_decls.push_back(decl);
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
  mlir::Value bid_idx =
      mlir::gpu::BlockIdOp::create(b, mlir::gpu::Dimension::x);
  mlir::Value tid_idx =
      mlir::gpu::ThreadIdOp::create(b, mlir::gpu::Dimension::x);
  mlir::Value bid = mlir::arith::IndexCastUIOp::create(b, i32_ty, bid_idx);
  mlir::Value tid = mlir::arith::IndexCastUIOp::create(b, i32_ty, tid_idx);

  int64_t max_threads = launch_dimensions_.num_threads_per_block();
  int64_t cluster_dim_x =
      cluster_dim_.has_value() ? static_cast<int64_t>(cluster_dim_->x) : 1;
  int64_t block_offset = 0;

  // Emit conditional launcher logic.
  for (size_t idx = 0; idx < nested_kernels_.size(); ++idx) {
    const NestedKernelSpec& spec = nested_kernels_[idx];
    int64_t nested_blocks = spec.launch_dims.num_blocks();
    int64_t allocated_blocks =
        spec.emitter_fusion_kind ==
                HloFusionAnalysis::EmitterFusionKind::kPhasedBlockCluster
            ? RoundUpTo<int64_t>(nested_blocks, cluster_dim_x)
            : nested_blocks;
    int64_t upper_bound = block_offset + allocated_blocks;
    mlir::Value bound_val =
        mlir::arith::ConstantIntOp::create(b, upper_bound, 32);
    mlir::Value in_range = mlir::arith::CmpIOp::create(
        b, mlir::arith::CmpIPredicate::ult, bid, bound_val);

    bool has_else = (idx + 1 < nested_kernels_.size());
    auto if_op =
        mlir::scf::IfOp::create(b, in_range, /*withElseRegion=*/has_else);

    b.setInsertionPointToStart(&if_op.getThenRegion().front());

    // Emit the relative block ID for the nested kernel.
    mlir::Value offset_val =
        mlir::arith::ConstantIntOp::create(b, block_offset, 32);
    mlir::Value rel_bid = mlir::arith::SubIOp::create(b, bid, offset_val);
    block_offset = upper_bound;

    // Gather arguments for the nested kernel call.
    llvm::SmallVector<mlir::Value> call_args;

    // Input arguments.
    for (const HloInstruction* operand : spec.fusion->operands()) {
      TF_RET_CHECK(operand->opcode() == HloOpcode::kParameter);
      call_args.push_back(arg_ptrs[operand->parameter_number()]);
    }

    // Output arguments.
    for (int64_t outer_root_idx : spec.outer_root_indices) {
      call_args.push_back(arg_ptrs[fusion.operand_count() + outer_root_idx]);
    }

    // Relative block ID.
    call_args.push_back(rel_bid);

    // For `PhasedBlockClusterFusion`, all `cluster_dim_x` CTAs and all
    // `max_threads` threads must enter the nested function so they execute its
    // aligned `nvvm.cluster.arrive` / `nvvm.cluster.wait` barriers; individual
    // phases inside `PhasedBlockClusterFusion` guard `bid` and `tid`.
    if (spec.emitter_fusion_kind !=
        HloFusionAnalysis::EmitterFusionKind::kPhasedBlockCluster) {
      // If the nested kernel has fewer threads per block than the launcher
      // kernel, we only need to launch the threads that are in range.
      int64_t nested_threads = spec.launch_dims.num_threads_per_block();
      if (nested_threads < max_threads) {
        mlir::Value max_nested_tid =
            mlir::arith::ConstantIntOp::create(b, nested_threads, 32);
        mlir::Value tid_active = mlir::arith::CmpIOp::create(
            b, mlir::arith::CmpIPredicate::ult, tid, max_nested_tid);
        auto tid_if =
            mlir::scf::IfOp::create(b, tid_active, /*withElseRegion=*/false);
        b.setInsertionPointToStart(&tid_if.getThenRegion().front());
      }
    }

    // Emit the nested kernel call.
    mlir::LLVM::CallOp::create(b, nested_decls[idx], call_args);

    if (has_else) {
      b.setInsertionPointToStart(&if_op.getElseRegion().front());
    }
  }

  b.setInsertionPointToEnd(entry_block);
  mlir::func::ReturnOp::create(
      b, entry_function.getArguments().drop_front(fusion.operand_count()));
  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
ThreadBlockSpecializedFusion::CreateAndLowerMLIRModule(
    mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
    absl::string_view entry_function_name,
    const BufferAssignment* buffer_assignment) const {
  mlir_context.appendDialectRegistry(MlirKernelEmitter::GetDialectRegistry());
  mlir_context.loadAllAvailableDialects();

  bool should_verify = (fusion.GetModule()
                            ->config()
                            .debug_options()
                            .xla_gpu_llvm_verification_level() >= 1);
  if constexpr (tsl::kIsDebugBuild) {
    should_verify = true;
  }

  int64_t max_shmem_bytes = 0;
  int64_t warp_size = device_info_.threads_per_warp();
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> nested_modules;
  std::vector<std::string> nested_fn_names;
  nested_modules.reserve(nested_kernels_.size());
  nested_fn_names.reserve(nested_kernels_.size());

  for (size_t i = 0; i < nested_kernels_.size(); ++i) {
    const NestedKernelSpec& nested_spec = nested_kernels_[i];
    std::string nested_fn_name =
        GetNestedKernelFunctionName(entry_function_name, i);
    nested_fn_names.push_back(nested_fn_name);

    HloFusionAnalysis nested_analysis =
        HloFusionAnalysis::Create(*nested_spec.fusion, device_info_);
    mlir::OwningOpRef<mlir::ModuleOp> nested_module;

    if (nested_spec.emitter_fusion_kind ==
        HloFusionAnalysis::EmitterFusionKind::kPhasedBlockCluster) {
      LaunchDimensions outer_cluster_launch_dims(
          cluster_dim_->x, launch_dimensions_.num_threads_per_block());
      PhasedBlockClusterFusion nested_emitter(nested_analysis,
                                              outer_cluster_launch_dims);
      ABSL_ASSIGN_OR_RETURN(nested_module,
                       nested_emitter.CreateAndLowerMLIRModule(
                           mlir_context, *nested_spec.fusion, nested_fn_name,
                           /*buffer_assignment=*/nullptr));
    } else {
      std::unique_ptr<MlirKernelEmitter> nested_emitter =
          CreateNativeNestedEmitter(nested_analysis);
      ABSL_ASSIGN_OR_RETURN(MlirKernelSource nested_source,
                       nested_emitter->Emit(&mlir_context, *nested_spec.fusion,
                                            nested_fn_name,
                                            /*buffer_assignment=*/nullptr));
      ABSL_ASSIGN_OR_RETURN(nested_source,
                       CompileMlirToLlvmDialect(
                           device_info_, *fusion.GetModule(), nested_fn_name,
                           nested_emitter->unroll_factor(), mlir_context,
                           std::move(nested_source)));
      nested_module = std::move(nested_source).TakeModule();
    }

    VirtualizeSubKernelPassOptions virtualize_options;
    virtualize_options.entry_function_name_ = nested_fn_name;
    virtualize_options.expected_num_tensor_args_ =
        nested_spec.fusion->operand_count() +
        nested_spec.outer_root_indices.size();
    virtualize_options.max_threads_ =
        launch_dimensions_.num_threads_per_block();
    virtualize_options.warp_size_ = warp_size;
    const se::BlockDim& block_counts = nested_spec.launch_dims.block_counts();
    virtualize_options.block_dim_x_ = block_counts.x;
    virtualize_options.block_dim_y_ = block_counts.y;
    virtualize_options.block_dim_z_ = block_counts.z;
    const se::ThreadDim& thread_counts =
        nested_spec.launch_dims.thread_counts_per_block();
    virtualize_options.thread_dim_x_ = thread_counts.x;
    virtualize_options.thread_dim_y_ = thread_counts.y;
    virtualize_options.thread_dim_z_ = thread_counts.z;

    mlir::PassManager virtualize_pm(&mlir_context);
    virtualize_pm.enableVerifier(should_verify);
    virtualize_pm.addPass(createVirtualizeSubKernelPass(virtualize_options));
    tsl::StatusScopedDiagnosticHandler diagnostic_handler(
        nested_module->getContext());
    ABSL_RETURN_IF_ERROR(
        diagnostic_handler.consumeStatus(virtualize_pm.run(*nested_module)));

    if (auto shmem_attr =
            nested_module->getOperation()->getAttrOfType<mlir::IntegerAttr>(
                "xla.shared_memory_bytes")) {
      max_shmem_bytes = std::max(max_shmem_bytes, shmem_attr.getInt());
    }

    nested_modules.push_back(std::move(nested_module));
  }

  mlir::OpBuilder builder(&mlir_context);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(fusion.name()));
  mlir::OwningOpRef<mlir::ModuleOp> launcher_module =
      llvm_ir::CreateMlirModuleOp(loc);

  ABSL_ASSIGN_OR_RETURN(mlir::func::FuncOp entry_func,
                   emitters::EmitKernelApi(
                       *launcher_module, fusion, buffer_assignment,
                       GetDefaultBufferAlignment(), entry_function_name));
  // Intermediate output buffers of `PhasedBlockClusterFusion` are written by
  // sibling CTAs before `barrier.cluster` and read in the next phase; marking
  // them `xla.aliased` tells `MergePointersToSameSlicePass` to omit
  // `llvm.noalias` on those parameters (and any aliased parameters sharing the
  // same `xla.slice_index`) so LLVM `BasicAA` cannot reorder accesses across
  // the barrier after inlining.
  for (const NestedKernelSpec& spec : nested_kernels_) {
    if (spec.emitter_fusion_kind !=
        HloFusionAnalysis::EmitterFusionKind::kPhasedBlockCluster) {
      continue;
    }
    for (size_t j = 0; j + 1 < spec.outer_root_indices.size(); ++j) {
      entry_func.setArgAttr(fusion.operand_count() + spec.outer_root_indices[j],
                            emitters::kXlaAliasedAttr, builder.getUnitAttr());
    }
  }
  SetBackendKind(&mlir_context, entry_func, BackendKind::kGpu);
  emitters::SetIndexDataLayout(*launcher_module, fusion);
  ABSL_RETURN_IF_ERROR(
      EmitLauncherBody(*launcher_module, entry_func, fusion, nested_modules));
  ABSL_RETURN_IF_ERROR(LowerLauncherModuleToLlvmDialect(
      *launcher_module, device_info_, should_verify));

  if (cluster_dim_.has_value()) {
    auto launcher_entry = launcher_module->lookupSymbol<mlir::LLVM::LLVMFuncOp>(
        entry_function_name);
    TF_RET_CHECK(launcher_entry != nullptr);
    launcher_entry->setAttr(
        mlir::NVVM::NVVMDialect::getClusterDimAttrName(),
        builder.getDenseI32ArrayAttr({static_cast<int32_t>(cluster_dim_->x),
                                      static_cast<int32_t>(cluster_dim_->y),
                                      static_cast<int32_t>(cluster_dim_->z)}));
  }

  for (size_t i = 0; i < nested_kernels_.size(); ++i) {
    ABSL_RETURN_IF_ERROR(SpliceNestedKernelModuleIntoLauncher(
        *launcher_module, *nested_modules[i], nested_fn_names[i]));
  }

  (*launcher_module)
      ->setAttr("xla.shared_memory_bytes",
                builder.getI64IntegerAttr(max_shmem_bytes));

  return launcher_module;
}

absl::StatusOr<ThreadBlockSpecializedFusion::LlvmKernelSourceWrapper>
ThreadBlockSpecializedFusion::CreateLLVMModule(
    const se::DeviceDescription& device, const HloFusionInstruction& fusion,
    absl::string_view entry_function_name,
    const BufferAssignment* buffer_assignment,
    mlir::MLIRContext& mlir_context) const {
  ABSL_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      CreateAndLowerMLIRModule(mlir_context, fusion, entry_function_name,
                               buffer_assignment));
  int64_t max_shmem_bytes =
      (*module)
          ->getAttrOfType<mlir::IntegerAttr>("xla.shared_memory_bytes")
          .getInt();

  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> llvm_module =
      mlir::translateModuleToLLVMIR(*module, *llvm_context);
  TF_RET_CHECK(llvm_module != nullptr)
      << "Failed to translate thread-block-specialized fusion MLIR module to "
         "LLVM IR.";

  return LlvmKernelSourceWrapper{
      LlvmKernelSource(std::move(llvm_context), std::move(llvm_module)),
      max_shmem_bytes};
}

AsyncThunkSequence ThreadBlockSpecializedFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  VLOG(4) << "ThreadBlockSpecializedFusion: "
          << fusion.fused_instructions_computation()->ToString();
  if (cluster_dim_.has_value()) {
    const auto* cuda_cc =
        ir_emitter_context.gpu_compute_capability().cuda_compute_capability();
    if (cuda_cc == nullptr || !cuda_cc->IsAtLeastHopper()) {
      return absl::UnimplementedError(absl::StrCat(
          "PhasedBlockClusterFusion requires CUDA Hopper (SM90+), got: ",
          ir_emitter_context.gpu_compute_capability().ToString()));
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      emitters::KernelArguments args,
      emitters::KernelArguments::Create(ir_emitter_context.buffer_assignment(),
                                        GetDefaultBufferAlignment(), &fusion));

  auto [future_entry, cached] = ir_emitter_context.kernel_cache().GetWithStatus(
      fusion.fused_instructions_computation(), args.args(),
      /*discriminator=*/"ThreadBlockSpecializedFusion",
      [&]() -> xla::Future<KernelReuseCache::Entry> {
        std::string kernel_name = ir_emitter_context.GetSanitizedUniqueName(
            std::string(fusion.name()));
        BorrowedMlirContext borrowed_context =
            ir_emitter_context.BorrowMlirContext();
        ABSL_ASSIGN_OR_RETURN(
            LlvmKernelSourceWrapper llvm_source_wrapper,
            CreateLLVMModule(
                ir_emitter_context.gpu_device_info(), fusion, kernel_name,
                &ir_emitter_context.buffer_assignment(), **borrowed_context));

        llvm::orc::ThreadSafeModule safe_module =
            std::move(llvm_source_wrapper.source).thread_safe_module();
        LaunchDimensions launch_dims = launch_dimensions();
        ABSL_RETURN_IF_ERROR(AnnotateMlirKernelLlvmModule(
            ir_emitter_context.gpu_device_info(), launch_dims, kernel_name,
            ir_emitter_context.data_layout(),
            ir_emitter_context.target_triple(),
            safe_module.getModuleUnlocked()));

        bool use_pdl =
            IsPdlEnabled(fusion.GetModule()->config().debug_options(),
                         ir_emitter_context.gpu_compute_capability());

        GpuKernelSpec spec{std::move(kernel_name), std::move(launch_dims), args,
                           std::nullopt, use_pdl};
        return ir_emitter_context.kernel_compiler()
            ->CompileToTargetBinary(LlvmKernelSource(std::move(safe_module)),
                                    spec)
            .Map([spec = std::move(spec), cluster_dim = cluster_dim_,
                  max_shmem_bytes = llvm_source_wrapper.max_shmem_bytes](
                     const std::vector<uint8_t>& cubin) mutable {
              KernelReuseCache::Entry entry{
                  std::string{spec.name()}, spec.launch_dimensions(),
                  cluster_dim, max_shmem_bytes,
                  std::make_shared<const std::vector<uint8_t>>(cubin)};
              entry.use_pdl = spec.use_pdl();
              return entry;
            });
      });

  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(
      &fusion, ir_emitter_context.GetNextThunkId());
  bool kernel_cached = cached;
  return future_entry.Map(
      [&fusion, thunk_info = std::move(thunk_info), args = std::move(args),
       kernel_cached,
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

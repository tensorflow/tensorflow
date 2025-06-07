/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/backends/gpu/codegen/emitters/emitter_base.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/mlir/tools/mlir_replay/public/compiler_trace.pb.h"
#include "xla/mlir/tools/mlir_replay/public/compiler_trace_instrumentation.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/llvm_gpu_backend/ptx_version_util.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using llvm::SmallVector;
using mlir::Value;
using mlir::ValueRange;
using mlir::func::FuncOp;

void AddRanges(llvm::Function* func, const LaunchDimensions& launch_dims,
               llvm::Module* module) {
  for (auto& block : *func) {
    for (auto& instr : block) {
      if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instr)) {
        if (auto* callee = call->getCalledFunction()) {
          switch (callee->getIntrinsicID()) {
            case llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x:
              llvm_ir::AddRangeMetadata(
                  0, launch_dims.thread_counts_per_block().x, call, module);
              break;
            case llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y:
              llvm_ir::AddRangeMetadata(
                  0, launch_dims.thread_counts_per_block().y, call, module);
              break;
            case llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z:
              llvm_ir::AddRangeMetadata(
                  0, launch_dims.thread_counts_per_block().z, call, module);
              break;
            case llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
              llvm_ir::AddRangeMetadata(0, launch_dims.block_counts().x, call,
                                        module);
              break;
            case llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
              llvm_ir::AddRangeMetadata(0, launch_dims.block_counts().y, call,
                                        module);
              break;
            case llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
              llvm_ir::AddRangeMetadata(0, launch_dims.block_counts().z, call,
                                        module);
              break;
          }
        }
      }
    }
  }
}

absl::Status RunPassPipeline(mlir::ModuleOp module, const HloModule& hlo_module,
                             mlir::PassManager& pm,
                             absl::string_view entry_function_name) {
  bool should_dump_mlir_passes =
      DumpingEnabledForHloModule(hlo_module) &&
      DumpingEnabledForHloPass("mlir-fusion-emitter",
                               hlo_module.config().debug_options());

  std::string mlir_passes_dump_result;
  llvm::raw_string_ostream log_stream(mlir_passes_dump_result);
  mlir::interpreter::MlirCompilationTrace trace;

  if (should_dump_mlir_passes) {
    module.getContext()->disableMultithreading();

    auto print_always = [](mlir::Pass*, mlir::Operation*) { return true; };
    pm.enableIRPrinting(/*shouldPrintBeforePass=*/print_always,
                        /*shouldPrintAfterPass=*/print_always,
                        /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/false,
                        /*printAfterOnlyOnFailure=*/true, log_stream,
                        /*opPrintingFlags=*/{});
    pm.printAsTextualPipeline(log_stream);
    log_stream.write("\n\n", 2);

    pm.addInstrumentation(
        std::make_unique<mlir::interpreter::MlirCompilerTraceInstrumentation>(
            trace));
  }

  tsl::StatusScopedDiagnosticHandler diagnostic_handler(module.getContext());
  (void)pm.run(module);

  if (should_dump_mlir_passes) {
    DumpPerModuleProtobufToFile(
        hlo_module, trace, hlo_module.config().debug_options(),
        absl::StrCat(entry_function_name, ".mlir-trace"));

    DumpToFileInDirOrStdout(
        hlo_module, "", absl::StrCat(entry_function_name, ".mlir-passes.log"),
        mlir_passes_dump_result);
  }

  return diagnostic_handler.consumeStatus();
}

}  // namespace

Value EmitterBase::EmitWorkGroupId(mlir::ImplicitLocOpBuilder& builder,
                                   WorkGroupDimension dim) const {
  const auto& counts = launch_dimensions().block_counts();
  int64_t count = dim == WorkGroupDimension::x   ? counts.x
                  : dim == WorkGroupDimension::y ? counts.y
                                                 : counts.z;
  auto block_id = builder.create<WorkGroupIdOp>(dim);
  block_id->setAttr("xla.range", builder.getIndexArrayAttr({0, count - 1}));
  return block_id;
}

Value EmitterBase::EmitBlockId(mlir::ImplicitLocOpBuilder& builder,
                               int dim) const {
  const auto& counts = launch_dimensions().block_counts();
  int64_t count = dim == 0 ? counts.x : dim == 1 ? counts.y : counts.z;
  auto block_id = builder.create<mlir::gpu::BlockIdOp>(
      static_cast<mlir::gpu::Dimension>(dim));
  block_id->setAttr("xla.range", builder.getIndexArrayAttr({0, count - 1}));
  return block_id;
}

Value EmitterBase::EmitThreadId(mlir::ImplicitLocOpBuilder& builder,
                                int dim) const {
  const auto& counts = launch_dimensions().thread_counts_per_block();
  int64_t count = dim == 0 ? counts.x : dim == 1 ? counts.y : counts.z;
  auto thread_id = builder.create<mlir::gpu::ThreadIdOp>(
      static_cast<mlir::gpu::Dimension>(dim));
  thread_id->setAttr("xla.range", builder.getIndexArrayAttr({0, count - 1}));
  return thread_id;
}

llvm::SmallVector<Value> EmitterBase::EmitThreadAndBlockIds(
    mlir::ImplicitLocOpBuilder& builder) const {
  auto& b = builder;
  return {EmitThreadId(b, 0), EmitThreadId(b, 1), EmitThreadId(b, 2),
          EmitBlockId(b, 0),  EmitBlockId(b, 1),  EmitBlockId(b, 2)};
}

absl::StatusOr<FusionEmissionResult> EmitterBase::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  VLOG(4) << "Fusion: " << fusion.fused_instructions_computation()->ToString();
  TF_ASSIGN_OR_RETURN(auto args, emitters::KernelArguments::Create(
                                     ir_emitter_context.buffer_assignment(),
                                     GetDefaultBufferAlignment(), &fusion));
  auto launch_dims = launch_dimensions();
  auto [status_or_entry, cached] =
      ir_emitter_context.kernel_cache().GetWithStatus(
          fusion.fused_instructions_computation(), args.args(),
          /*discriminator=*/"",
          [&]() -> absl::StatusOr<KernelReuseCache::Entry> {
            std::string kernel_name =
                ir_emitter_context.name_uniquer()->GetUniqueName(
                    llvm_ir::SanitizeFunctionName(std::string(fusion.name())));
            if (ir_emitter_context.emit_kernels()) {
              TF_ASSIGN_OR_RETURN(
                  auto module,
                  CreateLLVMModule(
                      *ir_emitter_context.mlir_context(),
                      ir_emitter_context.llvm_module()->getContext(),
                      ir_emitter_context.gpu_device_info(), fusion, kernel_name,
                      &ir_emitter_context.buffer_assignment()));
              auto* kernel_func = module->getFunction(kernel_name);
              AddRanges(kernel_func, launch_dims, module.get());

              auto* target = ir_emitter_context.llvm_module();
              module->setDataLayout(target->getDataLayout());
              module->setTargetTriple(target->getTargetTriple());

              llvm::IRBuilder<> builder(module->getContext());
              AnnotateFunctionAsGpuKernel(module.get(), kernel_func, &builder);
              TF_RETURN_IF_ERROR(AnnotateKernelLaunchDimensions(
                  ir_emitter_context.gpu_device_info(), launch_dims,
                  kernel_func, module.get()));

              // Use override flag because libdevice functions can be present in
              // both.
              CHECK(!llvm::Linker::linkModules(
                  *target, std::move(module),
                  llvm::Linker::Flags::OverrideFromSrc));
            } else {
              VLOG(3) << "Skipped kernel compilation.";
            }

            return KernelReuseCache::Entry{kernel_name, launch_dims,
                                           std::nullopt,
                                           /*shmem_bytes=*/0};
          });
  TF_ASSIGN_OR_RETURN(const KernelReuseCache::Entry* entry, status_or_entry);

  if (cached) {
    VLOG(3) << "Reuse: " << fusion.name() << " -> " << entry->kernel_name;
  }

  FusionEmissionResult result;
  result.thunks.emplace_back(std::make_unique<KernelThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(&fusion), entry->kernel_name,
      args.args(), launch_dims, entry->cluster_dim, entry->shmem_bytes));
  return result;
}

absl::StatusOr<std::unique_ptr<llvm::Module>> EmitterBase::CreateLLVMModule(
    mlir::MLIRContext& mlir_context, llvm::LLVMContext& llvm_context,
    const se::DeviceDescription& device, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment* buffer_assignment) const {
  mlir_context.appendDialectRegistry(GetDialectRegistry());
  mlir_context.loadAllAvailableDialects();
  TF_ASSIGN_OR_RETURN(
      auto module, CreateMLIRModule(mlir_context, fusion, entry_function_name,
                                    buffer_assignment));

  mlir::PassManager pm(&mlir_context);
  AddXlaGpuOpsOptimizationPasses(pm);
  AddLoopTransformationPasses(pm, device);
  AddLoweringPasses(pm, device);

  auto pipeline_status = RunPassPipeline(module.get(), *fusion.GetModule(), pm,
                                         entry_function_name);

  auto llvm_module = mlir::translateModuleToLLVMIR(module.get(), llvm_context);
  TF_RET_CHECK(llvm_module != nullptr)
      << "Failed to translate module to LLVM IR.";

  return llvm_module;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitterBase::CreateMLIRModule(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment* buffer_assignment) const {
  mlir::OpBuilder builder(&context);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(fusion.name()));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);

  TF_ASSIGN_OR_RETURN(mlir::func::FuncOp entry_func,
                      emitters::EmitKernelApi(
                          *module, fusion, buffer_assignment,
                          GetDefaultBufferAlignment(), entry_function_name));
  SetBackendKind(&context, entry_func, BackendKind::kGpu);

  TF_RETURN_IF_ERROR(EmitMlir(module.get(), entry_func, fusion));
  return module;
}

emitters::EpilogueSpecification EmitterBase::GetEpilogueForOutputIndexing(
    const HloFusionAnalysis& analysis,
    const std::vector<const HloInstruction*>& heroes,
    const std::vector<const HloInstruction*>& roots,
    mlir::MLIRContext* mlir_context) const {
  emitters::EpilogueSpecification result;

  absl::flat_hash_map<const HloInstruction*, const HloInstruction*>
      root_to_hero;
  for (auto [root, hero] :
       llvm::zip(analysis.fusion_roots(), analysis.fusion_heroes())) {
    root_to_hero[&root.instruction()] = &hero.instruction();
  }
  absl::flat_hash_map<const HloInstruction*, int> root_to_index;
  for (auto [index, root] : llvm::enumerate(analysis.fusion_roots())) {
    root_to_index[&root.instruction()] = root_to_index.size();
  }

  result.root_indexing.reserve(roots.size());
  for (auto* root : roots) {
    auto indexing =
        ComputeThreadIdToOutputIndexing(root_to_index[root], mlir_context);
    if (result.index_ranges.empty()) {
      result.index_ranges.reserve(indexing->GetDimensionCount() +
                                  indexing->GetSymbolCount());
      for (const auto& dim : indexing->GetDimensionBounds()) {
        result.index_ranges.push_back(dim.upper + 1);
      }
      for (const auto& sym : indexing->GetSymbolBounds()) {
        result.index_ranges.push_back(sym.upper + 1);
      }
    }
    auto* hero = root_to_hero[root];
    auto epilogue_indexing = ComputeEpilogueInputToOutputIndexing(
        {*hero, &analysis.fusion()}, {*root, &analysis.fusion()}, mlir_context);
    result.root_indexing.push_back(
        ComposeIndexingMaps(*indexing, epilogue_indexing));
  }
  result.heroes = heroes;
  result.roots = roots;
  return result;
}

mlir::DialectRegistry EmitterBase::GetDialectRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<
      mlir::DLTIDialect, mlir::NVVM::NVVMDialect, mlir::ROCDL::ROCDLDialect,
      mlir::affine::AffineDialect, mlir::arith::ArithDialect,
      mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
      mlir::gpu::GPUDialect, mlir::math::MathDialect, mlir::mhlo::MhloDialect,
      mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
      mlir::vector::VectorDialect, xla::XlaDialect, xla::gpu::XlaGpuDialect>();
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::func::registerInlinerExtension(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  return registry;
}

absl::Status EmitterBase::EmitMlir(mlir::ModuleOp module, FuncOp entry_function,
                                   const HloFusionInstruction& fusion) const {
  std::vector<emitters::EpilogueSpecification> epilogues =
      GetEpilogues(fusion, module->getContext());
  emitters::PartitionedComputations computations(
      fusion.fused_instructions_computation(), module->getContext(), epilogues);

  TF_ASSIGN_OR_RETURN(auto call_targets, emitters::EmitPartitionedComputations(
                                             module, computations));

  emitters::SetIndexDataLayout(module, fusion);

  return EmitEntryFunction(computations, call_targets, entry_function, fusion);
}

absl::flat_hash_map<const HloInstruction*, ValueRange>
EmitterBase::EmitEpilogue(
    int epilogue_index, const emitters::PartitionedComputations& computations,
    FuncOp entry_fn,
    const absl::flat_hash_map<const HloInstruction*, llvm::SmallVector<Value>>&
        injected,
    ValueRange output_indices, mlir::ImplicitLocOpBuilder& builder) const {
  const auto& epilogue = computations.epilogues().at(epilogue_index);
  if (epilogue.roots.empty()) {
    return {};
  }
  auto epilogue_fn = mlir::cast<FuncOp>(
      entry_fn->getParentOfType<mlir::ModuleOp>().lookupSymbol(epilogue.name));
  SmallVector<Value> operands = ValueRange(entry_fn.getArguments().take_front(
      computations.fusion()->num_parameters()));
  absl::c_copy(output_indices, std::back_inserter(operands));
  int injected_offset = operands.size();
  operands.resize(injected_offset + epilogue.num_injected_values);
  for (auto [injected_instruction, start] : epilogue.injected_value_starts) {
    absl::c_copy(injected.at(injected_instruction),
                 operands.begin() + injected_offset + start);
  }

  ValueRange results =
      builder.create<PureCallOp>(epilogue_fn, operands).getResults();
  absl::flat_hash_map<const HloInstruction*, ValueRange> results_per_root;
  for (auto* root : epilogue.roots) {
    int arity =
        root->shape().IsTuple() ? root->shape().tuple_shapes().size() : 1;
    results_per_root[root] = results.take_front(arity);
    results = results.drop_front(arity);
  }
  CHECK_EQ(results.size(), 0);
  return results_per_root;
}

void AddXlaGpuOpsOptimizationPasses(mlir::OpPassManager& pm) {
  pm.addNestedPass<FuncOp>(emitters::CreateSimplifyArithPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(emitters::CreateEraseDeadFunctionsPass());
  pm.addPass(mlir::createCSEPass());
}

void AddLoopTransformationPasses(mlir::OpPassManager& pm,
                                 const se::DeviceDescription& device) {
  pm.addNestedPass<FuncOp>(CreateLowerXlaSharedPass());
  pm.addNestedPass<FuncOp>(
      emitters::CreateLowerXlaToScfPass(device.threads_per_warp()));
  pm.addNestedPass<FuncOp>(CreateFuseLoopsPass());
  pm.addPass(mlir::createInlinerPass({}, [&](mlir::OpPassManager& pm) {
    // CSE after inlining because inlining can introduce duplicates.
    pm.addPass(mlir::createCSEPass());
  }));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<FuncOp>(CreatePeelLoopsPass());
  pm.addNestedPass<FuncOp>(emitters::CreateLowerXlaLoopsToScfPass());
  pm.addPass(mlir::mhlo::createConvertToSignlessPass());
  pm.addPass(emitters::CreatePropagateSliceIndicesPass());
  pm.addPass(emitters::CreateFlattenTensorsPass());
  // We need LICM before unswitching loops, because our loop unswitcher only
  // detects for loops with a single if inside them.
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addNestedPass<FuncOp>(emitters::CreateUnswitchLoopsPass());
  // We need LICM again after unswitching, because that can introduce new
  // opportunities for LICM. This would not be necessary if LICM also moved
  // instructions over ifs.
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addNestedPass<FuncOp>(emitters::CreateVectorizeLoadsAndStoresPass(device));
  pm.addNestedPass<FuncOp>(CreateOptimizeLoopsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void AddLoweringPasses(mlir::OpPassManager& pm,
                       const se::DeviceDescription& device) {
  pm.addNestedPass<FuncOp>(emitters::CreateConvertPureCallOpsPass());
  pm.addPass(emitters::CreateLowerTensorsPass(device));
  pm.addPass(mlir::createConvertComplexToStandardPass());
  pm.addPass(emitters::CreateMergePointersToSameSlicePass());

  // LowerTensors creates new affine.apply ops. Fold and CSE them so
  // simplify-affine has maximally folded expressions to work with.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addNestedPass<FuncOp>(emitters::CreateSimplifyArithPass());
  pm.addPass(emitters::CreateSimplifyAffinePass());
  pm.addPass(CreateConvertIndexTypePass());
  // simplify-affine lowers most affine.apply ops, but if it can't prove a
  // division or modulo is unsigned, affine.apply ops will remain.
  pm.addPass(mlir::createLowerAffinePass());

  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCSEPass());

  // This pass has to run before `ExpandFloatOpsPass`.
  if (auto* cc = std::get_if<se::CudaComputeCapability>(
          &device.gpu_compute_capability())) {
    se::SemanticVersion ptx_version =
        nvptx::DetermineHighestSupportedPtxVersionFromCudaVersion(
            device.runtime_version());

    // FP8 conversion intrinsics are available on sm89 since ptx 8.1
    // Older ptx versions only support FP8 conversion for sm90
    if ((ptx_version >= se::SemanticVersion(8, 1, 0) && cc->IsAtLeast(8, 9)) ||
        (ptx_version >= se::SemanticVersion(7, 8, 0) && cc->IsAtLeast(9, 0))) {
      pm.addPass(CreateConvertFloatNvidiaPass());
    }
  } else if (auto* cc = std::get_if<se::RocmComputeCapability>(
                 &device.gpu_compute_capability())) {
    if (cc->has_fp8_support()) {
      pm.addPass(CreateConvertFloatAMDPass(*cc));
    }
    pm.addPass(CreateRecoverExp2Pass());
  }

  pm.addPass(emitters::CreateExpandFloatOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(emitters::CreateLowerToLLVMPass(device));
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

}  // namespace gpu
}  // namespace xla

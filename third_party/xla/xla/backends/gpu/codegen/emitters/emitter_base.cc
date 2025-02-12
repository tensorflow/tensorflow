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
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Casting.h"
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
#include "mlir/IR/Types.h"
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
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/mlir/tools/mlir_replay/public/compiler_trace.pb.h"
#include "xla/mlir/tools/mlir_replay/public/compiler_trace_instrumentation.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/framework/mlir/status_scoped_diagnostic_handler.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

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

bool Needs64Bits(const Shape& shape) {
  return shape.IsArray() ? !IsInt32(ShapeUtil::ElementsIn(shape))
                         : absl::c_any_of(shape.tuple_shapes(), Needs64Bits);
}

bool Is64BitIndex(const HloInstruction* instr, int operand) {
  const auto& shape = instr->operand(operand)->shape();
  return shape.element_type() == PrimitiveType::S64 ||
         shape.element_type() == PrimitiveType::U64;
}

bool Needs64BitIndices(const HloComputation* computation) {
  for (auto* instr : computation->instructions()) {
    // Check if any HLO instructions directly take 64 bit indices as operands.
    switch (instr->opcode()) {
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice:
        for (int i = 1; i < instr->operand_count(); ++i) {
          if (Is64BitIndex(instr, i)) return true;
        }
        break;
      case HloOpcode::kGather:
      case HloOpcode::kScatter:
        CHECK(instr->shape().IsArray()) << "Variadic scatter is unsupported.";
        if (Is64BitIndex(instr, 1)) return true;
        break;
      default:
        break;
    }

    if (Needs64Bits(instr->shape()) ||
        absl::c_any_of(instr->called_computations(), Needs64BitIndices)) {
      return true;
    }
  }
  return false;
}

}  // namespace

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
  TF_ASSIGN_OR_RETURN(
      auto args,
      KernelArguments::Create(ir_emitter_context.buffer_assignment(), &fusion));
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
                  kernel_name, module.get()));

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
      &fusion, entry->kernel_name, args.args(), launch_dims, entry->cluster_dim,
      entry->shmem_bytes));
  return result;
}

absl::StatusOr<std::unique_ptr<llvm::Module>> EmitterBase::CreateLLVMModule(
    mlir::MLIRContext& mlir_context, llvm::LLVMContext& llvm_context,
    const se::DeviceDescription& device, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment* buffer_assignment) const {
  HloModule* hlo_module = fusion.GetModule();
  std::unique_ptr<mlir::interpreter::MlirCompilationTrace> trace = nullptr;
  if (DumpingEnabledForHloModule(*hlo_module) &&
      DumpingEnabledForHloPass("mlir-fusion-emitter",
                               hlo_module->config().debug_options())) {
    trace = std::make_unique<mlir::interpreter::MlirCompilationTrace>();
  }
  TF_ASSIGN_OR_RETURN(
      auto module, CreateMLIRModule(mlir_context, fusion, entry_function_name,
                                    buffer_assignment));

  mlir::PassManager pm(&mlir_context);
  AddXlaGpuOpsOptimizationPasses(pm);
  AddLoopTransformationPasses(pm, device);
  AddLoweringPasses(pm, device);
  auto pipeline_status = RunPassPipeline(module.get(), pm, trace.get());
  if (trace) {
    DumpPerModuleProtobufToFile(
        *hlo_module, *trace, hlo_module->config().debug_options(),
        absl::StrCat(entry_function_name, ".mlir-trace"));
  }
  TF_RETURN_IF_ERROR(pipeline_status);

  auto llvm_module = mlir::translateModuleToLLVMIR(module.get(), llvm_context);
  TF_RET_CHECK(llvm_module != nullptr)
      << "Failed to translate module to LLVM IR.";

  return llvm_module;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitterBase::CreateMLIRModule(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment* buffer_assignment,
    mlir::interpreter::MlirCompilationTrace* trace) const {
  context.loadDialect<
      mlir::DLTIDialect, mlir::NVVM::NVVMDialect, mlir::ROCDL::ROCDLDialect,
      mlir::affine::AffineDialect, mlir::arith::ArithDialect,
      mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
      mlir::gpu::GPUDialect, mlir::math::MathDialect, mlir::mhlo::MhloDialect,
      mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
      mlir::vector::VectorDialect, xla::XlaDialect, xla::gpu::XlaGpuDialect>();
  mlir::DialectRegistry registry;
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::func::registerInlinerExtension(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  context.appendDialectRegistry(registry);

  mlir::OpBuilder builder(&context);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(fusion.name()));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);

  // Create the entry function.
  SmallVector<mlir::Type> param_types;
  std::optional<KernelArguments> args;
  if (buffer_assignment != nullptr) {
    TF_ASSIGN_OR_RETURN(args,
                        KernelArguments::Create(*buffer_assignment, &fusion));
  }
  // Annotate tensors with the buffer indices. This way, the buffer propagation
  // pass can clean them up later.
  int next_slice_index = 0;
  absl::flat_hash_map<BufferAllocation::Slice, std::optional<int>>
      slice_indices;
  auto get_arg_attrs = [&](int index) -> absl::StatusOr<mlir::Attribute> {
    if (!args) {
      return builder.getDictionaryAttr({builder.getNamedAttr(
          "xla.slice_index", builder.getIndexAttr(next_slice_index++))});
    }

    const auto& arg = args->args()[index];
    SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr(
        "xla.slice_index", builder.getIndexAttr(arg.llvm_arg_index())));
    attrs.push_back(
        builder.getNamedAttr(mlir::LLVM::LLVMDialect::getAlignAttrName(),
                             builder.getIndexAttr(arg.alignment())));
    attrs.push_back(builder.getNamedAttr(
        mlir::LLVM::LLVMDialect::getDereferenceableAttrName(),
        builder.getIndexAttr(arg.slice().size())));
    if (!arg.written()) {
      attrs.push_back(
          builder.getNamedAttr("xla.invariant", builder.getUnitAttr()));
    }
    return builder.getDictionaryAttr(attrs);
  };

  SmallVector<mlir::Attribute> arg_attrs;
  int arg_index = 0;
  for (auto* param : fusion.operands()) {
    param_types.push_back(
        emitters::TensorShapeToMlirType(param->shape(), builder));
    TF_ASSIGN_OR_RETURN(arg_attrs.emplace_back(), get_arg_attrs(arg_index++));
  }

  auto result_types = emitters::ShapeToMlirTypes(fusion.shape(), builder);
  param_types.append(result_types.begin(), result_types.end());
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      fusion.shape(), [&](const auto& shape, const ShapeIndex& index) {
        if (shape.IsArray()) {
          TF_ASSIGN_OR_RETURN(arg_attrs.emplace_back(),
                              get_arg_attrs(arg_index++));
        }
        return absl::OkStatus();
      }));

  builder.setInsertionPointToStart(module->getBody());
  auto entry_func = builder.create<FuncOp>(
      loc, entry_function_name,
      mlir::FunctionType::get(&context, param_types, result_types),
      /*sym_visibility=*/mlir::StringAttr{},
      mlir::ArrayAttr::get(&context, arg_attrs),
      /*res_attrs=*/mlir::ArrayAttr{});
  entry_func->setAttr("xla.entry", mlir::UnitAttr::get(&context));
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

absl::Status EmitterBase::EmitMlir(mlir::ModuleOp module, FuncOp entry_function,
                                   const HloFusionInstruction& fusion) const {
  std::vector<emitters::EpilogueSpecification> epilogues =
      GetEpilogues(fusion, module->getContext());
  emitters::PartitionedComputations computations(
      fusion.fused_instructions_computation(), module->getContext(), epilogues);
  auto subgraph_to_mlir_fn = computations.DeclareFunctions(module);

  // Erase subgraphs for all heroes that aren't used anywhere else. This is
  // necessary because the instructions may not have elemental implementations
  // (scatter).
  for (const auto& epilogue : epilogues) {
    for (auto* custom : epilogue.heroes) {
      if (custom->user_count() == 0) {
        subgraph_to_mlir_fn.extract(&computations.FindSubgraph(custom))
            .mapped()
            .erase();
      }
    }
  }

  // The epilogue functions replace the root tuple.
  auto* root = fusion.fused_instructions_computation()->root_instruction();
  if (root->opcode() == HloOpcode::kTuple && !epilogues.empty()) {
    subgraph_to_mlir_fn.extract(&computations.FindSubgraph(root))
        .mapped()
        .erase();
  }

  auto call_targets =
      computations.CreateCallTargetProvider(subgraph_to_mlir_fn);
  for (const auto& comp : computations.partitioned_computations()) {
    for (const auto& subgraph : comp.subgraphs()) {
      if (subgraph_to_mlir_fn.contains(&subgraph)) {
        TF_RETURN_IF_ERROR(emitters::SubgraphToMlirFunction(
            comp, subgraph, subgraph_to_mlir_fn[&subgraph], call_targets));
      }
    }
  }
  for (const auto& epilogue : computations.epilogues()) {
    if (epilogue.roots.empty()) continue;
    TF_RETURN_IF_ERROR(emitters::SubgraphToMlirFunction(
        computations.FindPartitionedComputation(
            fusion.fused_instructions_computation()),
        epilogue, subgraph_to_mlir_fn[&epilogue], call_targets));
  }

  int index_bitwidth =
      Needs64BitIndices(fusion.fused_instructions_computation()) ? 64 : 32;
  mlir::OpBuilder b(module->getContext());
  auto index_layout = mlir::DataLayoutEntryAttr::get(
      b.getIndexType(), b.getI32IntegerAttr(index_bitwidth));
  module->setAttr(
      mlir::DLTIDialect::kDataLayoutAttrName,
      mlir::DataLayoutSpecAttr::get(module->getContext(), {index_layout}));

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

absl::Status EmitterBase::RunPassPipeline(
    mlir::ModuleOp module, mlir::PassManager& pm,
    mlir::interpreter::MlirCompilationTrace* trace) const {
  if (VLOG_IS_ON(5)) {
    module.getContext()->disableMultithreading();
    pm.enableIRPrinting();
  }
  if (trace) {
    module.getContext()->disableMultithreading();
    pm.addInstrumentation(
        std::make_unique<mlir::interpreter::MlirCompilerTraceInstrumentation>(
            *trace));
  }

  tsl::StatusScopedDiagnosticHandler diagnostic_handler(module.getContext());
  (void)pm.run(module);
  return diagnostic_handler.consumeStatus();
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
  pm.addNestedPass<FuncOp>(CreateVectorizeLoadsAndStoresPass(device));
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
  auto maybe_convert_fp8 = MaybeCreateConvertFloatNvidiaPass(device);
  if (maybe_convert_fp8.has_value()) {
    pm.addPass(std::move(*maybe_convert_fp8));
  }

  pm.addPass(emitters::CreateExpandFloatOpsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(emitters::CreateLowerToLLVMPass(device));
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

}  // namespace gpu
}  // namespace xla

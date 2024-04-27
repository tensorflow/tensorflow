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
#include "xla/service/gpu/fusions/mlir/mlir_fusion_emitter.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/DLTI/DLTI.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Interfaces/DataLayoutInterfaces.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/fusions/mlir/passes.h"
#include "xla/service/gpu/fusions/mlir/type_util.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/runtime/kernel_thunk.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using llvm::SmallVector;
using mlir::Value;
using mlir::ValueRange;

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

bool Needs64BitIndices(const HloComputation* computation) {
  for (auto* instr : computation->instructions()) {
    if (Needs64Bits(instr->shape()) ||
        absl::c_any_of(instr->called_computations(), Needs64BitIndices)) {
      return true;
    }
  }
  return false;
}

}  // namespace

Value MlirFusionEmitterBase::EmitBlockId(mlir::ImplicitLocOpBuilder& builder,
                                         int dim) const {
  const auto& counts = launch_dimensions().block_counts();
  int64_t count = dim == 0 ? counts.x : dim == 1 ? counts.y : counts.z;
  auto block_id = builder.create<mlir::gpu::BlockIdOp>(
      static_cast<mlir::gpu::Dimension>(dim));
  block_id->setAttr("xla.range", builder.getIndexArrayAttr({0, count - 1}));
  return block_id;
}

Value MlirFusionEmitterBase::EmitThreadId(mlir::ImplicitLocOpBuilder& builder,
                                          int dim) const {
  const auto& counts = launch_dimensions().thread_counts_per_block();
  int64_t count = dim == 0 ? counts.x : dim == 1 ? counts.y : counts.z;
  auto thread_id = builder.create<mlir::gpu::ThreadIdOp>(
      static_cast<mlir::gpu::Dimension>(dim));
  thread_id->setAttr("xla.range", builder.getIndexArrayAttr({0, count - 1}));
  return thread_id;
}

llvm::SmallVector<Value> MlirFusionEmitterBase::EmitThreadAndBlockIds(
    mlir::ImplicitLocOpBuilder& builder) const {
  auto& b = builder;
  return {EmitThreadId(b, 0), EmitThreadId(b, 1), EmitThreadId(b, 2),
          EmitBlockId(b, 0),  EmitBlockId(b, 1),  EmitBlockId(b, 2)};
}

absl::StatusOr<FusionEmissionResult> MlirFusionEmitterBase::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
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

absl::StatusOr<std::unique_ptr<llvm::Module>>
MlirFusionEmitterBase::CreateLLVMModule(
    mlir::MLIRContext& mlir_context, llvm::LLVMContext& llvm_context,
    const se::DeviceDescription& device, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment* buffer_assignment) const {
  bool is_amd = std::holds_alternative<se::RocmComputeCapability>(
      device.gpu_compute_capability());
  TF_RET_CHECK(!is_amd) << "Unsupported device type: " << device.name();
  TF_ASSIGN_OR_RETURN(
      auto module, CreateMLIRModule(mlir_context, fusion, entry_function_name,
                                    buffer_assignment));

  mlir::PassManager pm(&mlir_context);
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::mhlo::createConvertToSignlessPass());
  pm.addPass(CreatePropagateSliceIndicesPass());
  pm.addPass(CreateLowerFuncPass());
  pm.addPass(CreateLowerXlaGpuToScfPass());
  pm.addPass(CreateLowerTensorsPass(
      is_amd, is_amd ? device.rocm_compute_capability().gcn_arch_name()
                     : device.cuda_compute_capability().ToString()));
  pm.addPass(mlir::createConvertComplexToStandardPass());
  pm.addPass(CreateMergePointersToSameSlicePass());

  // LowerTensors creates new affine.apply ops. Fold and CSE them so
  // simplify-affine has maximally folded expressions to work with.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(CreateSimplifyAffinePass());
  // Replace comparisons that result in constant values (e.g. due to ranges not
  // overlapping). This pass must run after SimplifyAffinePass, since that
  // generates the range information.
  pm.addPass(CreateSimplifyArithPass());

  // simplify-affine lowers most affine.apply ops, but if it can't prove a
  // division or modulo is unsigned, affine.apply ops will remain.
  pm.addPass(mlir::createLowerAffinePass());

  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(CreateLowerTensorsPass());
  pm.addPass(CreateExpandFloatOpsPass(
      !device.cuda_compute_capability().IsAtLeastAmpere()));
  pm.addPass(CreateLowerToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (pm.run(module.get()).failed()) {
    std::string module_dump;
    llvm::raw_string_ostream os(module_dump);
    module->print(os);
    return absl::InternalError(absl::StrFormat(
        "Failed create LLVM module.\nHloFusionInstruction "
        "computation:\n%s\nMLIR module:\n%s",
        fusion.fused_instructions_computation()->ToString(), module_dump));
  }

  auto llvm_module = mlir::translateModuleToLLVMIR(module.get(), llvm_context);
  TF_RET_CHECK(llvm_module != nullptr)
      << "Failed to translate module to LLVM IR.";

  return llvm_module;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
MlirFusionEmitterBase::CreateMLIRModule(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment* buffer_assignment) const {
  context.loadDialect<mlir::DLTIDialect, mlir::tensor::TensorDialect,
                      mlir::func::FuncDialect, mlir::affine::AffineDialect,
                      mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                      mlir::math::MathDialect, mlir::scf::SCFDialect,
                      mlir::mhlo::MhloDialect, mlir::gpu::GPUDialect,
                      mlir::NVVM::NVVMDialect, xla::gpu::XlaGpuDialect>();
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
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
        mlir_converter::TensorShapeToMlirType(param->shape(), builder));
    TF_ASSIGN_OR_RETURN(arg_attrs.emplace_back(), get_arg_attrs(arg_index++));
  }

  auto result_types = mlir_converter::ShapeToMlirTypes(fusion.shape(), builder);
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
  auto entry_func = builder.create<mlir::func::FuncOp>(
      loc, entry_function_name,
      mlir::FunctionType::get(&context, param_types, result_types),
      /*sym_visibility=*/mlir::StringAttr{},
      mlir::ArrayAttr::get(&context, arg_attrs),
      /*res_attrs=*/mlir::ArrayAttr{});
  entry_func->setAttr("xla.entry", mlir::UnitAttr::get(&context));

  TF_RETURN_IF_ERROR(EmitMlir(module.get(), entry_func, fusion));

  // Run a minimal simplification pipeline.
  mlir::PassManager pm(&context);
  pm.addPass(CreateSimplifyArithPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (pm.run(module.get()).failed()) {
    std::string module_dump;
    llvm::raw_string_ostream os(module_dump);
    module->print(os);
    return absl::InternalError(absl::StrFormat(
        "Failed to simplify module.\nHloFusionInstruction "
        "computation:\n%s\nMLIR module:\n%s",
        fusion.fused_instructions_computation()->ToString(), module_dump));
  }

  return module;
}

SmallVector<Value> MlirFusionEmitterBase::EmitThreadLoopNest(
    mlir::ImplicitLocOpBuilder& b, ValueRange outputs,
    const IndexingMap& indexing_map,
    const std::function<
        SmallVector<Value>(ValueRange outputs_tensors, ValueRange dim_values,
                           ValueRange symbol_values)>& create_body) const {
  return mlir_converter::EmitLoopNest(b, EmitThreadAndBlockIds(b), outputs,
                                      indexing_map, create_body);
}

absl::Status MlirFusionEmitterBase::EmitMlir(
    mlir::ModuleOp module, mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  std::vector<mlir_converter::EpilogueSpecification> epilogues =
      GetEpilogues(fusion, module->getContext());
  mlir_converter::PartitionedComputations computations(
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
  if (!epilogues.empty() && root->opcode() == HloOpcode::kTuple) {
    subgraph_to_mlir_fn.extract(&computations.FindSubgraph(root))
        .mapped()
        .erase();
  }

  auto call_targets =
      computations.CreateCallTargetProvider(subgraph_to_mlir_fn);
  for (const auto& comp : computations.partitioned_computations()) {
    for (const auto& subgraph : comp.subgraphs()) {
      if (subgraph_to_mlir_fn.contains(&subgraph)) {
        TF_RETURN_IF_ERROR(mlir_converter::SubgraphToMlirFunction(
            comp, subgraph, subgraph_to_mlir_fn[&subgraph], call_targets));
      }
    }
  }
  for (const auto& epilogue : computations.epilogues()) {
    TF_RETURN_IF_ERROR(mlir_converter::SubgraphToMlirFunction(
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
MlirFusionEmitterBase::EmitEpilogue(
    int epilogue_index,
    const mlir_converter::PartitionedComputations& computations,
    mlir::func::FuncOp entry_fn,
    const absl::flat_hash_map<const HloInstruction*, llvm::SmallVector<Value>>&
        injected,
    ValueRange output_indices, mlir::ImplicitLocOpBuilder& builder) const {
  const auto& epilogue = computations.epilogues().at(epilogue_index);
  auto epilogue_fn = mlir::cast<mlir::func::FuncOp>(
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

}  // namespace gpu
}  // namespace xla

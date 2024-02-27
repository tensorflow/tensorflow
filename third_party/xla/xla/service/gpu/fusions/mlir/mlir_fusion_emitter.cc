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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Casting.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Arith/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
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
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/fusions/mlir/passes.h"
#include "xla/service/gpu/fusions/mlir/type_util.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/runtime/kernel_thunk.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace gpu {
namespace {

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

}  // namespace

mlir::Value MlirFusionEmitterBase::EmitBlockId(
    mlir::ImplicitLocOpBuilder& builder, int dim) const {
  const auto& counts = launch_dimensions().block_counts();
  int64_t count = dim == 0 ? counts.x : dim == 1 ? counts.y : counts.z;
  auto block_id = builder.create<mlir::gpu::BlockIdOp>(
      static_cast<mlir::gpu::Dimension>(dim));
  block_id->setAttr("xla.range", builder.getIndexArrayAttr({0, count - 1}));
  return block_id;
}

mlir::Value MlirFusionEmitterBase::EmitThreadId(
    mlir::ImplicitLocOpBuilder& builder, int dim) const {
  const auto& counts = launch_dimensions().thread_counts_per_block();
  int64_t count = dim == 0 ? counts.x : dim == 1 ? counts.y : counts.z;
  auto thread_id = builder.create<mlir::gpu::ThreadIdOp>(
      static_cast<mlir::gpu::Dimension>(dim));
  thread_id->setAttr("xla.range", builder.getIndexArrayAttr({0, count - 1}));
  return thread_id;
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
  TF_RET_CHECK(device.cuda_compute_capability().major >= 1)
      << "Unsupported device type: " << device.name();
  TF_ASSIGN_OR_RETURN(
      auto module, CreateMLIRModule(mlir_context, fusion, entry_function_name,
                                    buffer_assignment));

  mlir::PassManager pm(&mlir_context);
  // TODO(jreiffers): Proper inlining and CSE of function calls.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(CreatePropagateSliceIndicesPass());
  pm.addPass(CreateLowerFuncPass());
  pm.addPass(CreateLowerTensorsPass());
  pm.addPass(CreateMergePointersToSameSlicePass());

  // LowerTensors creates new affine.apply ops. Fold and CSE them so
  // simplify-affine has maximally folded expressions to work with.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(CreateSimplifyAffinePass());

  // simplify-affine lowers most affine.apply ops, but if it can't prove a
  // division or modulo is unsigned, affine.apply ops will remain.
  pm.addPass(mlir::createLowerAffinePass());

  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(CreateLowerTensorsPass());
  pm.addPass(CreateExpandFloatConversionsPass(
      !device.cuda_compute_capability().IsAtLeastAmpere()));
  pm.addPass(CreateLowerToLLVMPass());
  TF_RET_CHECK(pm.run(module.get()).succeeded());

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
  context.loadDialect<mlir::tensor::TensorDialect, mlir::func::FuncDialect,
                      mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                      mlir::math::MathDialect, mlir::scf::SCFDialect,
                      mlir::mhlo::MhloDialect, mlir::gpu::GPUDialect,
                      mlir::NVVM::NVVMDialect>();
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  context.appendDialectRegistry(registry);

  mlir::OpBuilder builder(&context);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(fusion.name()));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(loc);

  // Create the entry function.
  llvm::SmallVector<mlir::Type> param_types;
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
    llvm::SmallVector<mlir::NamedAttribute> attrs;
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

  llvm::SmallVector<mlir::Attribute> arg_attrs;
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
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  TF_RET_CHECK(pm.run(module.get()).succeeded());
  return module;
}

absl::StatusOr<llvm::SmallVector<mlir::Value>>
MlirFusionEmitterBase::EmitLoopNest(
    mlir::ImplicitLocOpBuilder& b, mlir::ValueRange output_tensors,
    const IndexingMap& indexing_map,
    const std::function<absl::StatusOr<llvm::SmallVector<mlir::Value>>(
        mlir::ValueRange outputs_tensors, mlir::ValueRange dim_values,
        mlir::ValueRange symbol_values)>& create_body) const {
  llvm::SmallVector<mlir::Value> map_dims{
      EmitThreadId(b, 0), EmitThreadId(b, 1), EmitThreadId(b, 2),
      EmitBlockId(b, 0),  EmitBlockId(b, 1),  EmitBlockId(b, 2)};
  llvm::SmallVector<mlir::Value> map_symbols;

  auto cst = [&](int64_t v) {
    return b.create<mlir::arith::ConstantOp>(b.getIndexAttr(v));
  };

  std::function<absl::StatusOr<llvm::SmallVector<mlir::Value>>(
      int, mlir::ValueRange)>
      make_loops;
  make_loops = [&](int i, mlir::ValueRange current_outputs)
      -> absl::StatusOr<llvm::SmallVector<mlir::Value>> {
    if (i < indexing_map.GetAffineMap().getNumSymbols()) {
      auto range = indexing_map.GetSymbolRange(i);
      auto for_op = b.create<mlir::scf::ForOp>(cst(range.lower_bound),
                                               cst(range.upper_bound + 1),
                                               cst(1), current_outputs);
      map_symbols.push_back(for_op.getInductionVar());
      b.setInsertionPointToStart(for_op.getBody());
      TF_ASSIGN_OR_RETURN(auto results,
                          make_loops(i + 1, for_op.getRegionIterArgs()));
      b.create<mlir::scf::YieldOp>(results);
      b.setInsertionPointAfter(for_op);
      return for_op.getResults();
    }
    auto is_in_bounds = mlir_converter::CheckConstraints(indexing_map, map_dims,
                                                         map_symbols, b);
    auto if_op = b.create<mlir::scf::IfOp>(mlir::TypeRange{current_outputs},
                                           is_in_bounds, true, true);
    b.setInsertionPointToStart(if_op.getBody(0));
    TF_ASSIGN_OR_RETURN(auto results,
                        create_body(current_outputs, map_dims, map_symbols));
    b.create<mlir::scf::YieldOp>(results);
    b.setInsertionPointToStart(if_op.getBody(1));
    b.create<mlir::scf::YieldOp>(current_outputs);
    b.setInsertionPointAfter(if_op);
    return if_op.getResults();
  };

  return make_loops(0, output_tensors);
}

}  // namespace gpu
}  // namespace xla

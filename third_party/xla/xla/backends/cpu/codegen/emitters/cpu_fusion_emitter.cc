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

#include "xla/backends/cpu/codegen/emitters/cpu_fusion_emitter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Linker/Linker.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_ops.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_types.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/mlir/tools/mlir_replay/public/compiler_trace.pb.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/dump.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {
namespace {

using llvm::SmallVector;
using mlir::func::FuncOp;

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
      case HloOpcode::kScatter: {
        int indices_operand_index = instr->operand_count() / 2;
        if (Is64BitIndex(instr, indices_operand_index)) return true;
        break;
      }
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

using mlir::AffineExpr;

IndexingMap GetDefaultIndexingMap(absl::Span<const int64_t> thread_tile_sizes,
                                  absl::Span<const int64_t> shape,
                                  mlir::MLIRContext* mlir_context) {
  CHECK_EQ(thread_tile_sizes.size(), shape.size())
      << "thread_tile_sizes and shape must have the same size";
  SmallVector<int64_t> thread_tile_counts;
  thread_tile_counts.reserve(thread_tile_sizes.size());
  for (auto [tile_size, dim_size] : llvm::zip(thread_tile_sizes, shape)) {
    thread_tile_counts.push_back(CeilDiv(dim_size, tile_size));
  }
  // Delinearize thread_expr w.r.t. number of thread tiles per dimension.
  auto thread_expr = mlir::getAffineDimExpr(0, mlir_context);
  SmallVector<AffineExpr, 4> thread_ids =
      DelinearizeInBoundsIndex(thread_expr, thread_tile_counts);
  SmallVector<AffineExpr, 4> result;
  result.reserve(thread_ids.size());
  auto linear_index = mlir::getAffineSymbolExpr(0, mlir_context);
  SmallVector<AffineExpr, 4> indices_in_tile =
      DelinearizeInBoundsIndex(linear_index, thread_tile_sizes);
  SmallVector<std::pair<AffineExpr, Interval>, 4> constraints;
  constraints.reserve(thread_ids.size());
  for (auto [tile_size, thread_id, index_in_tile, dim] :
       llvm::zip(thread_tile_sizes, thread_ids, indices_in_tile, shape)) {
    result.push_back(thread_id * tile_size + index_in_tile);
    constraints.push_back(std::make_pair(result.back(), Interval{0, dim - 1}));
  }
  int64_t num_threads = Product(thread_tile_counts);
  int64_t num_tile_elements = Product(thread_tile_sizes);

  auto affine_map = mlir::AffineMap::get(/*num_dims=*/1, /*num_symbols=*/1,
                                         result, mlir_context);
  return IndexingMap(
      affine_map, {IndexingMap::Variable({0, num_threads - 1, "thread_id"})},
      {IndexingMap::Variable({0, num_tile_elements - 1, "linear_index"})}, {},
      constraints);
}

absl::StatusOr<mlir::func::FuncOp> EmitFusionKernelApi(
    mlir::ModuleOp fusion_module, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment& buffer_assignment) {
  auto* context = fusion_module.getContext();
  mlir::OpBuilder builder(context);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(fusion.name()));
  TF_ASSIGN_OR_RETURN(
      std::vector<KernelApiIrBuilder::KernelParameter> arguments,
      KernelApiIrBuilder::GetKernelArgumentsParameters(&fusion,
                                                       &buffer_assignment));
  TF_ASSIGN_OR_RETURN(std::vector<KernelApiIrBuilder::KernelParameter> results,
                      KernelApiIrBuilder::GetKernelResultsParameters(
                          &fusion, &buffer_assignment));

  // TBD: Annotate tensors with the buffer indices. This way, the buffer
  // propagation pass can clean them up later.
  auto get_arg_attrs = [&](int index, BufferAllocation::Slice& slice,
                           bool is_result) -> absl::StatusOr<mlir::Attribute> {
    SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr(
        "xla.slice_index",
        builder.getIndexAttr(index + (is_result ? arguments.size() : 0))));
    attrs.push_back(builder.getNamedAttr(
        mlir::LLVM::LLVMDialect::getDereferenceableAttrName(),
        builder.getIndexAttr(slice.size())));
    attrs.push_back(
        builder.getNamedAttr(mlir::LLVM::LLVMDialect::getAlignAttrName(),
                             builder.getIndexAttr(MinAlign())));
    return builder.getDictionaryAttr(attrs);
  };

  // First argument is the thread id.
  SmallVector<mlir::Attribute> arg_attrs{builder.getDictionaryAttr(
      builder.getNamedAttr("xla.invariant", builder.getUnitAttr()))};
  SmallVector<mlir::Type> param_types{builder.getIndexType()};

  for (const auto& [index, arg] : llvm::enumerate(arguments)) {
    param_types.push_back(emitters::TensorShapeToMlirType(arg.shape, builder));
    TF_ASSIGN_OR_RETURN(
        arg_attrs.emplace_back(),
        get_arg_attrs(index - 1, arg.slice, /*is_result=*/false));
  }

  auto result_types = emitters::ShapeToMlirTypes(fusion.shape(), builder);
  param_types.append(result_types.begin(), result_types.end());
  for (const auto& [index, result] : llvm::enumerate(results)) {
    TF_ASSIGN_OR_RETURN(arg_attrs.emplace_back(),
                        get_arg_attrs(index, result.slice, /*is_result=*/true));
  }

  builder.setInsertionPointToStart(fusion_module.getBody());
  auto entry_func = builder.create<FuncOp>(
      loc, entry_function_name,
      mlir::FunctionType::get(context, param_types, result_types),
      /*sym_visibility=*/mlir::StringAttr{},
      mlir::ArrayAttr::get(context, arg_attrs),
      /*res_attrs=*/mlir::ArrayAttr{});
  entry_func->setAttr("xla.entry", mlir::UnitAttr::get(context));
  SetBackendKind(context, entry_func, xla::BackendKind::kCpu);
  entry_func.setPrivate();

  // Create wrapper for the entry function. This function has one call_frame
  // argument and call the entry function.
  auto error_type = cpu::ErrorType::get(context);
  auto call_frame_type = CallFrameType::get(context);
  auto call_frame_func = builder.create<FuncOp>(
      loc, fusion.name(),
      builder.getFunctionType(/*arg_types=*/{call_frame_type},
                              /*result_types=*/{error_type}));
  builder.setInsertionPointToStart(call_frame_func.addEntryBlock());
  mlir::Value call_frame_arg = call_frame_func.getArgument(0);
  SmallVector<mlir::Value> extracted_values;
  extracted_values.reserve(arguments.size() + results.size() + 1);
  extracted_values.push_back(builder.create<cpu::ThreadIdOp>(
      loc, builder.getIndexType(), call_frame_arg));

  for (int i = 1; i < param_types.size(); ++i) {
    extracted_values.push_back(builder.create<cpu::LoadOp>(
        loc, param_types[i], call_frame_arg, i - 1));
  }
  auto call_results =
      builder.create<xla::PureCallOp>(loc, entry_func, extracted_values);
  call_results->setAttr("noinline", mlir::UnitAttr::get(context));
  for (auto [index, call_result] : llvm::enumerate(call_results.getResults())) {
    builder.create<cpu::StoreOp>(loc, call_result, call_frame_arg,
                                 index + arguments.size());
  }
  auto error = builder.create<cpu::SuccessOp>(loc, error_type);
  builder.create<mlir::func::ReturnOp>(loc, error.getResult());

  return entry_func;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
absl::StatusOr<emitters::CallTargetProvider> EmitCallTargets(
    mlir::ModuleOp module, const HloFusionInstruction& fusion,
    const emitters::PartitionedComputations& computations,
    const std::vector<emitters::EpilogueSpecification>& epilogues) {
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

  return call_targets;
}

void SetDataLayoutAttribute(mlir::ModuleOp module,
                            const HloFusionInstruction& fusion) {
  int index_bitwidth =
      Needs64BitIndices(fusion.fused_instructions_computation()) ? 64 : 32;
  mlir::OpBuilder b(module->getContext());
  auto index_layout = mlir::DataLayoutEntryAttr::get(
      b.getIndexType(), b.getI32IntegerAttr(index_bitwidth));
  module->setAttr(
      mlir::DLTIDialect::kDataLayoutAttrName,
      mlir::DataLayoutSpecAttr::get(module->getContext(), {index_layout}));
}

absl::StatusOr<absl::flat_hash_set<int64_t>> SetKernelFunctionAttributes(
    llvm::Module& module, const BufferAssignment& buffer_assignment,
    const HloFusionInstruction* fusion) {
  const HloModule* hlo_module = fusion->GetModule();
  if (hlo_module == nullptr) {
    return Internal("HloModule is null");
  }

  // Create a Kernel API Builder and a throwaway kernel prototype in order to
  // extract useful info from them, e.g. noalias, invariant_arguments and
  // entry function attributes.
  // TODO(ecg): find a way to obtain the same info without wasting work by
  // creating a throwaway module. All of this additional info should probably be
  // explicit in the generated MLIR, not added afterwards like we're doing here.
  // TODO(ecg): some attributes on the final loads are missing wrt those
  // generated via KernelApiIrBuilder, e.g. noalias. Add them.
  llvm::LLVMContext& context = module.getContext();
  KernelApiIrBuilder kernel_api_ir_builder(
      context,
      KernelApiIrBuilder::Options::FromHloModuleConfig(hlo_module->config()));
  std::unique_ptr<llvm::Module> throwaway_llvm_module =
      KernelApiIrBuilder::CreateModule(
          absl::StrCat(fusion->name(), "_throwaway_module"), context);
  TF_ASSIGN_OR_RETURN(KernelApiIrBuilder::KernelPrototype kernel_prototype,
                      kernel_api_ir_builder.EmitKernelPrototype(
                          *throwaway_llvm_module, fusion, &buffer_assignment,
                          "_throwaway_kernel_prototype"));
  llvm::Function* kernel_function = module.getFunction(fusion->name());
  kernel_api_ir_builder.SetKernelFunctionAttributes(kernel_function);

  return kernel_prototype.invariant_arguments;
}

int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
}  // namespace cpu
}  // namespace xla

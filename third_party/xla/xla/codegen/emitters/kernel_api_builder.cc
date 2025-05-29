/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/emitters/kernel_api_builder.h"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/runtime/work_group.h"
#include "xla/runtime/work_item.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::emitters {

static constexpr absl::string_view kXlaEntryAttr = "xla.entry";
static constexpr absl::string_view kXlaSliceIndexAttr = "xla.slice_index";
static constexpr absl::string_view kXlaInvariantAttr = "xla.invariant";
static constexpr std::array<int, 3> kIndexingMapWorkItemDims = {0, 1, 2};
static constexpr std::array<int, 3> kIndexingMapWorkGroupDims = {3, 4, 5};

static bool Needs64Bits(const Shape& shape) {
  return shape.IsArray() ? !IsInt32(ShapeUtil::ElementsIn(shape))
                         : absl::c_any_of(shape.tuple_shapes(), Needs64Bits);
}

static bool Is64BitIndex(const HloInstruction* instr, int operand) {
  const auto& shape = instr->operand(operand)->shape();
  return shape.element_type() == PrimitiveType::S64 ||
         shape.element_type() == PrimitiveType::U64;
}

static bool Needs64BitIndices(const HloInstruction* instr) {
  // Check if any HLO instructions directly take 64 bit indices as operands.
  switch (instr->opcode()) {
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
      for (int i = 1; i < instr->operand_count(); ++i) {
        if (Is64BitIndex(instr, i)) {
          return true;
        }
      }
      break;
    case HloOpcode::kGather:
    case HloOpcode::kScatter: {
      // Divide by two to get the operand index of the indices tensor.
      // (works in the case of variadic operand/updates).
      int indices_operand_index = instr->operand_count() / 2;
      if (Is64BitIndex(instr, indices_operand_index)) {
        return true;
      }
      break;
    }
    default:
      break;
  }

  if (Needs64Bits(instr->shape())) {
    return true;
  }

  for (HloComputation* computation : instr->called_computations()) {
    if (absl::c_any_of(computation->instructions(), Needs64BitIndices)) {
      return true;
    }
  }

  return false;
}

static std::vector<IndexingMap::Variable> DimVarsFromWorkDimensions(
    const WorkDimensions& work_dimensions) {
  const NumWorkItems& num_work_items = work_dimensions.num_work_items;
  const NumWorkGroups& num_work_groups = work_dimensions.num_work_groups;

  // TODO(willfroom): Change the names to work items/groups.
  return DimVarsFromGPUGrid({static_cast<int64_t>(num_work_items.x),
                             static_cast<int64_t>(num_work_items.y),
                             static_cast<int64_t>(num_work_items.z),
                             static_cast<int64_t>(num_work_groups.x),
                             static_cast<int64_t>(num_work_groups.y),
                             static_cast<int64_t>(num_work_groups.z)});
}

absl::StatusOr<mlir::func::FuncOp> EmitKernelApi(
    mlir::ModuleOp module, const HloInstruction& hlo_instruction,
    const BufferAssignment* buffer_assignment,
    const KernelArguments::BufferAlignment& buffer_alignment,
    absl::string_view entry_function_name) {
  mlir::ImplicitLocOpBuilder builder(module.getLoc(), module);
  mlir::MLIRContext* context = builder.getContext();

  // Create the entry function.
  llvm::SmallVector<mlir::Type> param_types;
  std::optional<KernelArguments> args;
  if (buffer_assignment != nullptr) {
    TF_ASSIGN_OR_RETURN(
        args, KernelArguments::Create(*buffer_assignment, buffer_alignment,
                                      &hlo_instruction));
  }
  // Annotate tensors with the buffer indices. This way, the buffer propagation
  // pass can clean them up later.
  auto get_arg_attrs = [&](int index) -> mlir::Attribute {
    if (!args) {
      return builder.getDictionaryAttr({builder.getNamedAttr(
          kXlaSliceIndexAttr, builder.getIndexAttr(index))});
    }

    const auto& arg = args->args()[index];
    llvm::SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr(
        kXlaSliceIndexAttr, builder.getIndexAttr(arg.llvm_arg_index())));
    attrs.push_back(
        builder.getNamedAttr(mlir::LLVM::LLVMDialect::getAlignAttrName(),
                             builder.getIndexAttr(arg.alignment())));
    attrs.push_back(builder.getNamedAttr(
        mlir::LLVM::LLVMDialect::getDereferenceableAttrName(),
        builder.getIndexAttr(arg.slice().size())));
    if (!arg.written()) {
      attrs.push_back(
          builder.getNamedAttr(kXlaInvariantAttr, builder.getUnitAttr()));
    }
    return builder.getDictionaryAttr(attrs);
  };

  auto result_types =
      emitters::ShapeToMlirTypes(hlo_instruction.shape(), builder);

  llvm::SmallVector<mlir::Attribute> arg_attrs;
  arg_attrs.reserve(hlo_instruction.operands().size() + result_types.size());

  for (auto [arg_index, param] : llvm::enumerate(hlo_instruction.operands())) {
    param_types.push_back(
        emitters::TensorShapeToMlirType(param->shape(), builder));
    arg_attrs.push_back(get_arg_attrs(arg_index));
  }

  for (auto [result_index, type] : llvm::enumerate(result_types)) {
    param_types.push_back(type);
    arg_attrs.push_back(
        get_arg_attrs(hlo_instruction.operands().size() + result_index));
  }

  builder.setInsertionPointToStart(module.getBody());
  auto entry_func = builder.create<mlir::func::FuncOp>(
      entry_function_name,
      mlir::FunctionType::get(context, param_types, result_types),
      /*sym_visibility=*/mlir::StringAttr{},
      mlir::ArrayAttr::get(context, arg_attrs),
      /*res_attrs=*/mlir::ArrayAttr{});
  entry_func->setAttr(kXlaEntryAttr, mlir::UnitAttr::get(context));

  return entry_func;
}

void SetIndexDataLayout(mlir::ModuleOp module,
                        const HloInstruction& hlo_instruction) {
  int index_bitwidth = Needs64BitIndices(&hlo_instruction) ? 64 : 32;
  mlir::OpBuilder b(module->getContext());
  auto index_layout = mlir::DataLayoutEntryAttr::get(
      b.getIndexType(), b.getI32IntegerAttr(index_bitwidth));
  module->setAttr(
      mlir::DLTIDialect::kDataLayoutAttrName,
      mlir::DataLayoutSpecAttr::get(module->getContext(), {index_layout}));
}

IndexingMap GetDefaultWorkItemIndexingMap(const WorkDimensions& work_dimensions,
                                          int unroll_factor, const Shape& shape,
                                          mlir::MLIRContext* ctx) {
  std::vector<mlir::AffineExpr> output_dims(shape.dimensions().size());

  const NumWorkItems& num_work_items = work_dimensions.num_work_items;
  const NumWorkGroups& num_work_groups = work_dimensions.num_work_groups;

  std::array<uint64_t, 3> work_item_array{num_work_items.x, num_work_items.y,
                                          num_work_items.z};

  std::array<uint64_t, 3> total_item_array{
      num_work_items.x * num_work_groups.x,
      num_work_items.y * num_work_groups.y,
      num_work_items.z * num_work_groups.z};

  uint64_t total_items =
      total_item_array[0] * total_item_array[1] * total_item_array[2];

  // ParallelLoopEmitter makes some assumptions about launch dimensions and
  // computes the linear index using only the x and y components.
  //
  // We implement the general formula instead and rely on the simplifier to
  // fix it.
  //
  // This means that this code supports some launch grids that the parallel
  // loop emitter doesn't support. This is safe, since the latter CHECK fails
  // if its assumptions are not fulfilled.
  mlir::AffineExpr c0 = mlir::getAffineConstantExpr(0, ctx);
  mlir::AffineExpr linear_index = c0;
  uint64_t stride = 1;
  for (int i = 0; i < 3; ++i) {
    auto coord = mlir::getAffineDimExpr(kIndexingMapWorkItemDims[i], ctx) +
                 mlir::getAffineDimExpr(kIndexingMapWorkGroupDims[i], ctx) *
                     work_item_array[i];
    auto linear_component = coord * stride;
    linear_index = linear_index + linear_component;
    stride *= total_item_array[i];
  }
  mlir::AffineExpr chunk_id = mlir::getAffineSymbolExpr(0, ctx);
  mlir::AffineExpr unroll_elem_id = mlir::getAffineSymbolExpr(1, ctx);

  linear_index = linear_index * unroll_factor +
                 chunk_id * unroll_factor * total_items + unroll_elem_id;

  // See IndexUtil::LinearIndexToMultidimensionalIndex.
  uint64_t divisor = 1;
  for (auto dimension : LayoutUtil::MinorToMajor(shape)) {
    output_dims[dimension] = (linear_index.floorDiv(divisor)) %
                             static_cast<uint64_t>(shape.dimensions(dimension));
    divisor *= shape.dimensions(dimension);
  }

  std::vector<IndexingMap::Variable> dim_vars =
      DimVarsFromWorkDimensions(work_dimensions);
  std::vector<IndexingMap::Variable> range_vars;
  int64_t num_elements = ShapeUtil::ElementsIn(shape);
  range_vars.push_back(IndexingMap::Variable{
      {0, CeilOfRatio(num_elements,
                      static_cast<int64_t>(total_items) * unroll_factor) -
              1}});
  range_vars.push_back({0, unroll_factor - 1});
  IndexingMap indexing_map(
      mlir::AffineMap::get(/*dimCount=*/6,
                           /*symbolCount=*/2, output_dims, ctx),
      std::move(dim_vars), std::move(range_vars), /*rt_vars=*/{});
  indexing_map.AddConstraint(linear_index, Interval{0, num_elements - 1});
  indexing_map.Simplify();
  return indexing_map;
}

llvm::SmallVector<mlir::Value> EmitWorkGroupIds(
    mlir::ImplicitLocOpBuilder& builder, const NumWorkGroups& num_work_groups) {
  auto set_range = [&](WorkGroupIdOp op, int64_t count) {
    op->setAttr("xla.range", builder.getIndexArrayAttr({0, count - 1}));
  };

  auto work_group_id_x = builder.create<WorkGroupIdOp>(WorkGroupDimension::x);
  set_range(work_group_id_x, num_work_groups.x);

  auto work_group_id_y = builder.create<WorkGroupIdOp>(WorkGroupDimension::y);
  set_range(work_group_id_y, num_work_groups.y);

  auto work_group_id_z = builder.create<WorkGroupIdOp>(WorkGroupDimension::z);
  set_range(work_group_id_z, num_work_groups.z);

  return {work_group_id_x, work_group_id_y, work_group_id_z};
}

}  // namespace xla::emitters

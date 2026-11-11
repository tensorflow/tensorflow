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

#include "xla/codegen/emitters/concatenate_kernel_emitter.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/emitters/utils.h"
#include "xla/codegen/hlo_fusion_spec.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/runtime/work_cluster.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/runtime/work_group.h"
#include "xla/runtime/work_item.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::emitters {

using ::mlir::MLIRContext;

ConcatenateFusionKernelEmitter::ConcatenateFusionKernelEmitter(
    MLIRContext& mlir_context, const HloFusionInstruction& fusion,
    const HloFusionSpec& fusion_spec, const BufferAssignment* buffer_assignment,
    KernelArguments::BufferAlignment buffer_alignment,
    WorkDimensions work_dimensions, absl::string_view entry_function_name,
    BackendKind backend_kind)
    : mlir_context_(mlir_context),
      fusion_(fusion),
      fusion_spec_(fusion_spec),
      buffer_assignment_(buffer_assignment),
      buffer_alignment_(std::move(buffer_alignment)),
      work_dimensions_(std::move(work_dimensions)),
      largest_shape_(GetIndexingShape(fusion_spec_)),
      entry_function_name_(entry_function_name),
      backend_kind_(backend_kind) {}

absl::StatusOr<ConcatenateFusionKernelEmitter::KernelDefinition>
ConcatenateFusionKernelEmitter::EmitKernelDefinition() {
  mlir::OpBuilder builder(&mlir_context_);
  auto loc = mlir::NameLoc::get(builder.getStringAttr(fusion_.name()));
  mlir::OwningOpRef<mlir::ModuleOp> module = llvm_ir::CreateMlirModuleOp(
      loc, absl::StrCat(fusion_.name(), "_kernel_module"));

  bool force_64_bit = backend_kind_ == BackendKind::kCpu;
  emitters::SetIndexDataLayout(*module, fusion_, force_64_bit);

  ABSL_ASSIGN_OR_RETURN(
      mlir::func::FuncOp entry_func,
      emitters::EmitKernelApi(*module, fusion_, buffer_assignment_,
                              buffer_alignment_, entry_function_name_));
  SetBackendKind(&mlir_context_, entry_func, backend_kind_);

  if (ShouldEmitFastCpuCopy()) {
    ABSL_RETURN_IF_ERROR(EmitFastCpuEntryFunction(entry_func, fusion_));
  } else {
    std::vector<emitters::EpilogueSpecification> epilogues =
        GetEpilogues(fusion_, &mlir_context_);
    emitters::PartitionedComputations computations(
        fusion_.fused_instructions_computation(), &mlir_context_, epilogues);
    ABSL_ASSIGN_OR_RETURN(auto call_targets, emitters::EmitPartitionedComputations(
                                            *module, computations));

    ABSL_RETURN_IF_ERROR(
        EmitEntryFunction(computations, call_targets, entry_func, fusion_));
  }

  ABSL_ASSIGN_OR_RETURN(auto kernel_spec,
                   GetKernelSpec(entry_function_name_, fusion_,
                                 buffer_assignment_, work_dimensions_));

  return KernelDefinition(std::move(kernel_spec),
                          MlirKernelSource(std::move(module)));
}

const Shape& ConcatenateFusionKernelEmitter::GetIndexingShape(
    const HloFusionSpec& fusion_spec) {
  const HloInstruction& concat = fusion_spec.fusion_hero(0).instruction();
  int64_t dim = concat.concatenate_dimension();
  auto less = [&](const HloInstruction* lhs, const HloInstruction* rhs) {
    return lhs->shape().dimensions(dim) < rhs->shape().dimensions(dim);
  };
  HloInstruction* operand = *absl::c_max_element(concat.operands(), less);
  return operand->shape();
}

int ConcatenateFusionKernelEmitter::GetValidUnrollFactor(
    const HloFusionSpec& fusion_spec, int max_unroll_factor) {
  auto& concat = fusion_spec.fusion_hero(0).instruction();
  int unroll_factor = max_unroll_factor;
  int64_t dim = concat.concatenate_dimension();
  for (const HloInstruction* operand : concat.operands()) {
    if (unroll_factor == 1) {
      return 1;
    }
    unroll_factor = std::gcd(unroll_factor, operand->shape().dimensions(dim));
  }
  return unroll_factor;
}

bool ConcatenateFusionKernelEmitter::CanEmitFastCpuCopy(
    const HloFusionSpec& fusion_spec, BackendKind backend_kind) {
  if (backend_kind != BackendKind::kCpu) {
    return false;
  }
  if (fusion_spec.fusion_root_count() != 1) {
    return false;
  }
  const HloInstruction& root = fusion_spec.fusion_root(0).instruction();
  if (root.opcode() != HloOpcode::kConcatenate) {
    return false;
  }
  const Shape& output_shape = root.shape();
  if (!output_shape.has_layout()) {
    return false;
  }
  int bit_width = primitive_util::StorageBitWidth(output_shape.element_type());
  if (bit_width < 8 || bit_width % 8 != 0) {
    return false;
  }
  for (const HloInstruction* operand : root.operands()) {
    if (operand->opcode() != HloOpcode::kParameter) {
      return false;
    }
    if (!operand->shape().has_layout() ||
        !LayoutUtil::Equal(operand->shape().layout(), output_shape.layout())) {
      return false;
    }
  }
  return true;
}

bool ConcatenateFusionKernelEmitter::ShouldEmitFastCpuCopy() const {
  if (!CanEmitFastCpuCopy(fusion_spec_, backend_kind_)) {
    return false;
  }
  if (work_dimensions_.num_work_clusters.x != 1 ||
      work_dimensions_.num_work_clusters.y != 1 ||
      work_dimensions_.num_work_clusters.z != 1 ||
      work_dimensions_.num_work_items.x != 1 ||
      work_dimensions_.num_work_items.y != 1 ||
      work_dimensions_.num_work_items.z != 1 ||
      work_dimensions_.num_work_groups.y != 1 ||
      work_dimensions_.num_work_groups.z != 1) {
    return false;
  }
  const HloInstruction& concat = fusion_spec_.fusion_root(0).instruction();
  const Shape& output_shape = concat.shape();
  auto min2maj = LayoutUtil::MinorToMajor(output_shape.layout());
  auto concat_dim_it = absl::c_find(min2maj, concat.concatenate_dimension());
  bool has_outer_dims = std::next(concat_dim_it) != min2maj.end();
  return has_outer_dims || work_dimensions_.num_work_groups.x == 1;
}

absl::Status ConcatenateFusionKernelEmitter::EmitFastCpuEntryFunction(
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  const auto* concat = Cast<HloConcatenateInstruction>(
      &fusion_spec_.fusion_root(0).instruction());
  const Shape& output_shape = concat->shape();
  int64_t concat_dim = concat->concatenate_dimension();
  auto output_min2maj = LayoutUtil::MinorToMajor(output_shape.layout());
  auto concat_dim_it = absl::c_find(output_min2maj, concat_dim);

  int64_t inner_dims_product = 1;
  for (auto it = output_min2maj.begin(); it != concat_dim_it; ++it) {
    inner_dims_product *= output_shape.dimensions(*it);
  }
  std::vector<int64_t> outer_dims(std::next(concat_dim_it),
                                  output_min2maj.end());

  mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());

  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  llvm::SmallVector<mlir::Value> input_tensors(
      entry_function.getArguments().take_front(num_inputs));
  mlir::Value output_tensor = entry_function.getArguments()[num_inputs];

  if (ShapeUtil::IsZeroElementArray(output_shape)) {
    mlir::func::ReturnOp::create(builder, output_tensor);
    return absl::OkStatus();
  }

  mlir::Value c0 = mlir::arith::ConstantIndexOp::create(builder, 0);
  llvm::SmallVector<mlir::Value> output_multi_index(
      output_shape.dimensions().size(), c0);

  auto emit_copies = [&](mlir::ImplicitLocOpBuilder& b,
                         mlir::Value current_dest) -> mlir::Value {
    int64_t concat_offset = 0;
    for (const HloInstruction* operand : concat->operands()) {
      int64_t operand_concat_dim = operand->shape().dimensions(concat_dim);
      int64_t element_count = inner_dims_product * operand_concat_dim;
      if (element_count > 0) {
        int64_t param_num =
            Cast<HloParameterInstruction>(operand)->parameter_number();
        mlir::Value src_tensor = input_tensors[param_num];
        llvm::SmallVector<mlir::Value> src_indices = output_multi_index;
        src_indices[concat_dim] = c0;
        llvm::SmallVector<mlir::Value> dst_indices = output_multi_index;
        dst_indices[concat_dim] =
            concat_offset == 0
                ? c0
                : mlir::arith::ConstantIndexOp::create(b, concat_offset)
                      .getResult();
        current_dest =
            CopySliceOp::create(b, src_tensor, current_dest, src_indices,
                                dst_indices, element_count);
      }
      concat_offset += operand_concat_dim;
    }
    return current_dest;
  };

  std::function<mlir::Value(int, mlir::ImplicitLocOpBuilder&, mlir::Value)>
      emit_outer_loops = [&](int outer_idx, mlir::ImplicitLocOpBuilder& b,
                             mlir::Value current_dest) -> mlir::Value {
    if (outer_idx < 0) {
      return emit_copies(b, current_dest);
    }
    int64_t dim = outer_dims[outer_idx];
    int64_t dim_size = output_shape.dimensions(dim);
    mlir::Value lb;
    mlir::Value ub;
    int64_t num_wg = static_cast<int64_t>(work_dimensions_.num_work_groups.x);
    if (outer_idx == static_cast<int>(outer_dims.size()) - 1 && num_wg > 1) {
      auto wg_id = WorkGroupIdOp::create(b, WorkGroupDimension::x);
      wg_id->setAttr("xla.range", b.getIndexArrayAttr({0, num_wg - 1}));
      int64_t wg_size = CeilOfRatio(dim_size, num_wg);
      mlir::Value wg_size_val =
          mlir::arith::ConstantIndexOp::create(b, wg_size);
      mlir::Value c1 = mlir::arith::ConstantIndexOp::create(b, 1);
      mlir::Value dim_size_val =
          mlir::arith::ConstantIndexOp::create(b, dim_size);
      lb = mlir::arith::MulIOp::create(b, wg_id, wg_size_val);
      mlir::Value next_wg_id = mlir::arith::AddIOp::create(b, wg_id, c1);
      mlir::Value ub_unclamped =
          mlir::arith::MulIOp::create(b, next_wg_id, wg_size_val);
      ub = mlir::arith::MinSIOp::create(b, ub_unclamped, dim_size_val);
    } else {
      lb = c0;
      ub = mlir::arith::ConstantIndexOp::create(b, dim_size);
    }
    mlir::Value step = mlir::arith::ConstantIndexOp::create(b, 1);
    auto for_op = mlir::scf::ForOp::create(
        b, lb, ub, step, mlir::ValueRange{current_dest},
        [&](mlir::OpBuilder& body_builder, mlir::Location loc, mlir::Value iv,
            mlir::ValueRange iter_args) {
          mlir::ImplicitLocOpBuilder nested_b(loc, body_builder);
          output_multi_index[dim] = iv;
          mlir::Value next_dest =
              emit_outer_loops(outer_idx - 1, nested_b, iter_args[0]);
          mlir::scf::YieldOp::create(nested_b, next_dest);
        });
    return for_op.getResult(0);
  };

  mlir::Value result = emit_outer_loops(static_cast<int>(outer_dims.size()) - 1,
                                        builder, output_tensor);
  mlir::func::ReturnOp::create(builder, result);
  return absl::OkStatus();
}

IndexingMap ConcatenateFusionKernelEmitter::ComputeWorkItemIdToOutputIndexing(
    const WorkDimensions& work_dimensions, const Shape& largest_shape,
    MLIRContext* ctx) {
  return GetDefaultWorkItemIndexingMap(work_dimensions, largest_shape, ctx);
}

IndexingMap ConcatenateFusionKernelEmitter::ComputeWorkItemIdToOutputIndexing(
    MLIRContext* ctx) const {
  return ComputeWorkItemIdToOutputIndexing(work_dimensions_, largest_shape_,
                                           ctx);
}

absl::Status ConcatenateFusionKernelEmitter::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  const auto& root_computation = computations.FindPartitionedComputation(
      fusion.fused_instructions_computation());

  mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());

  auto workgroup_ids =
      EmitWorkGroupIds(builder, work_dimensions_.num_work_groups);

  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  llvm::SmallVector<mlir::Value> input_tensors(
      entry_function.getArguments().take_front(num_inputs));
  auto output_tensor_args =
      entry_function.getArguments().drop_front(num_inputs);

  llvm::SmallVector<mlir::Value> result_tensors{output_tensor_args.begin(),
                                                output_tensor_args.end()};

  auto work_item_id_to_input_map =
      ComputeWorkItemIdToOutputIndexing(&mlir_context_);
  // UseConcatenateFusion requires equal root and hero shapes (layouts
  // included), which should make this indexing equal for every root. The
  // TF_RET_CHECK below guards against that not actually holding.
  auto epilogue_indexing = ComputeEpilogueInputToOutputIndexing(
      fusion_spec_.fusion_hero(0), fusion_spec_.fusion_root(0), &mlir_context_);
  for (int64_t i = 1; i < fusion_spec_.fusion_root_count(); ++i) {
    TF_RET_CHECK(ComputeEpilogueInputToOutputIndexing(
                     fusion_spec_.fusion_hero(i), fusion_spec_.fusion_root(i),
                     &mlir_context_) == epilogue_indexing)
        << "UseConcatenateFusion should have ensured all roots share the same "
           "epilogue indexing";
  }

  const auto* concat = &fusion_spec_.fusion_hero(0).instruction();
  // Several roots may share a hero, so compute each hero only once.
  llvm::SmallVector<const HloInstruction*> concats;
  for (const HloInstructionAdaptor& hero : fusion_spec_.fusion_heroes()) {
    if (!absl::c_linear_search(concats, &hero.instruction())) {
      concats.push_back(&hero.instruction());
    }
  }

  const auto forall_body_builder = [&](mlir::OpBuilder& builder,
                                       mlir::Location loc,
                                       mlir::ValueRange
                                           workitem_ids_and_outputs) {
    mlir::ImplicitLocOpBuilder nested_b(loc, builder);

    mlir::ValueRange workitem_ids = workitem_ids_and_outputs.take_front(3);
    mlir::ValueRange outputs = workitem_ids_and_outputs.drop_front(3);

    llvm::SmallVector<mlir::Value, 6> work_dims;
    work_dims.insert(work_dims.end(), workitem_ids.begin(), workitem_ids.end());
    work_dims.insert(work_dims.end(), workgroup_ids.begin(),
                     workgroup_ids.end());

    for (auto [operand_index, operand] : llvm::enumerate(concat->operands())) {
      IndexingMap input_to_output_map =
          ComputeInputToOutputIndexing(concat, /*input_id=*/operand_index,
                                       &mlir_context_)
              .indexing_maps.front()
              .begin()
              ->map();
      auto thread_id_to_output_map = ComposeIndexingMaps(
          ComposeIndexingMaps(work_item_id_to_input_map, input_to_output_map),
          epilogue_indexing);
      thread_id_to_output_map.Simplify();

      auto loop_nest_body_builder = [&, operand_index = operand_index](
                                        mlir::ImplicitLocOpBuilder& nested_b,
                                        mlir::ValueRange symbol_values,
                                        mlir::ValueRange output_indices,
                                        mlir::ValueRange output_tensors)
          -> llvm::SmallVector<mlir::Value> {
        auto input_indices = emitters::ApplyIndexing(
            work_item_id_to_input_map, work_dims, symbol_values, nested_b);

        absl::flat_hash_map<const HloInstruction*,
                            llvm::SmallVector<mlir::Value>>
            hero_values;
        for (const HloInstruction* hero : concats) {
          hero_values.try_emplace(
              hero, emitters::ProvideParameter(
                        root_computation, hero, operand_index, input_indices,
                        call_targets, entry_function, nested_b));
        }
        auto results_per_root =
            EmitEpilogue(/*epilogue_index=*/0, computations, entry_function,
                         hero_values, output_indices, nested_b);

        llvm::SmallVector<mlir::Value> result_tensors;
        result_tensors.reserve(output_tensor_args.size());
        for (auto [root, tensor] :
             llvm::zip(fusion_spec_.fusion_roots(), output_tensors)) {
          mlir::Value value = results_per_root.at(&root.instruction()).front();
          result_tensors.push_back(
              nested_b
                  .create<mlir::tensor::InsertOp>(value, tensor, output_indices)
                  .getResult());
        }

        return result_tensors;
      };
      result_tensors = emitters::EmitXlaLoopOp(
          nested_b, work_dims, result_tensors, thread_id_to_output_map,
          loop_nest_body_builder);
    }

    auto terminator = nested_b.create<mlir::scf::InParallelOp>();
    nested_b.setInsertionPointToStart(terminator.getBody());
    for (auto [result, output] : llvm::zip(result_tensors, outputs)) {
      auto output_tensor = mlir::cast<mlir::RankedTensorType>(output.getType());
      llvm::SmallVector<mlir::OpFoldResult> offsets(output_tensor.getRank(),
                                                    nested_b.getIndexAttr(0));
      llvm::SmallVector<mlir::OpFoldResult> sizes =
          mlir::getAsIndexOpFoldResult(&mlir_context_,
                                       output_tensor.getShape());
      llvm::SmallVector<mlir::OpFoldResult> strides(output_tensor.getRank(),
                                                    nested_b.getIndexAttr(1));
      nested_b.create<mlir::tensor::ParallelInsertSliceOp>(
          result, output, offsets, sizes, strides);
    }
  };

  const NumWorkItems& num_work_items = work_dimensions_.num_work_items;
  llvm::SmallVector<mlir::OpFoldResult> upper_bounds =
      mlir::getAsIndexOpFoldResult(&mlir_context_,
                                   {static_cast<int64_t>(num_work_items.x),
                                    static_cast<int64_t>(num_work_items.y),
                                    static_cast<int64_t>(num_work_items.z)});
  builder.create<mlir::func::ReturnOp>(
      builder
          .create<mlir::scf::ForallOp>(upper_bounds, output_tensor_args,
                                       std::nullopt, forall_body_builder)
          .getResults());

  return absl::OkStatus();
}

std::vector<emitters::EpilogueSpecification>
ConcatenateFusionKernelEmitter::GetEpilogues(const HloFusionInstruction& fusion,
                                             MLIRContext* mlir_context) const {
  // One epilogue for all roots, called with the same output indices.
  emitters::EpilogueSpecification epilogue =
      emitters::EpilogueSpecification::FromIdentityIndexing(
          &fusion_spec_.fusion_hero(0).instruction(),
          &fusion_spec_.fusion_root(0).instruction(), mlir_context);
  for (int64_t i = 1; i < fusion_spec_.fusion_root_count(); ++i) {
    const HloInstruction* root = &fusion_spec_.fusion_root(i).instruction();
    epilogue.heroes.push_back(&fusion_spec_.fusion_hero(i).instruction());
    epilogue.roots.push_back(root);
    epilogue.root_indexing.push_back(
        CreateIdentityMap(root->shape(), mlir_context));
  }
  return {std::move(epilogue)};
}

}  // namespace xla::emitters

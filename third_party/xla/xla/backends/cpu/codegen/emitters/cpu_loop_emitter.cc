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

#include "xla/backends/cpu/codegen/emitters/cpu_loop_emitter.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/backends/cpu/codegen/emitters/cpu_fusion_emitter.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {
namespace {

using llvm::SmallVector;
using mlir::Value;
using mlir::ValueRange;

const Shape& GetIndexShape(const Shape& shape) {
  return shape.IsTuple() ? shape.tuple_shapes(0) : shape;
}

}  // namespace

std::optional<IndexingMap> CpuLoopFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* ctx) const {
  CHECK(!fusion_->shape().IsTuple());
  auto root_shape = GetIndexShape(fusion_->fused_expression_root()->shape());

  SmallVector<int64_t> outer_dimension_partitions(root_shape.rank(), 1);
  auto backend_config = fusion_->backend_config<BackendConfig>();
  if (backend_config.ok() &&
      !backend_config->outer_dimension_partitions().empty()) {
    outer_dimension_partitions.assign(
        backend_config->outer_dimension_partitions().begin(),
        backend_config->outer_dimension_partitions().end());
  }
  SmallVector<int64_t> tile_sizes;
  tile_sizes.reserve(outer_dimension_partitions.size());
  for (auto [count, dim] :
       llvm::zip(root_shape.dimensions(), outer_dimension_partitions)) {
    tile_sizes.push_back(CeilDiv(count, dim));
  }
  return GetDefaultIndexingMap(tile_sizes, root_shape.dimensions(), ctx);
}

int64_t CpuLoopFusion::num_threads() const { return 1; }

std::optional<IndexingMap> CpuLoopFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
  std::optional<IndexingMap> thread_id_to_output_indexing =
      ComputeThreadIdToOutputIndexing(root_index, ctx);
  if (!thread_id_to_output_indexing.has_value()) {
    return std::nullopt;
  }

  auto fusion_root = fusion_->fused_expression_root();
  auto output_to_input_indexing =
      ComputeOutputToInputIndexing(fusion_root, /*output_id=*/0, ctx);
  IndexingMapSet output_to_input_indexing_set =
      output_to_input_indexing.indexing_maps[hero_operand_index];
  // Since we are computing the indexing for a non-fusion op, there is only one
  // indexing map per operand.
  CHECK_EQ(output_to_input_indexing_set.size(), 1);
  IndexingMap thread_id_to_input_indexing_map = ComposeIndexingMaps(
      *thread_id_to_output_indexing, *output_to_input_indexing_set.begin());
  thread_id_to_input_indexing_map.Simplify();
  return thread_id_to_input_indexing_map;
}

absl::Status CpuLoopFusion::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());

  auto indexing =
      ComputeThreadIdToOutputIndexing(0, entry_function.getContext());
  TF_RET_CHECK(indexing) << "Indexing is never nullopt";

  auto fusion_root =
      fusion.fused_instructions_computation()->root_instruction();
  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  auto output_tensor_args =
      entry_function.getArguments().drop_front(num_inputs + 1);
  llvm::SmallVector<const Shape*> result_shapes;
  if (fusion_root->shape().IsTuple()) {
    for (const auto& shape : fusion_root->shape().tuple_shapes()) {
      result_shapes.push_back(&shape);
    }
  } else {
    result_shapes.push_back(&fusion_root->shape());
  }
  Value thread_id = entry_function.getArgument(0);
  // Set range for the func thread id arg.
  Interval thread_id_bounds = indexing->GetDimVars().front().bounds;
  entry_function.setArgAttr(
      0, "xla.range",
      builder.getIndexArrayAttr(
          {thread_id_bounds.lower, thread_id_bounds.upper}));

  auto body_builder = [&](mlir::OpBuilder nested_builder, mlir::Location loc,
                          ValueRange symbol_values, ValueRange map_results,
                          ValueRange output_tensors) {
    auto ctx = nested_builder.getContext();
    auto root_fn = call_targets(fusion_root);
    // Generate the operands for the root function: input tensors +
    // output indices.
    SmallVector<Value> operands(
        entry_function.getArguments().drop_front().take_front(num_inputs));
    absl::c_copy(map_results, std::back_inserter(operands));
    auto result_scalars =
        nested_builder.create<xla::PureCallOp>(loc, root_fn, operands)
            .getResults();

    SmallVector<Value> result_tensors;
    result_tensors.reserve(output_tensor_args.size());
    for (auto [root_shape, tensor, value] :
         llvm::zip(result_shapes, output_tensors, result_scalars)) {
      llvm::SmallVector<Value> output_indices = emitters::ApplyIndexing(
          GetBitcastMap(*result_shapes.front(), *root_shape, ctx), map_results,
          {}, builder);
      result_tensors.push_back(nested_builder.create<mlir::tensor::InsertOp>(
          loc, value, tensor, output_indices));
    }
    nested_builder.create<xla::YieldOp>(loc, result_tensors);
  };

  auto loop =
      builder
          .create<xla::LoopOp>(*indexing, ValueRange{thread_id},
                               ValueRange{output_tensor_args}, body_builder)
          .getResults();
  builder.create<mlir::func::ReturnOp>(loop);
  //  entry_function->getParentOfType<mlir::ModuleOp>().dump();
  return absl::OkStatus();
}

}  // namespace cpu
}  // namespace xla

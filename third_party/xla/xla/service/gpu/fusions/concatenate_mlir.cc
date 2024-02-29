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

#include "xla/service/gpu/fusions/concatenate_mlir.h"

#include <cstdint>
#include <iterator>
#include <optional>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Interfaces/DataLayoutInterfaces.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/concatenate.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/shape.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

/*static*/ bool MlirConcatenateFusion::IsSupported(
    const HloFusionAnalysis& analysis) {
  if (analysis.fusion_roots().size() != 1) return false;

  return true;
}

LaunchDimensions MlirConcatenateFusion::launch_dimensions() const {
  return CalculateLaunchDimensions(GetLargestConcatOperandShape(analysis_),
                                   analysis_.device_info());
}

std::optional<IndexingMap>
MlirConcatenateFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* ctx) const {
  return std::nullopt;
}

std::optional<IndexingMap>
MlirConcatenateFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
  return GetDefaultThreadIdToOutputIndexingMap(
      launch_dimensions(), /*unroll_factor=*/1,
      GetLargestConcatOperandShape(analysis_), ctx);
}

absl::Status MlirConcatenateFusion::EmitMlir(
    mlir::ModuleOp module, mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  CHECK(IsSupported(analysis_));

  auto concat = analysis_.fusion_heroes()[0];

  mlir_converter::PartitionedComputations computations(
      fusion.fused_instructions_computation(),
      /*isolated_and_injected_instructions=*/
      {concat});

  const auto& root_computation = computations.FindPartitionedComputation(
      fusion.fused_instructions_computation());
  const auto& root_graph = root_computation.GetRootSubgraph();
  const auto& hero_graph = root_computation.FindSubgraph(concat);

  auto subgraph_to_mlir_fn = computations.DeclareFunctions(module);
  subgraph_to_mlir_fn.extract(&hero_graph).mapped().erase();

  // Concatenate is inlined in the entry function. This is needed for
  // mlir_converter::ProvideParameter to correctly get parameter value.
  subgraph_to_mlir_fn[&hero_graph] = entry_function;

  auto call_target_lookup =
      computations.CreateCallTargetProvider(subgraph_to_mlir_fn);

  for (const auto& comp : computations.partitioned_computations()) {
    for (const auto& subgraph : comp.subgraphs()) {
      if (&subgraph == &hero_graph) {
        continue;
      }

      TF_RETURN_IF_ERROR(mlir_converter::SubgraphToMlirFunction(
          comp, subgraph, subgraph_to_mlir_fn[&subgraph], call_target_lookup));
    }
  }

  mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());

  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  llvm::SmallVector<mlir::Value> input_tensors(
      entry_function.getArguments().take_front(num_inputs));
  auto output_tensor_args =
      entry_function.getArguments().drop_front(num_inputs);

  int64_t concat_dim = concat->concatenate_dimension();
  int64_t operand_offset = 0;

  llvm::SmallVector<mlir::Value> result_tensors{output_tensor_args.begin(),
                                                output_tensor_args.end()};

  auto thread_id_to_input_map = *ComputeThreadIdToInputIndexing(
      /*root_index=*/0, /*hero_operand_index=*/0, module.getContext());

  for (auto [operand_index, operand] : llvm::enumerate(concat->operands())) {
    int64_t operand_concat_dim_size = operand->shape().dimensions(concat_dim);

    auto loop_nest_body_builder = [&, operand_index = operand_index](
                                      mlir::ValueRange output_tensors,
                                      mlir::ValueRange dim_values,
                                      mlir::ValueRange symbol_values)
        -> absl::StatusOr<llvm::SmallVector<mlir::Value>> {
      auto input_indices =
          mlir_converter::ApplyAffineMap(thread_id_to_input_map.GetAffineMap(),
                                         dim_values, symbol_values, builder);

      auto operand_concat_dim_size_val =
          builder.create<mlir::arith::ConstantOp>(
              builder.getIndexAttr(operand_concat_dim_size));
      auto in_bounds = builder.create<mlir::arith::CmpIOp>(
          mlir::arith::CmpIPredicate::ult, input_indices[concat_dim],
          operand_concat_dim_size_val);

      auto then_body = [&, operand_index = operand_index](mlir::OpBuilder& b,
                                                          mlir::Location loc) {
        mlir::ImplicitLocOpBuilder builder(loc, b);

        auto result_scalars = mlir_converter::ProvideParameter(
                                  root_computation, concat, operand_index,
                                  input_indices, call_target_lookup, builder)
                                  .value();

        llvm::SmallVector<mlir::Value> output_indices(input_indices);
        output_indices[concat_dim] = builder.create<mlir::arith::AddIOp>(
            output_indices[concat_dim],
            builder.create<mlir::arith::ConstantOp>(
                builder.getIndexAttr(operand_offset)));

        if (&root_graph != &hero_graph) {
          // Concatenate is not the root of the computation. Call epilogue
          // function.
          auto epilogue_fn = subgraph_to_mlir_fn[&root_graph];

          llvm::SmallVector<mlir::Value> operands = input_tensors;
          absl::c_copy(output_indices, std::back_inserter(operands));
          absl::c_copy(result_scalars, std::back_inserter(operands));

          result_scalars =
              builder.create<PureCallOp>(epilogue_fn, operands).getResults();
        }

        llvm::SmallVector<mlir::Value> yield_tensors;
        yield_tensors.reserve(output_tensor_args.size());
        for (auto [tensor, value] : llvm::zip(output_tensors, result_scalars)) {
          yield_tensors.push_back(
              builder
                  .create<mlir::tensor::InsertOp>(value, tensor, output_indices)
                  .getResult());
        }

        builder.create<mlir::scf::YieldOp>(yield_tensors);
      };

      auto else_body = [&](mlir::OpBuilder& b, mlir::Location loc) {
        b.create<mlir::scf::YieldOp>(loc, output_tensors);
      };

      return builder.create<mlir::scf::IfOp>(in_bounds, then_body, else_body)
          .getResults();
    };

    TF_ASSIGN_OR_RETURN(result_tensors, EmitLoopNest(builder, result_tensors,
                                                     thread_id_to_input_map,
                                                     loop_nest_body_builder));

    operand_offset += operand_concat_dim_size;
  }

  builder.create<mlir::func::ReturnOp>(result_tensors);

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

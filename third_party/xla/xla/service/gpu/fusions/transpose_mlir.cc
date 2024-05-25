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
#include "xla/service/gpu/fusions/transpose_mlir.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::MLIRContext;
using mlir::RankedTensorType;
using mlir::Value;
using mlir::ValueRange;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;
using mlir::tensor::ExtractOp;
using mlir::tensor::InsertOp;
using mlir_converter::ApplyIndexing;

constexpr int kNumRows = 4;
constexpr int kBaseBlockSize = WarpSize();
constexpr int kNumThreadsPerBlock = 128;

}  // namespace

MlirTransposeFusion::MlirTransposeFusion(const HloFusionAnalysis& analysis)
    : analysis_(analysis),
      transpose_(analysis.tiled_transpose()),
      permutation_(transpose_.permutation),
      input_shape_(Permute(transpose_.dimensions, permutation_)) {
  ConstHloInstructionSet transposes_to_tile;
  int index = 0;
  int64_t shmem_usage = 0;
  int max_element_bytes = 0;
  for (auto [root, hero] :
       llvm::zip(analysis_.fusion_roots(), analysis_.fusion_heroes())) {
    if (auto transpose = GetDescriptionForTiledTransposeEmitter(
            root.instruction(), hero.instruction())) {
      transposes_to_tile.insert(&hero.instruction());
      shmem_transpose_roots_.push_back(&root.instruction());
      int size = primitive_util::ByteWidth(hero.shape().element_type());
      max_element_bytes = std::max(max_element_bytes, size);
      shmem_usage += kBaseBlockSize * (kBaseBlockSize + 1) * size;
      shmem_transpose_root_indices_.push_back(index);
    } else {
      side_output_roots_.push_back(&root.instruction());
      side_output_root_indices_.push_back(index);
    }
    ++index;
  }
  shmem_transposes_ = {transposes_to_tile.begin(), transposes_to_tile.end()};

  auto compute_block_sizes = [this](int vector_size) {
    vector_size_ = vector_size;
    block_size_ = kBaseBlockSize * vector_size_;
    block_sizes_ = {1, 1, block_size_};
    block_sizes_[permutation_[2]] = block_size_;
    block_counts_ = {CeilOfRatio(input_shape_[0], block_sizes_[0]),
                     CeilOfRatio(input_shape_[1], block_sizes_[1]),
                     CeilOfRatio(input_shape_[2], block_sizes_[2])};
  };
  // Compute initial block sizes without vectorization. We use the result to
  // determine whether we can vectorize.
  compute_block_sizes(1);

  // Enable vectorization if we have enough work, enough shared memory and
  // the input dimensions are divisible by the vector size. Vectorizing loads
  // for large data types does not help (there's already enough parallelism).
  const auto& device = analysis_.device_info();
  bool enough_work = Product(block_counts_) * kNumThreadsPerBlock >=
                     4 * device.core_count() * device.threads_per_core_limit();
  bool enough_shmem = shmem_usage * 4 <= device.shared_memory_per_block();
  bool aligned_dims =
      (input_shape_[2] % 2 == 0) && (input_shape_[permutation_[2]] % 2 == 0);
  if (max_element_bytes < 4 && enough_work && enough_shmem && aligned_dims) {
    compute_block_sizes(2);
  }
}

std::optional<IndexingMap> MlirTransposeFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, MLIRContext* mlir_context) const {
  const auto& hero = analysis_.fusion_hero(root_index).instruction();
  if (hero.opcode() != HloOpcode::kTranspose) {
    // The shape of non-transpose roots are bitcast compatible with the input
    // shape of transpose heroes.
    auto map = ComposeIndexingMaps(
        GetIndexing(/*input=*/true, hero.shape(), mlir_context),
        GetBitcastMap(
            hero.shape(),
            analysis_.fusion_roots()[root_index].instruction().shape(),
            mlir_context));
    map.Simplify();
    return map;
  }
  return GetIndexing(/*input=*/false, hero.shape(), mlir_context);
}

std::optional<IndexingMap> MlirTransposeFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    MLIRContext* mlir_context) const {
  const auto& hero = analysis_.fusion_hero(root_index).instruction();
  if (hero.opcode() != HloOpcode::kTranspose) {
    auto map = ComposeIndexingMaps(
        *ComputeThreadIdToOutputIndexing(root_index, mlir_context),
        *ComputeOutputToInputIndexing(
             &analysis_.fusion_root(root_index).instruction(), 0, mlir_context)
             .indexing_maps[hero_operand_index]
             .begin());
    map.Simplify();
    return map;
  }
  return GetIndexing(/*input=*/true, hero.operand(hero_operand_index)->shape(),
                     mlir_context);
}

LaunchDimensions MlirTransposeFusion::launch_dimensions() const {
  return LaunchDimensions(Product(block_counts_), kNumThreadsPerBlock);
}

IndexingMap MlirTransposeFusion::GetSharedMemoryIndexing(
    bool read, mlir::MLIRContext* ctx) const {
  auto thread_offsets =
      Permute(GetThreadOffsets(ctx), read ? Vector3{0, 1, 2} : permutation_);
  return {mlir::AffineMap::get(6, 2, thread_offsets, ctx),
          DimVarsFromTensorSizes({kNumThreadsPerBlock, 1, 1, 1, 1, 1}),
          RangeVarsFromTensorSizes({block_size_ / kNumRows, vector_size_}),
          {}};
}

MlirTransposeFusion::WriteResult MlirTransposeFusion::EmitWriteToShMemMlir(
    mlir::ImplicitLocOpBuilder& builder, FuncOp entry_function,
    const HloFusionInstruction& fusion,
    const mlir_converter::PartitionedComputation& root_computation,
    const mlir_converter::CallTargetProvider& call_target_provider,
    ValueRange output_args) const {
  MLIRContext* ctx = builder.getContext();
  auto shmem_tensor_size = block_sizes_;
  // Avoid bank conflicts.
  ++shmem_tensor_size.back();

  // Allocate shared memory.
  SmallVector<Value> inits;
  for (auto* transpose : shmem_transposes_) {
    auto elem_type = *ConvertPrimitiveTypeToMlirType(
        transpose->shape().element_type(), builder);
    inits.push_back(builder.create<AllocateSharedOp>(
        RankedTensorType::get(shmem_tensor_size, elem_type)));
  }

  // Add output arguments for side outputs.
  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  for (int index : side_output_root_indices_) {
    inits.push_back(entry_function.getArgument(num_inputs + index));
  }

  IndexingMap write_indexing = GetSharedMemoryIndexing(/*read=*/false, ctx);
  auto body_builder = [&](ValueRange output_tensors, ValueRange dim_values,
                          ValueRange symbol_values) -> SmallVector<Value> {
    auto input_indices = [&](const HloInstruction* instr) {
      return ApplyIndexing(GetIndexing(/*input=*/true, instr->shape(), ctx),
                           dim_values, symbol_values, builder);
    };
    SmallVector<Value> result_tensors;
    auto shmem_indices =
        ApplyIndexing(write_indexing, dim_values, symbol_values, builder);
    for (auto [transpose, output] :
         llvm::zip(shmem_transposes_, output_tensors)) {
      // Emit loop that writes subgraphs of transpose operands to shmem.
      auto result_scalar = mlir_converter::ProvideParameter(
          root_computation, transpose,
          /*operand_index=*/0, input_indices(transpose->operand(0)),
          call_target_provider, entry_function, builder)[0];
      result_tensors.push_back(
          builder.create<InsertOp>(result_scalar, output, shmem_indices));
    }

    // Produce all side outputs and then write them.
    SmallVector<Value> side_outputs;
    SmallVector<SmallVector<Value>> side_output_indices;
    auto* root_tuple = fusion.fused_expression_root();
    for (auto root : side_output_roots_) {
      side_output_indices.push_back(input_indices(root));
      side_outputs.append(mlir_converter::ProvideParameter(
          root_computation, root_tuple, root_tuple->operand_index(root),
          side_output_indices.back(), call_target_provider, entry_function,
          builder));
    }

    for (const auto& [value, indices, output] :
         llvm::zip(side_outputs, side_output_indices,
                   output_tensors.take_back(side_output_roots_.size()))) {
      result_tensors.push_back(
          builder.create<InsertOp>(value, output, indices));
    }

    return result_tensors;
  };

  auto indexing = GetIndexing(
      /*input=*/true, shmem_transposes_.front()->operand(0)->shape(), ctx);
  auto written_vector =
      EmitThreadLoopNest(builder, inits, indexing, body_builder);
  ValueRange written = written_vector;
  auto shmem_tensors = written.take_front(shmem_transposes_.size());

  WriteResult result;
  result.shmem_tensors =
      builder
          .create<SyncThreadsOp>(mlir::TypeRange(shmem_tensors), shmem_tensors)
          .getResults();
  result.updated_outputs = output_args;
  for (auto [index, side_output_result] :
       llvm::zip(side_output_root_indices_,
                 written.take_back(side_output_roots_.size()))) {
    result.updated_outputs[index] = side_output_result;
  }
  return result;
}

void MlirTransposeFusion::EmitReadFromShMemMlir(
    mlir::ImplicitLocOpBuilder& builder, FuncOp entry_function,
    const HloFusionInstruction& fusion,
    const mlir_converter::PartitionedComputations& computations,
    const WriteResult& written) const {
  auto* mlir_context = builder.getContext();
  auto output_indexing = *ComputeThreadIdToOutputIndexing(
      shmem_transpose_root_indices_[0], mlir_context);
  auto shmem_read_indexing =
      GetSharedMemoryIndexing(/*read=*/true, mlir_context);
  auto result_tensors = EmitThreadLoopNest(
      builder, written.updated_outputs, output_indexing,
      [&](ValueRange output_tensors, ValueRange dim_values,
          ValueRange symbol_values) -> SmallVector<Value> {
        auto shmem_indices = ApplyIndexing(shmem_read_indexing, dim_values,
                                           symbol_values, builder);
        absl::flat_hash_map<const HloInstruction*, llvm::SmallVector<Value>>
            transpose_values;
        for (auto [transpose, shmem] :
             llvm::zip(shmem_transposes_, written.shmem_tensors)) {
          transpose_values[transpose].push_back(
              builder.create<ExtractOp>(shmem, shmem_indices));
        }
        llvm::SmallVector<Value> epilogue_indices = dim_values;
        absl::c_copy(symbol_values, std::back_inserter(epilogue_indices));
        auto result_scalars =
            EmitEpilogue(/*epilogue_index=*/0, computations, entry_function,
                         transpose_values, epilogue_indices, builder);
        SmallVector<Value> results = output_tensors;
        for (auto [root, indexing, root_index] :
             llvm::zip(shmem_transpose_roots_,
                       computations.epilogues().front().root_indexing,
                       shmem_transpose_root_indices_)) {
          llvm::SmallVector<Value> indices =
              ApplyIndexing(indexing, dim_values, symbol_values, builder);
          results[root_index] = builder.create<InsertOp>(
              result_scalars.at(root).front(), results[root_index], indices);
        }
        return results;
      });

  builder.create<ReturnOp>(result_tensors);
}

std::vector<mlir_converter::EpilogueSpecification>
MlirTransposeFusion::GetEpilogues(const HloFusionInstruction& fusion,
                                  MLIRContext* mlir_context) const {
  return {mlir_converter::EpilogueSpecification::FromOutputIndexing(
      analysis_, shmem_transposes_, shmem_transpose_roots_, *this,
      mlir_context)};
}

absl::Status MlirTransposeFusion::EmitEntryFunction(
    const mlir_converter::PartitionedComputations& computations,
    const mlir_converter::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  const auto& root_computation = computations.FindPartitionedComputation(
      fusion.fused_instructions_computation());
  // Write intermediate results to shmem.
  mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());
  auto written = EmitWriteToShMemMlir(
      builder, entry_function, fusion, root_computation, call_targets,
      entry_function.getArguments().take_back(analysis_.fusion_roots().size()));
  // Read intermediate results from shmem and compute epilogues.
  EmitReadFromShMemMlir(builder, entry_function, fusion, computations, written);
  return absl::OkStatus();
}

llvm::SmallVector<mlir::AffineExpr, 4> MlirTransposeFusion::GetThreadOffsets(
    mlir::MLIRContext* ctx) const {
  auto thread = mlir::getAffineDimExpr(
      KernelFusionInterface::kIndexingMapThreadIdxDims[0], ctx);
  auto loop = mlir::getAffineSymbolExpr(0, ctx);
  auto vector = mlir::getAffineSymbolExpr(1, ctx);
  int loop_stride = block_size_ * kNumRows;
  auto linear_index = loop * loop_stride + thread * vector_size_ + vector;
  return DelinearizeInBoundsIndex(linear_index, block_sizes_);
}

IndexingMap MlirTransposeFusion::GetIndexing(bool input,
                                             const xla::Shape& shape,
                                             mlir::MLIRContext* ctx) const {
  auto raw_id = mlir::getAffineDimExpr(
      KernelFusionInterface::kIndexingMapBlockIdxDims[0], ctx);
  auto block_ids = Permute(DelinearizeInBoundsIndex(raw_id, block_counts_),
                           input ? Vector3{0, 1, 2} : permutation_);
  auto thread_offsets = GetThreadOffsets(ctx);
  llvm::SmallVector<AffineExpr, 3> offsets;
  for (auto [block_id, block_size, thread] :
       llvm::zip(block_ids, block_sizes_, thread_offsets)) {
    offsets.push_back(block_id * block_size + thread);
  }
  IndexingMap result{
      mlir::AffineMap::get(6, 2, offsets, ctx),
      DimVarsFromTensorSizes(
          {kNumThreadsPerBlock, 1, 1, Product(block_counts_), 1, 1}),
      RangeVarsFromTensorSizes({block_size_ / kNumRows, vector_size_}),
      {}};
  auto normalized_shape =
      input ? ShapeUtil::MakeShape(shape.element_type(), input_shape_)
            : ShapeUtil::MakeShape(shape.element_type(), transpose_.dimensions);
  for (auto [size, dim] : llvm::zip(normalized_shape.dimensions(),
                                    result.GetAffineMap().getResults())) {
    result.AddConstraint(dim, {0, size - 1});
  }
  result =
      ComposeIndexingMaps(result, GetBitcastMap(normalized_shape, shape, ctx));
  result.Simplify();
  return result;
}

}  // namespace gpu
}  // namespace xla

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
#include "xla/backends/gpu/codegen/emitters/transpose.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/emitters/emitter_base.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/codegen/emitters/utils.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using emitters::ApplyIndexing;
using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::getAffineConstantExpr;
using mlir::getAffineDimExpr;
using mlir::getAffineSymbolExpr;
using mlir::ImplicitLocOpBuilder;
using mlir::MLIRContext;
using mlir::RankedTensorType;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;
using mlir::VectorType;
using mlir::func::FuncOp;
using mlir::func::ReturnOp;

namespace mt = ::mlir::tensor;
namespace mv = ::mlir::vector;

constexpr int kTileSize = 32;
// Default values (CUDA and other backends).
constexpr int kNumRows = 4;
constexpr int64_t kNumThreadsPerBlock = kNumRows * kTileSize;  // 128
constexpr int kMaxVectorizedBytes = 4;
// ROCm-specific configuration.
constexpr int kNumRowsRocm = 8;
constexpr int64_t kNumThreadsPerBlockRocm = kNumRowsRocm * kTileSize;  // 256
constexpr int kMaxVectorizedBytesRocm = 16;

// Reads the 2D vector tile <vector_size x vector_size> from the shared memory
// at the given indices.
Value ReadVectorTileFromShmem(ImplicitLocOpBuilder& b, Value shmem,
                              ValueRange shmem_indices,
                              Value vector_tile_init) {
  int64_t vector_size =
      mlir::cast<VectorType>(vector_tile_init.getType()).getDimSize(0);
  Value vector_tile = vector_tile_init;
  SmallVector<Value> shmem_indices_vec(shmem_indices.begin(),
                                       shmem_indices.end());
  auto elem_type =
      mlir::cast<mlir::RankedTensorType>(shmem.getType()).getElementType();
  auto vector_type = mlir::VectorType::get({vector_size}, elem_type);
  for (int64_t i = 0; i < vector_size; ++i) {
    Value loaded_vector = mv::TransferReadOp::create(
        b, vector_type, shmem, shmem_indices_vec, /*padding=*/std::nullopt,
        llvm::ArrayRef<bool>{true});
    for (int64_t j = 0; j < vector_size; ++j) {
      Value elem =
          mv::ExtractOp::create(b, loaded_vector, SmallVector<int64_t>{j});
      vector_tile = mv::InsertOp::create(b, elem, vector_tile,
                                         SmallVector<int64_t>{j, i});
    }
    shmem_indices_vec.front() =
        mlir::arith::AddIOp::create(b, shmem_indices_vec.front(),
                                    mlir::arith::ConstantIndexOp::create(b, 1));
  }
  return vector_tile;
}

// Offsets each VECTOR_SIZE x VECTOR_SIZE tile in the shared memory by
// vector_size to the right. This is needed to avoid bank conflicts.
AffineExpr Swizzle(AffineExpr shmem_row, AffineExpr shmem_col,
                   int vector_size) {
  return (shmem_col + shmem_row.floorDiv(vector_size) * vector_size) %
         (kNumShmemBanks * vector_size);
}

}  // namespace

TransposeFusionBase::TransposeFusionBase(const HloFusionAnalysis& analysis,
                                         mlir::MLIRContext* mlir_context)
    : analysis_(analysis),
      mlir_context_(mlir_context),
      num_threads_per_block_(
          analysis.device_info().gpu_compute_capability().IsRocm()
              ? kNumThreadsPerBlockRocm
              : kNumThreadsPerBlock) {}

absl::Status TransposeFusionBase::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  const auto& root_computation = computations.FindPartitionedComputation(
      fusion.fused_instructions_computation());
  // Write intermediate results to shmem.
  mlir::ImplicitLocOpBuilder builder(entry_function.getLoc(), entry_function);
  builder.setInsertionPointToStart(entry_function.addEntryBlock());
  auto thread_and_block_ids = EmitThreadAndBlockIds(builder);
  auto written = EmitWriteToShMemMlir(
      builder, entry_function, fusion, root_computation, call_targets,
      entry_function.getArguments().take_back(analysis_.fusion_roots().size()),
      thread_and_block_ids);
  // Read intermediate results from shmem and compute epilogues.
  EmitReadFromShMemMlir(builder, entry_function, fusion, computations, written,
                        thread_and_block_ids);
  return absl::OkStatus();
}

std::vector<emitters::EpilogueSpecification> TransposeFusionBase::GetEpilogues(
    const HloFusionInstruction& fusion, MLIRContext* mlir_context) const {
  std::vector<emitters::EpilogueSpecification> epilogues{
      GetEpilogueForOutputIndexing(analysis_, shmem_transposes_,
                                   shmem_transpose_roots_, mlir_context)};
  // Add empty epilogues for the side outputs. This ensures their roots don't
  // get "fused" into the tuple function.
  for (const auto* root : side_output_roots_) {
    epilogues.push_back(emitters::EpilogueSpecification::FromIdentityIndexing(
        root, root, mlir_context));
  }
  return epilogues;
}

TransposeFusion::TransposeFusion(const HloFusionAnalysis& analysis,
                                 MLIRContext* mlir_context)
    : TransposeFusionBase(analysis, mlir_context),
      transpose_(analysis.tiled_transpose()),
      permutation_(transpose_.permutation),
      input_shape_(
          Permute(transpose_.dimensions, InversePermutation(permutation_))),
      base_block_size_(kTileSize),
      num_rows_(analysis.device_info().gpu_compute_capability().IsRocm()
                    ? kNumRowsRocm
                    : kNumRows) {
  ConstHloInstructionSet transposes_to_tile;
  int index = 0;
  int64_t shmem_usage = 0;
  int max_element_bytes = 0;
  for (auto [root, hero] :
       llvm::zip(analysis_.fusion_roots(), analysis_.fusion_heroes())) {
    if (auto transpose =
            GetDescriptionForTiledTransposeEmitter(hero.instruction())) {
      transposes_to_tile.insert(&hero.instruction());
      shmem_transpose_roots_.push_back(&root.instruction());
      int size = primitive_util::ByteWidth(hero.shape().element_type());
      // If the last dimension stays the same, we need to make it part of the
      // shared memory tile.
      if (MostMinorDimensionUnchanged()) {
        size *= input_shape_.back();
      }
      max_element_bytes = std::max(max_element_bytes, size);
      shmem_usage += base_block_size_ * (base_block_size_ + 1) * size;
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
    block_sizes_.assign(input_shape_.size(), 1);
    if (MostMinorDimensionUnchanged()) {
      block_size_ = base_block_size_;
      block_sizes_.back() = vector_size_;
      block_sizes_[block_sizes_.size() - 2] = block_size_;
      block_sizes_[permutation_[block_sizes_.size() - 2]] = block_size_;
    } else {
      block_size_ = base_block_size_ * vector_size_;
      block_sizes_.back() = block_size_;
      block_sizes_[permutation_.back()] = block_size_;
    }
    output_block_sizes_ = Permute(block_sizes_, permutation_);
    block_counts_.resize(block_sizes_.size());
    for (int64_t i = 0; i < block_sizes_.size(); ++i) {
      block_counts_[i] = CeilOfRatio(input_shape_[i], block_sizes_[i]);
    }
  };
  if (MostMinorDimensionUnchanged()) {
    compute_block_sizes(input_shape_.back());
  } else {
    // Compute initial block sizes without vectorization. We use the result to
    // determine whether we can vectorize.
    compute_block_sizes(1);

    // Enable vectorization if we have enough work, enough shared memory and
    // the input dimensions are divisible by the vector size. Vectorizing loads
    // for large data types does not help (there's already enough parallelism).
    const auto& device = analysis_.device_info();
    int max_vectorized_bytes = device.gpu_compute_capability().IsRocm()
                                   ? kMaxVectorizedBytesRocm
                                   : kMaxVectorizedBytes;
    for (int vec_size = max_vectorized_bytes / max_element_bytes; vec_size > 1;
         vec_size /= 2) {
      int elems_per_thread = vec_size * vec_size;
      bool enough_work = Product(block_counts_) * num_threads_per_block_ >=
                         elems_per_thread * device.core_count() *
                             device.threads_per_core_limit();
      bool enough_shmem =
          shmem_usage * elems_per_thread <= device.shared_memory_per_block();
      bool aligned_dims = (input_shape_.back() % vec_size == 0) &&
                          (input_shape_[permutation_.back()] % vec_size == 0);
      if (enough_work && enough_shmem && aligned_dims) {
        compute_block_sizes(vec_size);
        break;
      }
    }
  }
}

std::optional<IndexingMap> TransposeFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, MLIRContext* mlir_context) const {
  const auto& hero = analysis_.fusion_hero(root_index).instruction();
  if (!GetDescriptionForTiledTransposeEmitter(hero)) {
    // The shape of non-transpose roots are bitcast compatible with the input
    // shape of transpose heroes.
    return GetIndexing(/*input=*/true,
                       analysis_.fusion_root(root_index).shape(), mlir_context);
  }
  return GetIndexing(/*input=*/false, hero.shape(), mlir_context);
}

std::optional<std::vector<IndexingMap>>
TransposeFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, MLIRContext* mlir_context) const {
  const auto& hero = analysis_.fusion_hero(root_index).instruction();
  if (GetDescriptionForTiledTransposeEmitter(hero)) {
    return std::vector<IndexingMap>{GetIndexing(
        /*input=*/true, hero.operand(0)->shape(), mlir_context)};
  }
  std::vector<IndexingMap> result;
  result.reserve(hero.operand_count());
  auto thread_id_to_output_indexing =
      ComputeThreadIdToOutputIndexing(root_index, mlir_context);
  if (!thread_id_to_output_indexing.has_value()) {
    return std::nullopt;
  }
  for (int64_t operand_index = 0; operand_index < hero.operand_count();
       ++operand_index) {
    auto map = ComposeIndexingMaps(
        *thread_id_to_output_indexing,
        ComputeOutputToInputIndexing(
            &analysis_.fusion_root(root_index).instruction(), 0, mlir_context)
            .indexing_maps[operand_index]
            .begin()
            ->map());
    map.Simplify();
    result.push_back(map);
  }
  return result;
}

LaunchDimensions TransposeFusion::launch_dimensions() const {
  return LaunchDimensions(Product(block_counts_), num_threads_per_block_);
}

IndexingMap TransposeFusion::GetSharedMemoryIndexing(
    bool read, MLIRContext* mlir_context) const {
  auto thread_offsets = GetThreadOffsets(/*read=*/true, mlir_context);
  if (!read) {
    // Regarding shared memory indexing, the permutation we need to apply is
    // just a swap of the two dimensions that are tiled.
    if (MostMinorDimensionUnchanged()) {
      std::swap(thread_offsets[thread_offsets.size() - 2],
                thread_offsets[permutation_[permutation_.size() - 2]]);
    } else {
      std::swap(thread_offsets.back(), thread_offsets[permutation_.back()]);
    }
  }
  std::vector<int64_t> dim_var_sizes(6, 1);
  dim_var_sizes[KernelFusionInterface::kIndexingMapThreadIdxDims[0]] =
      num_threads_per_block_;
  dim_var_sizes[KernelFusionInterface::kIndexingMapBlockIdxDims[0]] =
      Product(block_counts_);
  return {mlir::AffineMap::get(6, 2, thread_offsets, mlir_context),
          DimVarsFromGPUGrid(dim_var_sizes),
          RangeVarsFromTensorSizes({block_size_ / num_rows_, vector_size_}),
          {}};
}

TransposeFusion::WriteResult TransposeFusion::EmitWriteToShMemMlir(
    mlir::ImplicitLocOpBuilder& builder, FuncOp entry_function,
    const HloFusionInstruction& fusion,
    const emitters::PartitionedComputation& root_computation,
    const emitters::CallTargetProvider& call_target_provider,
    ValueRange output_args, mlir::ValueRange thread_and_block_ids) const {
  auto shmem_tensor_size = block_sizes_;
  // Avoid bank conflicts.
  if (MostMinorDimensionUnchanged()) {
    // Increase the dimension that is actually iterated over. The most minor
    // dimension is always completely loaded into the shared memory tile.
    ++shmem_tensor_size[shmem_tensor_size.size() - 2];
  } else {
    ++shmem_tensor_size.back();
  }

  // Allocate shared memory.
  SmallVector<Value> inits;
  for (auto* transpose : shmem_transposes_) {
    auto elem_type = emitters::PrimitiveTypeToMlirType(
        transpose->shape().element_type(), builder);
    inits.push_back(AllocateSharedOp::create(
        builder, RankedTensorType::get(shmem_tensor_size, elem_type)));
  }

  // Add output arguments for side outputs.
  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  for (int index : side_output_root_indices_) {
    inits.push_back(entry_function.getArgument(num_inputs + index));
  }

  IndexingMap write_indexing =
      GetSharedMemoryIndexing(/*read=*/false, mlir_context_);
  auto body_builder = [&](ImplicitLocOpBuilder& nested_b,
                          ValueRange symbol_values, ValueRange map_results,
                          ValueRange output_tensors) -> SmallVector<Value> {
    auto input_indices = [&](const HloInstruction* instr) {
      return ApplyIndexing(
          GetIndexing(/*input=*/true, instr->shape(), mlir_context_),
          thread_and_block_ids, symbol_values, nested_b);
    };

    SmallVector<Value> result_tensors;
    auto shmem_indices = ApplyIndexing(write_indexing, thread_and_block_ids,
                                       symbol_values, builder);
    for (auto [transpose, output] :
         llvm::zip(shmem_transposes_, output_tensors)) {
      // Emit loop that writes subgraphs of transpose operands to shmem.
      auto result_scalar = emitters::ProvideParameter(
          root_computation, transpose,
          /*operand_index=*/0, input_indices(transpose->operand(0)),
          call_target_provider, entry_function, builder)[0];
      result_tensors.push_back(mlir::tensor::InsertOp::create(
          builder, result_scalar, output, shmem_indices));
    }

    // Produce all side outputs and then write them.
    SmallVector<Value> side_outputs;
    SmallVector<SmallVector<Value>> side_output_indices;
    auto* root_tuple = fusion.fused_expression_root();
    for (auto root : side_output_roots_) {
      side_output_indices.push_back(input_indices(root));
      ValueRange param_values = emitters::ProvideParameter(
          root_computation, root_tuple, root_tuple->operand_index(root),
          side_output_indices.back(), call_target_provider, entry_function,
          nested_b);
      side_outputs.append(param_values.begin(), param_values.end());
    }

    for (const auto& [value, indices, output] :
         llvm::zip(side_outputs, side_output_indices,
                   output_tensors.take_back(side_output_roots_.size()))) {
      result_tensors.push_back(
          mt::InsertOp::create(nested_b, value, output, indices));
    }

    return result_tensors;
  };

  auto indexing = GetIndexing(
      /*input=*/true, shmem_transposes_.front()->operand(0)->shape(),
      mlir_context_);
  auto written_vector = emitters::EmitXlaLoopOp(builder, thread_and_block_ids,
                                                inits, indexing, body_builder);
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

void TransposeFusion::EmitReadFromShMemMlir(
    mlir::ImplicitLocOpBuilder& builder, FuncOp entry_function,
    const HloFusionInstruction& fusion,
    const emitters::PartitionedComputations& computations,
    const WriteResult& written, mlir::ValueRange thread_and_block_ids) const {
  auto output_indexing = *ComputeThreadIdToOutputIndexing(
      shmem_transpose_root_indices_[0], mlir_context_);
  auto shmem_read_indexing =
      GetSharedMemoryIndexing(/*read=*/true, mlir_context_);
  auto result_tensors = emitters::EmitXlaLoopOp(
      builder, thread_and_block_ids, written.updated_outputs, output_indexing,
      [&](ImplicitLocOpBuilder& nested_b, ValueRange symbol_values,
          ValueRange map_results,
          ValueRange output_tensors) -> SmallVector<Value> {
        auto shmem_indices = ApplyIndexing(
            shmem_read_indexing, thread_and_block_ids, symbol_values, nested_b);
        absl::flat_hash_map<const HloInstruction*, llvm::SmallVector<Value>>
            transpose_values;
        for (auto [transpose, shmem] :
             llvm::zip(shmem_transposes_, written.shmem_tensors)) {
          transpose_values[transpose].push_back(
              mt::ExtractOp::create(nested_b, shmem, shmem_indices));
        }
        llvm::SmallVector<Value> epilogue_indices = thread_and_block_ids;
        absl::c_copy(symbol_values, std::back_inserter(epilogue_indices));
        auto result_scalars =
            EmitEpilogue(/*epilogue_index=*/0, computations, entry_function,
                         transpose_values, epilogue_indices, nested_b);
        SmallVector<Value> results = output_tensors;
        for (auto [root, indexing, root_index] :
             llvm::zip(shmem_transpose_roots_,
                       computations.epilogues().front().root_indexing,
                       shmem_transpose_root_indices_)) {
          llvm::SmallVector<Value> indices = ApplyIndexing(
              indexing, thread_and_block_ids, symbol_values, nested_b);
          results[root_index] =
              mt::InsertOp::create(nested_b, result_scalars.at(root).front(),
                                   results[root_index], indices);
        }
        return results;
      });

  ReturnOp::create(builder, result_tensors);
}

llvm::SmallVector<mlir::AffineExpr, 4> TransposeFusion::GetThreadOffsets(
    bool read, MLIRContext* mlir_context) const {
  auto thread = getAffineDimExpr(
      KernelFusionInterface::kIndexingMapThreadIdxDims[0], mlir_context);
  auto loop = getAffineSymbolExpr(0, mlir_context);
  auto vector = getAffineSymbolExpr(1, mlir_context);
  int loop_stride = block_size_ * num_rows_;
  if (MostMinorDimensionUnchanged()) {
    loop_stride *= vector_size_;
  }
  auto linear_index = loop * loop_stride + thread * vector_size_ + vector;
  return DelinearizeInBoundsIndex(linear_index,
                                  read ? block_sizes_ : output_block_sizes_);
}

IndexingMap TransposeFusion::GetIndexing(bool input, const xla::Shape& shape,
                                         MLIRContext* mlir_context) const {
  auto raw_id = getAffineDimExpr(
      KernelFusionInterface::kIndexingMapBlockIdxDims[0], mlir_context);
  auto block_ids = DelinearizeInBoundsIndex(raw_id, block_counts_);
  if (!input) {
    absl::c_copy(Permute(block_ids, permutation_), block_ids.begin());
  }
  auto thread_offsets = GetThreadOffsets(input, mlir_context);
  const auto& permuted_block_sizes = input ? block_sizes_ : output_block_sizes_;
  llvm::SmallVector<AffineExpr, 3> offsets;
  for (auto [block_id, block_size, thread] :
       llvm::zip(block_ids, permuted_block_sizes, thread_offsets)) {
    offsets.push_back(block_id * block_size + thread);
  }
  std::vector<int64_t> dim_var_sizes(6, 1);
  dim_var_sizes[KernelFusionInterface::kIndexingMapThreadIdxDims[0]] =
      num_threads_per_block_;
  dim_var_sizes[KernelFusionInterface::kIndexingMapBlockIdxDims[0]] =
      Product(block_counts_);
  IndexingMap result{
      mlir::AffineMap::get(6, 2, offsets, mlir_context),
      DimVarsFromTensorSizes(dim_var_sizes),
      RangeVarsFromTensorSizes({block_size_ / num_rows_, vector_size_}),
      {}};
  auto normalized_shape =
      input ? ShapeUtil::MakeShape(shape.element_type(), input_shape_)
            : ShapeUtil::MakeShape(shape.element_type(), transpose_.dimensions);
  for (auto [size, dim] : llvm::zip(normalized_shape.dimensions(),
                                    result.GetSymbolicMap().GetResults())) {
    result.AddConstraint(dim, {0, size - 1});
  }
  result = ComposeIndexingMaps(
      result, GetBitcastMap(normalized_shape, shape, mlir_context));
  result.Simplify();
  return result;
}

bool TransposeFusion::MostMinorDimensionUnchanged() const {
  return permutation_.back() == permutation_.size() - 1;
}

std::vector<int64_t> GetBlockCounts(absl::Span<const int64_t> shape,
                                    absl::Span<const int64_t> tile) {
  std::vector<int64_t> block_counts;
  for (auto [dim, tile_size] : llvm::zip(shape, tile)) {
    block_counts.push_back(CeilOfRatio(dim, tile_size));
  }
  return block_counts;
}

PackedTranspose::PackedTranspose(const HloFusionAnalysis& analysis,
                                 const PackedTransposeDescription& spec,
                                 absl::Span<const int64_t> output_block_tile,
                                 MLIRContext* mlir_context)
    : TransposeFusionBase(analysis, mlir_context),
      spec_(spec),
      output_tile_(output_block_tile.begin(), output_block_tile.end()),
      input_tile_(Permute(output_tile_, spec_.canonical_inv_permutation)),
      block_counts_(GetBlockCounts(spec_.canonical_output_shape, output_tile_)),
      num_shmem_groups_per_block_(num_threads_per_block_ / kNumShmemBanks),
      tile_size_t1_(input_tile_[spec_.dim_T1_input_id()]),
      tile_size_a_(input_tile_[spec_.dim_A_id()]),
      tile_size_t2_(input_tile_[spec_.dim_T2_input_id()]),
      populated_shmem_cols_(tile_size_a_ * tile_size_t1_),
      populated_shmem_rows_(tile_size_t2_) {
  VLOG(5) << "Transpose spec: " << spec.ToString()
          << "Output block tile: " << absl::StrJoin(output_block_tile, ", ")
          << "\nNumber of shmem groups: " << num_shmem_groups_per_block_
          << "\n";
  auto bits_per_element = GetBitwidth(spec_.elem_type());
  vector_size_ = kBankBitwidth / bits_per_element;
  CHECK_GE(vector_size_, 1);

  int64_t index = 0;
  ConstHloInstructionSet transposes_to_tile;
  for (auto [root, hero] :
       llvm::zip(analysis_.fusion_roots(), analysis_.fusion_heroes())) {
    if (auto transpose =
            GetDescriptionForTiledTransposeEmitter(hero.instruction())) {
      transposes_to_tile.insert(&hero.instruction());
      shmem_transpose_roots_.push_back(&root.instruction());
      shmem_transpose_root_indices_.push_back(index);
    } else {
      side_output_roots_.push_back(&root.instruction());
      side_output_root_indices_.push_back(index);
    }
    ++index;
  }
  shmem_transposes_ = {transposes_to_tile.begin(), transposes_to_tile.end()};
}

std::optional<IndexingMap> PackedTranspose::ComputeThreadIdToOutputIndexing(
    int64_t root_index, MLIRContext* mlir_context) const {
  auto map = GetOutputIndexing(mlir_context);
  auto hero_shape = analysis_.fusion_hero(root_index).shape();
  if (!ShapeUtil::SameDimensions(hero_shape, spec_.original_output_shape())) {
    auto bitcast =
        GetBitcastMap(spec_.original_output_shape(), hero_shape, mlir_context);
    map = ComposeIndexingMaps(map, bitcast);
    map.Simplify();
    map.RemoveUnusedSymbols();
  }
  return map;
}

std::optional<std::vector<IndexingMap>>
PackedTranspose::ComputeThreadIdToInputIndexing(
    int64_t root_index, MLIRContext* mlir_context) const {
  const auto& hero = analysis_.fusion_hero(root_index).instruction();
  if (GetDescriptionForTiledTransposeEmitter(hero)) {
    auto map = GetInputIndexing(mlir_context);
    auto operand_shape = hero.operand(0)->shape();
    if (!ShapeUtil::SameDimensions(operand_shape,
                                   spec_.original_input_shape())) {
      auto bitcast = GetBitcastMap(spec_.original_input_shape(), operand_shape,
                                   mlir_context);
      map = ComposeIndexingMaps(map, bitcast);
      map.Simplify();
      map.RemoveUnusedSymbols();
    }
    return std::vector<IndexingMap>{map};
  }
  std::vector<IndexingMap> result;
  result.reserve(hero.operand_count());
  auto thread_id_to_output_indexing =
      ComputeThreadIdToOutputIndexing(root_index, mlir_context);
  if (!thread_id_to_output_indexing.has_value()) {
    return std::nullopt;
  }
  for (int64_t operand_index = 0; operand_index < hero.operand_count();
       ++operand_index) {
    auto map = ComposeIndexingMaps(
        *thread_id_to_output_indexing,
        ComputeOutputToInputIndexing(
            &analysis_.fusion_root(root_index).instruction(), 0, mlir_context)
            .indexing_maps[operand_index]
            .begin()
            ->map());
    map.Simplify();
    result.push_back(map);
  }
  return result;
}

LaunchDimensions PackedTranspose::launch_dimensions() const {
  return LaunchDimensions(Product(block_counts_), num_threads_per_block_);
}

PackedTranspose::WriteResult PackedTranspose::EmitWriteToShMemMlir(
    mlir::ImplicitLocOpBuilder& builder, FuncOp entry_function,
    const HloFusionInstruction& fusion,
    const emitters::PartitionedComputation& root_computation,
    const emitters::CallTargetProvider& call_target_provider,
    ValueRange output_args, mlir::ValueRange thread_and_block_ids) const {
  IndexingMap input_indexing = GetInputIndexing(mlir_context_);
  IndexingMap shmem_write_indexing = GetShmemWriteIndexing(mlir_context_);

  int64_t shmem_dim = kNumShmemBanks * vector_size_;

  // Allocate all shared memory tensors upfront (one per transpose).
  SmallVector<Value> shmem_inits;
  for (auto* transpose : shmem_transposes_) {
    Type elem_type = emitters::PrimitiveTypeToMlirType(
        transpose->shape().element_type(), builder);
    Value shmem = AllocateSharedOp::create(
        builder, RankedTensorType::get({shmem_dim, shmem_dim}, elem_type));
    shmem_inits.push_back(shmem);
  }

  // Create a single loop that writes to all shared memory tensors.
  auto tids_and_bids = EmitThreadAndBlockIds(builder);
  SmallVector<Value> shmem_tensors = emitters::EmitXlaLoopOp(
      builder, tids_and_bids, shmem_inits, input_indexing,
      [&](ImplicitLocOpBuilder& nested_b, ValueRange ivs,
          ValueRange input_indices,
          ValueRange iter_args) -> SmallVector<Value> {
        SmallVector<Value> updated_shmems;
        auto shmem_indices = emitters::ApplyIndexing(
            shmem_write_indexing, tids_and_bids, ivs, nested_b);

        // Process all transposes in a single loop iteration.
        for (auto [transpose, shmem_tensor] :
             llvm::zip(shmem_transposes_, iter_args)) {
          // Compute bitcasted indices for this specific transpose if needed.
          ValueRange indices = input_indices;
          SmallVector<Value> indices_storage;
          if (!ShapeUtil::SameDimensions(transpose->operand(0)->shape(),
                                         spec_.original_input_shape())) {
            auto map =
                GetBitcastMap(spec_.original_input_shape(),
                              transpose->operand(0)->shape(), mlir_context_);
            indices_storage =
                emitters::ApplyIndexing(map, input_indices, {}, nested_b);
            indices = indices_storage;
          }

          Value input_element =
              emitters::ProvideParameter(root_computation, transpose,
                                         /*operand_index=*/0, indices,
                                         call_target_provider, entry_function,
                                         nested_b)
                  .front();
          updated_shmems.push_back(nested_b.create<mt::InsertOp>(
              input_element, shmem_tensor, shmem_indices));
        }
        return updated_shmems;
      });

  // Produce all side outputs and then write them.
  int num_inputs = fusion.fused_instructions_computation()->num_parameters();
  SmallVector<Value> side_output_inits;
  for (int index : side_output_root_indices_) {
    side_output_inits.push_back(entry_function.getArgument(num_inputs + index));
  }
  auto body_builder = [&](ImplicitLocOpBuilder& nested_b,
                          ValueRange symbol_values, ValueRange map_results,
                          ValueRange output_tensors) -> SmallVector<Value> {
    SmallVector<Value> side_outputs;
    SmallVector<SmallVector<Value>> side_output_indices;
    auto* root_tuple = fusion.fused_expression_root();
    for (auto root : side_output_roots_) {
      auto indexing = ComposeIndexingMaps(
          input_indexing, GetBitcastMap(spec_.original_input_shape(),
                                        root->shape(), mlir_context_));
      indexing.Simplify();
      side_output_indices.push_back(ApplyIndexing(
          indexing, thread_and_block_ids, symbol_values, nested_b));
      ValueRange param_values = emitters::ProvideParameter(
          root_computation, root_tuple, root_tuple->operand_index(root),
          side_output_indices.back(), call_target_provider, entry_function,
          nested_b);
      side_outputs.append(param_values.begin(), param_values.end());
    }

    SmallVector<Value> result_tensors;
    for (const auto& [value, indices, output] :
         llvm::zip(side_outputs, side_output_indices, output_tensors)) {
      result_tensors.push_back(
          mt::InsertOp::create(nested_b, value, output, indices));
    }

    return result_tensors;
  };
  mlir::ValueRange side_output_vector;
  if (!side_output_inits.empty()) {
    side_output_vector = emitters::EmitXlaLoopOp(builder, thread_and_block_ids,
                                                 side_output_inits,
                                                 input_indexing, body_builder);
  }

  WriteResult result;
  result.shmem_tensors =
      builder
          .create<SyncThreadsOp>(mlir::TypeRange(shmem_tensors), shmem_tensors)
          .getResults();
  result.updated_outputs = output_args;
  for (auto [index, side_output_result] :
       llvm::zip(side_output_root_indices_, side_output_vector)) {
    result.updated_outputs[index] = side_output_result;
  }
  return result;
}

Value GetZeroVector(ImplicitLocOpBuilder& b, PrimitiveType elem_type,
                    llvm::ArrayRef<int64_t> shape) {
  auto mlir_elem_type = emitters::PrimitiveTypeToMlirType(elem_type, b);
  auto accumulator_type = mlir::VectorType::get(shape, mlir_elem_type);
  return mlir::arith::ConstantOp::create(
      b, accumulator_type,
      emitters::GetZeroDenseElementsAttr(accumulator_type));
}

void PackedTranspose::EmitReadFromShMemMlir(
    mlir::ImplicitLocOpBuilder& builder, FuncOp entry_function,
    const HloFusionInstruction& fusion,
    const emitters::PartitionedComputations& computations,
    const WriteResult& written, mlir::ValueRange thread_and_block_ids) const {
  auto shmem_read_indexing = GetShmemReadIndexing(mlir_context_);
  auto outer_loop_indexing = ConvertRangeVariablesToDimensions(
      shmem_read_indexing, /*range_var_indices=*/{1, 2});
  auto output_indexing = GetOutputIndexing(mlir_context_);
  auto output_indexing_over_vectors = ConvertRangeVariablesToDimensions(
      output_indexing, /*range_var_indices=*/{0});

  auto c0 = mlir::arith::ConstantIndexOp::create(builder, 0);
  SmallVector<Value> grid_and_vector_ids{thread_and_block_ids};
  grid_and_vector_ids.append({c0, c0});
  absl::flat_hash_map<PrimitiveType, Value> elem_type_to_vector_tile;
  for (const HloInstruction* transpose : shmem_transposes_) {
    PrimitiveType elem_type = transpose->shape().element_type();
    if (!elem_type_to_vector_tile.contains(elem_type)) {
      elem_type_to_vector_tile[elem_type] =
          GetZeroVector(builder, elem_type, {vector_size_, vector_size_});
    }
  }
  absl::flat_hash_map<const HloInstruction*, llvm::SmallVector<Value>>
      transpose_values;
  // The outer loop reads <vector_size x vector_size> tiles.
  auto outer_loop_results = emitters::EmitXlaLoopOp(
      builder, grid_and_vector_ids, written.updated_outputs,
      outer_loop_indexing,
      [&](ImplicitLocOpBuilder& nested_b, ValueRange iv, ValueRange map_results,
          ValueRange output_tensors) -> SmallVector<Value> {
        SmallVector<Value> dims{thread_and_block_ids};
        dims.push_back(iv.front());
        SmallVector<Value> results;
        for (auto [transpose, shmem] :
             llvm::zip(shmem_transposes_, written.shmem_tensors)) {
          ValueRange shmem_indices = map_results;
          Value vector_tile =
              elem_type_to_vector_tile[transpose->shape().element_type()];
          vector_tile = ReadVectorTileFromShmem(nested_b, shmem, shmem_indices,
                                                vector_tile);
          transpose_values[transpose] = {vector_tile};
        }
        // The inner loop writes columns of the <vector_size x vector_size>
        // tiles.
        auto inner_loop_results = emitters::EmitXlaLoopOp(
            nested_b, dims, output_tensors, output_indexing_over_vectors,
            [&](ImplicitLocOpBuilder& nested_b_2, ValueRange ivs,
                ValueRange map_results,
                ValueRange output_tensors) -> SmallVector<Value> {
              for (auto [transpose, shmem] :
                   llvm::zip(shmem_transposes_, written.shmem_tensors)) {
                Value elem = mv::ExtractOp::create(
                    nested_b_2, transpose_values[transpose].front(),
                    getAsOpFoldResult(ivs));
                transpose_values[transpose] = {elem};
              }
              llvm::SmallVector<Value> epilogue_indices = thread_and_block_ids;
              absl::c_copy(iv, std::back_inserter(epilogue_indices));
              absl::c_copy(ivs, std::back_inserter(epilogue_indices));
              auto result_scalars = EmitEpilogue(
                  /*epilogue_index=*/0, computations, entry_function,
                  transpose_values, epilogue_indices, nested_b_2);
              SmallVector<Value> results = output_tensors;
              for (auto [root, indexing, root_index] :
                   llvm::zip(shmem_transpose_roots_,
                             computations.epilogues().front().root_indexing,
                             shmem_transpose_root_indices_)) {
                SmallVector<Value> symbols{iv};
                symbols.append(ivs.begin(), ivs.end());
                llvm::SmallVector<Value> indices = ApplyIndexing(
                    indexing, thread_and_block_ids, symbols, nested_b);
                results[root_index] = mt::InsertOp::create(
                    nested_b_2, result_scalars.at(root).front(),
                    results[root_index], indices);
              }
              return results;
            });
        return inner_loop_results;
      });
  ReturnOp::create(builder, outer_loop_results);
}

IndexingMap PackedTranspose::GetInputIndexing(MLIRContext* mlir_context) const {
  // Dimensions variables.
  auto thread_id = getAffineDimExpr(
      KernelFusionInterface::kIndexingMapThreadIdxDims[0], mlir_context);
  auto block_id = getAffineDimExpr(
      KernelFusionInterface::kIndexingMapBlockIdxDims[0], mlir_context);
  auto shmem_group_size = kNumShmemBanks;
  auto lane_id = thread_id % shmem_group_size;
  auto shmem_group_id = thread_id.floorDiv(shmem_group_size);
  std::vector<IndexingMap::Variable> dim_vars =
      DimVarsFromGPUGrid({num_shmem_groups_per_block_ * shmem_group_size, 1, 1,
                          Product(block_counts_), 1, 1});

  // Range variables.
  auto loop = getAffineSymbolExpr(0, mlir_context);
  auto vector_element_id = getAffineSymbolExpr(1, mlir_context);
  std::vector<IndexingMap::Variable> range_vars = RangeVarsFromTensorSizes(
      {{CeilOfRatio(tile_size_t2_, num_shmem_groups_per_block_),
        vector_size_}});

  // Block offsets.
  auto block_ids = DelinearizeInBoundsIndex(block_id, block_counts_);
  absl::c_copy(Permute(block_ids, spec_.canonical_inv_permutation),
               block_ids.begin());

  // Shmem expressions.
  auto shmem_row = loop * num_shmem_groups_per_block_ + shmem_group_id;
  auto shmem_col = lane_id * vector_size_ + vector_element_id;

  // Offsets within the block.
  auto c0 = getAffineConstantExpr(0, mlir_context);
  int64_t canonical_rank = spec_.canonical_rank();
  llvm::SmallVector<AffineExpr, 4> offsets_within_tile(canonical_rank, c0);
  offsets_within_tile[spec_.dim_A_id()] = shmem_col.floorDiv(tile_size_t1_);
  offsets_within_tile[spec_.dim_T1_input_id()] = shmem_col % tile_size_t1_;
  offsets_within_tile[spec_.dim_T2_input_id()] = shmem_row;

  // Canonical indexing.
  llvm::SmallVector<AffineExpr, 4> canonical_offsets;
  canonical_offsets.reserve(canonical_rank + 2);
  for (auto [thread_offset, block_index, tile_size] :
       llvm::zip(offsets_within_tile, block_ids, input_tile_)) {
    canonical_offsets.push_back(block_index * tile_size + thread_offset);
  }
  llvm::SmallVector<std::pair<AffineExpr, Interval>> constraints{
      {shmem_col, Interval{0, populated_shmem_cols_ - 1}},
      {shmem_row, Interval{0, populated_shmem_rows_ - 1}}};
  IndexingMap canonical_input_indexing{
      mlir::AffineMap::get(/*num_dims=*/6, /*num_symbols=*/2, canonical_offsets,
                           mlir_context),
      std::move(dim_vars), std::move(range_vars), /*rt_vars=*/{}, constraints};
  canonical_input_indexing.Simplify();

  // Actual indexing.
  auto canonical_input_shape_to_real_shape = GetBitcastMap(
      spec_.canonical_input_shape, spec_.original_input_shape(), mlir_context);
  // When we compose, the constraints w.r.t. to the input dimension sizes will
  // be added.
  auto input_indexing = ComposeIndexingMaps(
      canonical_input_indexing, canonical_input_shape_to_real_shape);
  input_indexing.Simplify();
  return input_indexing;
}

IndexingMap PackedTranspose::GetShmemWriteIndexing(
    MLIRContext* mlir_context) const {
  // Dimensions variables.
  auto thread_id = getAffineDimExpr(
      KernelFusionInterface::kIndexingMapThreadIdxDims[0], mlir_context);
  auto shmem_group_size = kNumShmemBanks;
  auto lane_id = thread_id % shmem_group_size;
  auto shmem_group_id = thread_id.floorDiv(shmem_group_size);
  std::vector<IndexingMap::Variable> dim_vars =
      DimVarsFromGPUGrid({num_shmem_groups_per_block_ * shmem_group_size, 1, 1,
                          Product(block_counts_), 1, 1});

  // Range variables.
  auto loop = getAffineSymbolExpr(0, mlir_context);
  auto vector_element_id = getAffineSymbolExpr(1, mlir_context);
  std::vector<IndexingMap::Variable> range_vars = RangeVarsFromTensorSizes(
      {CeilOfRatio(tile_size_t2_, num_shmem_groups_per_block_), vector_size_});

  // Shmem expressions.
  auto shmem_row = loop * num_shmem_groups_per_block_ + shmem_group_id;
  auto shmem_col = lane_id * vector_size_ + vector_element_id;
  llvm::SmallVector<std::pair<AffineExpr, Interval>> constraints{
      {shmem_col, Interval{0, populated_shmem_cols_ - 1}},
      {shmem_row, Interval{0, populated_shmem_rows_ - 1}}};
  shmem_col = Swizzle(shmem_row, shmem_col, vector_size_);

  IndexingMap shmem_write_indexing_map{
      mlir::AffineMap::get(6, 2, {shmem_row, shmem_col}, mlir_context),
      dim_vars, range_vars, /*rt_vars=*/{}, constraints};
  shmem_write_indexing_map.Simplify();
  return shmem_write_indexing_map;
}

IndexingMap PackedTranspose::GetShmemReadIndexing(
    MLIRContext* mlir_context) const {
  // Dimensions variables.
  auto thread_id = getAffineDimExpr(
      KernelFusionInterface::kIndexingMapThreadIdxDims[0], mlir_context);
  auto shmem_group_size = kNumShmemBanks;
  auto lane_id = thread_id % shmem_group_size;
  auto shmem_group_id = thread_id.floorDiv(shmem_group_size);
  std::vector<IndexingMap::Variable> dim_vars =
      DimVarsFromGPUGrid({num_shmem_groups_per_block_ * shmem_group_size, 1, 1,
                          Product(block_counts_), 1, 1});

  // Range variables.
  auto loop = getAffineSymbolExpr(0, mlir_context);
  auto vector_horizontal = getAffineSymbolExpr(1, mlir_context);
  auto vector_vertical = getAffineSymbolExpr(2, mlir_context);
  std::vector<IndexingMap::Variable> range_vars = RangeVarsFromTensorSizes(
      {CeilOfRatio(populated_shmem_cols_,
                   (vector_size_ * num_shmem_groups_per_block_)),
       vector_size_, vector_size_});

  // Shmem expressions.
  auto shmem_row = lane_id * vector_size_ + vector_vertical;
  auto shmem_col =
      (loop * num_shmem_groups_per_block_ + shmem_group_id) * vector_size_ +
      vector_horizontal;
  llvm::SmallVector<std::pair<AffineExpr, Interval>> constraints{
      {shmem_col, Interval{0, populated_shmem_cols_ - 1}},
      {shmem_row, Interval{0, populated_shmem_rows_ - 1}}};
  shmem_col = Swizzle(shmem_row, shmem_col, vector_size_);

  IndexingMap shmem_read_indexing_map{
      mlir::AffineMap::get(6, 3, {shmem_row, shmem_col}, mlir_context),
      dim_vars, range_vars, /*rt_vars=*/{}, constraints};
  shmem_read_indexing_map.Simplify();
  return shmem_read_indexing_map;
}

IndexingMap PackedTranspose::GetOutputIndexing(
    MLIRContext* mlir_context) const {
  // Dimensions variables.
  auto thread_id = getAffineDimExpr(
      KernelFusionInterface::kIndexingMapThreadIdxDims[0], mlir_context);
  auto block_id = getAffineDimExpr(
      KernelFusionInterface::kIndexingMapBlockIdxDims[0], mlir_context);
  auto shmem_group_size = kNumShmemBanks;
  auto lane_id = thread_id % shmem_group_size;
  auto shmem_group_id = thread_id.floorDiv(shmem_group_size);
  std::vector<IndexingMap::Variable> dim_vars =
      DimVarsFromGPUGrid({num_shmem_groups_per_block_ * shmem_group_size, 1, 1,
                          Product(block_counts_), 1, 1});

  // Range variables.
  auto loop = getAffineSymbolExpr(0, mlir_context);
  auto vector_horizontal = getAffineSymbolExpr(1, mlir_context);
  auto vector_vertical = getAffineSymbolExpr(2, mlir_context);
  std::vector<IndexingMap::Variable> range_vars = RangeVarsFromTensorSizes(
      {CeilOfRatio(populated_shmem_cols_,
                   vector_size_ * num_shmem_groups_per_block_),
       vector_size_, vector_size_});

  // Block offsets.
  auto block_ids = DelinearizeInBoundsIndex(block_id, block_counts_);

  // Shmem expressions.
  auto shmem_col =
      (loop * num_shmem_groups_per_block_ + shmem_group_id) * vector_size_ +
      vector_horizontal;
  auto shmem_row = lane_id * vector_size_ + vector_vertical;

  // Offsets within the block.
  auto c0 = getAffineConstantExpr(0, mlir_context);
  int64_t canonical_rank = spec_.canonical_rank();
  llvm::SmallVector<AffineExpr, 4> offsets_within_tile(canonical_rank, c0);
  offsets_within_tile[spec_.dim_A_id()] = shmem_col.floorDiv(tile_size_t1_);
  offsets_within_tile[spec_.dim_T1_output_id()] = shmem_col % tile_size_t1_;
  offsets_within_tile[spec_.dim_T2_output_id()] = shmem_row;

  // Canonical indexing.
  llvm::SmallVector<AffineExpr, 4> canonical_offsets;
  canonical_offsets.reserve(canonical_rank + 2);
  for (auto [thread_offset, block_index, tile_size] :
       llvm::zip(offsets_within_tile, block_ids, output_tile_)) {
    canonical_offsets.push_back(block_index * tile_size + thread_offset);
  }
  llvm::SmallVector<std::pair<AffineExpr, Interval>> constraints{
      {shmem_col, Interval{0, populated_shmem_cols_ - 1}},
      {shmem_row, Interval{0, populated_shmem_rows_ - 1}}};
  IndexingMap canonical_output_indexing{
      mlir::AffineMap::get(6, 3, canonical_offsets, mlir_context),
      std::move(dim_vars), std::move(range_vars), /*rt_vars=*/{}, constraints};
  canonical_output_indexing.Simplify();

  // Actual indexing.
  auto canonical_output_shape_to_real_shape =
      GetBitcastMap(spec_.canonical_output_shape, spec_.original_output_shape(),
                    mlir_context);
  // When we compose, the constraints w.r.t. to the output dimension sizes will
  // be added.
  auto output_indexing = ComposeIndexingMaps(
      canonical_output_indexing, canonical_output_shape_to_real_shape);
  output_indexing.Simplify();
  return output_indexing;
}

std::unique_ptr<EmitterBase> CreateTransposeFusion(
    const HloFusionAnalysis& analysis, MLIRContext* mlir_context) {
  PackedTransposeDescription spec(analysis.tiled_transpose());
  auto packed_transpose_tile = GetPackedTransposeTileSizes(spec);
  if (packed_transpose_tile.ok()) {
    return std::make_unique<PackedTranspose>(
        analysis, spec, *packed_transpose_tile, mlir_context);
  }
  return std::make_unique<TransposeFusion>(analysis, mlir_context);
}

}  // namespace gpu
}  // namespace xla

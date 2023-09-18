/*Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/fusions/tiling_util.h"

#include "llvm/IR/IRBuilder.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/kernel_support_library.h"

namespace xla {
namespace gpu {
namespace {

// Gets the output offset as calculated from thread_id.x (to be applied to the
// offset calculated from block_id and thread_id.y).
llvm::Value* GetStartOffsetX(const TilingScheme& tiling_scheme,
                             llvm::Value* thread_id_x, llvm::Type* index_ty,
                             llvm::IRBuilder<>* b) {
  int64_t multiplier =
      tiling_scheme.GetIndexingOrder() == TilingScheme::StridedIndexingX
          ? tiling_scheme.GetVectorSize()
          : tiling_scheme.GetTileSizeFor(TilingScheme::DimX);
  return b->CreateMul(thread_id_x,
                      llvm::ConstantInt::get(index_ty, multiplier));
}

// Emits loop through the minor (X) dimension of a tile, starting at a given
// offset.
//
// Rough pseudocode:
//
// Given: offset, callback
//
// for (int x = 0; x < x_tile_size / vector_size; ++x) {
//   for (int i = 0; i < vector_size; ++i) {
//      callback(offset + x * stride * vector_size + i);
//   }
// }
void EmitXTileLoop(const TilingThreadIdInfo& thread_id_info,
                   const llvm_ir::IrArray::Index& tile_origin_index,
                   const TilingScheme& tiling_scheme, bool check_x_tile_bounds,
                   llvm::Value* y_loc,
                   std::array<llvm::Value*, 2> tile_dimensions,
                   llvm::IRBuilder<>* b,
                   const EmitTileElementFunction* emit_elem_function) {
  llvm::Type* index_ty = tile_dimensions[1]->getType();
  KernelSupportLibrary ksl(b, llvm_ir::UnrollMode::kDefaultUnroll);
  auto constant = [&](int64_t val) {
    return llvm::ConstantInt::get(index_ty, val);
  };
  llvm::Value* start_offset_x =
      GetStartOffsetX(tiling_scheme, thread_id_info.thread_id_x, index_ty, b);

  int64_t vector_size = tiling_scheme.GetVectorSize();
  int64_t stride_x =
      tiling_scheme.GetIndexingOrder() == TilingScheme::LinearIndexingX
          ? 1
          : tiling_scheme.GetNumThreadsFor(TilingScheme::DimX);
  KernelSupportLibrary unrolled_ksl(b, llvm_ir::UnrollMode::kFullyUnroll);
  unrolled_ksl.For(
      "tile_loop",
      /*start=*/constant(0),
      /*end=*/
      constant(tiling_scheme.GetTileSizeFor(TilingScheme::DimX) / vector_size),
      /*step=*/1, [&](llvm::Value* x) {
        for (int64_t i = 0; i < vector_size; i++) {
          llvm::Value* x_offset = b->CreateAdd(
              b->CreateMul(x, constant(stride_x * vector_size)), constant(i));
          llvm::Value* x_loc = b->CreateAdd(x_offset, start_offset_x, "x_loc");
          llvm_ir::IrArray::Index source_idx_x =
              tile_origin_index
                  .AddOffsetToDim(y_loc, tiling_scheme.GetTilingDimension(0), b)
                  .AddOffsetToDim(x_loc, tiling_scheme.GetTilingDimension(1),
                                  b);
          auto emit_element = [&] {
            return (*emit_elem_function)(thread_id_info, source_idx_x, y_loc,
                                         x_loc);
          };
          if (check_x_tile_bounds) {
            ksl.If("x_in_tile", b->CreateICmpULT(x_loc, tile_dimensions[1]),
                   emit_element);
          } else {
            emit_element();
          }
        }
      });
}

}  // namespace

void EmitTile(llvm::IRBuilder<>* builder, const TilingScheme& tiling_scheme,
              const llvm_ir::IrArray::Index& tile_origin_index,
              const TilingThreadIdInfo& thread_id_info,
              std::array<llvm::Value*, 2> tile_dimensions,
              const EmitTileElementFunction& emit_elem_function) {
  llvm::Type* index_ty = tile_dimensions[0]->getType();
  auto constant = [&](int64_t val) {
    return llvm::ConstantInt::get(index_ty, val);
  };
  llvm::Value* num_threads_y = constant(
      tiling_scheme.GetNumThreadsFor(tiling_scheme.GetTilingDimension(0)));

  KernelSupportLibrary ksl(builder, llvm_ir::UnrollMode::kDefaultUnroll);

  ksl.For(
      "y_in_tile",
      /*start=*/thread_id_info.thread_id_y,
      /*end=*/
      tile_dimensions[0],
      /*step=*/num_threads_y, [&](llvm::Value* y_loc) {
        auto unroll_inner_tile_loop = [&](bool check_x_tile_bounds) {
          return EmitXTileLoop(thread_id_info, tile_origin_index, tiling_scheme,
                               check_x_tile_bounds, y_loc, tile_dimensions,
                               builder, &emit_elem_function);
        };

        // Only take this path when we unroll in a way vectorizable by
        // LLVM. Special case when the tile doesn't fit completely for even
        // row size. For odd row size every other row isn't aligned to the
        // vectorized size, so it can't be vectorized by LLVM.
        if (tiling_scheme.GetIndexingOrder() ==
            TilingScheme::StridedIndexingX) {
          ksl.If(
              "is_full_tile",
              builder->CreateICmpEQ(constant(tiling_scheme.GetBlockTileSizeFor(
                                        TilingScheme::DimX)),
                                    tile_dimensions[1]),
              [&] { unroll_inner_tile_loop(/*check_x_tile_bounds=*/false); },
              [&] { unroll_inner_tile_loop(/*check_x_tile_bounds=*/true); });
        } else {
          unroll_inner_tile_loop(/*check_x_tile_bounds=*/true);
        }
      });
}

namespace {

// Emits current block id.
llvm::Value* EmitBlockId(llvm::IRBuilder<>* builder, int32_t num_blocks,
                         llvm::Type* index_ty) {
  llvm::Value* block_id =
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockIdx, {}, {}, builder);
  if (num_blocks != 0) {
    llvm_ir::AddRangeMetadata(0, num_blocks,
                              llvm::cast<llvm::Instruction>(block_id));
  }
  return builder->CreateIntCast(block_id, index_ty, /*isSigned=*/true,
                                "block.id.x");
}

// Emits current thread id with the given type.
//
// Sets the return value range to [0, threads_per_block).
llvm::Value* EmitThreadId(llvm::IRBuilder<>* builder, int64_t threads_per_block,
                          llvm::Type* index_ty) {
  // Calculate (y, x) coordinates respectively in the 2D view of thread block,
  // defined by (num_thread_y, num_thread_x) from thread_id.
  llvm::CallInst* thread_id_raw =
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdx, {}, {}, builder);
  llvm_ir::AddRangeMetadata(0, threads_per_block, thread_id_raw);
  return builder->CreateIntCast(thread_id_raw, index_ty,
                                /*isSigned=*/true, "thread.id.x");
}

// Emits the LLVM values for thread_id, thread_id.x, thread_id.y and lane
// id.
//
// Returns a struct containting these values.
//
// In the presence of thread scaling in tiling scheme may return early if the
// combination of thread_id/block_id does not correspond to a real block.
// Assumes the current function returns void.
StatusOr<TilingThreadIdInfo> EmitThreadIdInfo(llvm::IRBuilder<>* builder,
                                              const TilingScheme& tiling_scheme,
                                              llvm::Type* index_ty) {
  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };
  llvm::Value* thread_id_physical = EmitThreadId(
      builder, tiling_scheme.GetNumThreadsPerBlockPhysical(), index_ty);
  int64_t num_blocks = tiling_scheme.GetNumberOfBlocksPhysical();
  if (num_blocks > (int64_t)std::numeric_limits<uint32_t>::max()) {
    return FailedPrecondition(
        "Number of physical blocks (%d) does not fit in an i32 in tiling "
        "scheme: %s",
        num_blocks, tiling_scheme.ToString());
  }
  llvm::Value* block_id_physical = EmitBlockId(builder, num_blocks, index_ty);

  // Wait this will break coalescing.
  llvm::Value* thread_id_logical = builder->CreateURem(
      thread_id_physical, constant(tiling_scheme.GetNumThreadsPerBlock()));
  llvm::Value* scaling = builder->CreateUDiv(
      thread_id_physical, constant(tiling_scheme.GetNumThreadsPerBlock()));
  llvm::Value* block_id_logical = builder->CreateAdd(
      builder->CreateMul(block_id_physical,
                         constant(tiling_scheme.GetThreadIdScalingFactor())),
      scaling);

  llvm::Value* num_threads_x_v =
      constant(tiling_scheme.GetNumThreadsFor(TilingScheme::DimX));

  llvm::Value* block_exists = builder->CreateICmpULT(
      block_id_logical, constant(tiling_scheme.GetNumberOfBlocks()));
  llvm_ir::EmitEarlyReturn(block_exists, builder);
  return {
      {thread_id_logical,
       /*thread_id_x=*/
       builder->CreateURem(thread_id_logical, num_threads_x_v, "thread_id.x"),
       /*thread_id_y=*/
       builder->CreateUDiv(thread_id_logical, num_threads_x_v, "thread_id.y"),
       /*lane_id=*/
       builder->CreateURem(thread_id_logical, constant(WarpSize()), "lane_id"),
       /*block_id=*/block_id_logical,
       /*scaling=*/scaling}};
}

}  // namespace

StatusOr<TilingKernelInfo> EmitTilingKernel(
    llvm::IRBuilder<>* builder, const TilingScheme& tiling_scheme,
    llvm::Type* index_ty, const TileElementGenerator& tile_element_generator) {
  absl::Span<const int64_t> dims_in_elems = tiling_scheme.GetDimsInElems();
  Vector3 dims_in_blocks = tiling_scheme.GetDimsInBlocks();
  auto constant = [&](uint64_t c) -> llvm::Constant* {
    return llvm::ConstantInt::get(index_ty, c);
  };

  TF_ASSIGN_OR_RETURN(TilingThreadIdInfo thread_id_info,
                      EmitThreadIdInfo(builder, tiling_scheme, index_ty));

  KernelSupportLibrary ksl(builder, llvm_ir::UnrollMode::kDefaultUnroll);

  int64_t non_tiling_dimension = tiling_scheme.GetTilingDimension(0) == 1
                                     ? TilingScheme::DimZ
                                     : TilingScheme::DimY;
  const llvm_ir::IrArray::Index block_coords(
      thread_id_info.block_id,
      ShapeUtil::MakeShapeWithDenseLayout(
          PRED /*arbitrary*/, dims_in_blocks,
          // This layout determines the iteration order. We want the
          // non-tiling dimension to be the slowest varying dimension.
          {2, 1 - non_tiling_dimension, non_tiling_dimension}),
      builder);

  std::array<llvm::Value*, 2> tile_dimensions;
  // Coordinate access is shifted: 0 corresponds to the first non-tiling
  // dimension and 1 corresponds to DimX.
  std::array<int64_t, 2> tiling_coords{1 - non_tiling_dimension,
                                       TilingScheme::DimX};
  for (int i = 0; i < 2; ++i) {
    int64_t tile_size_for_dim =
        tiling_scheme.GetBlockTileSizeFor(tiling_coords[i]);
    // Only last row or column may not have full size.
    llvm::Value* is_last =
        builder->CreateICmpEQ(block_coords[tiling_coords[i]],
                              constant(dims_in_blocks[tiling_coords[i]] - 1));
    int64_t partial_row =
        dims_in_elems[tiling_coords[i]] -
        (dims_in_blocks[tiling_coords[i]] - 1) * tile_size_for_dim;
    tile_dimensions[i] =
        builder->CreateSelect(is_last, constant(partial_row),
                              constant(tile_size_for_dim), "tile_bound");
  }

  llvm_ir::IrArray::Index tile_origin = [&] {
    std::vector<llvm::Value*> elem_multi_index = block_coords.multidim();
    llvm::Type* index_ty = block_coords.GetType();
    for (int i = 0; i < TilingScheme::DimTot; ++i) {
      elem_multi_index[i] = builder->CreateMul(
          block_coords[i],
          llvm::ConstantInt::get(index_ty,
                                 tiling_scheme.GetBlockTileSizeFor(i)),
          "tile_origin." + std::to_string(i));
    }
    return llvm_ir::IrArray::Index(elem_multi_index,
                                   tiling_scheme.GetDimsInElems(), index_ty);
  }();

  auto emit_tile = [&](const llvm_ir::IrArray::Index& tile) {
    tile_element_generator(thread_id_info, tile, tile_dimensions);
  };

  if (tiling_scheme.GetBlockTileSizeFor(non_tiling_dimension) == 1) {
    emit_tile(tile_origin);
  } else {
    llvm::Value* starting_tile_index_for_dim =
        tile_origin[non_tiling_dimension];
    llvm::Value* block_size_for_dim =
        constant(tiling_scheme.GetBlockTileSizeFor(non_tiling_dimension));
    llvm::Value* block_id_for_dim =
        builder->CreateUDiv(starting_tile_index_for_dim, block_size_for_dim);
    llvm::Value* last_block_for_dim =
        constant(dims_in_blocks[non_tiling_dimension] - 1);
    llvm::Value* last_block_size_for_dim =
        constant(dims_in_elems[non_tiling_dimension] -
                 (dims_in_blocks[non_tiling_dimension] - 1) *
                     tiling_scheme.GetBlockTileSizeFor(non_tiling_dimension));

    llvm::Value* num_tiles_in_block = builder->CreateSelect(
        builder->CreateICmpEQ(last_block_for_dim, block_id_for_dim),
        last_block_size_for_dim, block_size_for_dim);
    ksl.For("loop_z",
            /*start=*/constant(0),
            /*end=*/num_tiles_in_block,
            /*step=*/1, [&](llvm::Value* block_dim_induction_var) {
              llvm_ir::IrArray::Index tile_index = tile_origin.AddOffsetToDim(
                  block_dim_induction_var, non_tiling_dimension, builder);
              emit_tile(tile_index);
            });
  }

  return {{tile_dimensions, tile_origin, thread_id_info}};
}

llvm::Type* TilingThreadIdInfo::GEPIntoSharedMemoryType(
    llvm::GlobalVariable* shared,
    absl::Span<llvm::Value* const> idx_major_to_minor) const {
  std::vector<llvm::Value*> idxs_scaled;
  idxs_scaled.push_back(llvm::ConstantInt::get(scaling->getType(), 0));
  idxs_scaled.push_back(scaling);
  idxs_scaled.insert(idxs_scaled.end(), idx_major_to_minor.begin(),
                     idx_major_to_minor.end());
  return llvm::GetElementPtrInst::getIndexedType(shared->getValueType(),
                                                 idxs_scaled);
}

llvm::Value* TilingThreadIdInfo::GEPIntoSharedMemory(
    llvm::IRBuilder<>* b, llvm::GlobalVariable* shared,
    absl::Span<llvm::Value* const> idx_major_to_minor,
    const llvm::Twine& name) const {
  std::vector<llvm::Value*> idxs_scaled;
  idxs_scaled.push_back(llvm::ConstantInt::get(scaling->getType(), 0));
  idxs_scaled.push_back(scaling);
  idxs_scaled.insert(idxs_scaled.end(), idx_major_to_minor.begin(),
                     idx_major_to_minor.end());
  llvm::Value* gep =
      b->CreateInBoundsGEP(shared->getValueType(), shared, idxs_scaled, name);

  llvm::PointerType* pointer_in_addressspace =
      llvm::PointerType::getWithSamePointeeType(
          llvm::cast<llvm::PointerType>(gep->getType()), /*AddressSpace=*/0);

  // __shared__ memory uses a different address space, so we cast it to
  // global address space before writing or reading.
  return b->CreateAddrSpaceCast(gep, pointer_in_addressspace);
}

llvm_ir::IrArray::Index GetUnnormalizedIndex(
    const llvm_ir::IrArray::Index& normalized_shape_index,
    const Shape& unnormalized_shape, llvm::IRBuilder<>* builder,
    absl::Span<const int64_t> dims_in_elems) {
  CHECK_EQ(normalized_shape_index.size(), 3);
  // If the normalization only add a new dimensions of size 1,
  // generate simpler indexing. LLVM doesn't always simplify the more
  // complicated indexing and this prevents it from vectorizing some
  // cases. We do this only for major_to_minor memory layout.
  if (unnormalized_shape.rank() == 2 && unnormalized_shape.has_layout() &&
      unnormalized_shape.dimensions()[0] == normalized_shape_index.dims()[1] &&
      unnormalized_shape.dimensions()[1] == normalized_shape_index.dims()[2] &&
      unnormalized_shape.layout().minor_to_major(1) == 0) {
    CHECK_EQ(normalized_shape_index.dims()[0], 1);
    auto multidim = normalized_shape_index.multidim();
    return llvm_ir::IrArray::Index({multidim[1], multidim[2]},
                                   unnormalized_shape,
                                   normalized_shape_index.GetType());
  }
  if (unnormalized_shape.rank() == 2 && unnormalized_shape.has_layout() &&
      unnormalized_shape.dimensions()[0] == normalized_shape_index.dims()[2] &&
      unnormalized_shape.dimensions()[1] == normalized_shape_index.dims()[1] &&
      unnormalized_shape.layout().minor_to_major(1) == 1) {
    CHECK_EQ(normalized_shape_index.dims()[0], 1);
    auto multidim = normalized_shape_index.multidim();
    return llvm_ir::IrArray::Index({multidim[2], multidim[1]},
                                   unnormalized_shape,
                                   normalized_shape_index.GetType());
  }
  return normalized_shape_index.SourceIndexOfBitcast(
      ShapeUtil::MakeShape(F32, dims_in_elems), unnormalized_shape, builder);
}

}  // namespace gpu
}  // namespace xla

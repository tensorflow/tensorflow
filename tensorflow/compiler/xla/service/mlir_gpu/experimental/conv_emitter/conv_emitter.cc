/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This is an explorative prototype emitter for convolution using MLIR.
// This prototype is still under construction.
// TODO(timshen): Fix the documentation once it's implemented.
//
// Goals:
// * Autotune-able tiling.
// * Autotune-able memory accesses.
// * Autotune-able lowering logic (from a portable program to thread-oriented
//   CUDA program).
// * Use milr::AffineExpr to analyze all accesses. It aims to algorithmically
//   find memory access strategies for given input layouts and tiling configs.

#include "tensorflow/compiler/xla/service/mlir_gpu/experimental/conv_emitter/conv_emitter.h"

#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/AffineExpr.h"  // TF:llvm-project
#include "mlir/IR/AffineMap.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Transforms/LoopUtils.h"  // TF:llvm-project
#include "mlir/Transforms/RegionUtils.h"  // TF:llvm-project
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace mlir_gpu {
namespace {

// Various extracted information for input shapes.
struct ShapeInfo {
  // Buffer dimensions in the order of NCHW.
  std::vector<int64_t> nchw_dimensions;

  // Buffer dimensions in the order of major to minor;
  std::vector<int64_t> physical_dimensions;

  // The affine map that takes NCHW indices, and maps to the physical order.
  mlir::AffineMap affine_map;

  mlir::Type element_type;
};

ShapeInfo GetShapeInfo(const Shape& shape, int64 n_dim, int64 c_dim,
                       absl::Span<const int64> spatial_dims,
                       mlir::Builder builder) {
  ShapeInfo shape_info;

  std::vector<int64> physical_to_logical(
      shape.layout().minor_to_major().rbegin(),
      shape.layout().minor_to_major().rend());

  std::vector<int64> nchw_to_logical;

  nchw_to_logical.push_back(n_dim);
  nchw_to_logical.push_back(c_dim);
  for (int64 dim : spatial_dims) {
    nchw_to_logical.push_back(dim);
  }

  for (int64 dim : nchw_to_logical) {
    shape_info.nchw_dimensions.push_back(shape.dimensions(dim));
  }

  for (int64 dim : physical_to_logical) {
    shape_info.physical_dimensions.push_back(shape.dimensions(dim));
  }

  std::vector<mlir::AffineExpr> affine_exprs;
  // We want physical to nchw order.
  for (int64 dim : ComposePermutations(InversePermutation(nchw_to_logical),
                                       physical_to_logical)) {
    affine_exprs.push_back(builder.getAffineDimExpr(dim));
  }

  shape_info.affine_map = mlir::AffineMap::get(
      /*dimCount=*/2 + spatial_dims.size(), /*symbolCount=*/0, affine_exprs);

  shape_info.element_type = [&] {
    switch (shape.element_type()) {
      case xla::F16:
        return builder.getF16Type();
      case xla::F32:
        return builder.getF32Type();
      default:
        break;
    }
    CHECK(false);
  }();

  return shape_info;
}

bool IsSimpleLoop(mlir::AffineForOp loop) {
  return loop.getLowerBoundMap().isSingleConstant() &&
         loop.getLowerBoundMap().getSingleConstantResult() == 0 &&
         loop.getStep() == 1 && loop.getUpperBoundMap().getNumResults() == 1 &&
         std::next(loop.region().begin()) == loop.region().end();
}

struct BoundAffineMap {
  mlir::AffineMap affine_map;
  std::vector<mlir::Value> operands;
};

BoundAffineMap GetBoundAffineMapFrom(mlir::Operation* op) {
  if (auto load = mlir::dyn_cast<mlir::AffineLoadOp>(op)) {
    return {load.getAffineMap(),
            std::vector<mlir::Value>(load.getMapOperands().begin(),
                                     load.getMapOperands().end())};
  } else if (auto store = mlir::dyn_cast<mlir::AffineStoreOp>(op)) {
    return {store.getAffineMap(),
            std::vector<mlir::Value>(store.getMapOperands().begin(),
                                     store.getMapOperands().end())};
  } else {
    CHECK(false);
  }
}

mlir::Operation* CloneWithNewAffineMap(mlir::Operation* op,
                                       BoundAffineMap new_affine,
                                       mlir::OpBuilder builder) {
  if (auto load = mlir::dyn_cast<mlir::AffineLoadOp>(op)) {
    return builder.create<mlir::AffineLoadOp>(
        builder.getUnknownLoc(), load.getMemRef(), new_affine.affine_map,
        new_affine.operands);
  } else if (auto store = mlir::dyn_cast<mlir::AffineStoreOp>(op)) {
    return builder.create<mlir::AffineStoreOp>(
        builder.getUnknownLoc(), store.getValueToStore(), store.getMemRef(),
        new_affine.affine_map, new_affine.operands);
  } else {
    CHECK(false);
  }
}

void SetMemRef(mlir::Operation* op, mlir::Value memref) {
  if (auto load = mlir::dyn_cast<mlir::AffineLoadOp>(op)) {
    load.setMemRef(memref);
  } else if (auto store = mlir::dyn_cast<mlir::AffineStoreOp>(op)) {
    store.setMemRef(memref);
  } else {
    CHECK(false);
  }
}

std::vector<mlir::AffineForOp> CreateNestedSimpleLoops(
    absl::Span<const int64_t> upper_bounds, mlir::OpBuilder builder) {
  std::vector<mlir::AffineForOp> loops;
  loops.reserve(upper_bounds.size());
  for (int64_t dim : upper_bounds) {
    auto loop =
        builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 0, dim);
    loops.push_back(loop);
    builder = loop.getBodyBuilder();
  }
  return loops;
}

void SetBoundForSimpleLoop(mlir::AffineForOp loop, mlir::AffineExpr new_bound,
                           mlir::OpBuilder builder) {
  CHECK(IsSimpleLoop(loop));

  loop.setUpperBoundMap(mlir::AffineMap::get(
      loop.getUpperBoundMap().getNumDims(),
      loop.getUpperBoundMap().getNumSymbols(), {new_bound}));
}

// Tile a loop with trip count N by `size`. For now, N has to be a multiple of
// size, but later this constraint will be removed.
//
// The major loop (with trip count N / size) stays as-is, while the minor loop
// (with trip count `size`) will take over the body of `target`, and be placed
// as the new body of `target`.
//
// `target` has to be within the same "perfectly nested loop group" as `loop`.
// See the documentation for mlir::getPerfectlyNestedLoops.
//
// Example:
// Before tiling `loop` with tile size X:
//   for (loop in N)
//     for (unrelated_loop in ...)
//       for (target in ...)
//         // pass loop into affine maps
// After:
//   for (loop in N / X)
//     for (unrelated_loop in ...)
//       for (target in ...)
//         for (tiled_loop in X)
//           // rewrite all affine exprs from loop to `loop * X + tiled_loop`.
//
// Design note:
// TileLoop is different from mlir::tile. At the moment, mlir::tile is not well
// documented about the exact tiling semantics, but the observed behavior is:
//   for (i from 0 to N)
//     for (unrelated_loop in ...)
//       for (target in ...)
//         // pass i into affine maps
// =>
//   for (i from 0 to N, step = X)
//     for (unrelated_loop in ...)
//       for (target in ...)
//         for (j from i to min(i + X, N), step = 1)
//           // pass j into affine maps
//
// There are two differences between mlir::tile and TileLoop:
// * TileLoop always puts the tiling logic "stepping" logic into AffineExprs.
//   With that all index calculation is done in AffineExprs and easier to
//   analyze in a single place.
// * TileLoop doesn't plan to use use max() and min() to resolve the issue when
//   N % X != 0. max() and min() are not representable in AffineExprs.
//   TODO(timshen): support the case where N % X != 0.
//
// TODO(timshen): consider the possibility to reuse mlir::tile's logic to
// achieve the same goal.
mlir::AffineForOp TileLoop(mlir::AffineForOp loop, int64_t size,
                           mlir::AffineForOp target) {
  CHECK(IsSimpleLoop(loop));
  CHECK(IsSimpleLoop(target));
  {
    llvm::SmallVector<mlir::AffineForOp, 4> all_loops;
    getPerfectlyNestedLoops(all_loops, loop);
    CHECK(absl::c_linear_search(all_loops, target));
  }

  auto builder = target.getBodyBuilder();

  auto inner_loop =
      builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 0, size);
  {
    auto& inner_operations = inner_loop.getBody()->getOperations();
    auto& target_operations = target.getBody()->getOperations();

    inner_operations.splice(inner_operations.begin(), target_operations,
                            target_operations.begin(),
                            std::prev(target_operations.end(), 2));

    mlir::AffineExpr length = loop.getUpperBoundMap().getResult(0);
    CHECK_EQ(0, length.cast<mlir::AffineConstantExpr>().getValue() % size);
    SetBoundForSimpleLoop(loop, length.ceilDiv(size), builder);
  }

  for (auto& use :
       llvm::make_early_inc_range(loop.getInductionVar().getUses())) {
    mlir::Operation* owner = use.getOwner();
    BoundAffineMap affine_map = GetBoundAffineMapFrom(owner);
    unsigned new_dim = affine_map.operands.size();
    affine_map.operands.push_back(inner_loop.getInductionVar());
    std::vector<mlir::AffineExpr> replacements;
    for (int i = 0; i < affine_map.affine_map.getNumDims(); i++) {
      if (affine_map.operands[i] == loop.getInductionVar()) {
        replacements.push_back(builder.getAffineDimExpr(i) * size +
                               builder.getAffineDimExpr(new_dim));
      } else {
        replacements.push_back(builder.getAffineDimExpr(i));
      }
    }
    affine_map.affine_map = affine_map.affine_map.replaceDimsAndSymbols(
        replacements, {}, affine_map.operands.size(), 0);
    auto new_op =
        CloneWithNewAffineMap(owner, affine_map, mlir::OpBuilder(owner));
    owner->replaceAllUsesWith(new_op);
    owner->erase();
  }
  return inner_loop;
}

// Hoist operations out of `where`. [begin_op, end_op) must be the first
// operations of their parent loop, and `where` must be an ancestor of that
// parent loop.
//
// It always preserves the semantics of the program, therefore it may modify the
// hoisted operations or add extra loops at the hoisted place.
mlir::Operation* HoistAndFix(llvm::iplist<mlir::Operation>::iterator begin_op,
                             llvm::iplist<mlir::Operation>::iterator end_op,
                             mlir::AffineForOp where) {
  // All loops to hoist through.
  llvm::SmallVector<mlir::AffineForOp, 4> ancestors;
  getPerfectlyNestedLoops(ancestors, where);
  {
    int i;
    for (i = 0; i < ancestors.size(); i++) {
      if (&ancestors[i].getBody()->front() == &*begin_op) {
        break;
      }
    }
    CHECK(i < ancestors.size());
    ancestors.resize(i + 1);
  }

  std::vector<int64_t> ancestor_dimensions;
  for (auto ancestor : ancestors) {
    CHECK(IsSimpleLoop(ancestor));
    ancestor_dimensions.push_back(
        ancestor.getUpperBoundMap().getSingleConstantResult());
  }

  if (auto alloc = mlir::dyn_cast<mlir::AllocOp>(begin_op)) {
    CHECK(std::next(begin_op) == end_op)
        << "alloc() needs to be hoisted by its own";

    mlir::OpBuilder builder(where);
    mlir::MemRefType type = alloc.getType();
    CHECK(type.getAffineMaps().empty());
    ancestor_dimensions.insert(ancestor_dimensions.end(),
                               type.getShape().begin(), type.getShape().end());
    mlir::MemRefType new_type =
        mlir::MemRefType::get(ancestor_dimensions, type.getElementType());
    auto new_alloc =
        builder.create<mlir::AllocOp>(builder.getUnknownLoc(), new_type);

    std::vector<mlir::Value> indvars;
    for (auto ancestor : ancestors) {
      indvars.push_back(ancestor.getInductionVar());
    }
    for (auto& use : llvm::make_early_inc_range(alloc.getResult().getUses())) {
      mlir::Operation* owner = use.getOwner();
      BoundAffineMap affine_map = GetBoundAffineMapFrom(owner);
      affine_map.operands.insert(affine_map.operands.begin(), indvars.begin(),
                                 indvars.end());
      CHECK(affine_map.affine_map.isIdentity());
      affine_map.affine_map = mlir::AffineMap::getMultiDimIdentityMap(
          affine_map.operands.size(), builder.getContext());

      mlir::Operation* new_op =
          CloneWithNewAffineMap(owner, affine_map, mlir::OpBuilder(owner));
      SetMemRef(new_op, new_alloc);
      owner->replaceAllUsesWith(new_op);
      owner->erase();
    }
    alloc.erase();
    return new_alloc;
  }

  const bool any_op_is_loop_variant = [&] {
    for (mlir::Operation& op : llvm::make_range(begin_op, end_op)) {
      if (mlir::isa<mlir::AffineForOp>(op) ||
          mlir::isa<mlir::AffineStoreOp>(op)) {
        return true;
      }
    }
    return false;
  }();

  if (any_op_is_loop_variant) {
    auto builder = mlir::OpBuilder(where);
    std::vector<mlir::AffineForOp> new_loops;
    for (auto dim : ancestor_dimensions) {
      auto where =
          builder.create<mlir::AffineForOp>(builder.getUnknownLoc(), 0, dim);
      new_loops.push_back(where);
      builder = where.getBodyBuilder();
    }
    for (mlir::Operation& op :
         llvm::make_early_inc_range(llvm::make_range(begin_op, end_op))) {
      op.moveBefore(&new_loops.back().getBody()->back());
    }
    CHECK_EQ(ancestors.size(), new_loops.size());
    for (int i = 0; i < ancestors.size(); i++) {
      replaceAllUsesInRegionWith(ancestors[i].getInductionVar(),
                                 new_loops[i].getInductionVar(),
                                 new_loops.back().region());
    }
    return new_loops.front();
  }
  CHECK(false);
}

mlir::Operation* HoistAndFix(mlir::Operation* op, mlir::AffineForOp where) {
  return HoistAndFix(op->getIterator(), std::next(op->getIterator()), where);
}

// Sinks a segment of perfectly nested loops to the bottom. It implements this
// by rotating the loop nest by rotate_amount.
void SinkPerfectlyNestedLoops(absl::Span<const mlir::AffineForOp> loops,
                              int rotate_amount) {
  CHECK_GE(rotate_amount, 0);
  std::vector<unsigned> permutation(loops.size());
  std::iota(permutation.begin(), permutation.end(), unsigned(0));
  std::rotate(permutation.begin(),
              permutation.begin() + loops.size() - rotate_amount,
              permutation.end());
  mlir::interchangeLoops(
      llvm::ArrayRef<mlir::AffineForOp>(loops.begin(), loops.end()),
      permutation);
}

struct InitialMlirConvAnchors {
  std::vector<mlir::AffineForOp> cartesian_product_loops;
  std::vector<mlir::AffineForOp> reduction_loops;
  mlir::AllocOp output_acc;
};

// Return the following IR with the anchors set to corresponding operations.
//   for (cartesian loops...) {
//     %output_acc = alloc() : memref(f32)
//     output_acc[] = 0
//     for (reduction loops...) {
//       output_acc[] += input[...] * filter[...]
//     }
//     output[...] = output_acc[]
//   }
StatusOr<InitialMlirConvAnchors> CreateNaiveMlirConv(
    mlir::Value input, mlir::Value filter, mlir::Value output,
    const ShapeInfo& input_shape_info, const ShapeInfo& filter_shape_info,
    const ShapeInfo& output_shape_info, const Window& window,
    mlir::OpBuilder builder) {
  CHECK(input_shape_info.element_type == builder.getF16Type());
  CHECK(filter_shape_info.element_type == builder.getF16Type());
  CHECK(output_shape_info.element_type == builder.getF16Type());

  auto location = mlir::UnknownLoc::get(builder.getContext());

  std::vector<mlir::AffineForOp> cartesian_product_loops =
      CreateNestedSimpleLoops(output_shape_info.nchw_dimensions, builder);

  builder = cartesian_product_loops.back().getBodyBuilder();

  mlir::AllocOp output_acc = builder.create<mlir::AllocOp>(
      location, mlir::MemRefType::get({}, builder.getF32Type()));

  builder.create<mlir::AffineStoreOp>(
      location,
      builder.create<mlir::ConstantOp>(
          location, mlir::FloatAttr::get(builder.getF32Type(), 0)),
      output_acc, llvm::ArrayRef<mlir::Value>());

  std::vector<mlir::AffineForOp> reduction_loops;
  reduction_loops = CreateNestedSimpleLoops(
      absl::MakeSpan(filter_shape_info.nchw_dimensions).subspan(1), builder);

  mlir::AffineForOp loop_n = cartesian_product_loops[0];
  mlir::AffineForOp loop_o = cartesian_product_loops[1];
  mlir::AffineForOp loop_c = reduction_loops[0];

  std::vector<mlir::Value> output_spatial_indvars;
  for (auto loop : absl::MakeSpan(cartesian_product_loops).subspan(2)) {
    output_spatial_indvars.push_back(loop.getInductionVar());
  }
  std::vector<mlir::Value> filter_spatial_indvars;
  for (auto loop : absl::MakeSpan(reduction_loops).subspan(1)) {
    filter_spatial_indvars.push_back(loop.getInductionVar());
  }
  int num_spatial_dims = output_spatial_indvars.size();
  CHECK_EQ(num_spatial_dims, filter_spatial_indvars.size());

  builder = reduction_loops.back().getBodyBuilder();

  mlir::Value loaded_input = [&] {
    std::vector<mlir::AffineExpr> input_indices;
    input_indices.push_back(builder.getAffineDimExpr(0));
    input_indices.push_back(builder.getAffineDimExpr(1));

    // For spatial dimensions, generate input_index * stride + filter_index -
    // left_pad
    //
    // TODO(timshen): guard out-of-bound loads and stores brought by padding.
    for (int i = 0; i < num_spatial_dims; i++) {
      const WindowDimension& window_dim = window.dimensions(i);
      input_indices.push_back(
          builder.getAffineDimExpr(i + 2) * window_dim.stride() +
          builder.getAffineDimExpr(2 + num_spatial_dims + i) -
          window_dim.padding_low());
    }
    std::vector<mlir::Value> input_vars;
    input_vars.push_back(loop_n.getInductionVar());
    input_vars.push_back(loop_c.getInductionVar());
    input_vars.insert(input_vars.end(), output_spatial_indvars.begin(),
                      output_spatial_indvars.end());
    input_vars.insert(input_vars.end(), filter_spatial_indvars.begin(),
                      filter_spatial_indvars.end());

    return builder.create<mlir::FPExtOp>(
        location,
        builder.createOrFold<mlir::AffineLoadOp>(
            location, input,
            mlir::AffineMap(input_shape_info.affine_map)
                .compose(
                    mlir::AffineMap::get(/*dimCount=*/2 + num_spatial_dims * 2,
                                         /*symbolCount=*/0, input_indices)),
            input_vars),
        builder.getF32Type());
  }();

  mlir::Value loaded_filter = [&] {
    std::vector<mlir::Value> filter_vars;
    filter_vars.push_back(loop_o.getInductionVar());
    filter_vars.push_back(loop_c.getInductionVar());
    filter_vars.insert(filter_vars.end(), filter_spatial_indvars.begin(),
                       filter_spatial_indvars.end());

    return builder.create<mlir::FPExtOp>(
        location,
        builder.createOrFold<mlir::AffineLoadOp>(
            location, filter, filter_shape_info.affine_map, filter_vars),
        builder.getF32Type());
  }();

  builder.createOrFold<mlir::AffineStoreOp>(
      location,
      builder.create<mlir::AddFOp>(
          location,
          builder.createOrFold<mlir::AffineLoadOp>(location, output_acc),
          builder.create<mlir::MulFOp>(location, loaded_input, loaded_filter)),
      output_acc, llvm::ArrayRef<mlir::Value>());

  builder.setInsertionPointAfter(reduction_loops[0]);
  {
    std::vector<mlir::Value> output_vars;
    output_vars.push_back(loop_n.getInductionVar());
    output_vars.push_back(loop_o.getInductionVar());
    output_vars.insert(output_vars.end(), output_spatial_indvars.begin(),
                       output_spatial_indvars.end());
    builder.createOrFold<mlir::AffineStoreOp>(
        location,
        builder.create<mlir::FPTruncOp>(
            location,
            builder.createOrFold<mlir::AffineLoadOp>(location, output_acc),
            builder.getF16Type()),
        output, output_shape_info.affine_map, output_vars);
  }

  return InitialMlirConvAnchors{cartesian_product_loops, reduction_loops,
                                output_acc};
}

// Contains the following pattern with anchors:
//   for (cartesian loops...) {
//     %output_acc = alloc() : memref(..., f32)
//     for (reduction loops...) {
//       for (tiled cartesian loops...) {
//         output_acc[...] = 0
//       }
//       for (tiled cartesian loops...) {
//         for (reduction loops...) {
//           output_acc[] += input[...] * filter[...]
//         }
//       }
//       for (tiled cartesian loops...) {
//         output[...] = output_acc[...]
//       }
//     }
//   }
struct TransformedMlirConvAnchors {
  std::vector<mlir::AffineForOp> cartesian_product_loops;
  std::vector<mlir::AffineForOp> reduction_loops;
};

StatusOr<TransformedMlirConvAnchors> TransformMlirConv(
    InitialMlirConvAnchors anchors) {
  std::vector<mlir::AffineForOp> cartesian_product_loops =
      anchors.cartesian_product_loops;
  std::vector<mlir::AffineForOp> reduction_loops = anchors.reduction_loops;
  mlir::AllocOp output_acc = anchors.output_acc;

  // TODO(timshen): consider using pattern matchers for transformations
  //
  // Initial form:
  //   for (cartesian loops...) {
  //     %output_acc = alloc() : memref(f32)
  //     output_acc[] = 0
  //     for (reduction loops...) {
  //       output_acc[] += input[...] * filter[...]
  //     }
  //     output[...] = output_acc[]
  //   }

  // Tile cartesian loops to:
  //   for (cartesian loops...) {
  //     for (tiled cartesian loops...) {
  //       %output_acc = alloc() : memref(f32)
  //       output_acc[] = 0
  //       for (reduction loops...) {
  //         output_acc[] += input[...] * filter[...]
  //       }
  //       output[...] = output_acc[]
  //     }
  //   }
  TileLoop(reduction_loops[0], 4, reduction_loops.back());

  std::vector<mlir::AffineForOp> tiled_cartesian_loops;
  tiled_cartesian_loops.push_back(
      TileLoop(cartesian_product_loops[1], 32, cartesian_product_loops.back()));

  tiled_cartesian_loops.push_back(TileLoop(cartesian_product_loops.back(), 16,
                                           tiled_cartesian_loops.back()));

  // Two hoist operations to interleave the allocation, computation, and
  // writebacks to output_acc:
  // After first hoist:
  //   for (cartesian loops...) {
  //     %output_acc = alloc() : memref(..., f32)
  //     for (tiled cartesian loops...) {
  //       output_acc[...] = 0
  //       for (reduction loops...) {
  //         output_acc[...] += input[...] * filter[...]
  //       }
  //       output[...] = output_acc[...]
  //     }
  //   }
  output_acc = llvm::cast<mlir::AllocOp>(
      HoistAndFix(output_acc, tiled_cartesian_loops.front()));

  // Hoist everything before reduction loops (aka zero initializations of
  // output_acc):
  //   for (cartesian loops...) {
  //     %output_acc = alloc() : memref(..., f32)
  //     for (tiled cartesian loops...) {
  //       output_acc[...] = 0
  //     }
  //     for (tiled cartesian loops...) {
  //       for (reduction loops...) {
  //         output_acc[...] += input[...] * filter[...]
  //       }
  //       output[...] = output_acc[...]
  //     }
  //   }
  HoistAndFix(tiled_cartesian_loops.back().getBody()->begin(),
              reduction_loops.front().getOperation()->getIterator(),
              tiled_cartesian_loops.front());

  // Now hoist all reduction loops outside of tiled cartesian loops.
  // Notice that HoistAndFix automatically add a new set of tiled cartesian
  // loops for hoisted reduction loops to keep the semantics correct.
  //
  // After second hoist:
  //   for (cartesian loops...) {
  //     %output_acc = alloc() : memref(..., f32)
  //     for (tiled cartesian loops...) {
  //       output_acc[...] = 0
  //     }
  //     for (tiled cartesian loops...) {
  //       for (reduction loops...) {
  //         output_acc[] += input[...] * filter[...]
  //       }
  //     }  // compute loop
  //     for (tiled cartesian loops...) {
  //       output[...] = output_acc[...]
  //     }
  //   }
  {
    auto compute_loop = llvm::cast<mlir::AffineForOp>(
        HoistAndFix(reduction_loops.front(), tiled_cartesian_loops[0]));

    // Fix tiled_cartesian_loops to make them point to the tiled compute loops,
    // not the writeback loops to output buffer.
    llvm::SmallVector<mlir::AffineForOp, 4> all_loops;
    getPerfectlyNestedLoops(all_loops, compute_loop);
    absl::c_copy_n(all_loops, tiled_cartesian_loops.size(),
                   tiled_cartesian_loops.data());
  }

  // After exchanging tiled cartesian compute loops with reduction loops:
  //   for (cartesian loops...) {
  //     %output_acc = alloc() : memref(..., f32)
  //     for (tiled cartesian loops...) {
  //       output_acc[...] = 0
  //     }
  //     for (reduction loops...) {
  //       for (tiled cartesian loops...) {
  //         output_acc[] += input[...] * filter[...]
  //       }
  //     }
  //     for (tiled cartesian loops...) {
  //       output[...] = output_acc[...]
  //     }
  //   }
  //
  // ...so that later tiled cartesian loops (with computations in it) can be
  // replaced by CUDA MMA instructions.
  {
    std::vector<mlir::AffineForOp> loops;
    loops.insert(loops.end(), tiled_cartesian_loops.begin(),
                 tiled_cartesian_loops.end());
    loops.insert(loops.end(), reduction_loops.begin(), reduction_loops.end());
    SinkPerfectlyNestedLoops(loops, tiled_cartesian_loops.size());
  }
  return TransformedMlirConvAnchors{cartesian_product_loops, reduction_loops};
}

}  // namespace

StatusOr<mlir::FuncOp> EmitConvolutionForwardAsMlir(
    HloInstruction* conv, absl::string_view function_name,
    mlir::MLIRContext* context) {
  mlir::OpBuilder builder(context);

  const auto& dim_nums = conv->convolution_dimension_numbers();
  ShapeInfo input_shape_info =
      GetShapeInfo(conv->operand(0)->shape(), dim_nums.input_batch_dimension(),
                   dim_nums.input_feature_dimension(),
                   dim_nums.input_spatial_dimensions(), builder);

  ShapeInfo filter_shape_info = GetShapeInfo(
      conv->operand(1)->shape(), dim_nums.kernel_output_feature_dimension(),
      dim_nums.kernel_input_feature_dimension(),
      dim_nums.kernel_spatial_dimensions(), builder);

  ShapeInfo output_shape_info = GetShapeInfo(
      conv->shape().tuple_shapes(0), dim_nums.output_batch_dimension(),
      dim_nums.output_feature_dimension(), dim_nums.output_spatial_dimensions(),
      builder);

  auto function = mlir::FuncOp::create(
      mlir::UnknownLoc::get(builder.getContext()),
      llvm_ir::AsStringRef(function_name),
      builder.getFunctionType(
          {mlir::MemRefType::get(output_shape_info.physical_dimensions,
                                 output_shape_info.element_type, {}),
           mlir::MemRefType::get(input_shape_info.physical_dimensions,
                                 input_shape_info.element_type, {}),
           mlir::MemRefType::get(filter_shape_info.physical_dimensions,
                                 filter_shape_info.element_type, {})},
          {}));

  auto* entry_block = function.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);
  builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(entry_block);

  mlir::Value input = entry_block->getArgument(1);
  mlir::Value filter = entry_block->getArgument(2);
  mlir::Value output = entry_block->getArgument(0);

  TF_RETURN_IF_ERROR(ConvIsImplemented(conv));

  TF_ASSIGN_OR_RETURN(
      InitialMlirConvAnchors initial_anchors,
      CreateNaiveMlirConv(input, filter, output, input_shape_info,
                          filter_shape_info, output_shape_info, conv->window(),
                          builder));

  TF_ASSIGN_OR_RETURN(TransformedMlirConvAnchors transformed_anchors,
                      TransformMlirConv(initial_anchors));

  // TODO(timshen): Implement a transformation that collects loads to a given
  // buffer, create a local alloc() for the accessed part, redirects all loads
  // and stores to that local alloc(), and create code to initialize /
  // writeback the local alloc() if needed.

  // TODO(timshen): Implement CUDA-specific lowering.

  return function;
}

Status ConvIsImplemented(const HloInstruction* conv) {
  if (conv->feature_group_count() != 1 || conv->batch_group_count() != 1) {
    return Unimplemented("group count is not implemented.");
  }
  if (window_util::HasWindowReversal(conv->window())) {
    return Unimplemented("Window reversal is not implemented.");
  }
  if (window_util::HasDilation(conv->window())) {
    return Unimplemented("Dilation is not implemented.");
  }
  return Status::OK();
}

}  // namespace mlir_gpu
}  // namespace xla

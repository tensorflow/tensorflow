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

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AffineExpr.h"  // TF:local_config_mlir
#include "mlir/IR/AffineMap.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Transforms/LoopUtils.h"  // TF:local_config_mlir
#include "mlir/Transforms/RegionUtils.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace mlir_gpu {
namespace {

class NameLookupTable {
 public:
  llvm::StringRef FindOrCreate(mlir::Value* key) {
    auto it = names_.find(key);
    if (it != names_.end()) {
      return it->second;
    }
    names_.try_emplace(key, std::string("%") + std::to_string(count_++));
    return names_.at(key);
  }

  void Insert(mlir::Value* key, llvm::StringRef value) {
    CHECK(names_.try_emplace(key, std::string(value)).second);
  }

 private:
  int64 count_ = 0;
  absl::flat_hash_map<mlir::Value*, std::string> names_;
};

std::vector<int64_t> GetLogicalToPhysicalPermutationOrDie(
    mlir::AffineMap affine_map) {
  CHECK_EQ(0, affine_map.getNumSymbols());
  CHECK_EQ(affine_map.getNumDims(), affine_map.getNumResults());

  std::vector<int64_t> permutation(affine_map.getNumResults(), -1);
  int count = 0;
  for (auto expr : affine_map.getResults()) {
    int64_t dim = expr.cast<mlir::AffineDimExpr>().getPosition();
    CHECK_GE(dim, 0);
    CHECK_EQ(-1, std::exchange(permutation[dim], count));
    count++;
  }
  return permutation;
}

std::vector<int64_t> GetLogicalDimensions(mlir::MemRefType type) {
  CHECK_EQ(1, type.getAffineMaps().size());

  auto affine_map = type.getAffineMaps()[0];
  CHECK(affine_map.isPermutation());

  std::vector<int64_t> permutation =
      GetLogicalToPhysicalPermutationOrDie(affine_map);
  std::vector<int64_t> dimensions(type.getShape().size());
  for (int i = 0; i < type.getShape().size(); i++) {
    dimensions[i] = type.getShape()[permutation[i]];
  }
  return dimensions;
}

mlir::Value* NormalizeMlirShapeToLogicalNchw(
    int64 n_dim, int64 c_dim, absl::Span<const int64> spatial_dims,
    mlir::Value* buffer, mlir::OpBuilder builder) {
  mlir::MemRefType type = buffer->getType().cast<mlir::MemRefType>();
  CHECK_EQ(1, type.getAffineMaps().size());

  std::vector<mlir::AffineExpr> exprs;
  exprs.push_back(builder.getAffineDimExpr(n_dim));
  exprs.push_back(builder.getAffineDimExpr(c_dim));
  for (int64 dim : spatial_dims) {
    exprs.push_back(builder.getAffineDimExpr(dim));
  }

  return builder.createOrFold<mlir::MemRefCastOp>(
      builder.getUnknownLoc(),
      builder.getMemRefType(
          type.getShape(), type.getElementType(),
          mlir::AffineMap(type.getAffineMaps()[0])
              .compose(builder.getAffineMap(exprs.size(), 0, exprs))),
      buffer);
}

void PrintExprWithBindings(mlir::AffineExpr expr,
                           absl::Span<mlir::Value* const> indices,
                           NameLookupTable* names, llvm::raw_ostream& p) {
  if (auto bin = expr.dyn_cast<mlir::AffineBinaryOpExpr>()) {
    p << "(";
    PrintExprWithBindings(bin.getLHS(), indices, names, p);
    switch (expr.getKind()) {
      case mlir::AffineExprKind::Add:
        p << " + ";
        break;
      case mlir::AffineExprKind::Mul:
        p << " * ";
        break;
      case mlir::AffineExprKind::Mod:
        p << " % ";
        break;
      case mlir::AffineExprKind::FloorDiv:
        p << " floordiv ";
        break;
      case mlir::AffineExprKind::CeilDiv:
        p << " ceildiv ";
        break;
      default:
        p << " <unknown affine expr> ";
    }
    PrintExprWithBindings(bin.getRHS(), indices, names, p);
    p << ")";
    return;
  }
  if (auto constant = expr.dyn_cast<mlir::AffineConstantExpr>()) {
    p << constant.getValue();
    return;
  }
  if (auto dim_expr = expr.dyn_cast<mlir::AffineDimExpr>()) {
    p << names->FindOrCreate(indices[dim_expr.getPosition()]);
    return;
  }
  CHECK(false);
}

bool IsSimpleLoop(mlir::AffineForOp loop) {
  return loop.getLowerBoundMap().isSingleConstant() &&
         loop.getLowerBoundMap().getSingleConstantResult() == 0 &&
         loop.getStep() == 1 && loop.getUpperBoundMap().getNumResults() == 1 &&
         std::next(loop.region().begin()) == loop.region().end();
}

void PrintRangeForAffineLoop(mlir::AffineForOp loop, NameLookupTable* names,
                             llvm::raw_ostream& p) {
  CHECK(IsSimpleLoop(loop));
  p << names->FindOrCreate(loop.getInductionVar());
  p << " in ";
  auto affine_map = loop.getUpperBoundMap();
  CHECK_EQ(1, affine_map.getNumResults());
  PrintExprWithBindings(
      affine_map.getResult(0),
      std::vector<mlir::Value*>(loop.getUpperBoundOperands().begin(),
                                loop.getUpperBoundOperands().end()),
      names, p);
}

void PrintIndices(mlir::AffineMap affine_map,
                  absl::Span<mlir::Value* const> indices,
                  NameLookupTable* names, llvm::raw_ostream& p) {
  bool first = true;
  for (mlir::AffineExpr expr : affine_map.getResults()) {
    if (!std::exchange(first, false)) {
      p << ", ";
    }
    PrintExprWithBindings(expr, indices, names, p);
  }
}

void ToStringImpl(mlir::Operation* node, NameLookupTable* names,
                  int indent_level, llvm::raw_ostream& p) {
  if (mlir::isa<mlir::AffineTerminatorOp>(node)) {
    return;
  }
  p << "\n";
  for (int i = 0; i < indent_level; i++) {
    p << "  ";
  }
  if (auto loop = mlir::dyn_cast<mlir::AffineForOp>(node)) {
    llvm::SmallVector<mlir::AffineForOp, 4> all_loops;
    getPerfectlyNestedLoops(all_loops, loop);
    CHECK(!all_loops.empty());
    p << "for (";
    bool first = true;
    for (const auto& loop : all_loops) {
      if (!std::exchange(first, false)) {
        p << ", ";
      }
      PrintRangeForAffineLoop(loop, names, p);
    }
    CHECK_EQ(1, all_loops.back().region().getBlocks().size());
    p << ") {";
    for (mlir::Operation& op : all_loops.back().getBody()->getOperations()) {
      ToStringImpl(&op, names, indent_level + 1, p);
    }
    p << "\n";
    for (int i = 0; i < indent_level; i++) {
      p << "  ";
    }
    p << "}";
    return;
  }
  if (auto load = mlir::dyn_cast<mlir::AffineLoadOp>(node)) {
    p << names->FindOrCreate(load.getResult());
    p << " = ";
    p << names->FindOrCreate(load.getMemRef());
    p << "[";
    PrintIndices(load.getAffineMap(),
                 std::vector<mlir::Value*>(load.getMapOperands().begin(),
                                           load.getMapOperands().end()),
                 names, p);
    p << "]";
    return;
  }
  if (auto store = mlir::dyn_cast<mlir::AffineStoreOp>(node)) {
    p << names->FindOrCreate(store.getMemRef());
    p << "[";
    PrintIndices(store.getAffineMap(),
                 std::vector<mlir::Value*>(store.getMapOperands().begin(),
                                           store.getMapOperands().end()),
                 names, p);
    p << "]";
    p << " = ";
    p << names->FindOrCreate(store.getValueToStore());
    return;
  }
  if (auto alloca = mlir::dyn_cast<mlir::AllocOp>(node)) {
    p << names->FindOrCreate(alloca.getResult());
    p << " = alloc() : ";
    alloca.getType().print(p);
    return;
  }
  if (auto add = mlir::dyn_cast<mlir::AddFOp>(node)) {
    p << names->FindOrCreate(add.getResult());
    p << " = addf(";
    p << names->FindOrCreate(add.lhs());
    p << ", ";
    p << names->FindOrCreate(add.rhs());
    p << ")";
    return;
  }
  if (auto mul = mlir::dyn_cast<mlir::MulFOp>(node)) {
    p << names->FindOrCreate(mul.getResult());
    p << " = mulf(";
    p << names->FindOrCreate(mul.lhs());
    p << ", ";
    p << names->FindOrCreate(mul.rhs());
    p << ")";
    return;
  }
  if (auto constant = mlir::dyn_cast<mlir::ConstantOp>(node)) {
    p << names->FindOrCreate(constant.getResult());
    p << " = ";
    constant.getValue().print(p);
    return;
  }
  CHECK(false);
}

std::string ToString(mlir::Operation* node, const NameLookupTable& names) {
  auto local_names = names;
  std::string str;
  llvm::raw_string_ostream strstream(str);
  ToStringImpl(node, &local_names, 0, strstream);
  return str;
}

struct BoundAffineMap {
  mlir::AffineMap affine_map;
  std::vector<mlir::Value*> operands;
};

BoundAffineMap GetBoundAffineMapFrom(mlir::Operation* op) {
  if (auto load = mlir::dyn_cast<mlir::AffineLoadOp>(op)) {
    return {load.getAffineMap(),
            std::vector<mlir::Value*>(load.getMapOperands().begin(),
                                      load.getMapOperands().end())};
  } else if (auto store = mlir::dyn_cast<mlir::AffineStoreOp>(op)) {
    return {store.getAffineMap(),
            std::vector<mlir::Value*>(store.getMapOperands().begin(),
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

void SetMemRef(mlir::Operation* op, mlir::Value* memref) {
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

  loop.setUpperBoundMap(builder.getAffineMap(
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

  for (mlir::IROperand& use :
       llvm::make_early_inc_range(loop.getInductionVar()->getUses())) {
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
// It always preseves the semantics of the program, therefore it may modify the
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
        builder.getMemRefType(ancestor_dimensions, type.getElementType());
    auto new_alloc =
        builder.create<mlir::AllocOp>(builder.getUnknownLoc(), new_type);

    std::vector<mlir::Value*> indvars;
    for (auto ancestor : ancestors) {
      indvars.push_back(ancestor.getInductionVar());
    }
    for (mlir::IROperand& use :
         llvm::make_early_inc_range(alloc.getResult()->getUses())) {
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
    HloInstruction* custom_call, mlir::Value* input, mlir::Value* filter,
    mlir::Value* output, mlir::OpBuilder builder, NameLookupTable* names) {
  TF_RETURN_IF_ERROR(ConvIsImplemented(custom_call));

  const auto& dim_nums = custom_call->convolution_dimension_numbers();
  input = NormalizeMlirShapeToLogicalNchw(
      dim_nums.input_batch_dimension(), dim_nums.input_feature_dimension(),
      dim_nums.input_spatial_dimensions(), input, builder);

  filter = NormalizeMlirShapeToLogicalNchw(
      dim_nums.kernel_output_feature_dimension(),
      dim_nums.kernel_input_feature_dimension(),
      dim_nums.kernel_spatial_dimensions(), filter, builder);

  output = NormalizeMlirShapeToLogicalNchw(
      dim_nums.output_batch_dimension(), dim_nums.output_feature_dimension(),
      dim_nums.output_spatial_dimensions(), output, builder);

  auto location = mlir::UnknownLoc::get(builder.getContext());
  auto input_shape = input->getType().cast<mlir::MemRefType>();
  auto filter_shape = filter->getType().cast<mlir::MemRefType>();
  auto output_shape = output->getType().cast<mlir::MemRefType>();

  CHECK(input_shape.hasStaticShape());
  CHECK(filter_shape.hasStaticShape());
  CHECK(output_shape.hasStaticShape());

  names->Insert(input, "input");
  names->Insert(filter, "filter");
  names->Insert(output, "output");

  std::vector<mlir::AffineForOp> cartesian_product_loops =
      CreateNestedSimpleLoops(GetLogicalDimensions(output_shape), builder);
  // batch dimension
  names->Insert(cartesian_product_loops[0].getInductionVar(), "n");
  // output feature dimension
  names->Insert(cartesian_product_loops[1].getInductionVar(), "o");
  // height index
  names->Insert(cartesian_product_loops[2].getInductionVar(), "hi");
  // width index
  names->Insert(cartesian_product_loops[3].getInductionVar(), "wi");

  builder = cartesian_product_loops.back().getBodyBuilder();

  mlir::AllocOp output_acc = builder.create<mlir::AllocOp>(
      location, builder.getMemRefType({}, builder.getF32Type()));

  builder.create<mlir::AffineStoreOp>(
      location,
      builder.create<mlir::ConstantOp>(
          location, builder.getFloatAttr(builder.getF32Type(), 0)),
      output_acc, llvm::ArrayRef<mlir::Value*>());

  std::vector<mlir::AffineForOp> reduction_loops;
  {
    std::vector<int64_t> dims = GetLogicalDimensions(filter_shape);
    reduction_loops =
        CreateNestedSimpleLoops(absl::MakeSpan(dims).subspan(1), builder);
  }
  // input feature dimension
  names->Insert(reduction_loops[0].getInductionVar(), "c");
  // filter height index
  names->Insert(reduction_loops[1].getInductionVar(), "fhi");
  // filter width index
  names->Insert(reduction_loops[2].getInductionVar(), "fwi");

  mlir::AffineForOp loop_n = cartesian_product_loops[0];
  mlir::AffineForOp loop_o = cartesian_product_loops[1];
  mlir::AffineForOp loop_c = reduction_loops[0];

  std::vector<mlir::Value*> output_spatial_indvars;
  for (auto loop : absl::MakeSpan(cartesian_product_loops).subspan(2)) {
    output_spatial_indvars.push_back(loop.getInductionVar());
  }
  std::vector<mlir::Value*> filter_spatial_indvars;
  for (auto loop : absl::MakeSpan(reduction_loops).subspan(1)) {
    filter_spatial_indvars.push_back(loop.getInductionVar());
  }
  int num_spatial_dims = output_spatial_indvars.size();
  CHECK_EQ(num_spatial_dims, filter_spatial_indvars.size());

  builder = reduction_loops.back().getBodyBuilder();

  mlir::Value* loaded_input = [&] {
    std::vector<mlir::AffineExpr> input_indices;
    input_indices.push_back(builder.getAffineDimExpr(0));
    input_indices.push_back(builder.getAffineDimExpr(1));

    // For spatial dimensions, generate input_index * stride + filter_index -
    // left_pad
    //
    // TODO(timshen): guard out-of-bound loads and stores brought by padding.
    for (int i = 0; i < num_spatial_dims; i++) {
      const WindowDimension& window = custom_call->window().dimensions(i);
      input_indices.push_back(
          builder.getAffineDimExpr(i + 2) * window.stride() +
          builder.getAffineDimExpr(2 + num_spatial_dims + i) -
          window.padding_low());
    }
    std::vector<mlir::Value*> input_vars;
    input_vars.push_back(loop_n.getInductionVar());
    input_vars.push_back(loop_c.getInductionVar());
    input_vars.insert(input_vars.end(), output_spatial_indvars.begin(),
                      output_spatial_indvars.end());
    input_vars.insert(input_vars.end(), filter_spatial_indvars.begin(),
                      filter_spatial_indvars.end());

    return builder.createOrFold<mlir::AffineLoadOp>(
        location, input,
        builder.getAffineMap(2 + num_spatial_dims * 2, 0, input_indices),
        input_vars);
  }();

  mlir::Value* loaded_filter = [&] {
    std::vector<mlir::Value*> filter_vars;
    filter_vars.push_back(loop_o.getInductionVar());
    filter_vars.push_back(loop_c.getInductionVar());
    filter_vars.insert(filter_vars.end(), filter_spatial_indvars.begin(),
                       filter_spatial_indvars.end());

    return builder.createOrFold<mlir::AffineLoadOp>(location, filter,
                                                    filter_vars);
  }();

  builder.createOrFold<mlir::AffineStoreOp>(
      location,
      builder.create<mlir::AddFOp>(
          location,
          builder.createOrFold<mlir::AffineLoadOp>(location, output_acc),
          builder.create<mlir::MulFOp>(location, loaded_input, loaded_filter)),
      output_acc, llvm::ArrayRef<mlir::Value*>());

  builder.setInsertionPointAfter(reduction_loops[0]);
  {
    std::vector<mlir::Value*> output_vars;
    output_vars.push_back(loop_n.getInductionVar());
    output_vars.push_back(loop_o.getInductionVar());
    output_vars.insert(output_vars.end(), output_spatial_indvars.begin(),
                       output_spatial_indvars.end());
    builder.createOrFold<mlir::AffineStoreOp>(
        location,
        builder.createOrFold<mlir::AffineLoadOp>(location, output_acc), output,
        output_vars);
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

  // Hoist everyting before reduction loops (aka zero initializations of
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
  //     }
  //     for (tiled cartesian loops...) {
  //       output[...] = output_acc[...]
  //     }
  //   }
  HoistAndFix(reduction_loops.front(), tiled_cartesian_loops.front());

  return TransformedMlirConvAnchors{cartesian_product_loops, reduction_loops};
}

}  // namespace

Status EmitConvolutionForwardAsMlir(HloInstruction* custom_call,
                                    mlir::Value* input, mlir::Value* filter,
                                    mlir::Value* output,
                                    mlir::OpBuilder builder) {
  NameLookupTable names;

  TF_ASSIGN_OR_RETURN(
      InitialMlirConvAnchors initial_anchors,
      CreateNaiveMlirConv(custom_call, input, filter, output, builder, &names));

  TF_ASSIGN_OR_RETURN(TransformedMlirConvAnchors transformed_anchors,
                      TransformMlirConv(initial_anchors));

  // TODO(timshen): Implement a transformation that collects loads to a given
  // buffer, create a local alloc() for the accessed part, redirects all loads
  // and stores to that local alloc(), and create code to ininitialize /
  // writeback the local alloc() if needed.

  // TODO(timshen): Implement CUDA-specific lowering.

  VLOG(1) << ToString(transformed_anchors.cartesian_product_loops[0], names);

  return Status::OK();
}

Status ConvIsImplemented(const HloInstruction* custom_call) {
  if (custom_call->feature_group_count() != 1 ||
      custom_call->batch_group_count() != 1) {
    return Unimplemented("group count is not implemented.");
  }
  if (window_util::HasWindowReversal(custom_call->window())) {
    return Unimplemented("Window reversal is not implemented.");
  }
  if (window_util::HasDilation(custom_call->window())) {
    return Unimplemented("Dilation is not implemented.");
  }
  return Status::OK();
}

}  // namespace mlir_gpu
}  // namespace xla

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file provides basic utilities for the elemental lowering of
// each node

#include "mlir-hlo/Dialect/mhlo/transforms/lhlo_elemental_utils.h"

#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/map_lmhlo_to_scalar_op.h"
#include "mlir-hlo/utils/codegen_utils.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using mlir::memref::DimOp;
using mlir::memref::LoadOp;
using mlir::memref::StoreOp;

namespace mlir {
namespace lmhlo {

Value createLoadOrUseCachedValue(Location loc, OpBuilder* b, Value memref,
                                 ValueRange indices,
                                 OpBuilder::InsertPoint insert_point) {
  // Check if there are any cached value that can be reused,
  // within the current Block. Alternatively we can do this for
  // all the Blocks that dominant this Block, but that will be
  // complicated anyway.
  std::vector<StoreOp> store_ops;
  insert_point.getBlock()->walk(
      insert_point.getBlock()->begin(), insert_point.getPoint(),
      [&](StoreOp store_op) {
        if (store_op.getOperation()->getBlock() != insert_point.getBlock())
          return;
        if ((store_op.getMemRef() == memref) &&
            (store_op.getIndices() == indices))
          store_ops.emplace_back(store_op);
      });
  if (!store_ops.empty()) return store_ops[0].getOperand(0);
  int rank = memref.getType().dyn_cast<MemRefType>().getRank();
  return rank > 0 ? b->create<LoadOp>(loc, memref, indices)
                  : b->create<LoadOp>(loc, memref);
}

DenseSet<Operation*> NoLoaderUser(SmallVectorImpl<Operation*>& ops) {
  SmallVector<Operation*, 4> worklist;
  DenseSet<Operation*> has_loader_ops;
  for (Operation* op : ops) {
    Value memref = cast<LmhloOp>(op).getResultBuffer();
    if (memref == nullptr) continue;
    for (auto* user : memref.getUsers()) {
      if (isa<memref::LoadOp>(user)) {
        worklist.push_back(op);
        has_loader_ops.insert(op);
      }
    }
  }

  while (!worklist.empty()) {
    Operation* op = worklist.pop_back_val();
    int num_operands = op->getNumOperands();
    for (int i = 0; i < num_operands - 1; ++i) {
      Value memref = op->getOperand(i);
      for (Operation* user : memref.getUsers()) {
        if ((!isa<LmhloOp>(user)) || has_loader_ops.count(user)) continue;
        if (cast<LmhloOp>(user).getResultBuffer() == memref) {
          worklist.push_back(user);
          has_loader_ops.insert(user);
        }
      }
    }
  }

  DenseSet<Operation*> no_loader_ops;
  for (Operation* op : ops)
    if (!has_loader_ops.count(op)) no_loader_ops.insert(op);
  return no_loader_ops;
}

void cleanUnusedLhloOps(Block* parent) {
  SmallVector<Operation*, 4> lhlo_ops;
  for (Operation& op : parent->getOperations()) {
    if (op.getDialect() == op.getContext()->getLoadedDialect("lmhlo") &&
        (!isa<lmhlo::TerminatorOp>(op)))
      lhlo_ops.push_back(&op);
  }
  const DenseSet<Operation*>& no_loader_user = NoLoaderUser(lhlo_ops);
  for (auto* lhlo_op : no_loader_user) lhlo_op->erase();
}

template <typename LHLO_OpTy>
Value elementalLower(OpBuilder* b, Location loc, LHLO_OpTy op,
                     ValueRange output_index, bool check_cache);

template <>
Value elementalLower<lmhlo::RealDynamicSliceOp>(OpBuilder* b, Location loc,
                                                lmhlo::RealDynamicSliceOp op,
                                                ValueRange output_index,
                                                bool check_cache) {
  Value start_indices_memref = op->getOperand(1);
  Value strides_memref = op->getOperand(3);
  int rank = output_index.size();
  SmallVector<Value, 4> input_index;
  for (int dim = 0; dim < rank; ++dim) {
    SmallVector<Value, 4> dim_index;
    dim_index.push_back(b->create<ConstantOp>(
        loc, b->getIndexType(), b->getIntegerAttr(b->getIndexType(), dim)));
    auto start_index_load =
        b->create<LoadOp>(loc, start_indices_memref, ValueRange{dim_index});
    auto start_index =
        b->create<IndexCastOp>(loc, b->getIndexType(), start_index_load);
    auto stride_load =
        b->create<LoadOp>(loc, strides_memref, ValueRange{dim_index});
    auto stride = b->create<IndexCastOp>(loc, b->getIndexType(), stride_load);
    // input_dim = out_dim * stride + start_index
    auto input_dim = b->create<AddIOp>(
        loc, b->create<MulIOp>(loc, output_index[dim], stride), start_index);
    input_index.push_back(input_dim);
  }

  Value operand_memref = *(op->getOperands().begin());

  if (!check_cache) return b->create<LoadOp>(loc, operand_memref, input_index);
  return createLoadOrUseCachedValue(loc, b, operand_memref, input_index,
                                    b->saveInsertionPoint());
}

namespace {

template <typename T>
Value elementalLowerImplForBroadcastInDimOps(OpBuilder* b, Location loc,
                                             T broadcast_in_dim,
                                             ValueRange output_index,
                                             bool check_cache) {
  auto broadcast_dimensions =
      broadcast_in_dim.broadcast_dimensions().template getValues<int64_t>();
  int out_rank = output_index.size();
  Value operand_memref = broadcast_in_dim->getOperand(0);
  SmallVector<Value, 4> input_index;
  for (int64_t dim = 0; dim < out_rank; ++dim) {
    auto it = std::find(broadcast_dimensions.begin(),
                        broadcast_dimensions.end(), dim);

    bool is_broadcast_dim = (it != broadcast_dimensions.end());
    if (is_broadcast_dim) {
      int input_dim = std::distance(broadcast_dimensions.begin(), it);
      int64_t static_dim_size =
          operand_memref.getType().cast<MemRefType>().getShape()[input_dim];
      if (static_dim_size == 1) {
        // we know this dim is to be broadcasted at compile time
        auto zero = b->create<ConstantOp>(
            loc, b->getIndexType(), b->getIntegerAttr(b->getIndexType(), 0));
        input_index.push_back(zero);
      } else if (static_dim_size == ShapedType::kDynamicSize) {
        // we are not sure if this dim is to be broadcasted at compile time
        auto dim_size = b->create<DimOp>(loc, operand_memref, input_dim);
        auto one = b->create<ConstantOp>(
            loc, b->getIndexType(), b->getIntegerAttr(b->getIndexType(), 1));
        auto zero = b->create<ConstantOp>(
            loc, b->getIndexType(), b->getIntegerAttr(b->getIndexType(), 0));
        auto dim_size_is_1 =
            b->create<CmpIOp>(loc, CmpIPredicate::eq, dim_size, one);
        input_index.push_back(b->create<mlir::SelectOp>(
            loc, dim_size_is_1, zero, output_index[dim]));
      } else {
        // we know this dim is not to be broadcasted at compile time
        input_index.push_back(output_index[dim]);
      }
    }
  }

  if (!check_cache) {
    int rank = operand_memref.getType().dyn_cast<MemRefType>().getRank();
    return (rank > 0) ? b->create<LoadOp>(loc, operand_memref, input_index)
                      : b->create<LoadOp>(loc, operand_memref, ValueRange());
  }
  return createLoadOrUseCachedValue(loc, b, operand_memref, input_index,
                                    b->saveInsertionPoint());
}

}  // namespace

template <>
Value elementalLower<lmhlo::DynamicBroadcastInDimOp>(
    OpBuilder* b, Location loc, lmhlo::DynamicBroadcastInDimOp op,
    ValueRange output_index, bool check_cache) {
  return elementalLowerImplForBroadcastInDimOps(b, loc, op, output_index,
                                                check_cache);
}

template <>
Value elementalLower<lmhlo::BroadcastInDimOp>(OpBuilder* b, Location loc,
                                              lmhlo::BroadcastInDimOp op,
                                              ValueRange output_index,
                                              bool check_cache) {
  return elementalLowerImplForBroadcastInDimOps(b, loc, op, output_index,
                                                check_cache);
}

scf::ForOp createLoopAndSetInsPt(OpBuilder& b, Location loc, Value& var,
                                 Value lb, Value ub, Value step,
                                 ArrayRef<Value> init_values) {
  auto for_op = b.create<scf::ForOp>(loc, lb, ub, step, init_values);
  b.setInsertionPointToStart(for_op.getBody());
  var = for_op.getInductionVar();
  return for_op;
}

scf::ParallelOp createParallelAndSetInsPt(OpBuilder& b, Location loc,
                                          SmallVectorImpl<Value>& vars,
                                          ArrayRef<Value> lbs,
                                          ArrayRef<Value> ubs,
                                          ArrayRef<Value> steps,
                                          ArrayRef<Value> init_values) {
  auto par_op = b.create<scf::ParallelOp>(loc, lbs, ubs, steps, init_values,
                                          /*bodyBuilderFn=*/nullptr);
  b.setInsertionPointToStart(par_op.getBody());
  vars.append(par_op.getInductionVars().begin(),
              par_op.getInductionVars().end());
  return par_op;
}

// reinterpret_cast the input memref into 1D
memref::ReinterpretCastOp createMemRef1DReinterpretCast(OpBuilder& b,
                                                        Location loc,
                                                        Value memref) {
  auto memref_ty = memref.getType().cast<MemRefType>();
  assert(memref_ty.getAffineMaps().empty());
  Value size = codegen_utils::emitNumElementsComputation(b, loc, memref);
  Value stride = b.create<mlir::ConstantOp>(
      loc, b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 1));
  Value zero = b.create<mlir::ConstantOp>(
      loc, b.getIndexType(), b.getIntegerAttr(b.getIndexType(), 0));
  auto memref_1d_type =
      MemRefType::get({MemRefType::kDynamicSize}, memref_ty.getElementType(),
                      memref_ty.getAffineMaps(), memref_ty.getMemorySpace());
  return b.create<memref::ReinterpretCastOp>(
      loc, memref_1d_type, memref, zero, ValueRange{size}, ValueRange{stride});
}

void createOffsetStore(OpBuilder& b, Location loc, Value res, Value memref,
                       Value offset) {
  Value memref_1d = createMemRef1DReinterpretCast(b, loc, memref);
  b.create<memref::StoreOp>(loc, res, memref_1d, ValueRange{offset});
}

memref::LoadOp createOffsetLoad(OpBuilder& b, Location loc, Value memref,
                                Value offset) {
  Value memref_1d = createMemRef1DReinterpretCast(b, loc, memref);
  return b.create<memref::LoadOp>(loc, memref_1d, ValueRange{offset});
}

}  // namespace lmhlo
}  // namespace mlir

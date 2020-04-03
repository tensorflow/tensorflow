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

// This file implements logic for lowering LHLO dialect to Affine dialect.
//
#include <iostream>
#include "absl/memory/memory.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/Dialect/LoopOps/EDSC/Builders.h" 
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"

namespace mlir {
namespace xla_lhlo {

namespace {

constexpr StringRef kDhloMemrefFuncAttr = "xla_dhlo.memref_func";

enum class ScheduleKind {
  kLoop,
  kRowReduction,
  kColReduction,
};

std::vector<Operation*> getRootOps(FuncOp& func) {
  std::vector<Operation*> root_ops;
  auto arg_list = func.getArguments();
  func.walk([&](Operation* op) {
    if (op->getDialect() !=
        op->getContext()->getRegisteredDialect("xla_lhlo")) {
      return;
    }
    if (op->getParentOp() != func.getOperation()) {
      return;
    }
    auto num_operands = op->getNumOperands();
    if (num_operands == 0) return;
    // for lhlo_op the last operand is always the result.
    // TODO: any exceptions?
    auto result = op->getOperand(num_operands - 1);
    for (auto arg : arg_list) {
      if (arg == result) {
        root_ops.emplace_back(op);
        return;
      }
    }
  });
  return root_ops;
}

// TODO: Only rank 2 to rank 1 reduction can be
// supported for now
bool isRowReduction(Operation* op) {
  if (!isa<xla_lhlo::ReduceOp>(op)) {
    return false;
  }
  auto reduce_op = cast<xla_lhlo::ReduceOp>(op);
  if (reduce_op.dimensions().getNumElements() != 1) {
    return false;
  }
  auto rank =
      op->getOperand(0).getType().cast<MemRefType>().getRank();
  auto dimensions = reduce_op.dimensions().getValues<int64_t>();
  return ((*dimensions.begin() == 1) && (rank == 2));
}

// TODO: Only rank 2 to rank 1 reduction can be
// supported for now
bool isColReduction(Operation* op) {
  if (!isa<xla_lhlo::ReduceOp>(op)) {
    return false;
  }
  auto reduce_op = cast<xla_lhlo::ReduceOp>(op);
  if (reduce_op.dimensions().getNumElements() != 1) {
    return false;
  }
  auto rank =
      op->getOperand(0).getType().cast<MemRefType>().getRank();
  auto dimensions = reduce_op.dimensions().getValues<int64_t>();
  return ((*(dimensions.begin() + 1) == 0) && (rank == 2));
}

bool hasLhloUsersOtherThan(
    Operation* op, std::vector<Operation*>& ops) {
  auto num_operands = op->getNumOperands();
  auto memref = op->getOperand(num_operands - 1);
  for (auto* user : memref.getUsers()) {
    if ((user->getDialect()->getNamespace() == "xla_lhlo") &&
        (user != op) &&
        (std::find(ops.begin(), ops.end(), op) == ops.end())) {
      return true;
    }
  }
  return false;
}

Operation* chooseDominant(std::vector<Operation*> root_ops,
                          ScheduleKind& schedule_kind) {
  Operation* dominant = nullptr;
  assert((root_ops.size()) > 0 && "root_ops is empty");
  for (auto* root : root_ops) {
    if (isRowReduction(root)) {
      dominant = root;
      schedule_kind = ScheduleKind::kRowReduction; 
    } else if (isColReduction(root)) {
      if (schedule_kind == ScheduleKind::kLoop) {
        dominant = root;
        schedule_kind = ScheduleKind::kColReduction; 
      }
    } else {
      if (dominant == nullptr) {
        dominant = root;
        schedule_kind = ScheduleKind::kLoop; 
      }
    }
  }
  return dominant;
}

Value createLoadOrUseCachedValue(
    Location loc, OpBuilder& b,
    Value memref, ValueRange indices,
    OpBuilder::InsertPoint insert_point) {
  // Check if there are any cached value that can be reused, 
  // within the current Block. In fact we can do this for
  // all the Blocks that dominant this Block, but that will be
  // compilacated anyway. 
  std::vector<mlir::StoreOp> store_ops;
  insert_point.getBlock()->walk(
      insert_point.getBlock()->begin(),
      insert_point.getPoint(),
      [&](mlir::StoreOp store_op) {
        if ((store_op.getMemRef() == memref) &&
            (store_op.getIndices() == indices)) {
          store_ops.emplace_back(store_op);
        }
      });
  if (store_ops.size() > 0) {
    return store_ops[0].getOperand(0); 
  }
  return b.create<LoadOp>(loc, memref, indices);
}

template <typename LHLO_OpTy>
bool binaryLowerHelper(
    OpBuilder &b, Location loc, Operation* op,
    const SmallVector<Value, 4>& input_index) {
  if (!isa<LHLO_OpTy>(op)) {
    return false;
  }
  auto lhs_memref = *op->getOperands().begin();
  auto rhs_memref = *(op->getOperands().begin() + 1);
  auto res_memref = *(op->getOperands().begin() + 2);

  auto lhs_data = createLoadOrUseCachedValue(
          loc, b, lhs_memref, input_index, b.saveInsertionPoint());
  auto rhs_data = createLoadOrUseCachedValue(
          loc, b, rhs_memref, input_index, b.saveInsertionPoint());
	auto res = b.create<AddFOp>(loc, lhs_data, rhs_data);
  b.create<StoreOp>(loc, res, res_memref, input_index);
  return true;
}

/*
<h, w, w_i, h_i>
loop.parallel %h = 0 to %sh step tile_h {
	loop.parallel %w = 0 to %sw step tile_w {
		loop.for %w_i = 0 to tile_w step 1 {
			alloc %acc;
			loop.for %h_i = 0 to tile_h step 1 {
				if (w_inbound && h_inbound) {
					%val = load %data[%h + %h_i, %w + %w_i]
					%acc = std.addf %acc, %val
				}
			}
			std.atomic_add %w*tile_w + %w_i, %acc     
		}
	}
}
*/
LogicalResult lowerWithScheduleColReduction(OpBuilder& b,
    std::vector<Operation*> root_ops, Operation* dominant_op) {

  auto lhs = *dominant_op->getOperands().begin();
  const auto& lhs_type = lhs.getType().template cast<MemRefType>();
  const auto& element_type = lhs_type.getElementType();
	const int c_tilesize_h = 128;
	const int c_tilesize_w = 2;
  if (!isColReduction(dominant_op)) {
    return failure();
  }
  const auto& shape = lhs_type.getShape();
  const auto loc = dominant_op->getLoc();
  b.setInsertionPoint(dominant_op);
	auto zero = b.create<ConstantOp>(
      loc, b.getIndexType(),
      b.getIntegerAttr(b.getIndexType(), 0)); 
	auto one = b.create<ConstantOp>(
      loc, b.getIndexType(),
      b.getIntegerAttr(b.getIndexType(), 1)); 
	auto tilesize_h = b.create<ConstantOp>(
      loc, b.getIndexType(),
      b.getIntegerAttr(b.getIndexType(), c_tilesize_h)); 
	auto tilesize_w = b.create<ConstantOp>(
      loc, b.getIndexType(),
      b.getIntegerAttr(b.getIndexType(), c_tilesize_w)); 
  auto shape_h = b.create<DimOp>(loc, lhs, 0);
  auto shape_w = b.create<DimOp>(loc, lhs, 1);
  
	// h outter
  auto forOp_ho = b.create<loop::ForOp>(loc, zero, shape_h, tilesize_h);
  forOp_ho.getBody()->clear();
  b.setInsertionPointToStart(forOp_ho.getBody());
	auto var_ho = forOp_ho.getInductionVar();
	// w outter
  auto forOp_wo = b.create<loop::ForOp>(loc, zero, shape_w, tilesize_w);
  forOp_wo.getBody()->clear();
  b.setInsertionPointToStart(forOp_wo.getBody());
	auto var_wo = forOp_wo.getInductionVar();
	// w inner
  auto forOp_wi = b.create<loop::ForOp>(loc, zero, tilesize_w, one);
  forOp_wi.getBody()->clear();
  b.setInsertionPointToStart(forOp_wi.getBody());
	auto var_wi = forOp_wi.getInductionVar();
	// w_inbound = wo + wi < size_w
	auto w_idx = b.create<AddIOp>(loc, var_wo, var_wi);
	auto w_inbound = b.create<CmpIOp>(
  		loc, CmpIPredicate::slt, w_idx, shape_w);
	// TODO: init value should be from rhs
  SmallVector<Value, 4> init_values;
  SmallVector<Type, 4> init_values_types;
  for (auto* root_op : root_ops) {
    if (isColReduction(root_op)) {
      auto root_lhs = *(root_op->getOperands().begin());
      const auto& root_lhs_type = root_lhs.getType().template cast<MemRefType>();
      const auto& root_element_type = root_lhs_type.getElementType();
	    auto init_value = b.create<ConstantOp>(
          loc, root_element_type, b.getFloatAttr(root_element_type, 0.0)); 
      init_values.push_back(init_value);
      init_values_types.push_back(init_value.getType());
    }
  }
  auto forOp_hi = b.create<loop::ForOp>(
      loc, zero, tilesize_h, one, init_values);
  forOp_hi.getBody()->clear();
  b.setInsertionPointToStart(forOp_hi.getBody());
	auto var_hi = forOp_hi.getInductionVar();
	// h_inbound = ho + hi < size_h
	auto h_idx = b.create<AddIOp>(loc, var_ho, var_hi);
	auto h_inbound = b.create<CmpIOp>(
  		loc, CmpIPredicate::slt, h_idx, shape_h);
	auto inbound = b.create<mlir::AndOp>(loc, h_inbound, w_inbound);
  auto if_inbound_op = b.create<loop::IfOp>(
      loc, /*resultTypes*/init_values_types,
      inbound, /*hasElseRegion*/true);
  if_inbound_op.thenRegion().front().clear();
  if_inbound_op.elseRegion().front().clear();
  b.setInsertionPointToStart(&if_inbound_op.thenRegion().front());

  // emit all the calculations of root_ops
  // TODO: emit affine.apply if not all the shapes of root_ops are the same.
  // for example, root_shapes like [16, 256] and [4, 4, 256] may exist in 
  // one fused pattern.
  SmallVector<Value, 4> input_index;
  input_index.push_back(h_idx);
  input_index.push_back(w_idx);
  SmallVector<Value, 4> yield_values_then_branch;
  SmallVector<Value, 4> yield_values_else_branch;
  // TODO: check the order of root_ops by op.walk() is as expected
  for (auto* root_op : root_ops) {
    if (isColReduction(root_op)) {
      // then branch
      b.setInsertionPointToEnd(&if_inbound_op.thenRegion().front());
      auto data = createLoadOrUseCachedValue(
          loc, b, lhs, input_index, b.saveInsertionPoint());
	    auto acc = b.create<AddFOp>(
          loc, *forOp_hi.getRegionIterArgs().begin(), data);
      yield_values_then_branch.push_back(acc);
      // else branch 
      b.setInsertionPointToEnd(&if_inbound_op.elseRegion().front());
	    auto acc_init = *forOp_hi.getRegionIterArgs().begin();
      yield_values_else_branch.push_back(acc_init);
    } else if (isRowReduction(root_op)) {
      assert(false && "unexpected row_reduction");
    } else {
      // then branch
      b.setInsertionPointToEnd(&if_inbound_op.thenRegion().front());
      // TODO: potential affine map and apply
      if (binaryLowerHelper<xla_lhlo::AddOp>(b, loc, root_op, input_index) ||
          binaryLowerHelper<xla_lhlo::SubOp>(b, loc, root_op, input_index) ||
          binaryLowerHelper<xla_lhlo::MulOp>(b, loc, root_op, input_index) ||
          binaryLowerHelper<xla_lhlo::DivOp>(b, loc, root_op, input_index)) {
      } else {
        assert(false && "unsupported lhlo_op");
      }
    }
  }
  b.setInsertionPointToEnd(&if_inbound_op.thenRegion().front());
  b.create<loop::YieldOp>(loc, yield_values_then_branch);
  b.setInsertionPointToEnd(&if_inbound_op.elseRegion().front());
  b.create<loop::YieldOp>(loc, yield_values_else_branch);

	b.setInsertionPointToEnd(forOp_hi.getBody());
  b.create<loop::YieldOp>(
      loc, ValueRange({*if_inbound_op.results().begin()}));

	b.setInsertionPointToEnd(forOp_wi.getBody());
	b.create<AtomicRMWOp>(loc, element_type,
			AtomicRMWKind::addf, *forOp_hi.results().begin(),
      dominant_op->getOperand(2), ValueRange({w_idx}));
  b.create<loop::YieldOp>(loc, ValueRange({}));

	b.setInsertionPointToEnd(forOp_wo.getBody());
  b.create<loop::YieldOp>(loc, ValueRange({}));
	
	b.setInsertionPointToEnd(forOp_ho.getBody());
  b.create<loop::YieldOp>(loc, ValueRange({}));

  // remote the root_op if it has no other users except the memref
  for (auto* root_op : root_ops) {
    if (!hasLhloUsersOtherThan(root_op, root_ops)) {
      root_op->erase();
    }
  }

  return success();
}

}  // namespace

struct LhloLegalizeToLoop : public FunctionPass<LhloLegalizeToLoop> {
  
  void runOnFunction() override {
    auto func = getFunction();
    if (!func.getAttrOfType<UnitAttr>(kDhloMemrefFuncAttr)) {
      return;
    }
    auto root_ops = getRootOps(func);

    // 1, If any reduce op among the 'root_ops', follow the schedule of it;
    //    or else, follow the schedule of kLoop.
    // 2, If there are a mixer of column reductions and row reductions,
    //    follow the schedule of the row reduction, and implement all the 
    //    column reduction with the 'pure atomic' way, which has no
    //    requirement on the schedule.
    // TODO: the support of row reduction and 'pure atomic' reduction
    ScheduleKind schedule_kind;
    auto dominant_op = chooseDominant(root_ops, schedule_kind);

    OpBuilder b(func);
    switch (schedule_kind) {
      case ScheduleKind::kRowReduction: {
        assert(false && "not implemented yet");
        break;
      }
      case ScheduleKind::kColReduction: {
        auto r = lowerWithScheduleColReduction(b, root_ops, dominant_op);
        assert(!failed(r) && "lowerWithScheduleColReduction failed");
        break;
      }
      case ScheduleKind::kLoop: {
        assert(false && "not implemented yet");
        break;
      }
      default:
        assert(false && "unknown schedule_kind");
    }
  }
};

std::unique_ptr<OpPassBase<FuncOp>> createLhloLegalizeToLoopPass() {
  return absl::make_unique<LhloLegalizeToLoop>();
}

static PassRegistration<LhloLegalizeToLoop> legalize_pass(
    "lhlo-legalize-to-loops", "Legalize from LHLO dialect to loop dialect");

}  // namespace xla_lhlo
}  // namespace mlir

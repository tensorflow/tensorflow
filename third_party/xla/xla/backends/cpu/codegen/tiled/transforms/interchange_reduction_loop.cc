/* Copyright 2026 The OpenXLA Authors.

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

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"  // IWYU pragma: keep
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/analysis/interval.h"

namespace xla::cpu {

#define GEN_PASS_DEF_INTERCHANGEREDUCTIONLOOPPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

namespace ma = ::mlir::arith;
namespace mm = ::mlir::memref;
namespace ms = ::mlir::scf;
namespace mv = ::mlir::vector;
namespace MemoryEffects = ::mlir::MemoryEffects;

using ::llvm::DenseSet;
using ::llvm::SmallVector;
using ::mlir::APInt;
using ::mlir::Block;
using ::mlir::Location;
using ::mlir::MemRefType;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::Value;
using ::mlir::ValueRange;
using ::mlir::VectorType;

// Unwraps view/cast and bitcast chains to find the underlying memref buffer.
Value GetBaseMemRef(Value memref) {
  while (memref) {
    if (auto view = memref.getDefiningOp<mlir::ViewLikeOpInterface>()) {
      memref = view.getViewSource();
      continue;
    }
    if (auto bitcast = memref.getDefiningOp<xtile::MemRefBitcastOp>()) {
      memref = bitcast.getSource();
      continue;
    }
    break;
  }
  return memref;
}

// Computes the transitive closure of values in `block` that depend on
// `seeds`, ignoring dataflow edges through `excluded_val`.
void PropagateDependence(Block* block, DenseSet<Value>& seeds,
                         Value excluded_val = nullptr) {
  bool changed = true;
  while (changed) {
    changed = false;
    block->walk([&](Operation* op) {
      if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        return;
      }
      if (llvm::any_of(op->getOperands(), [&](Value v) {
            return v != excluded_val && seeds.contains(v);
          })) {
        for (Value res : op->getResults()) {
          if (seeds.insert(res).second) {
            changed = true;
          }
        }
      }
    });
  }
}

// Returns true if `op` has side effects or control flow unsafe to reorder:
// nested loops, non-`scf.if` regions, alloc/free, unmodeled effects, or
// writes when `allow_writes` is false.
bool HasDisallowedSideEffects(Operation* op, bool allow_writes) {
  mlir::WalkResult result = op->walk([&](Operation* nested) {
    if (nested->hasTrait<mlir::OpTrait::IsTerminator>()) {
      return mlir::WalkResult::advance();
    }
    if (mlir::isa<ms::ForOp, ms::WhileOp>(nested)) {
      return mlir::WalkResult::interrupt();
    }
    if (nested->getNumRegions() > 0 && !mlir::isa<ms::IfOp>(nested)) {
      return mlir::WalkResult::interrupt();
    }
    if (auto mem_iface =
            mlir::dyn_cast<mlir::MemoryEffectOpInterface>(nested)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      mem_iface.getEffects(effects);
      for (const MemoryEffects::EffectInstance& eff : effects) {
        if (mlir::isa<MemoryEffects::Allocate, MemoryEffects::Free>(
                eff.getEffect())) {
          return mlir::WalkResult::interrupt();
        }
        if (!allow_writes && mlir::isa<MemoryEffects::Write>(eff.getEffect())) {
          return mlir::WalkResult::interrupt();
        }
      }
    } else if (!mlir::isMemoryEffectFree(nested) &&
               !mlir::isa<ms::IfOp>(nested)) {
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return result.wasInterrupted();
}

struct InterchangePlan {
  ms::ForOp outer_loop;
  ms::ForOp inner_loop;
  VectorType acc_type;
  int64_t max_outer_trips;
  SmallVector<Operation*> pre_ops;
  SmallVector<Operation*> post_ops;
};

struct MemRefReadInfo {
  Value memref;
  mlir::OperandRange indices;
  int64_t read_rank;
};

std::optional<MemRefReadInfo> GetMemRefReadInfo(Operation* op) {
  if (auto tr = mlir::dyn_cast<mv::TransferReadOp>(op)) {
    return MemRefReadInfo{tr.getBase(), tr.getIndices(),
                          tr.getVectorType().getRank()};
  }
  if (auto vl = mlir::dyn_cast<mv::LoadOp>(op)) {
    return MemRefReadInfo{vl.getBase(), vl.getIndices(),
                          vl.getVectorType().getRank()};
  }
  if (auto ml = mlir::dyn_cast<mm::LoadOp>(op)) {
    return MemRefReadInfo{ml.getMemref(), ml.getIndices(), /*read_rank=*/1};
  }
  return std::nullopt;
}

Value GetMemRefStoreInfo(Operation* op) {
  if (auto write = mlir::dyn_cast<mv::TransferWriteOp>(op)) {
    return write.getBase();
  }
  if (auto store = mlir::dyn_cast<mv::StoreOp>(op)) {
    return store.getBase();
  }
  if (auto store = mlir::dyn_cast<mm::StoreOp>(op)) {
    return store.getMemref();
  }
  return nullptr;
}

std::optional<int64_t> GetMaxOuterTrips(ms::ForOp outer_loop) {
  std::optional<Interval> lb_range = xla::GetRange(outer_loop.getLowerBound());
  if (!lb_range) {
    return std::nullopt;
  }
  std::optional<Interval> ub_range = xla::GetRange(outer_loop.getUpperBound());
  if (!ub_range || lb_range->lower < 0 ||
      ub_range->upper > std::numeric_limits<int64_t>::max() / 2) {
    return std::nullopt;
  }
  int64_t max_outer_trips = ub_range->upper - lb_range->lower;
  if (max_outer_trips < 2) {
    return std::nullopt;
  }
  return max_outer_trips;
}

// Returns true if `inner_loop` is read-only and `post_ops` do not write to any
// memref read by `inner_loop` (including through view/bitcast aliases), so
// reordering `(outer_loop, inner_loop)` into separate init, reduction, and
// store-back loops preserves semantics.
bool CanReorderMemoryEffects(ms::ForOp inner_loop,
                             mlir::ArrayRef<Operation*> post_ops) {
  // Ensure inner_loop has no writes, allocs, or nested loops.
  for (Operation& op : inner_loop.getBody()->without_terminator()) {
    if (HasDisallowedSideEffects(&op, /*allow_writes=*/false)) {
      return false;
    }
  }
  // Collect underlying memrefs read in inner_loop; bail on unmodeled reads.
  DenseSet<Value> inner_read_memrefs;
  mlir::WalkResult read_result = inner_loop.walk([&](Operation* nested) {
    if (nested == inner_loop || mlir::isa<ms::IfOp>(nested) ||
        nested->hasTrait<mlir::OpTrait::IsTerminator>()) {
      return mlir::WalkResult::advance();
    }
    if (std::optional<MemRefReadInfo> read = GetMemRefReadInfo(nested)) {
      inner_read_memrefs.insert(GetBaseMemRef(read->memref));
      return mlir::WalkResult::advance();
    }
    if (mlir::mightHaveEffect<MemoryEffects::Read>(nested)) {
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (read_result.wasInterrupted()) {
    return false;
  }
  // Ensure post_ops do not contain nested loops, allocate memory, perform
  // unmodeled writes, or write to any memref read in inner_loop.
  for (Operation* op : post_ops) {
    if (HasDisallowedSideEffects(op, /*allow_writes=*/true)) {
      return false;
    }
    mlir::WalkResult write_result = op->walk([&](Operation* nested) {
      if (mlir::isa<ms::IfOp>(nested) ||
          nested->hasTrait<mlir::OpTrait::IsTerminator>()) {
        return mlir::WalkResult::advance();
      }
      if (Value written_memref = GetMemRefStoreInfo(nested)) {
        if (inner_read_memrefs.contains(GetBaseMemRef(written_memref))) {
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      }
      if (mlir::mightHaveEffect<MemoryEffects::Write>(nested)) {
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (write_result.wasInterrupted()) {
      return false;
    }
  }
  return true;
}

// Returns true if `inner_loop` reads a 2D+ identity-layout memref with
// `outer_loop` on the minor-most dim and `inner_loop` on a major dim, without
// any `outer_loop`-strided or `inner_loop`-minor reads.
bool HasContiguousNonKMinorReads(ms::ForOp outer_loop, ms::ForOp inner_loop) {
  DenseSet<Value> tid_deps{outer_loop.getInductionVar()};
  PropagateDependence(outer_loop.getBody(), tid_deps,
                      /*excluded_val=*/inner_loop.getRegionIterArg(0));

  DenseSet<Value> k_deps{inner_loop.getInductionVar()};
  PropagateDependence(inner_loop.getBody(), k_deps);

  bool found_contiguous_non_k_minor_read = false;
  bool found_k_minor_or_strided_read = false;
  inner_loop.walk([&](Operation* op) {
    std::optional<MemRefReadInfo> read = GetMemRefReadInfo(op);
    if (!read) {
      return;
    }
    auto mem_type = mlir::dyn_cast<MemRefType>(read->memref.getType());
    if (!mem_type || mem_type.getRank() < 2 || read->indices.size() < 2) {
      return;
    }
    bool last_depends_on_tid = tid_deps.contains(read->indices.back());
    bool last_depends_on_k = k_deps.contains(read->indices.back());
    bool major_depends_on_tid =
        llvm::any_of(read->indices.drop_back(),
                     [&](Value v) { return tid_deps.contains(v); });
    bool major_depends_on_k = llvm::any_of(
        read->indices.drop_back(), [&](Value v) { return k_deps.contains(v); });

    if (major_depends_on_tid || last_depends_on_k) {
      found_k_minor_or_strided_read = true;
      return;
    }
    if (last_depends_on_tid && major_depends_on_k && read->read_rank == 1 &&
        mem_type.getLayout().isIdentity()) {
      found_contiguous_non_k_minor_read = true;
    }
  });
  return !found_k_minor_or_strided_read && found_contiguous_non_k_minor_read;
}

std::optional<InterchangePlan> MatchLoopNest(ms::ForOp outer_loop) {
  if (!outer_loop.getInitArgs().empty()) {
    return std::nullopt;
  }
  APInt outer_step;
  if (!mlir::matchPattern(outer_loop.getStep(),
                          mlir::m_ConstantInt(&outer_step)) ||
      !outer_step.isOne()) {
    return std::nullopt;
  }
  std::optional<int64_t> max_outer_trips = GetMaxOuterTrips(outer_loop);
  if (!max_outer_trips) {
    return std::nullopt;
  }

  ms::ForOp inner_loop;
  SmallVector<Operation*> pre_ops;
  SmallVector<Operation*> post_ops;
  for (Operation& op : outer_loop.getBody()->without_terminator()) {
    if (auto for_op = mlir::dyn_cast<ms::ForOp>(&op)) {
      if (inner_loop) {
        return std::nullopt;
      }
      inner_loop = for_op;
      continue;
    }
    if (!inner_loop) {
      if (op.getNumRegions() > 0 || !mlir::isMemoryEffectFree(&op)) {
        return std::nullopt;
      }
      pre_ops.push_back(&op);
    } else {
      post_ops.push_back(&op);
    }
  }
  if (!inner_loop || inner_loop.getNumRegionIterArgs() != 1) {
    return std::nullopt;
  }

  auto acc_type = mlir::dyn_cast<VectorType>(inner_loop.getResult(0).getType());
  if (!acc_type || acc_type.getRank() != 1 || acc_type.isScalable() ||
      !acc_type.getElementType().isIntOrFloat()) {
    return std::nullopt;
  }
  // Cap stack scratch allocation to avoid thread stack overflow.
  constexpr int64_t kMaxScratchBytes = 256 * 1024;
  int64_t elem_bytes =
      (static_cast<int64_t>(acc_type.getElementTypeBitWidth()) + 7) / 8;
  int64_t vec_len = acc_type.getNumElements();
  if (elem_bytes <= 0 || vec_len <= 0 || elem_bytes > kMaxScratchBytes ||
      vec_len > kMaxScratchBytes / elem_bytes ||
      *max_outer_trips > kMaxScratchBytes / (vec_len * elem_bytes)) {
    return std::nullopt;
  }
  // Require inner loop bounds, step, and init arg to be invariant wrt
  // outer_loop.
  if (!outer_loop.isDefinedOutsideOfLoop(inner_loop.getLowerBound()) ||
      !outer_loop.isDefinedOutsideOfLoop(inner_loop.getUpperBound()) ||
      !outer_loop.isDefinedOutsideOfLoop(inner_loop.getStep()) ||
      !outer_loop.isDefinedOutsideOfLoop(inner_loop.getInitArgs()[0])) {
    return std::nullopt;
  }
  APInt inner_lb_const, inner_ub_const, inner_step_const;
  if (mlir::matchPattern(inner_loop.getLowerBound(),
                         mlir::m_ConstantInt(&inner_lb_const)) &&
      mlir::matchPattern(inner_loop.getUpperBound(),
                         mlir::m_ConstantInt(&inner_ub_const)) &&
      mlir::matchPattern(inner_loop.getStep(),
                         mlir::m_ConstantInt(&inner_step_const))) {
    int64_t trips = (inner_ub_const - inner_lb_const + inner_step_const - 1)
                        .sdiv(inner_step_const)
                        .getSExtValue();
    if (trips < 2) {
      return std::nullopt;
    }
  }
  if (!CanReorderMemoryEffects(inner_loop, post_ops) ||
      !HasContiguousNonKMinorReads(outer_loop, inner_loop)) {
    return std::nullopt;
  }
  return InterchangePlan{outer_loop,         inner_loop,
                         acc_type,           *max_outer_trips,
                         std::move(pre_ops), std::move(post_ops)};
}

Value GetScratchIndex(OpBuilder& b, Location loc, Value tid, Value outer_lb,
                      bool lb_is_zero) {
  return lb_is_zero ? tid
                    : ma::SubIOp::create(b, loc, tid, outer_lb).getResult();
}

void RewriteLoop(const InterchangePlan& plan) {
  ms::ForOp outer_loop = plan.outer_loop;
  ms::ForOp inner_loop = plan.inner_loop;
  Location loc = outer_loop.getLoc();
  OpBuilder builder(outer_loop);

  Value c0 = ma::ConstantIndexOp::create(builder, loc, 0);
  Value outer_lb = outer_loop.getLowerBound();
  Value outer_ub = outer_loop.getUpperBound();
  Value outer_step = outer_loop.getStep();
  bool lb_is_zero = mlir::matchPattern(outer_lb, mlir::m_Zero());

  int64_t vec_len = plan.acc_type.getNumElements();
  auto scratch_type = MemRefType::get({plan.max_outer_trips, vec_len},
                                      plan.acc_type.getElementType());
  Value scratch = mm::AllocaOp::create(builder, loc, scratch_type,
                                       builder.getI64IntegerAttr(64));

  Value outer_tid = outer_loop.getInductionVar();
  Value inner_k = inner_loop.getInductionVar();
  Value inner_iter_arg = inner_loop.getRegionIterArg(0);
  Value inner_init_arg = inner_loop.getInitArgs()[0];
  Value inner_yield_val =
      mlir::cast<ms::YieldOp>(inner_loop.getBody()->getTerminator())
          .getOperand(0);

  // Initialize scratch buffer across tile iterations.
  ms::ForOp::create(builder, loc, outer_lb, outer_ub, outer_step, ValueRange{},
                    [&](OpBuilder& ib, Location iloc, Value tid, ValueRange) {
                      Value idx =
                          GetScratchIndex(ib, iloc, tid, outer_lb, lb_is_zero);
                      mv::StoreOp::create(ib, iloc, inner_init_arg, scratch,
                                          ValueRange{idx, c0});
                      ms::YieldOp::create(ib, iloc);
                    });

  // Reduction loop wrapping tile loop.
  ms::ForOp::create(
      builder, loc, inner_loop.getLowerBound(), inner_loop.getUpperBound(),
      inner_loop.getStep(), ValueRange{},
      [&](OpBuilder& kb, Location kloc, Value k, ValueRange) {
        ms::ForOp::create(
            kb, kloc, outer_lb, outer_ub, outer_step, ValueRange{},
            [&](OpBuilder& tb, Location tloc, Value tid, ValueRange) {
              Value idx = GetScratchIndex(tb, tloc, tid, outer_lb, lb_is_zero);
              Value acc = mv::LoadOp::create(tb, tloc, plan.acc_type, scratch,
                                             ValueRange{idx, c0})
                              .getResult();
              mlir::IRMapping map;
              map.map(outer_tid, tid);
              map.map(inner_k, k);
              map.map(inner_iter_arg, acc);
              for (Operation* op : plan.pre_ops) {
                tb.clone(*op, map);
              }
              for (Operation& op : inner_loop.getBody()->without_terminator()) {
                tb.clone(op, map);
              }
              Value yielded = map.lookupOrDefault(inner_yield_val);
              mv::StoreOp::create(tb, tloc, yielded, scratch,
                                  ValueRange{idx, c0});
              ms::YieldOp::create(tb, tloc);
            });
        ms::YieldOp::create(kb, kloc);
      });

  // Final store back using original post-loop ops.
  ms::ForOp::create(
      builder, loc, outer_lb, outer_ub, outer_step, ValueRange{},
      [&](OpBuilder& wb, Location wloc, Value tid, ValueRange) {
        Value idx = GetScratchIndex(wb, wloc, tid, outer_lb, lb_is_zero);
        Value final_val = mv::LoadOp::create(wb, wloc, plan.acc_type, scratch,
                                             ValueRange{idx, c0})
                              .getResult();
        mlir::IRMapping map;
        map.map(outer_tid, tid);
        map.map(inner_loop.getResult(0), final_val);
        for (Operation* op : plan.pre_ops) {
          wb.clone(*op, map);
        }
        for (Operation* op : plan.post_ops) {
          wb.clone(*op, map);
        }
        ms::YieldOp::create(wb, wloc);
      });

  outer_loop.erase();
}

class InterchangeReductionLoopPass
    : public impl::InterchangeReductionLoopPassBase<
          InterchangeReductionLoopPass> {
 public:
  using InterchangeReductionLoopPassBase::InterchangeReductionLoopPassBase;

  void runOnOperation() override {
    mlir::func::FuncOp func_op = getOperation();
    if (func_op.getBody().empty()) {
      return;
    }
    SmallVector<InterchangePlan> plans;
    for (Operation& op : func_op.getBody().front()) {
      auto outer_loop = mlir::dyn_cast<ms::ForOp>(&op);
      if (!outer_loop) {
        continue;
      }
      if (std::optional<InterchangePlan> plan = MatchLoopNest(outer_loop)) {
        plans.push_back(std::move(*plan));
      }
    }
    for (const InterchangePlan& plan : plans) {
      RewriteLoop(plan);
    }
  }
};

}  // namespace
}  // namespace xla::cpu

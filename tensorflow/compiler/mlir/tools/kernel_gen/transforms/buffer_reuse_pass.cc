/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <vector>

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/BufferAliasAnalysis.h"  // from @llvm-project
#include "mlir/Analysis/Liveness.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

// Needed to build `llvm::EquivalenceClasses` of `mlir::Value`s.
namespace mlir {
static bool operator<(const Value &lhs, const Value &rhs) {
  return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
}
}  // namespace mlir

constexpr llvm::StringRef
    mlir::kernel_gen::tf_framework::TFAllocOp::kReuseOutputAttrName;
constexpr llvm::StringRef
    mlir::kernel_gen::tf_framework::TFAllocOp::kReuseInputCandidatesAttrName;
constexpr llvm::StringRef
    mlir::kernel_gen::tf_framework::TFFrameworkDialect::kTFEntryAttrName;

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

/// A temporary buffer size analysis that is correct but may be incomplete.
class BufferSizeAnalysis {
 public:
  BufferSizeAnalysis(FuncOp f, const BufferAliasAnalysis &aliases) {
    build(f, aliases);
  }

  bool is_same_size(Value a, Value b) { return ecs_.isEquivalent(a, b); }

 private:
  void build(FuncOp &f, const BufferAliasAnalysis &aliases) {
    auto buffers = find_buffer_values(f);

    // Memrefs with statically known same shape and same symbol-free affine maps
    // must be of the same size.
    int n = buffers.size();
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        Value a = buffers[i];
        Value b = buffers[j];
        auto a_ty = a.getType().dyn_cast<MemRefType>();
        auto b_ty = b.getType().dyn_cast<MemRefType>();
        if (a_ty && b_ty && a_ty.hasStaticShape() && b_ty.hasStaticShape() &&
            a_ty.getNumElements() == b_ty.getNumElements() &&
            a_ty.getElementType() == b_ty.getElementType() &&
            affine_maps_symbol_free_and_equal(a_ty.getAffineMaps(),
                                              b_ty.getAffineMaps())) {
          ecs_.unionSets(a, b);
        }
      }
    }

    // Operands to `linalg.generic` with equal affine maps must be of same size.
    f.walk([&](linalg::GenericOp genericOp) {
      auto operand_buffers = genericOp.getShapedOperands();
      int n = operand_buffers.size();
      for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
          Value a = operand_buffers[i];
          Value b = operand_buffers[j];
          auto a_ty = a.getType().dyn_cast<MemRefType>();
          auto b_ty = b.getType().dyn_cast<MemRefType>();
          if (a_ty && b_ty && a_ty.getElementType() == b_ty.getElementType() &&
              a_ty.getAffineMaps() == b_ty.getAffineMaps()) {
            AffineMap map_i = genericOp.getIndexingMap(i);
            AffineMap map_j = genericOp.getIndexingMap(j);
            if (map_i == map_j && map_i.isPermutation()) ecs_.unionSets(a, b);
          }
        }
      }
    });

    // All aliases of a memref must be of the same underlying buffer size.
    for (auto e : aliases) {
      Value value = e.getFirst();
      if (!value.getType().isa<BaseMemRefType>()) continue;
      for (Value alias : e.getSecond()) {
        assert(alias.getType().isa<BaseMemRefType>() &&
               "Expected aliases of memref to be memrefs.");
        ecs_.unionSets(value, alias);
      }
    }
  }

  bool affine_maps_symbol_free_and_equal(ArrayRef<AffineMap> as,
                                         ArrayRef<AffineMap> bs) {
    auto is_symbol_free = [](AffineMap map) {
      return map.getNumSymbols() == 0;
    };
    return llvm::all_of(as, is_symbol_free) &&
           llvm::all_of(bs, is_symbol_free) && as == bs;
  }

  llvm::SmallVector<Value, 8> find_buffer_values(FuncOp f) {
    llvm::SmallVector<Value, 8> buffers;
    f.walk([&](Operation *op) {
      for (Value val : op->getResults())
        if (val.getType().isa<BaseMemRefType>()) buffers.push_back(val);
    });
    f.walk([&](Block *block) {
      for (Value val : block->getArguments()) {
        if (val.getType().isa<BaseMemRefType>()) buffers.push_back(val);
      }
    });
    return buffers;
  }

  llvm::EquivalenceClasses<Value> ecs_;
};

class BufferReuseAnalysis {
 public:
  explicit BufferReuseAnalysis(FuncOp f) { build(f); }

  static constexpr int32_t kIndexAmbiguous = -1;

  Optional<SmallVector<int32_t, 2>> get_reuse_candiates(AllocOp op) {
    auto it = reuse_candidates_.find(op);
    if (it == reuse_candidates_.end()) return llvm::None;
    return it->second;
  }

  Optional<int32_t> get_output_index(AllocOp op) {
    auto it = output_indices_.find(op);
    if (it == output_indices_.end()) return llvm::None;
    return it->second;
  }

 private:
  void build(FuncOp &f) {
    BufferAliasAnalysis aliases(f);
    find_output_indices(f, aliases);
    find_reuse_candiates(f, aliases);
  }

  void find_output_indices(FuncOp &f, BufferAliasAnalysis &aliases) {
    f.walk([&](AllocOp alloc_op) {
      int32_t output_index = kIndexAmbiguous;
      int count_return_uses = 0;
      auto buffer_aliases = aliases.resolve(alloc_op.getResult());
      for (Value alias : buffer_aliases) {
        for (auto &use : alias.getUses()) {
          if (isa<ReturnOp>(use.getOwner())) {
            int32_t index = use.getOperandNumber();
            if (count_return_uses++ == 0)
              output_index = index;
            else if (output_index != index)
              output_index = kIndexAmbiguous;
          }
        }
      }
      output_indices_[alloc_op] = output_index;
    });
  }

  void find_reuse_candiates(FuncOp &f, BufferAliasAnalysis &aliases) {
    Liveness liveness(f);
    BufferSizeAnalysis size_equivalences(f, aliases);
    f.walk([&](Block *block) {
      find_reuse_candiates(block, aliases, liveness.getLiveness(block),
                           size_equivalences, f.getArguments());
    });
  }

  void find_reuse_candiates(Block *block, BufferAliasAnalysis &aliases,
                            const LivenessBlockInfo *liveness,
                            BufferSizeAnalysis &size_equivalences,
                            ArrayRef<BlockArgument> arguments) {
    for (Operation &op : *block) {
      auto alloc_op = dyn_cast<AllocOp>(op);
      if (!alloc_op) continue;

      // Find first use of the newly allocated buffer within this block.
      Value new_buffer = alloc_op.getResult();
      Operation *first_reuse = find_first_use_in_block(new_buffer, block);
      assert((first_reuse == nullptr || first_reuse->getBlock() == block) &&
             "Expected first use in same block if found.");

      // Find reuse candidates for the regarded allocation.
      SmallVector<int32_t, 2> local_reuse_candidates;
      for (BlockArgument old_buffer : arguments) {
        if (!old_buffer.getType().isa<BaseMemRefType>()) continue;

        // Size criterion: Do not reuse buffers of different size as they may be
        // too small.
        if (!size_equivalences.is_same_size(new_buffer, old_buffer)) continue;

        // Lifetime criterion: Only reuse buffers that are no longer used on
        // first reuse, i.e. they are no longer alive.
        bool lifetimes_compatible = true;
        for (Value old_buffer_alias : aliases.resolve(old_buffer)) {
          if (first_reuse == nullptr) {
            // If the first use is beyond the end of this block we look at the
            // block end. An argument buffer that is already reusable there is
            // certainly reusable at any later actual use. Otherwise, lifetimes
            // are incompatible.
            if (liveness->isLiveOut(old_buffer_alias)) {
              lifetimes_compatible = false;
              break;
            }
          } else {
            // A buffer is reusable if
            //   i)  its last use is before the point of reuse, or
            //   ii) its last use is also its first reuse and the operation
            //       allows for local reuse.
            // Otherwise, lifetimes are incompatible.
            Operation *last_use =
                liveness->getEndOperation(old_buffer_alias, &block->front());
            assert(last_use != nullptr && last_use->getBlock() == block &&
                   "Expected last use in same block.");
            if (first_reuse->isBeforeInBlock(last_use)) {
              lifetimes_compatible = false;
              break;
            }
            if (first_reuse == last_use &&
                !can_reuse_locally(first_reuse, old_buffer_alias, new_buffer)) {
              lifetimes_compatible = false;
              break;
            }
          }
        }

        if (lifetimes_compatible) {
          // All criteria are fulfilled 🙂.
          int32_t old_buffer_index = old_buffer.getArgNumber();
          local_reuse_candidates.push_back(old_buffer_index);
        }
      }

      reuse_candidates_[&op] = local_reuse_candidates;
    }
  }

  Operation *find_first_use_in_block(Value value, Block *block) {
    Operation *first_use = nullptr;
    for (Operation *op : value.getUsers()) {
      Operation *ancestor_op = block->findAncestorOpInBlock(*op);
      if (ancestor_op == nullptr) continue;
      if (first_use == nullptr || ancestor_op->isBeforeInBlock(first_use))
        first_use = ancestor_op;
    }
    return first_use;
  }

  std::vector<Value> get_buffer_arguments(FuncOp &f) {
    std::vector<Value> buffer_arguments;
    for (BlockArgument arg : f.getArguments()) {
      if (arg.getType().isa<BaseMemRefType>()) buffer_arguments.push_back(arg);
    }
    return buffer_arguments;
  }

  bool can_reuse_locally(Operation *op, Value old_buffer, Value new_buffer) {
    // For now, we support only memrefs with the same memory layout.
    auto old_buffer_ty = old_buffer.getType().dyn_cast<MemRefType>();
    auto new_buffer_ty = old_buffer.getType().dyn_cast<MemRefType>();
    if (!old_buffer_ty || !new_buffer_ty ||
        old_buffer_ty.getAffineMaps() != new_buffer_ty.getAffineMaps())
      return false;

    if (auto generic_op = dyn_cast<linalg::GenericOp>(op)) {
      assert(llvm::find(op->getOperands(), old_buffer) !=
                 op->getOperands().end() &&
             llvm::find(op->getOperands(), new_buffer) !=
                 op->getOperands().end() &&
             "Expect `old/new_buffer` to be operand of `op`.");

      // If `linalg.generic` indexing maps are the same for input and output
      // buffer then the last use of the input buffer happens before its first
      // reuse (per memory location).
      auto operand_buffers = generic_op.getShapedOperands();
      int old_index =
          llvm::find(operand_buffers, old_buffer) - operand_buffers.begin();
      int new_index =
          llvm::find(operand_buffers, new_buffer) - operand_buffers.begin();
      AffineMap old_indexing_map = generic_op.getIndexingMap(old_index);
      AffineMap new_indexing_map = generic_op.getIndexingMap(new_index);
      return old_indexing_map == new_indexing_map &&
             old_indexing_map.isPermutation();
    }
    return false;
  }

  DenseMap<Operation *, SmallVector<int32_t, 2>> reuse_candidates_;
  DenseMap<Operation *, int32_t> output_indices_;
};

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct BufferReusePass : public BufferReusePassBase<BufferReusePass> {
  void runOnFunction() override {
    if (!getFunction()->getAttrOfType<UnitAttr>(
            tf_framework::TFFrameworkDialect::kTFEntryAttrName))
      return;

    BufferReuseAnalysis analysis(getFunction());

    // Annotate IR with reuse candidates and output indices per allocation.
    Builder builder(&getContext());
    getFunction().walk([&](AllocOp op) {
      if (auto output_index = analysis.get_output_index(op)) {
        auto attr = builder.getI32IntegerAttr(*output_index);
        op.getOperation()->setAttr(
            tf_framework::TFAllocOp::kReuseOutputAttrName, attr);
      }
      if (auto reuse_candiates = analysis.get_reuse_candiates(op)) {
        auto attr = builder.getI32ArrayAttr(*reuse_candiates);
        op.getOperation()->setAttr(
            tf_framework::TFAllocOp::kReuseInputCandidatesAttrName, attr);
      }
    });
  }
};

}  // namespace

std::unique_ptr<FunctionPass> CreateBufferReusePass() {
  return std::make_unique<BufferReusePass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir

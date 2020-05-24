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

// This file implements logic for computing proper alloc and dealloc positions.
// The main class is the BufferAssignment class that realizes this analysis.
// In order to put allocations and deallocations at safe positions, it is
// significantly important to put them into the proper blocks. However, the
// liveness analysis does not pay attention to aliases, which can occur due to
// branches (and their associated block arguments) in general. For this purpose,
// BufferAssignment firstly finds all possible aliases for a single value (using
// the BufferAssignmentAliasAnalysis class). Consider the following example:
//
// ^bb0(%arg0):
//   cond_br %cond, ^bb1, ^bb2
// ^bb1:
//   br ^exit(%arg0)
// ^bb2:
//   %new_value = ...
//   br ^exit(%new_value)
// ^exit(%arg1):
//   return %arg1;
//
// Using liveness information on its own would cause us to place the allocs and
// deallocs in the wrong block. This is due to the fact that %new_value will not
// be liveOut of its block. Instead, we have to place the alloc for %new_value
// in bb0 and its associated dealloc in exit. Using the class
// BufferAssignmentAliasAnalysis, we will find out that %new_value has a
// potential alias %arg1. In order to find the dealloc position we have to find
// all potential aliases, iterate over their uses and find the common
// post-dominator block. In this block we can safely be sure that %new_value
// will die and can use liveness information to determine the exact operation
// after which we have to insert the dealloc. Finding the alloc position is
// highly similar and non- obvious. Again, we have to consider all potential
// aliases and find the common dominator block to place the alloc.
//
// TODO(dfki):
// The current implementation does not support loops. The only thing that
// is currently missing is a high-level loop analysis that allows us to move
// allocs and deallocs outside of the loop blocks.

#include "tensorflow/compiler/mlir/xla/transforms/buffer_assignment.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Function.h"                 // TF:llvm-project
#include "mlir/IR/Operation.h"                // TF:llvm-project
#include "mlir/Pass/Pass.h"                   // TF:llvm-project
#include "absl/memory/memory.h"

namespace mlir {
namespace xla {
namespace {

//===----------------------------------------------------------------------===//
// BufferAssignmentAliasAnalysis
//===----------------------------------------------------------------------===//

/// A straight-forward alias analysis which ensures that all aliases of all
/// values will be determined. This is a requirement for the BufferAssignment
/// class since you need to determine safe positions to place alloc and
/// deallocs.
class BufferAssignmentAliasAnalysis {
 public:
  using ValueSetT = SmallPtrSet<Value, 16>;

 public:
  /// Constructs a new alias analysis using the op provided.
  BufferAssignmentAliasAnalysis(Operation* op) { build(op->getRegions()); }

  /// Finds all immediate and indirect aliases this value could potentially
  /// have. Note that the resulting set will also contain the value provided as
  /// it is an alias of itself.
  ValueSetT resolve(Value value) const {
    ValueSetT result;
    resolveRecursive(value, result);
    return result;
  }

 private:
  /// Recursively determines alias information for the given value. It stores
  /// all newly found potential aliases in the given result set.
  void resolveRecursive(Value value, ValueSetT& result) const {
    if (!result.insert(value).second) {
      return;
    }
    auto it = aliases.find(value);
    if (it == aliases.end()) return;
    for (auto alias : it->second) {
      resolveRecursive(alias, result);
    }
  }

  /// This function constructs a mapping from values to its immediate aliases.
  /// It iterates over all blocks, gets their predecessors, determines the
  /// values that will be passed to the corresponding block arguments and
  /// inserts them into map.
  void build(MutableArrayRef<Region> regions) {
    for (Region& region : regions) {
      for (Block& block : region) {
        // Iterate over all predecessor and get the mapped values to their
        // corresponding block arguments values.
        for (auto pred : block.getPredecessors()) {
          // Determine the current successor index of the current predecessor.
          unsigned successorIndex = std::distance(
              pred->getSuccessors().begin(),
              llvm::find_if(pred->getSuccessors(), [&](Block* successor) {
                return successor == &block;
              }));
          // Get the terminator and the values that will be passed to our block.
          if (auto branchInterface =
                  dyn_cast<BranchOpInterface>(pred->getTerminator())) {
            // Query the branch op interace to get the successor operands.
            auto successorOps =
                branchInterface.getSuccessorOperands(successorIndex);
            if (successorOps.hasValue()) {
              // Build the actual mapping of values to their immediate aliases.
              for (auto arg : block.getArguments()) {
                Value predecessorArgValue =
                    successorOps.getValue()[arg.getArgNumber()];
                aliases[predecessorArgValue].insert(arg);
              }
            }
          }
        }
      }
    }
  }

  /// Maps values to all immediate aliases this value can have.
  llvm::DenseMap<Value, ValueSetT> aliases;
};

//===----------------------------------------------------------------------===//
// BufferAssignmentPositions
//===----------------------------------------------------------------------===//

/// Stores proper alloc and dealloc positions to place dialect-specific alloc
/// and dealloc operations.
struct BufferAssignmentPositions {
 public:
  BufferAssignmentPositions()
      : allocPosition(nullptr), deallocPosition(nullptr) {}

  /// Creates a new positions tuple including alloc and dealloc positions.
  BufferAssignmentPositions(Operation* allocPosition,
                            Operation* deallocPosition)
      : allocPosition(allocPosition), deallocPosition(deallocPosition) {}

  /// Returns the alloc position before which the alloc operation has to be
  /// inserted.
  Operation* getAllocPosition() const { return allocPosition; }

  /// Returns the dealloc position after which the dealloc operation has to be
  /// inserted.
  Operation* getDeallocPosition() const { return deallocPosition; }

 private:
  Operation* allocPosition;
  Operation* deallocPosition;
};

//===----------------------------------------------------------------------===//
// BufferAssignmentAnalysis
//===----------------------------------------------------------------------===//

// The main buffer assignment analysis used to place allocs and deallocs.
class BufferAssignmentAnalysis {
 public:
  using DeallocSetT = SmallPtrSet<Operation*, 2>;

 public:
  BufferAssignmentAnalysis(Operation* op)
      : operation(op),
        liveness(op),
        dominators(op),
        postDominators(op),
        aliases(op) {}

  /// Computes the actual positions to place allocs and deallocs for the given
  /// value.
  BufferAssignmentPositions computeAllocAndDeallocPositions(Value value) const {
    if (value.use_empty()) {
      return BufferAssignmentPositions(value.getDefiningOp(),
                                       value.getDefiningOp());
    }
    // Get all possible aliases
    auto possibleValues = aliases.resolve(value);
    return BufferAssignmentPositions(getAllocPosition(value, possibleValues),
                                     getDeallocPosition(value, possibleValues));
  }

  /// Finds all associated dealloc nodes for the alloc nodes using alias
  /// information.
  DeallocSetT findAssociatedDeallocs(AllocOp alloc) const {
    DeallocSetT result;
    auto possibleValues = aliases.resolve(alloc);
    for (auto alias : possibleValues) {
      for (auto user : alias.getUsers()) {
        if (isa<DeallocOp>(user)) result.insert(user);
      }
    }
    return result;
  }

  /// Dumps the buffer assignment information to the given stream.
  void print(raw_ostream& os) const {
    os << "// ---- Buffer Assignment -----\n";

    for (Region& region : operation->getRegions())
      for (Block& block : region)
        for (Operation& operation : block)
          for (Value result : operation.getResults()) {
            BufferAssignmentPositions positions =
                computeAllocAndDeallocPositions(result);
            os << "Positions for ";
            result.print(os);
            os << "\n Alloc: ";
            positions.getAllocPosition()->print(os);
            os << "\n Dealloc: ";
            positions.getDeallocPosition()->print(os);
            os << "\n";
          }
  }

 private:
  /// Finds a proper placement block to store alloc/dealloc node according to
  /// the algorithm described at the top of the file. It supports dominator and
  /// post-dominator analyses via template arguments.
  template <typename AliasesT, typename DominatorT>
  Block* findPlacementBlock(Value value, const AliasesT& aliases,
                            const DominatorT& doms) const {
    assert(!value.isa<BlockArgument>() && "Cannot place a block argument");
    // Start with the current block the value is defined in.
    Block* dom = value.getDefiningOp()->getBlock();
    // Iterate over all aliases and their uses to find a safe placement block
    // according to the given dominator information.
    for (auto alias : aliases) {
      for (auto user : alias.getUsers()) {
        // Move upwards in the dominator tree to find an appropriate
        // dominator block that takes the current use into account.
        dom = doms.findNearestCommonDominator(dom, user->getBlock());
      }
    }
    return dom;
  }

  /// Finds a proper alloc positions according to the algorithm described at the
  /// top of the file.
  template <typename AliasesT>
  Operation* getAllocPosition(Value value, const AliasesT& aliases) const {
    // Determine the actual block to place the alloc and get liveness
    // information.
    auto placementBlock = findPlacementBlock(value, aliases, dominators);
    auto livenessInfo = liveness.getLiveness(placementBlock);

    // We have to ensure that the alloc will be before the first use of all
    // aliases of the given value. We first assume that there are no uses in the
    // placementBlock and that we can safely place the alloc before the
    // terminator at the end of the block.
    Operation* startOperation = placementBlock->getTerminator();
    // Iterate over all aliases and ensure that the startOperation will point to
    // the first operation of all potential aliases in the placementBlock.
    for (auto alias : aliases) {
      auto aliasStartOperation = livenessInfo->getStartOperation(alias);
      // Check whether the aliasStartOperation lies in the desired block and
      // whether it is before the current startOperation. If yes, this will be
      // the new startOperation.
      if (aliasStartOperation->getBlock() == placementBlock &&
          aliasStartOperation->isBeforeInBlock(startOperation)) {
        startOperation = aliasStartOperation;
      }
    }
    // startOperation is the first operation before which we can safely store
    // the alloc taking all potential aliases into account.
    return startOperation;
  }

  /// Finds a proper dealloc positions according to the algorithm described at
  /// the top of the file.
  template <typename AliasesT>
  Operation* getDeallocPosition(Value value, const AliasesT& aliases) const {
    // Determine the actual block to place the dealloc and get liveness
    // information.
    auto placementBlock = findPlacementBlock(value, aliases, postDominators);
    auto livenessInfo = liveness.getLiveness(placementBlock);

    // We have to ensure that the dealloc will be after the last use of all
    // aliases of the given value. We first assume that there are no uses in the
    // placementBlock and that we can safely place the dealloc at the beginning.
    Operation* endOperation = &placementBlock->front();
    // Iterate over all aliases and ensure that the endOperation will point to
    // the last operation of all potential aliases in the placementBlock.
    for (auto alias : aliases) {
      auto aliasEndOperation =
          livenessInfo->getEndOperation(alias, endOperation);
      // Check whether the aliasEndOperation lies in the desired block and
      // whether it is behind the current endOperation. If yes, this will be the
      // new endOperation.
      if (aliasEndOperation->getBlock() == placementBlock &&
          endOperation->isBeforeInBlock(aliasEndOperation)) {
        endOperation = aliasEndOperation;
      }
    }
    // endOperation is the last operation behind which we can safely store the
    // dealloc taking all potential aliases into account.
    return endOperation;
  }

  /// The operation this transformation was constructed from.
  Operation* operation;

  /// The underlying liveness analysis to compute fine grained information about
  /// alloc and dealloc positions.
  Liveness liveness;

  /// The dominator analysis to place allocs in the appropriate blocks.
  DominanceInfo dominators;

  /// The post dominator analysis to place deallocs in the appropriate blocks.
  PostDominanceInfo postDominators;

  /// The internal alias analysis to ensure that allocs and deallocs take all
  /// their potential aliases into account.
  BufferAssignmentAliasAnalysis aliases;
};

//===----------------------------------------------------------------------===//
// BufferAssignmentPass
//===----------------------------------------------------------------------===//

/// The actual buffer assignment pass that moves alloc and dealloc nodes into
/// the right positions. It uses the algorithm described at the top of the file.
// TODO(dfki): create a templated version that allows to match dialect-specific
// alloc/dealloc nodes and to insert dialect-specific dealloc node.
struct BufferAssignmentPass
    : mlir::PassWrapper<BufferAssignmentPass, FunctionPass> {
  void runOnFunction() override {
    // Get required analysis information first.
    auto& analysis = getAnalysis<BufferAssignmentAnalysis>();

    // Compute an initial placement of all nodes.
    llvm::SmallDenseMap<Value, BufferAssignmentPositions, 16> placements;
    getFunction().walk([&](AllocOp alloc) {
      placements[alloc] = analysis.computeAllocAndDeallocPositions(alloc);
    });

    // Move alloc (and dealloc - if any) nodes into the right places
    // and insert dealloc nodes if necessary.
    getFunction().walk([&](AllocOp alloc) {
      // Find already associated dealloc nodes.
      auto deallocs = analysis.findAssociatedDeallocs(alloc);
      assert(deallocs.size() < 2 &&
             "Not supported number of associated dealloc operations");

      // Move alloc node to the right place.
      BufferAssignmentPositions& positions = placements[alloc];
      Operation* allocOperation = alloc.getOperation();
      allocOperation->moveBefore(positions.getAllocPosition());

      // If there is an existing dealloc, move it to the right place.
      if (deallocs.size()) {
        Operation* nextOp = positions.getDeallocPosition()->getNextNode();
        assert(nextOp && "Invalid Dealloc operation position");
        (*deallocs.begin())->moveBefore(nextOp);
      } else {
        // If there is no dealloc node, insert one in the right place.
        OpBuilder builder(alloc);
        builder.setInsertionPointAfter(positions.getDeallocPosition());
        builder.create<DeallocOp>(allocOperation->getLoc(), alloc);
      }
    });
  };
};

}  // namespace

//===----------------------------------------------------------------------===//
// BufferAssignmentPlacer
//===----------------------------------------------------------------------===//

/// Creates a new assignment placer.
BufferAssignmentPlacer::BufferAssignmentPlacer(Operation* op)
    : operation(op), dominators(op) {}

/// Computes the actual position to place allocs for the given value.
OpBuilder::InsertPoint BufferAssignmentPlacer::computeAllocPosition(
    Value value) {
  Operation* insertOp = value.getDefiningOp();
  assert(insertOp && "There is not a defining operation for the input value");
  OpBuilder opBuilder(insertOp);
  return opBuilder.saveInsertionPoint();
}

//===----------------------------------------------------------------------===//
// FunctionAndBlockSignatureConverter
//===----------------------------------------------------------------------===//

// Performs the actual signature rewriting step.
LogicalResult FunctionAndBlockSignatureConverter::matchAndRewrite(
    FuncOp funcOp, ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const {
  auto toMemrefConverter = [&](Type t) -> Type {
    if (auto tensorType = t.dyn_cast<RankedTensorType>()) {
      return MemRefType::get(tensorType.getShape(),
                             tensorType.getElementType());
    }
    return t;
  };
  // Converting tensor-type function arguments to memref-type.
  auto funcType = funcOp.getType();
  TypeConverter::SignatureConversion conversion(funcType.getNumInputs());
  for (auto argType : llvm::enumerate(funcType.getInputs())) {
    conversion.addInputs(argType.index(), toMemrefConverter(argType.value()));
  }
  for (auto resType : funcType.getResults()) {
    conversion.addInputs(toMemrefConverter(resType));
  }
  rewriter.updateRootInPlace(funcOp, [&] {
    funcOp.setType(
        rewriter.getFunctionType(conversion.getConvertedTypes(), llvm::None));
    rewriter.applySignatureConversion(&funcOp.getBody(), conversion);
  });
  // Converting tensor-type block arugments of all blocks inside the
  // function region to memref-type except for the entry block.
  for (auto& block : funcOp.getBlocks()) {
    if (block.isEntryBlock()) continue;
    for (int i = 0, e = block.getNumArguments(); i < e; ++i) {
      auto oldArg = block.getArgument(i);
      auto newArg =
          block.insertArgument(i, toMemrefConverter(oldArg.getType()));
      oldArg.replaceAllUsesWith(newArg);
      block.eraseArgument(i + 1);
    }
  }
  return success();
}

/// A helper method to make the functions, whose all block argument types are
/// Memref or non-shaped type, legal. BufferAssignmentPlacer expects all
/// function and block argument types are in Memref or non-shaped type. Using
/// this helper method and additionally, FunctionAndBlockSignatureConverter as a
/// pattern conversion make sure that the type of block arguments are compatible
/// with using BufferAssignmentPlacer.
void FunctionAndBlockSignatureConverter::addDynamicallyLegalFuncOp(
    ConversionTarget& target) {
  auto isLegalBlockArg = [](BlockArgument arg) -> bool {
    auto type = arg.getType();
    return type.isa<MemRefType>() || !type.isa<ShapedType>();
  };
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) {
    bool legality = true;
    for (auto& block2 : funcOp.getBlocks()) {
      legality &= llvm::all_of(block2.getArguments(), isLegalBlockArg);
      if (!legality) break;
    }
    return legality;
  });
}

//===----------------------------------------------------------------------===//
// Buffer assignment pass registrations
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> createBufferAssignmentPass() {
  return absl::make_unique<BufferAssignmentPass>();
}

static PassRegistration<BufferAssignmentPass> buffer_assignment_pass(
    "buffer-assignment",
    "Executes buffer assignment pass to automatically move alloc and dealloc "
    "operations into their proper positions");

}  // namespace xla
}  // namespace mlir

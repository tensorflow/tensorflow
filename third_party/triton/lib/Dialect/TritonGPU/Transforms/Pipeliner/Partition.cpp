#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/SCCIterator.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

//===----------------------------------------------------------------------===//
// PartitionGraph
//===----------------------------------------------------------------------===//

namespace {
// A temporary node structure that can be used to build a graph of partitions.
// The consumers have to be precomputed in order for the SCC iterator to have an
// acceptable runtime complexity. This assumes the underlying loop is immutable.
struct PartitionNode {
  PartitionNode(const WarpSchedule::Partition *partition)
      : partition(partition) {}

  // The partition this node represents.
  const WarpSchedule::Partition *partition;
  // Partitions that consume the outputs of this partition.
  SmallVector<std::pair<const PartitionNode *, OpOperand *>> consumers;
};

// A graph of partitions that can be used to check for cycles and other schedule
// invariants.
struct PartitionGraph {
  PartitionGraph(scf::ForOp loop, const WarpSchedule &schedule);

  PartitionNode root;
  llvm::MapVector<const WarpSchedule::Partition *, PartitionNode> nodes;
};
} // namespace

PartitionGraph::PartitionGraph(scf::ForOp loop, const WarpSchedule &schedule)
    : root(schedule.getRootPartition()) {
  // Create the nodes at once. Afterwards, the map won't re-allocate and the
  // pointers will be stable.
  for (WarpSchedule::Partition &partition : schedule.getPartitions())
    nodes.try_emplace(&partition, &partition);

  // Wire up the graph. Consider the root node to be consumed by all other
  // partitions so that it can be used as a virtual root.
  for (PartitionNode &node : llvm::make_second_range(nodes))
    root.consumers.emplace_back(&node, nullptr);

  // Check the users of the partition outputs to wire the rest of the graph.
  for (auto &[partition, node] : nodes) {
    auto callback = [&, node = &node](Operation *owner, OpOperand &use) {
      // Ignore uses in subsequent iterations.
      if (isa<scf::YieldOp>(owner))
        return;
      PartitionNode &consumer =
          nodes.find(schedule.getPartition(owner))->second;
      node->consumers.emplace_back(&consumer, &use);
    };
    schedule.iterateOutputs(loop, partition, callback);
  }
}

namespace llvm {
template <> struct GraphTraits<PartitionGraph> {
  using NodeRef = std::pair<const PartitionNode *, mlir::OpOperand *>;
  static NodeRef getEntryNode(const PartitionGraph &graph) {
    return {&graph.root, nullptr};
  }

  using ChildIteratorType = SmallVector<NodeRef>::const_iterator;
  static ChildIteratorType child_begin(NodeRef node) {
    return node.first->consumers.begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return node.first->consumers.end();
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// WarpSchedule
//===----------------------------------------------------------------------===//

WarpSchedule::Partition *WarpSchedule::addPartition(unsigned stage) {
  partitions.push_back(std::make_unique<Partition>(partitions.size(), stage));
  return partitions.back().get();
}

WarpSchedule::Partition *WarpSchedule::getPartition(Operation *op) {
  return opToPartition.at(op);
}
const WarpSchedule::Partition *WarpSchedule::getPartition(Operation *op) const {
  return opToPartition.at(op);
}

WarpSchedule::Partition *WarpSchedule::getPartition(unsigned idx) {
  return partitions[idx].get();
}
const WarpSchedule::Partition *WarpSchedule::getPartition(unsigned idx) const {
  return partitions[idx].get();
}

FailureOr<WarpSchedule> WarpSchedule::deserialize(scf::ForOp loop) {
  auto stages = loop->getAttrOfType<ArrayAttr>(kPartitionStagesAttrName);
  if (!stages) {
    return mlir::emitWarning(loop.getLoc(), "missing '")
           << kPartitionStagesAttrName << "' attribute";
  }

  WarpSchedule result;
  for (auto [idx, attr] : llvm::enumerate(stages)) {
    auto stage = dyn_cast<IntegerAttr>(attr);
    if (!stage || stage.getInt() < 0) {
      return mlir::emitWarning(loop.getLoc(), "partition stages attribute '")
             << kPartitionStagesAttrName << "' has invalid element " << attr;
    }

    result.partitions.push_back(
        std::make_unique<Partition>(idx, stage.getInt()));
  }

  for (Operation &op : loop.getBody()->without_terminator()) {
    Partition *partition = result.getRootPartition();
    if (auto attr = op.getAttrOfType<IntegerAttr>(kPartitionAttrName)) {
      int64_t idx = attr.getInt();
      if (idx < 0 || idx >= result.partitions.size()) {
        return mlir::emitWarning(op.getLoc(), "invalid partition index ")
               << idx;
      }
      partition = result.partitions[idx].get();
    }

    partition->insert(&op);
    result.opToPartition[&op] = partition;
  }

  return result;
}

void WarpSchedule::serialize(scf::ForOp loop) const {
  SmallVector<Attribute> stages;
  Builder b(loop.getContext());
  for (auto [i, partition] :
       llvm::enumerate(llvm::make_pointee_range(partitions))) {
    stages.push_back(b.getI32IntegerAttr(partition.getStage()));
    for (Operation *op : partition.getOps()) {
      op->setAttr(kPartitionAttrName, b.getI32IntegerAttr(i));
    }
  }
  loop->setAttr(kPartitionStagesAttrName, b.getArrayAttr(stages));
}

LogicalResult WarpSchedule::verify(scf::ForOp loop) const {
  // The root partition is only allowed to transitively depend on itself.
  bool failed = false;
  iterateInputs(loop, getRootPartition(), [&](OpOperand &input) {
    auto [def, distance] = getDefiningOpAndDistance(loop, input.get());
    // Ignore values defined outside the loop.
    if (!def || def->getParentOp() != loop)
      return;
    const Partition *defPartition = opToPartition.at(def);
    if (defPartition == getRootPartition())
      return;
    InFlightDiagnostic diag = mlir::emitWarning(input.getOwner()->getLoc());
    diag << "operation in the root partition depends on a value that "
            "originates from a non-root partition through operand #"
         << input.getOperandNumber();
    diag.attachNote(def->getLoc())
        << "operand defined here in partition #" << defPartition->getIndex()
        << " at distance " << distance;
    failed = true;
  });
  if (failed)
    return failure();

  // Within a loop iteration, the partitions must form a DAG. For example, the
  // following is invalid:
  //
  //   scf.for %i = %lb to %ub step %step
  //     %0 = op_a()     {ttg.partition = 0}
  //     %1 = op_b(%0)   {ttg.partition = 1}
  //     op_c(%1)        {ttg.partition = 0}
  //
  PartitionGraph graph(loop, *this);
  for (auto it = llvm::scc_begin(graph); !it.isAtEnd(); ++it) {
    if (!it.hasCycle())
      continue;
    InFlightDiagnostic diag =
        mlir::emitWarning(loop.getLoc(), "warp schedule contains a cycle");
    for (auto [node, use] : *it) {
      assert(use && "already checked that the root partition has no ancestors");
      diag.attachNote(use->getOwner()->getLoc())
          << "operation in partition #" << node->partition->getIndex()
          << " uses value defined in partition #"
          << opToPartition.at(use->get().getDefiningOp())->getIndex();
    }
    return failure();
  }

  // Each partition's stage must be strictly less than all of its consumers plus
  // the distance.
  for (Partition &partition : getPartitions()) {
    bool failed = false;
    auto callback = [&](OpResult output, OpOperand &use, unsigned distance) {
      const Partition *consumer = opToPartition.at(use.getOwner());
      if (partition.getStage() < consumer->getStage() + distance)
        return;
      InFlightDiagnostic diag =
          mlir::emitWarning(loop.getLoc(), "partition #")
          << partition.getIndex() << " has stage " << partition.getStage()
          << " but is consumed by partition #" << consumer->getIndex()
          << " with stage " << consumer->getStage() << " at distance "
          << distance;
      diag.attachNote(use.getOwner()->getLoc())
          << "use of value defined in partition #" << partition.getIndex()
          << " at " << distance << " iterations in the future";
      diag.attachNote(output.getLoc())
          << "value defined here in partition #" << partition.getIndex();
      failed = true;
    };
    iterateUses(loop, &partition, callback);
    if (failed)
      return failure();
  }

  return success();
}

void WarpSchedule::eraseFrom(scf::ForOp loop) {
  for (Operation &op : loop.getBody()->without_terminator())
    op.removeAttr(kPartitionAttrName);
  loop->removeAttr(kPartitionStagesAttrName);
}

void WarpSchedule::iterateInputs(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(OpOperand &)> callback) const {
  for (Operation *op : partition->getOps()) {
    visitNestedOperands(op, [&](OpOperand &operand) {
      // Ignore implicit captures.
      Value value = operand.get();
      if (value.getParentBlock() != loop.getBody())
        return;
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        assert(arg.getOwner() == loop.getBody());
        // Ignore the induction variable.
        if (arg == loop.getInductionVar())
          return;
        // This value originates from a previous iteration.
        assert(llvm::is_contained(loop.getRegionIterArgs(), arg));
        callback(operand);
      } else if (opToPartition.at(value.getDefiningOp()) != partition) {
        // This value originates from a different partition in the same
        // iteration.
        assert(value.getDefiningOp()->getParentOp() == loop);
        callback(operand);
      }
    });
  }
}

void WarpSchedule::iterateOutputs(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(Operation *, OpOperand &)> callback) const {
  for (Operation *op : partition->getOps()) {
    for (OpResult result : op->getOpResults()) {
      for (OpOperand &use : result.getUses()) {
        Operation *owner =
            loop.getBody()->findAncestorOpInBlock(*use.getOwner());
        if (isa<scf::YieldOp>(owner)) {
          // This value is used in a subsequent iteration.
          callback(owner, use);
        } else if (opToPartition.at(owner) != partition) {
          // This value is used in a different partition in the same iteration.
          callback(owner, use);
        }
      }
    }
  }
}

void WarpSchedule::iterateDefs(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(OpResult, unsigned)> callback) const {
  iterateInputs(loop, partition, [&](OpOperand &input) {
    auto [def, distance] = getDefinitionAndDistance(loop, input.get());
    if (def && def.getParentBlock() == loop.getBody())
      callback(def, distance);
  });
}

void WarpSchedule::iterateUses(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(OpResult, OpOperand &, unsigned)> callback) const {
  SmallVector<std::tuple<OpResult, OpOperand *, unsigned>> uses;
  iterateOutputs(loop, partition, [&](Operation *owner, OpOperand &use) {
    uses.emplace_back(cast<OpResult>(use.get()), &use, 0);
  });
  while (!uses.empty()) {
    auto [output, use, distance] = uses.pop_back_val();
    if (!isa<scf::YieldOp>(use->getOwner())) {
      callback(output, *use, distance);
      continue;
    }
    BlockArgument arg = loop.getRegionIterArg(use->getOperandNumber());
    for (OpOperand &use : arg.getUses())
      uses.emplace_back(output, &use, distance + 1);
  }
}

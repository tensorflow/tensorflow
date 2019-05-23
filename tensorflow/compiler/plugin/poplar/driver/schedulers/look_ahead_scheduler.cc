#include "tensorflow/compiler/plugin/poplar/driver/schedulers/look_ahead_scheduler.h"

#include <map>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace poplarplugin {
namespace {

using ::tensorflow::strings::HumanReadableNumBytes;

/*
  A scheduler designed to schedule instructions in groups by looking past local
  minima to find short groups of instructions which can be scheduled together
  for a net gain. Each node is grouped to ONE cluster which may or may not
  contain multiple nodes.


  1. Firstly the instructions are grouped into clusters. This is implemented by
  the ClusterHelper structure. The default clustering algorithm is to group all
  continous chains of instructions and ending each cluster only when the last
  instruction has multiple children.

  The scheduler is agnostic to this part and different algorithms can be used
  in the future.

  2. The clusters are then added to a priority queue based on their memory
  usage. Firstly clusters with no dependencies are added. Then as we reduce the
  number of pending dependencies we add any and all nodes with zero pending
  dependcies to the priority queue.

  They will be added based on ClusterComparitor. Adding a new comparitor class
  won't have any impact on the rest of the scheduler or clusterer.

*/

class LookAheadScheduler {
 public:
  // Construct and return a memory-minimizing sequence of HLO instructions
  // containing the given HLO computation.
  static StatusOr<HloInstructionSequence> Run(
      HloComputation* computation,
      const TuplePointsToAnalysis& points_to_analysis,
      const LogicalBuffer::SizeFunction& size_function,
      const absl::flat_hash_map<const HloComputation*, int64>&
          memory_by_computation,
      int64 max_syncs) {
    LookAheadScheduler scheduler(computation, points_to_analysis, size_function,
                                 memory_by_computation, max_syncs);
    return scheduler.CreateSchedule();
  }

  // Returns whether the memory used by the given HLO should be ignored by the
  // scheduling heuristic.
  static bool IgnoreInstruction(const HloInstruction& instruction) {
    return instruction.opcode() == HloOpcode::kParameter ||
           instruction.opcode() == HloOpcode::kConstant;
  }

 private:
  // Helper structure to represent a group of instructions. Each instruction
  // which depends on another instruction that is not in this cluster will track
  // that dependency at the cluster level rather than individual instruction.
  // These are tracked by the dependencies set and the reverse_dependencies list
  // tracks all the clusters which in turn depend on this cluster.
  struct Cluster {
    using Ref = Cluster*;

    Cluster() : nodes({}), dependencies({}), net_memory_usage(0) {}

    // All of the instructions grouped by this cluster.
    std::list<HloInstruction*> nodes;

    // The set of all the nodes which this cluster depends on.
    absl::flat_hash_set<Ref> dependencies;

    // A list of all the clusters which have a dependency on this cluster.
    std::list<Ref> reverse_dependencies;

    // Estimate of the net memory saved/accumulated from all scheduling all the
    // instructions in nodes.
    int64 net_memory_usage;

    bool is_all_reduce;
  };

  // So we can store Cluster::Ref in a priority queue or any other structure
  // that needs a custom comparitor.
  struct ClusterComparitor {
    bool operator()(const Cluster::Ref& lhs, const Cluster::Ref& rhs) const {
      if (lhs->net_memory_usage != rhs->net_memory_usage) {
        return lhs->net_memory_usage < rhs->net_memory_usage;
      }

      if (rhs->nodes.empty()) {
        // Nothing compares less than an empty set.
        return false;
      }

      if (lhs->nodes.empty()) {
        return true;
      }

      // Compare the minimum HloInstruction id between the two clusters.
      auto compare_id = [](const HloInstruction* inst1,
                           const HloInstruction* inst2) {
        return inst1->unique_id() < inst2->unique_id();
      };
      return compare_id(*absl::c_min_element(lhs->nodes, compare_id),
                        *absl::c_min_element(rhs->nodes, compare_id));
    }
  };

  // To keep the clustering as modular and independent to the scheduler as
  // possible we use this helper class to do the actual clustering with the idea
  // that we may in future want to swap between multiple different clustering
  // algorithms.
  class ClusterHelper {
   public:
    ClusterHelper(LookAheadScheduler* parent_scheduler)
        : parent(parent_scheduler) {}

    // Compute the full cluster graph and store it in parent.
    void ClusterNodes();

    const std::set<Cluster::Ref, ClusterComparitor>& GetRoots() {
      return root_nodes;
    }

   private:
    // Cluster instructions by following through continous chains of
    // instructions. Stopping only when an instruction has multiple paths.
    void GroupChainsOfInstructions();

    // After the instructions have been clusters we need to go over them again
    // to add the dependency information.
    void BuildDependencyGraph();

    LookAheadScheduler* parent;

    // Roots of the cluster graph. These are nodes which have no strict
    // dependencies (a dependency that isn't itself) so can be scheduled
    // immediately.
    std::set<Cluster::Ref, ClusterComparitor> root_nodes;

    absl::flat_hash_map<HloInstruction*, Cluster::Ref>
        previously_clustered_node;
  };

  LookAheadScheduler(HloComputation* computation,
                     const TuplePointsToAnalysis& points_to_analysis,
                     const LogicalBuffer::SizeFunction& size_function,
                     const absl::flat_hash_map<const HloComputation*, int64>&
                         memory_by_computation,
                     int64 max_syncs_)
      : computation_(computation),
        points_to_analysis_(points_to_analysis),
        size_function_(size_function),
        memory_by_computation_(memory_by_computation),
        max_syncs(max_syncs_),
        pending_syncs_in_bytes(0) {}

  // Returns whether the memory used by the given buffer should be ignored by
  // the scheduling heuristic.
  static bool IgnoreBuffer(const LogicalBuffer& buffer) {
    return IgnoreInstruction(*buffer.instruction());
  }

  int64 GetBufferMemoryFreed(const HloInstruction* parent,
                             const HloInstruction* operand);

  int64 BytesFreedIfScheduled(const HloInstruction* instruction);

  // Add a cluster node to the wait queue or if it has no dependencies, straight
  // to the ready queue.
  void AddClusterWaitOrReadyQueue(Cluster::Ref node_to_add);

  // Add all of the dependencies of the node_just_added to the wait queue if
  // they have other unscheduled parents or add them to the ready queue if this
  // is the last parent to be scheduled.
  void AddDepsToWaitOrReadyQueue(Cluster::Ref node_just_added);

  // Dump the cluster to VLOG as a dot file. Provided order parameter is used to
  // determine the order that each cluster was added in.
  void DumpClusterAsDot(
      const absl::flat_hash_map<Cluster::Ref, uint32_t>& order) const;

  // Add to the ready or sync list queues.
  void AddToReady(Cluster::Ref node);

  // Pop the next instruction either from the ready queue or from the all reduce
  // list.
  Cluster::Ref PopFromQueue(bool& should_all_reduce);

  HloInstructionSequence CreateSchedule();

  HloComputation* computation_;
  const TuplePointsToAnalysis& points_to_analysis_;
  const LogicalBuffer::SizeFunction& size_function_;

  // Computations are analyzed in post-order. When scheduling an instruction
  // that includes subcomputations, such as a while loop, we use this map to
  // look up the memory needed by subcomputations.
  const absl::flat_hash_map<const HloComputation*, int64>&
      memory_by_computation_;

  // The underlaying structure used to manage the storage of all the clusters.
  std::list<Cluster> cluster_memory_storage;

  // Tracker to show which of the clusters have already been scheduled.
  absl::flat_hash_set<Cluster::Ref> scheduled_clusters;

  // This will determine the order instructions are insterted based on
  // ClusterComparitor. As an instruction is inserted the queue is updated if
  // any more nodes are no longer blocked by their dependencies.
  std::priority_queue<Cluster::Ref, std::vector<Cluster::Ref>,
                      ClusterComparitor>
      ready_queue;

  // We want all the AllReduce operations to be at the back of the queue
  // until we decide to schedule them once the size of sync_queue (in bytes)
  // is greater than max_syncs or if we have to due to ready queue being empty.
  std::list<Cluster::Ref> sync_queue;

  // Map of nodes waiting to be scheduled to their dependencies which have not
  // been scheduled.
  absl::flat_hash_map<Cluster::Ref, absl::flat_hash_set<Cluster::Ref>>
      wait_queue;

  // Maximum number of AllReduces we should keep at the back of the queue.
  int64 max_syncs;

  // Total size of syncs that we have pending in the sync_queue .
  int64 pending_syncs_in_bytes;
};

int64 LookAheadScheduler::GetBufferMemoryFreed(const HloInstruction* parent,
                                               const HloInstruction* operand) {
  int64 size = 0;
  // Calculate the total memory used by this operands output.
  points_to_analysis_.GetPointsToSet(operand).ForEachElement(
      [&](const ShapeIndex& /*index*/, const PointsToSet::BufferList& buffers) {
        std::for_each(buffers.begin(), buffers.end(),
                      [&](const LogicalBuffer* buffer) {
                        size += this->size_function_(*buffer);
                      });
      });

  return size;
}

int64 LookAheadScheduler::BytesFreedIfScheduled(
    const HloInstruction* instruction) {
  auto opcode = instruction->opcode();

  int64 freed_bytes = 0;
  for (const HloInstruction* operand : instruction->operands()) {
    freed_bytes += GetBufferMemoryFreed(instruction, operand);
  }

  // We only count the memory usage of the largest subcomputation, instead of
  // adding them all, because subcomputations won't execute in parallel.
  int64 max_subcomputation_bytes = 0;
  for (const auto* c : instruction->called_computations()) {
    auto it = memory_by_computation_.find(c);
    if (it != memory_by_computation_.end()) {
      int64 subcomputation_bytes = it->second;
      if (subcomputation_bytes > max_subcomputation_bytes) {
        max_subcomputation_bytes = subcomputation_bytes;
      }
    }
  }

  int64 bytes_defined = 0;
  if (max_subcomputation_bytes > 0 &&
      (opcode == HloOpcode::kWhile || opcode == HloOpcode::kCall ||
       opcode == HloOpcode::kConditional)) {
    // The output buffer of while/call/conditional is always aliased with the
    // output buffer of the root instruction in the body. Don't double count.
    bytes_defined = max_subcomputation_bytes;
  } else {
    // Calculate bytes defined.
    for (auto* buffer :
         points_to_analysis_.GetBuffersDefinedByInstruction(instruction)) {
      if (!IgnoreBuffer(*buffer)) {
        bytes_defined += size_function_(*buffer);
      }
    }
  }
  return freed_bytes - bytes_defined;
}

// Dump the cluster as a dot file. Each cluster is a node with the
// dependencies between clusters being the edges. If the graph is large it may
// be advisable to use "dot -gslimit=1" to improve processing time.
void LookAheadScheduler::DumpClusterAsDot(
    const absl::flat_hash_map<Cluster::Ref, uint32_t>& order) const {
  std::stringstream dot;
  dot << "digraph clusters { ";

  for (auto& pair : order) {
    const Cluster::Ref cluster = pair.first;
    uint32_t position_in_schedule = pair.second;

    dot << (uint64_t)cluster << " [label=\"";

    for (const HloInstruction* node : cluster->nodes) {
      dot << node->metadata().op_type() << ": " << node->name() << "\\n";
    }

    dot << HumanReadableNumBytes(cluster->net_memory_usage) << "\"";

    dot << ", xlabel=\"" << position_in_schedule << "\"];";

    for (const Cluster::Ref dependency : cluster->dependencies) {
      dot << (uint64_t)dependency << " -> " << (uint64_t)cluster << ";";
    }
  }

  dot << "}";
  VLOG(0) << dot.str() << "\n\n";
}

void LookAheadScheduler::ClusterHelper::GroupChainsOfInstructions() {
  // For each instruction in the graph group it into a cluster with its
  // neighbours if possible.
  for (HloInstruction* instruction : parent->computation_->instructions()) {
    // If this instruction is already in another cluster, skip.
    if (previously_clustered_node.count(instruction) != 0) continue;

    // Add an empty cluster to be populated to the reference tracker.
    parent->cluster_memory_storage.push_back({});

    // And keeep a reference to it.
    Cluster::Ref ref = &parent->cluster_memory_storage.back();

    // Add the first node to the cluster.
    ref->nodes.push_back(instruction);
    previously_clustered_node.insert({instruction, ref});
    ref->net_memory_usage += parent->BytesFreedIfScheduled(instruction);

    // We mark these so they can be added to their own seperately managed queue.
    ref->is_all_reduce = instruction->opcode() == HloOpcode::kAllReduce;

    // Add the child nodes to the cluster.
    HloInstruction* current_instruction = instruction;
    while (current_instruction && current_instruction->user_count() == 1 &&
           current_instruction->control_successors().empty() &&
           !ref->is_all_reduce) {
      HloInstruction* user = current_instruction->users()[0];
      // Check the child hasn't already been added to a cluster. We want all
      // reduce instructions to be in their own cluster.
      if (previously_clustered_node.count(user) != 0 ||
          user->opcode() == HloOpcode::kAllReduce)
        break;

      // Add the child.
      ref->nodes.push_back(user);
      previously_clustered_node.insert({user, ref});
      ref->net_memory_usage += parent->BytesFreedIfScheduled(user);

      current_instruction = user;
    }
  }
}

void LookAheadScheduler::ClusterHelper::BuildDependencyGraph() {
  // Build the dependency map of the clusters.
  for (Cluster& cluster : parent->cluster_memory_storage) {
    // Could be a root node. We will remove if this is invalidated later.
    if (cluster.dependencies.empty()) root_nodes.insert(&cluster);

    for (HloInstruction* node : cluster.nodes) {
      // Add any users of this cluster to the dependency set.
      for (HloInstruction* user : node->users()) {
        auto itr = previously_clustered_node.find(user);
        assert(itr != previously_clustered_node.end());

        // Check that the dependency is between clusters and not between nodes
        // in this cluster.
        if (itr->second != &cluster) {
          auto pair = itr->second->dependencies.insert(&cluster);

          if (pair.second) cluster.reverse_dependencies.push_back(itr->second);

          if (root_nodes.count(itr->second) != 0) root_nodes.erase(itr->second);
        }
      }

      // Add any control link in this cluster to the dependency set.
      for (HloInstruction* control : node->control_successors()) {
        auto itr = previously_clustered_node.find(control);
        assert(itr != previously_clustered_node.end());

        if (itr->second != &cluster) {
          auto pair = itr->second->dependencies.insert(&cluster);

          if (pair.second) cluster.reverse_dependencies.push_back(itr->second);

          if (root_nodes.count(itr->second) != 0) root_nodes.erase(itr->second);
        }
      }
    }
  }
}

void LookAheadScheduler::ClusterHelper::ClusterNodes() {
  GroupChainsOfInstructions();
  BuildDependencyGraph();
}

void LookAheadScheduler::AddToReady(Cluster::Ref node_to_add) {
  if (node_to_add->is_all_reduce) {
    sync_queue.push_back(node_to_add);
    pending_syncs_in_bytes +=
        ShapeUtil::ByteSizeOf((*node_to_add->nodes.begin())->shape());
  } else {
    ready_queue.push(node_to_add);
  }
}

LookAheadScheduler::Cluster::Ref LookAheadScheduler::PopFromQueue(
    bool& should_all_reduce) {
  if (sync_queue.empty()) should_all_reduce = false;

  // If the ready queue is empty at any point or we exceede the maximum number
  // of pending AllReduce ops, empty the list. We maintain the bool
  // should_all_reduce to ensure once we start queueing all reduce we don't stop
  // until the list is empty.
  if (ready_queue.empty() || pending_syncs_in_bytes > max_syncs ||
      should_all_reduce) {
    should_all_reduce = true;

    Cluster::Ref r = sync_queue.front();
    sync_queue.pop_front();

    pending_syncs_in_bytes -=
        ShapeUtil::ByteSizeOf((*r->nodes.begin())->shape());

    return r;
  } else {
    // The normal path is to just take from the top of the priority queue.
    Cluster::Ref r = ready_queue.top();
    ready_queue.pop();
    return r;
  }
}

// Add a cluster node to the wait queue or if it has no dependencies, straight
// to the ready queue.
void LookAheadScheduler::AddClusterWaitOrReadyQueue(Cluster::Ref node_to_add) {
  // If all the parents of this node have already been scheduled we can
  // just schedule this node directly.
  bool canJustSchedule = true;

  // Check if the parents have been scheduled.
  for (Cluster::Ref parent_dependency : node_to_add->dependencies) {
    // If the parent hasn't been scheduled add it to the set of
    // dependences that are pending.
    if (scheduled_clusters.count(parent_dependency) == 0) {
      wait_queue[node_to_add].insert(parent_dependency);
      canJustSchedule = false;
    }
  }

  // Put it in the ready queue.
  if (canJustSchedule) {
    AddToReady(node_to_add);
  }
}

// Add all of the dependencies of the node_just_added to the wait queue if
// they have other unscheduled parents or add them to the ready queue if this
// is the last parent to be scheduled.
void LookAheadScheduler::AddDepsToWaitOrReadyQueue(
    Cluster::Ref node_just_added) {
  // Add all of the next nodes into either the wait queue or the ready queue.
  for (Cluster::Ref child_dependency : node_just_added->reverse_dependencies) {
    if (wait_queue.count(child_dependency) != 0) {
      absl::flat_hash_set<Cluster::Ref>& pending_deps =
          wait_queue[child_dependency];

      pending_deps.erase(node_just_added);

      if (pending_deps.size() == 0) {
        AddToReady(child_dependency);
      }
    } else {
      AddClusterWaitOrReadyQueue(child_dependency);
    }
  }
}

HloInstructionSequence LookAheadScheduler::CreateSchedule() {
  HloInstructionSequence schedule;

  bool should_dump_dot = tensorflow::GetPoplarXlaFlags().dump_schedule_as_dot;

  // Tracker to make sure at the end we have added the correct number of
  // instructions.
  size_t number_of_instructions_added = 0;

  // A debug structure to indentify the scheduled order that each node has been
  // inserted in.
  absl::flat_hash_map<Cluster::Ref, uint32_t> order;

  // Group all nodes into their clusters.
  ClusterHelper clustering_helper{this};
  clustering_helper.ClusterNodes();

  // Start with the roots of the graph and add them to the queue first.
  for (Cluster::Ref ref : clustering_helper.GetRoots()) {
    AddToReady(ref);
  }

  bool should_all_reduce = false;

  while (!ready_queue.empty() || !sync_queue.empty()) {
    // Deque.
    Cluster::Ref r = PopFromQueue(should_all_reduce);

    if (scheduled_clusters.count(r) != 0) continue;

    // Schedule each instruction.
    for (HloInstruction* instruction : r->nodes) {
      schedule.push_back(instruction);

      number_of_instructions_added++;
    }

    if (should_dump_dot) {
      order.insert({r, order.size()});
    }

    scheduled_clusters.insert(r);

    // Add the dependencies of the last cluster to the ready queue.
    AddDepsToWaitOrReadyQueue(r);
  }

  if (should_dump_dot) {
    DumpClusterAsDot(order);
  }

  CHECK_EQ(schedule.size(), computation_->instruction_count());
  CHECK_EQ(number_of_instructions_added, computation_->instruction_count());

  return schedule;
}

StatusOr<HloInstructionSequence> LookAheadScheduler(
    HloComputation* computation,
    const TuplePointsToAnalysis& points_to_analysis,
    const LogicalBuffer::SizeFunction& size_function,
    const absl::flat_hash_map<const HloComputation*, int64>&
        memory_by_computation,
    int64 max_syncs) {
  return LookAheadScheduler::Run(computation, points_to_analysis, size_function,
                                 memory_by_computation, max_syncs);
}
}  // namespace

// Create a functor which performs the look-ahead scheduling.
MemorySchedulerAlgorithm CreateLookAheadMemoryScheduler(int64 max_syncs) {
  return [=](HloComputation* computation,
             const TuplePointsToAnalysis& points_to_analysis,
             const LogicalBuffer::SizeFunction& size_function,
             const absl::flat_hash_map<const HloComputation*, int64>&
                 memory_by_computation) {
    return LookAheadScheduler(computation, points_to_analysis, size_function,
                              memory_by_computation, max_syncs);
  };
}

}  // namespace poplarplugin
}  // namespace xla

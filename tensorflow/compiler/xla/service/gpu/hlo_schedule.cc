/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <deque>
#include <memory>
#include <unordered_map>

#include "tensorflow/compiler/xla/service/gpu/hlo_schedule.h"

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace gpu {

namespace {

// An HLO partial ordering based on the actual stream assignment and thunk
// launch order.
class GpuHloOrdering : public PredecessorHloOrdering {
 public:
  GpuHloOrdering(const HloModule* module,
                 const StreamAssignment& stream_assignment,
                 const std::vector<const HloInstruction*>& thunk_launch_order);
  ~GpuHloOrdering() override = default;

  // Only the entry computation can possibly be sequentially ordered, and only
  // if we've assigned all instructions to a single stream.
  const std::vector<const HloInstruction*>* SequentialOrder(
      const HloComputation& computation) const override {
    return &computation == module_->entry_computation() ? entry_sequence_.get()
                                                        : nullptr;
  }

  string ToString() const override { return ToStringHelper("GpuHloOrdering"); }

 private:
  std::unique_ptr<std::vector<const HloInstruction*>> entry_sequence_;
};

GpuHloOrdering::GpuHloOrdering(
    const HloModule* module, const StreamAssignment& stream_assignment,
    const std::vector<const HloInstruction*>& thunk_launch_order)
    : PredecessorHloOrdering(module) {
  // The entry computation has a total order when there's only one stream.
  if (stream_assignment.StreamCount() == 1) {
    entry_sequence_ =
        MakeUnique<std::vector<const HloInstruction*>>(thunk_launch_order);
  }

  // The ordering of instructions for the entry computation is determined by the
  // total order of thunk launches, and stream assignment. Instructions are
  // sequential within a stream and concurrent across streams. In addition, the
  // GpuExecutable adds cross-stream dependency edges to ensure each instruction
  // waits for its operands before executing.
  //
  // The predecessor map is built incrementally, in thunk launch order. We
  // record the instructions already visited per stream in
  // 'instructions_per_stream'. This lets us quickly determine the same-stream
  // predecessors of each instruction. To capture cross-stream dependency edges,
  // we use the predecessor map to insert each operand as well as its transitive
  // closure of dependencies.

  // Compute the set of all instructions we will want to set reachability on
  auto predecessor_map = MakeUnique<HloComputation::ReachabilityMap>(
      module->entry_computation()->MakeInstructionPostOrder());

  std::vector<std::vector<const HloInstruction*>> instructions_per_stream(
      stream_assignment.StreamCount());

  for (const HloInstruction* hlo : thunk_launch_order) {
    if (stream_assignment.HasStreamAssigned(*hlo)) {
      // All ops already queued on the same instruction stream, and their
      // transitive predecessors, are predecessors. Since the relation is
      // transitive, we just set the transitive closure of the previous op.
      const int stream_no = stream_assignment.StreamNumberForHlo(*hlo);
      std::vector<const HloInstruction*>* instructions =
          &instructions_per_stream[stream_no];
      if (!instructions->empty()) {
        const HloInstruction* back = instructions->back();
        predecessor_map->SetReachableAndTransitiveClosure(hlo, back);
      }
      // All operands and their transitive predecessors are predecessors. Each
      // operand must already exist in 'predecessor_map', since we're iterating
      // in thunk launch order.
      for (const HloInstruction* operand : hlo->operands()) {
        predecessor_map->SetReachableAndTransitiveClosure(hlo, operand);
      }
      instructions->push_back(hlo);
    } else {
      // Only parameters and constants don't have an assigned stream, since they
      // don't require a thunk. These ops don't have any predecessors.
      CHECK(hlo->opcode() == HloOpcode::kParameter ||
            hlo->opcode() == HloOpcode::kConstant);
      CHECK_EQ(hlo->operand_count(), 0);
    }
  }
  strict_predecessors_.emplace(module->entry_computation(),
                               std::move(predecessor_map));

  // The ordering of instructions in subcomputations is based solely on data
  // dependencies. I.e. the strict predecessors of each subcomputation
  // instruction is its transitive operands.
  //
  // TODO(toddw): Each subcomputation is actually emitted as a function in DFS
  // postorder, so we can do better and establish the total order here. We don't
  // do that yet since it's hard to ensure that the order here is the order used
  // by IrEmitterNested. And mismatched ordering bugs would be hard to find.
  for (auto& computation : module->computations()) {
    if (computation.get() != module->entry_computation()) {
      strict_predecessors_.emplace(computation.get(),
                                   computation->ComputeTransitiveOperands());
    }
  }
}

// Computes a topological launch_order that is close to a breadth-first
// order. This heuristic works well for graphs where concurrent kernels are
// located at the same layer. It can often reduce dependency between concurrent
// GEMMs due to intra-stream total orders.  E.g. consider the following HLO
// graph where the numbers in the parens indicate the stream assigned to each
// HLO.
//
//   A(0) -> D(0) -> E(1)
//    |
//    v
//   B(0)
//    |
//    v
//   C(0)
//
// If the total order is A,B,C,D,E, then C and E would be sequentialized
// because C completes before D starts in stream 0, and E depends on D.
// However, if the total order is A,B,D,C,E, then C and E can run
// concurrently.
void BFSLaunchOrder(const HloComputation* computation,
                    std::vector<const HloInstruction*>* launch_order) {
  // This topological sort uses two data structures:
  // 1. `incoming_edge_count` which keeps track of the number of incoming
  // edges to each HLO;
  // 2. `queue` which contains all HLOs with no incoming edges.
  //
  // The sorting algorithm repeatedly pops the top from the queue and deletes
  // that HLO from the graph, making more HLOs incoming-edge free.
  std::deque<const HloInstruction*> queue;
  std::unordered_map<const HloInstruction*, int64> incoming_edge_count;
  for (const auto& hlo : computation->instructions()) {
    if (hlo->operand_count() == 0) {
      queue.push_back(hlo.get());
    } else {
      incoming_edge_count[hlo.get()] =
          std::set<HloInstruction*>(hlo->operands().begin(),
                                    hlo->operands().end())
              .size();
    }
  }

  while (!queue.empty()) {
    const HloInstruction* x = queue.front();
    queue.pop_front();
    launch_order->push_back(x);
    for (const HloInstruction* y : x->users()) {
      --incoming_edge_count[y];
      if (incoming_edge_count[y] == 0) {
        queue.push_back(y);
      }
    }
  }
}

}  // end namespace

HloSchedule::HloSchedule() {}

/* static */
StatusOr<std::unique_ptr<HloSchedule>> HloSchedule::Build(
    const HloModule& module, const StreamAssignment& stream_assignment,
    int64 pointer_size) {
  std::unique_ptr<HloSchedule> schedule(new HloSchedule);

  // Initialize thunk_launch_order_, the total order of thunk launches.
  const HloComputation* entry_computation = module.entry_computation();
  if (stream_assignment.StreamCount() == 1) {
    // All kernels are launched on a single stream, so there's no loss of
    // concurrency by optimizing for minimal memory usage.
    TF_ASSIGN_OR_RETURN(
        schedule->thunk_launch_order_,
        CreateMemoryMinimizingSequence(
            *entry_computation, [pointer_size](const LogicalBuffer& buffer) {
              return ShapeUtil::ByteSizeOf(buffer.shape(), pointer_size);
            }));
  } else {
    // BFS tends to increase concurrency, but also increases memory usage.
    BFSLaunchOrder(entry_computation, &schedule->thunk_launch_order_);
  }

  schedule->hlo_ordering_ = MakeUnique<GpuHloOrdering>(
      &module, stream_assignment, schedule->thunk_launch_order_);

  return std::move(schedule);
}

}  // namespace gpu
}  // namespace xla

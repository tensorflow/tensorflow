/* Copyright 2017 The OpenXLA Authors.

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

// Call graph for an HLO module.

#ifndef XLA_SERVICE_CALL_GRAPH_H_
#define XLA_SERVICE_CALL_GRAPH_H_

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "tsl/platform/logging.h"

namespace xla {

// The context in which a computation is called by another computation.
enum class CallContext {
  // In an embedded call context, the body of the function cannot allocate
  // buffers.
  kEmbedded,

  // A control flow call context can allocate buffers.
  kControlFlow,

  // A computation is called from both an embedded and control flow context.
  kBoth,

  // During call graph construction kNone is used to indicate that the context
  // has not been determined. This is the top value for the context
  // lattice. After construction, no call sites or call graph nodes should have
  // this value.
  kNone
};

std::string CallContextToString(CallContext context);
std::ostream& operator<<(std::ostream& out, const CallContext& context);

CallContext GetInstructionCallContext(HloOpcode opcode);

// Represents an HLO instruction which calls one or more computations.
class CallSite {
 public:
  CallSite(HloInstruction* instruction,
           absl::Span<HloComputation* const> called_computations,
           CallContext context)
      : instruction_(CHECK_NOTNULL(instruction)),
        called_computations_(called_computations.begin(),
                             called_computations.end()),
        context_(context) {}

  // Returns the instruction associated with this call site.
  HloInstruction* instruction() const { return instruction_; }

  // Returns the computations called at this call site.
  absl::Span<HloComputation* const> called_computations() const {
    return called_computations_;
  }

  // Returns the context in which computations are called at this call site.
  CallContext context() const { return context_; }

  std::string ToString() const;

 private:
  // The calling instruction.
  HloInstruction* instruction_;

  // The computations called by this callsite.
  const absl::InlinedVector<HloComputation*, 2> called_computations_;

  // The context in which the computations are called.
  const CallContext context_;
};

// A node in the call graph representing an HLO computation.
class CallGraphNode {
 public:
  explicit CallGraphNode(HloComputation* computation);

  // Returns the computation represented by this call graph node.
  HloComputation* computation() const { return computation_; }

  // Returns the call sites in this computation. These are the instructions in
  // this computation which call other computations.
  absl::Span<const CallSite> callsites() const { return callsites_; }

  // Returns the callsite associated with the given instruction. If this
  // instruction calls no computations nullptr is returned.
  // Prerequisite: instruction is in the computation associated with this call
  // graph node.
  const CallSite* GetCallSite(const HloInstruction* instruction) const;

  // Returns the computations called by this computation.
  absl::Span<HloComputation* const> callees() const { return callees_; }

  // Returns the call sites in other computations which call this computation.
  absl::Span<const CallSite> caller_callsites() const {
    return caller_callsites_;
  }

  // Returns the computations which call this computation.
  absl::Span<HloComputation* const> callers() const { return callers_; }

  // Returns the context in which this computation is called.
  CallContext context() const { return context_; }

  // Returns the depth of this node in the call graph. The depth is defined as
  // the length of the longest call chain from a computation with no callers
  // (usually the entry computation node) to this node.
  int depth() const { return depth_; }

  absl::string_view ToString() const;

  CallGraphNode(const CallGraphNode&) = delete;
  CallGraphNode& operator=(const CallGraphNode&) = delete;
  CallGraphNode(CallGraphNode&&) = default;
  CallGraphNode& operator=(CallGraphNode&&) noexcept = default;

 private:
  // Only CallGraph can modify CallGraphNode.
  friend class CallGraph;

  // Sets the context in which this computation is called.
  void set_context(CallContext value) { context_ = value; }

  // Sets the depth of this node in the graph.
  void set_depth(int value) { depth_ = value; }

  // Adds a callsite which calls this computation. Updates callers to include
  // the calling computation.
  void AddCallerCallSite(const CallSite& caller_callsite);

  // If instruction calls any computations adds a call site for this instruction
  // to the call graph node. If the instruction calls no computations then no
  // call site is added.
  void AddCallSiteForInstruction(
      HloInstruction* instruction,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {});

  // Computation represented by this call graph node.
  HloComputation* computation_;

  // The computations called by this computation. The vector is used for a
  // stable ordering and the set enables fast membership testing.
  absl::InlinedVector<HloComputation*, 1> callees_;
  absl::flat_hash_set<HloComputation*> callee_set_;

  // The computations which call this computation. The vector is used for a
  // stable ordering and the set enables fast membership testing.
  absl::InlinedVector<HloComputation*, 1> callers_;
  absl::flat_hash_set<HloComputation*> caller_set_;

  // The call sites in this computation
  absl::InlinedVector<CallSite, 1> callsites_;

  // The map from instruction to index in callsites_ for looking up the callsite
  // (if any) associated with a particular instruction in this computation.
  absl::flat_hash_map<const HloInstruction*, int64_t> callsite_instructions_;

  // The call sites in other computations which call this computation.
  absl::InlinedVector<CallSite, 1> caller_callsites_;

  // The context in which this computation is called.
  CallContext context_ = CallContext::kNone;

  // The depth of this node in the call graph.
  int depth_ = 0;
};

// The call graph for an HLO module. The graph includes a node for each
// computation in the module.
class CallGraph {
 public:
  using VisitorFunction = absl::FunctionRef<absl::Status(const CallGraphNode&)>;

  // Builds and returns a call graph for the given HLO module. If a non-empty
  // execution_threads is provided, only computations that are in
  // execution_threads will be part of the returned call graph.
  static std::unique_ptr<CallGraph> Build(
      const HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {});

  // Returns the node associated with the given computation.
  const CallGraphNode& GetNode(const HloComputation* computation) const;
  CallGraphNode& GetNode(const HloComputation* computation);

  // Returns the vector of all nodes in the call graph.
  const std::vector<CallGraphNode>& nodes() const { return nodes_; }

  // Calls the given function on each node in the call graph. Nodes are visited
  // in post order (callees before callers). If visit_unreachable_nodes is true
  // then all nodes in the call graph are visited. Otherwise only those nodes
  // reachable from the entry computation are visited.
  absl::Status VisitNodes(VisitorFunction visitor_func,
                          bool visit_unreachable_nodes = true) const;

  // Returns true if 'a' dominates 'b' in the call graph. Computation 'a'
  // dominates computation 'b' iff all callgraph paths in the caller-to-callee
  // direction from a root computation to 'b' pass through computation
  // 'a'. Trivially, a computation dominates itself.
  bool Dominates(const HloComputation* a, const HloComputation* b) const;

  // Returns true if 'a' can reach 'b' in the call graph. 'a' can reach 'b' if
  // 'a' is 'b' or 'a' can reach one of the callers of 'b'.
  bool CanReach(const HloComputation* a, const HloComputation* b) const;

  // Returns whether 'instruction' is contained in 'computation' either directly
  // ('instruction->parent' is 'computation') or indirectly ('computation'
  // dominates 'instruction->parent' in the call graph).
  bool InstructionIsNestedIn(const HloInstruction* instruction,
                             const HloComputation* computation) const {
    return Dominates(computation, instruction->parent());
  }

  // Returns the nearest call graph ancestors of instructions 'a' and 'b' for
  // which the ancestors are in the same computation. An instruction is an call
  // graph ancestor of 'a' if the instruction calls the computation containing
  // 'a' either directly or transitively. Degeneratively an instruction is an
  // ancestor of itself. nullptr is returned if there is no common ancestor or
  // if the caller chain of 'a' or 'b' diverges (has multiple callers) before
  // the nearest common ancestor.
  //
  // Example:
  //
  // Entry computation:
  //   %x = Call(A, {Constant(42.0)})
  //   %y = Call(B, {%x})
  //
  // Computation A:
  //   %a = Negate(Param())
  //
  // Computation B:
  //   %b = Exp(Param());
  //
  // If called with %a and %b, this function would return (%x, %y). %x is an
  // ancestor of %a, and %y is an ancestor of %b, and %x and %y are in the same
  // computation.
  std::pair<HloInstruction*, HloInstruction*> NearestAncestorsInSameComputation(
      HloInstruction* a, HloInstruction* b) const;

  // Given a set of instructions within a computation, returns nearest common
  // ancestors as Hlo instructions (There could be multiple nearest common
  // ancestors in a DAG). If the given instructions are not in the same
  // computation, this function would report FAILURE.
  //
  // Unlike the `NearestAncestorsInSameComputation` defined above, it:
  //
  // (1) Only compute the nearest common ancestors within a computation, instead
  // of across computations (that's the function
  // `ComputationsNearestCommonAncestors` that defined below).
  //
  // (2) Takes in **a set of** Hlo instructions, instead of two Hlo
  // instructions, and find their nearest common ancestors.
  //
  // Example:
  //
  // Computation A:
  //   %p0   = Param(0)
  //   %p1   = Param(1)
  //   %p2   = Param(2)
  //   %add0 = Add(%p0, %p1)
  //   %mul0 = Mul(%p1, %p2)
  //   %sub0 = Sub(%add0, %mul0)
  //
  // If called with {%p0, %p1}, this function would return {%add0}.
  //
  // Please check the detailed example in
  // `CallGraphTest.NearestCommonAncestorInstructions`.
  absl::flat_hash_set<const HloInstruction*> NearestCommonAncestorInstructions(
      std::vector<const HloInstruction*> instructions);

  // Given a set of computations within a module, returns nearest common
  // ancestors as Hlo computations (There could be multiple nearest common
  // ancestors in a DAG).
  //
  // Entry_computation:
  //   %x = Call(A, {Constant(42.0)})
  //   %y = Call(B, {%x})
  //
  // Computation_A:
  //   %a = Negate(Param())
  //
  // Computation_B:
  //   %b = Exp(Param());
  //
  // If called with {Computation_A, Computation_B}, this function would return
  // {Entry_computation}.
  //
  // Please check the detailed example in
  // `CallGraphTest.NearestCommonAncestorComputations`.
  absl::flat_hash_set<const HloComputation*> NearestCommonAncestorComputations(
      std::vector<const HloComputation*> computations);

  // A template helper function that computes the nearest common ancestors among
  // instructions/computations. `T` can be either `HloInstruction` or
  // `HloComputation`. Computing nearest common ancestors are basically the same
  // for HloInstruction and HloComputation. The only difference is that they
  // require different ways to access the ancestors of one node. Specifically,
  // the ancestors are users_instruction for instructions, and are
  // caller_computations for computations.
  //
  // The overall idea is to conduct BFS from the `starting_nodes`, and keep
  // track of the visited ancestors of each node. For each BFS step, we check if
  // there is a common node in all the visited ancestors, and if yes, that
  // common node is the nearest ancestor we are looking for. Note that, since we
  // are traversing DAG, there could be multiple nearest common ancestors. And
  // there must be at least one common ancestor (i.e., entry computations among
  // computations or root instruction among instructions).
  template <typename T>
  absl::flat_hash_set<const T*> NearestCommonAncestorsHelper(
      std::vector<const T*>& starting_nodes);

  // Returns whether the call graph is flattened. A call graph is flattened if
  // every computation called in a sequential context (eg, kWhile or kCall) has
  // zero or one callsite, and no computation is called from both a parallel and
  // sequential context. The call graph of a module can be flattened with
  // FlattenCallGraph.
  bool IsFlattened() const;

  // Returns a vector of instructions calling the passed computation.
  // (Often a vector of size 1.)
  std::vector<HloInstruction*> GetComputationCallers(
      const HloComputation* c) const;

  std::string ToString() const;

 private:
  explicit CallGraph(
      const HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {});

  // Not copyable.
  CallGraph(const CallGraph&) = delete;
  CallGraph& operator=(const CallGraph&) = delete;

  // Sets the call contexts for every node in the graph.
  void SetCallContexts();

  // Sets the call node depths for every node in the graph.
  void SetNodeDepths();

  // Helper method for VisitNodes(). Traverses the call graph from 'node' in DFS
  // post order (callee before caller) calling visitor_func on each node. Adds
  // nodes to 'visited' as each node is visited. Skips nodes already in
  // 'visited'.
  absl::Status VisitNodesInternal(
      VisitorFunction visitor_func, const CallGraphNode& node,
      absl::flat_hash_set<const CallGraphNode*>* visited) const;

  // Recursive helper for computing whether 'a' dominates 'b' in the call
  // graph. 'b_ancestor' is the currently visited node (which starts at 'b'),
  // and 'visited' is the set of computations which have been visited.
  bool DominatesHelper(
      const HloComputation* a, const HloComputation* b,
      absl::flat_hash_set<const HloComputation*>* visited) const;

  // The HLO module represented by this call graph.
  const HloModule* module_ = nullptr;

  // Vector of all nodes in the call graph.
  std::vector<CallGraphNode> nodes_;

  // Map from HLO computation to the index of the corresponding call graph node
  // in nodes_.
  absl::flat_hash_map<const HloComputation*, int64_t> node_indices_;

  // The execution threads that the call graph is built for.
  absl::flat_hash_set<absl::string_view> execution_threads_;
};

}  // namespace xla

#endif  // XLA_SERVICE_CALL_GRAPH_H_

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

// Call graph for an HLO module.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CALL_GRAPH_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CALL_GRAPH_H_

#include <ostream>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"

namespace xla {

// The context in which a computation is called by another computation.
enum class CallContext {
  // In a parallel contex the computation is applied to each element of the
  // array argument(s). kMap and kReduce instructions call computations in
  // parallel context.
  kParallel,

  // In a sequential context the computation is applied to the entire argument
  // shape(s). kCall and kWhile (body and condition) call computations in
  // sequential context.
  kSequential,

  // A computation is called from both a parallel and sequential context.
  kBoth,

  // During call graph construction kNone is used to indicate that the context
  // has not been determined. This is the top value for the context
  // lattice. After construction, no call sites or call graph nodes should have
  // this value.
  kNone
};

string CallContextToString(CallContext context);
std::ostream& operator<<(std::ostream& out, const CallContext& context);

CallContext GetInstructionCallContext(HloOpcode opcode);

// Represents an HLO instruction which calls one or more computations.
class CallSite {
 public:
  CallSite(HloInstruction* instruction,
           const std::vector<HloComputation*>& called_computations,
           CallContext context)
      : instruction_(CHECK_NOTNULL(instruction)),
        called_computations_(called_computations),
        context_(context) {}

  // Returns the instruction associated with this call site.
  HloInstruction* instruction() const { return instruction_; }

  // Returns the computations called at this call site.
  const std::vector<HloComputation*>& called_computations() const {
    return called_computations_;
  }

  // Returns the context in which computations are called at this call site.
  CallContext context() const { return context_; }

  string ToString() const;

 private:
  // The calling instruction.
  HloInstruction* instruction_;

  // The computations called by this callsite.
  const std::vector<HloComputation*> called_computations_;

  // The context in which the computations are called.
  const CallContext context_;
};

// A node in the call graph representing an HLO computation.
class CallGraphNode {
 public:
  CallGraphNode(HloComputation* computation);

  // Returns the computation represented by this call graph node.
  HloComputation* computation() const { return computation_; }

  // Returns the call sites in this computation. These are the instructions in
  // this computation which call other computations.
  const std::vector<CallSite>& callsites() const { return callsites_; }

  // Returns the callsite associated with the given instruction. If this
  // instruction calls no computations nullptr is returned.
  // Prerequisite: instruction is in the computation associated with this call
  // graph node.
  const CallSite* GetCallSite(const HloInstruction* instruction) const;

  // Returns the computations called by this computation.
  const std::vector<HloComputation*>& callees() const { return callees_; }

  // Returns the call sites in other computations which call this computation.
  const std::vector<CallSite>& caller_callsites() const {
    return caller_callsites_;
  }

  // Returns the computations which call this computation.
  const std::vector<HloComputation*>& callers() const { return callers_; }

  // Returns the context in which this computation is called.
  CallContext context() const { return context_; }

  string ToString() const;

 private:
  // Only CallGraph can modify CallGraphNode.
  friend class CallGraph;

  // Sets the context in which this computation is called.
  void set_context(CallContext value) { context_ = value; }

  // Adds a callsite which calls this computation. Updates callers to include
  // the calling computation.
  void AddCallerCallSite(const CallSite& caller_callsite);

  // If instruction calls any computations adds a call site for this instruction
  // to the call graph node. If the instruction calls no computations then no
  // call site is added.
  void AddCallSiteForInstruction(HloInstruction* instruction);

  // Computation represented by this call graph node.
  HloComputation* computation_;

  // The computations called by this computation. The vector is used for a
  // stable ordering and the set enables fast membership testing.
  std::vector<HloComputation*> callees_;
  tensorflow::gtl::FlatSet<HloComputation*> callee_set_;

  // The computations which call this computation. The vector is used for a
  // stable ordering and the set enables fast membership testing.
  std::vector<HloComputation*> callers_;
  tensorflow::gtl::FlatSet<HloComputation*> caller_set_;

  // The call sites in this computation
  std::vector<CallSite> callsites_;

  // The map from instruction to index in callsites_ for looking up the callsite
  // (if any) associated with a particular instruction in this computation.
  tensorflow::gtl::FlatMap<const HloInstruction*, int64> callsite_instructions_;

  // The call sites in other computations which call this computation.
  std::vector<CallSite> caller_callsites_;

  // The context in which this computation is called.
  CallContext context_ = CallContext::kNone;
};

// The call graph for an HLO module. The graph includes a node for each
// computation in the module.
class CallGraph {
 public:
  using VisitorFunction = std::function<Status(const CallGraphNode&)>;

  // Builds and returns a call graph for the given HLO module.
  static std::unique_ptr<CallGraph> Build(const HloModule* module);

  // Returns the node associated with the given computation.
  const CallGraphNode& GetNode(const HloComputation* computation) const;
  CallGraphNode& GetNode(const HloComputation* computation);

  // Returns the vector of all nodes in the call graph.
  const std::vector<CallGraphNode>& nodes() const { return nodes_; }

  // Calls the given function on each node in the call graph. Nodes are visited
  // in post order (callees before callers). If visit_unreachable_nodes is true
  // then all nodes in the call graph are visited. Otherwise only those nodes
  // reachable from the entry computation are visited.
  Status VisitNodes(const VisitorFunction& visitor_func,
                    bool visit_unreachable_nodes = true) const;

  // Returns true if 'a' dominates 'b' in the call graph. Computation 'a'
  // dominates computation 'b' iff all callgraph paths in the caller-to-callee
  // direction from a root computation to 'b' pass through computation
  // 'a'. Trivially, a computation dominates itself.
  bool Dominates(const HloComputation* a, const HloComputation* b) const;

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

  // Returns whether the call graph is flattened. A call graph is flattened if
  // every computation called in a sequential context (eg, kWhile or kCall) has
  // zero or one callsite, and no computation is called from both a parallel and
  // sequential context. The call graph of a module can be flattened with
  // FlattenCallGraph.
  bool IsFlattened() const;

  string ToString() const;

 private:
  CallGraph(const HloModule* module);

  // Sets the call contexts for every node in the graph.
  void SetCallContexts();

  // Helper method for VisitNodes(). Traverses the call graph from 'node' in DFS
  // post order (callee before caller) calling visitor_func on each node. Adds
  // nodes to 'visited' as each node is visited. Skips nodes already in
  // 'visited'.
  Status VisitNodesInternal(
      const VisitorFunction& visitor_func, const CallGraphNode& node,
      tensorflow::gtl::FlatSet<const CallGraphNode*>* visited) const;

  // Recursive helper for computing whether 'a' dominates 'b' in the call
  // graph. 'b_ancestor' is the currently visited node (which starts at 'b'),
  // and 'visited' is the set of computations which have been visited.
  bool DominatesHelper(
      const HloComputation* a, const HloComputation* b,
      tensorflow::gtl::FlatSet<const HloComputation*>* visited) const;

  // The HLO module represented by this call graph.
  const HloModule* module_ = nullptr;

  // Vector of all nodes in the call graph.
  std::vector<CallGraphNode> nodes_;

  // Map from HLO computation to the index of the corresponding call graph node
  // in nodes_.
  tensorflow::gtl::FlatMap<const HloComputation*, int64> node_indices_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CALL_GRAPH_H_

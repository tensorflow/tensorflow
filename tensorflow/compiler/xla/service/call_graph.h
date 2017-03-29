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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CALL_GRAPH_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CALL_GRAPH_H_

#include <ostream>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/statusor.h"
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
  Status AddCallSiteForInstruction(HloInstruction* instruction);

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
  static StatusOr<std::unique_ptr<CallGraph>> Build(const HloModule* module);

  // Returns the node associated with the given computation.
  StatusOr<const CallGraphNode*> GetNode(
      const HloComputation* computation) const;
  StatusOr<CallGraphNode*> GetNode(const HloComputation* computation);

  // Returns the vector of all nodes in the call graph.
  const std::vector<CallGraphNode>& nodes() const { return nodes_; }

  // Calls the given function on each node in the call graph. Nodes are visited
  // in post order (callees before callers). If visit_unreachable_nodes is true
  // then all nodes in the call graph are visited. Otherwise only those nodes
  // reachable from the entry computation are visited.
  Status VisitNodes(const VisitorFunction& visitor_func,
                    bool visit_unreachable_nodes = true) const;

  string ToString() const;

 private:
  CallGraph(const HloModule* module);

  // Sets the call contexts for every node in the graph.
  Status SetCallContexts();

  // Helper method for VisitNodes(). Traverses the call graph from 'node' in DFS
  // post order (callee before caller) calling visitor_func on each node. Adds
  // nodes to 'visited' as each node is visited. Skips nodes already in
  // 'visited'.
  Status VisitNodesInternal(
      const VisitorFunction& visitor_func, const CallGraphNode* node,
      tensorflow::gtl::FlatSet<const CallGraphNode*>* visited) const;

  // The HLO module represented by this call graph.
  const HloModule* module_ = nullptr;

  // Vector of all nodes in the call graph.
  std::vector<CallGraphNode> nodes_;

  // Map from HLO computation to the index of the corresponding call graph node
  // in nodes_.
  tensorflow::gtl::FlatMap<const HloComputation*, int64> node_indices_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CALL_GRAPH_H_

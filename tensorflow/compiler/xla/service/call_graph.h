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

// Represents an instruction calling a particular computation in an HLO
// module. Some instructions such as kWhile can call more than one computation
// and may be represented with more than one CallSite, one for each computation
// called.
struct CallSite {
  // The calling instruction.
  HloInstruction* instruction;

  // The computation the instruction is calling.
  HloComputation* called_computation;

  // The context in which the computation is called.
  CallContext context;

  string ToString() const;
};

// A node in the call graph representing an HLO computation.
class CallGraphNode {
 public:
  CallGraphNode(HloComputation* computation);

  // Return the computation represented by this call graph node.
  HloComputation* computation() const { return computation_; }

  // Return the call sites in this computation. These are the instructions in
  // this computation which call other computations.
  const std::vector<CallSite>& callsites() const { return callsites_; }

  // Return the computations called by this computation.
  const std::vector<HloComputation*>& callees() const { return callees_; }

  // Return the call sites in other computations which call this computation.
  const std::vector<CallSite>& caller_callsites() const {
    return caller_callsites_;
  }

  // Return the computations which call this computation.
  const std::vector<HloComputation*>& callers() const { return callers_; }

  // Return or set the context in which this computation is called.
  CallContext context() const { return context_; }
  void set_context(CallContext value) { context_ = value; }

  // Add a callsite which calls this computation. Updates callers to include the
  // calling computation.
  void AddCallerCallSite(const CallSite& caller_callsite);

  // Add a call site to this computation. Updates callees to include the called
  // computation.
  void AddCallSite(const CallSite& callsite);

  // Add all the call sites (if any) for this instruction. Instruction must be
  // an instruction in this node's computation.
  void AddCallSitesInInstruction(HloInstruction* instruction);

  string ToString() const;

 private:
  // Computation represented by this call graph node.
  HloComputation* computation_;

  // The computations called by this computation. The vector is used for a
  // stable ordering and the set enables fast membership testing.
  std::vector<HloComputation*> callees_;
  std::unordered_set<HloComputation*> callee_set_;

  // The computations which call this computation. The vector is used for a
  // stable ordering and the set enables fast membership testing.
  std::vector<HloComputation*> callers_;
  std::unordered_set<HloComputation*> caller_set_;

  // The call sites in this computation
  std::vector<CallSite> callsites_;

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

  // Build and return a call graph for the given HLO module.
  static StatusOr<std::unique_ptr<CallGraph>> Build(const HloModule* module);

  // Public default constructor required for StatusOr<CallGraph>.
  CallGraph() = default;

  // Return the node associated with the given computation.
  StatusOr<const CallGraphNode*> GetNode(
      const HloComputation* computation) const;
  StatusOr<CallGraphNode*> GetNode(const HloComputation* computation);

  // Return the vector of all nodes in the call graph.
  const std::vector<CallGraphNode>& nodes() const { return nodes_; }

  // Call the given function on each node in the call graph. Nodes are visited
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
      std::unordered_set<const CallGraphNode*>* visited) const;

  // The HLO module represented by this call graph.
  const HloModule* module_ = nullptr;

  // Vector of all nodes in the call graph.
  std::vector<CallGraphNode> nodes_;

  // Map from HLO computation to the index of the corresponding call graph node
  // in nodes_.
  std::unordered_map<const HloComputation*, int64> node_indices_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CALL_GRAPH_H_

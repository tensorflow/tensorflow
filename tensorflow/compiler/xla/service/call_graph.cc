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

#include "tensorflow/compiler/xla/service/call_graph.h"

#include <queue>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

using ::tensorflow::strings::Appendf;
using ::tensorflow::strings::StrCat;

string CallContextToString(CallContext context) {
  switch (context) {
    case CallContext::kNone:
      return "kNone";
    case CallContext::kSequential:
      return "kSequential";
    case CallContext::kParallel:
      return "kParallel";
    case CallContext::kBoth:
      return "kBoth";
  }
}

std::ostream& operator<<(std::ostream& out, const CallContext& context) {
  out << CallContextToString(context);
  return out;
}

CallContext GetInstructionCallContext(const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kWhile:
      return CallContext::kSequential;
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kFusion:
      return CallContext::kParallel;
    default:
      return CallContext::kNone;
  }
}

string CallSite::ToString() const {
  return StrCat(instruction()->name(), " calls in context ",
                CallContextToString(context()), ": ",
                tensorflow::str_util::Join(
                    called_computations(), ", ",
                    [](string* out, const HloComputation* computation) {
                      out->append(computation->name());
                    }));
}

CallGraphNode::CallGraphNode(HloComputation* computation)
    : computation_(computation) {}

const CallSite* CallGraphNode::GetCallSite(
    const HloInstruction* instruction) const {
  auto it = callsite_instructions_.find(instruction);
  if (it == callsite_instructions_.end()) {
    return nullptr;
  }
  return &callsites_[it->second];
}

void CallGraphNode::AddCallerCallSite(const CallSite& caller_callsite) {
  caller_callsites_.push_back(caller_callsite);
  HloComputation* caller = caller_callsite.instruction()->parent();
  if (!ContainsKey(caller_set_, caller)) {
    callers_.push_back(caller);
    caller_set_.insert(caller);
  }
}

void CallGraphNode::AddCallSiteForInstruction(HloInstruction* instruction) {
  CHECK_EQ(instruction->parent(), computation());
  const CallContext context = GetInstructionCallContext(instruction);
  if (!instruction->called_computations().empty()) {
    CHECK(context == CallContext::kSequential ||
          context == CallContext::kParallel);
    callsite_instructions_.insert({instruction, callsites_.size()});
    callsites_.push_back(
        CallSite(instruction, instruction->called_computations(), context));
    // Update callee computations to include any new computations called by this
    // instruction.
    for (auto* callee : callsites_.back().called_computations()) {
      if (!ContainsKey(callee_set_, callee)) {
        callees_.push_back(callee);
        callee_set_.insert(callee);
      }
    }
  }
}

CallGraph::CallGraph(const HloModule* module) : module_(module) {}

const CallGraphNode& CallGraph::GetNode(
    const HloComputation* computation) const {
  auto it = node_indices_.find(computation);
  CHECK(it != node_indices_.end());
  return nodes_[it->second];
}

CallGraphNode& CallGraph::GetNode(const HloComputation* computation) {
  auto it = node_indices_.find(computation);
  CHECK(it != node_indices_.end());
  return nodes_[it->second];
}

namespace {

// Returns the call context of a computation which is called from contexts 'a'
// and 'b'.
CallContext UnionContexts(CallContext a, CallContext b) {
  if (a == CallContext::kNone) {
    return b;
  } else if (b == CallContext::kNone) {
    return a;
  } else if (a == b) {
    return a;
  } else {
    // Contexts are different and neither is kNone, ie one is kSequential and
    // the other is kParallel.
    return CallContext::kBoth;
  }
}

}  // namespace

void CallGraph::SetCallContexts() {
  std::queue<CallGraphNode*> worklist;

  // Initialize worklist with all roots of the call graph (computations without
  // callers).
  for (const std::unique_ptr<HloComputation>& computation :
       module_->computations()) {
    CallGraphNode& node = GetNode(computation.get());
    if (node.callers().empty()) {
      node.set_context(CallContext::kSequential);
      worklist.push(&node);
    }
  }

  while (!worklist.empty()) {
    CallGraphNode* node = worklist.front();
    worklist.pop();

    for (const CallSite& callsite : node->callsites()) {
      for (const HloComputation* callee : callsite.called_computations()) {
        CallGraphNode& callee_node = GetNode(callee);

        // Update context of callee computation based on the callsite and its
        // current context.
        CallContext context_to_add;
        if (callsite.context() == CallContext::kParallel) {
          context_to_add = CallContext::kParallel;
        } else {
          CHECK_EQ(callsite.context(), CallContext::kSequential);
          context_to_add = node->context();
        }
        CallContext new_context =
            UnionContexts(context_to_add, callee_node.context());

        if (new_context != callee_node.context()) {
          // Context of computation has been changed so add node to worklist.
          callee_node.set_context(new_context);
          worklist.push(&callee_node);
        }
      }
    }
  }

  // No node should have a kNone calling context.
  for (const std::unique_ptr<HloComputation>& computation :
       module_->computations()) {
    CHECK_NE(GetNode(computation.get()).context(), CallContext::kNone);
  }
}

/* static */
std::unique_ptr<CallGraph> CallGraph::Build(const HloModule* module) {
  // Constructor for CallGraph is private so MakeUnique can't be used.
  auto call_graph = WrapUnique<CallGraph>(new CallGraph(module));

  VLOG(2) << "Building call graph for:";
  XLA_VLOG_LINES(2, module->ToString());

  // Construct nodes of the call graph and populate the callsites.
  for (const std::unique_ptr<HloComputation>& computation :
       module->computations()) {
    auto it_added = call_graph->node_indices_.insert(
        {computation.get(), call_graph->nodes_.size()});
    // All computations should be unique, so the computation should not already
    // exist in the map.
    CHECK(it_added.second);
    call_graph->nodes_.emplace_back(computation.get());

    // Add all callsites in this computation.
    for (const std::unique_ptr<HloInstruction>& instruction :
         computation->instructions()) {
      call_graph->nodes_.back().AddCallSiteForInstruction(instruction.get());
    }
  }

  // Add caller callsites to each node.
  for (const std::unique_ptr<HloComputation>& computation :
       module->computations()) {
    for (const CallSite& callsite :
         call_graph->GetNode(computation.get()).callsites()) {
      for (auto* callee : callsite.called_computations()) {
        // Add caller callsites.
        call_graph->GetNode(callee).AddCallerCallSite(callsite);
      }
    }
  }

  call_graph->SetCallContexts();
  XLA_VLOG_LINES(1, call_graph->ToString());

  return call_graph;
}

Status CallGraph::VisitNodesInternal(
    const VisitorFunction& visitor_func, const CallGraphNode& node,
    tensorflow::gtl::FlatSet<const CallGraphNode*>* visited) const {
  auto pair = visited->insert(&node);
  if (!pair.second) {
    // Node was not inserted. Node has already been visited.
    return Status::OK();
  }

  for (const HloComputation* computation : node.callees()) {
    TF_RETURN_IF_ERROR(
        VisitNodesInternal(visitor_func, GetNode(computation), visited));
  }

  return visitor_func(node);
}

Status CallGraph::VisitNodes(const VisitorFunction& visitor_func,
                             bool visit_unreachable_nodes) const {
  tensorflow::gtl::FlatSet<const CallGraphNode*> visited;
  if (visit_unreachable_nodes) {
    // Traverse from all roots in the call graph.
    for (const CallGraphNode& node : nodes()) {
      if (node.callers().empty()) {
        TF_RETURN_IF_ERROR(VisitNodesInternal(visitor_func, node, &visited));
      }
    }
  } else {
    // Traverse only from the entry computation.
    TF_RETURN_IF_ERROR(VisitNodesInternal(
        visitor_func, GetNode(module_->entry_computation()), &visited));
  }

  return Status::OK();
}

string CallGraph::ToString() const {
  string out;
  Appendf(&out, "Call graph for module %s:\n", module_->name().c_str());
  for (const CallGraphNode& node : nodes()) {
    Appendf(&out, "Computation %s:\n", node.computation()->name().c_str());
    Appendf(&out, "  calls:\n");
    for (const HloComputation* callee : node.callees()) {
      Appendf(&out, "    %s\n", callee->name().c_str());
    }
    Appendf(&out, "  called by:\n");
    for (const HloComputation* caller : node.callers()) {
      Appendf(&out, "    %s\n", caller->name().c_str());
    }
    Appendf(&out, "  callsites:\n");
    for (const CallSite& callsite : node.callsites()) {
      Appendf(&out, "    %s\n", callsite.ToString().c_str());
    }
  }
  return out;
}

}  // namespace xla

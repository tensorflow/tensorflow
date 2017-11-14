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

#include "tensorflow/compiler/jit/mark_for_compilation_pass.h"

#include <atomic>
#include <deque>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/graphcycles/graphcycles.h"
#include "tensorflow/compiler/jit/legacy_flags/mark_for_compilation_pass_flags.h"
#include "tensorflow/compiler/jit/union_find.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/memory_types.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

const char* const kXlaClusterAttr = "_XlaCluster";

namespace {

bool HasXLAKernel(const Node& node, const DeviceType& jit_device_type) {
  // There is a SymbolicGradient kernel on the XLA_JIT device, but the gradient
  // is really a kind of function call and will be handled by
  // IsCompilableCall().
  if (node.type_string() == "SymbolicGradient") return false;
  return FindKernelDef(jit_device_type, node.def(), nullptr, nullptr).ok();
}

// Make sure we don't recurse infinitely on recursive functions.
const int kMaxRecursionDepth = 10;

bool IsCompilableCall(const NodeDef& call_def,
                      const DeviceType& jit_device_type, int depth,
                      FunctionLibraryRuntime* lib_runtime);

// Tests whether 'while_node' is a completely compilable loop.
// Every operator in the condition and body functions must be compilable for a
// while loop to be compilable.
bool IsCompilableWhile(const Node& while_node,
                       const DeviceType& jit_device_type, int depth,
                       FunctionLibraryRuntime* lib_runtime) {
  VLOG(2) << "Loop marking: " << while_node.type_string();

  const NameAttrList* name_attr;
  NodeDef call;
  Status status;
  status = GetNodeAttr(while_node.attrs(), "cond", &name_attr);
  if (!status.ok()) {
    VLOG(2) << "Missing 'cond' attribute on While node.";
    return false;
  }
  const string cond_func = name_attr->name();
  call.set_name("while_cond");
  call.set_op(cond_func);
  *call.mutable_attr() = name_attr->attr();
  if (!IsCompilableCall(call, jit_device_type, depth + 1, lib_runtime)) {
    VLOG(2) << "Can't compile loop condition: " << cond_func;
    return false;
  }
  status = GetNodeAttr(while_node.attrs(), "body", &name_attr);
  if (!status.ok()) {
    VLOG(2) << "Missing 'body' attribute on While node.";
    return false;
  }
  const string body_func = name_attr->name();
  call.set_name("while_body");
  call.set_op(body_func);
  *call.mutable_attr() = name_attr->attr();
  if (!IsCompilableCall(call, jit_device_type, depth + 1, lib_runtime)) {
    VLOG(2) << "Can't compile loop body: " << body_func;
    return false;
  }
  VLOG(2) << "Loop is compilable.";
  return true;
}

// Tests whether 'call_def' is a call to a completely compilable function.
// Every operator in the function must be compilable for a function to be
// compilable.
bool IsCompilableCall(const NodeDef& call_def,
                      const DeviceType& jit_device_type, int depth,
                      FunctionLibraryRuntime* lib_runtime) {
  VLOG(2) << "Function marking: " << call_def.op();

  if (depth > kMaxRecursionDepth) {
    VLOG(2) << "Function depth limit exceeded";
    return false;
  }

  FunctionLibraryRuntime::Handle handle;
  Status status =
      lib_runtime->Instantiate(call_def.op(), AttrSlice(call_def), &handle);
  if (!status.ok()) {
    VLOG(2) << "Could not instantiate " << call_def.op() << ": " << status;
    return false;
  }
  const FunctionBody* fbody = lib_runtime->GetFunctionBody(handle);
  CHECK(fbody);
  const FunctionDef& fdef = fbody->fdef;
  bool noinline = false;
  if (GetNodeAttr(AttrSlice(&fdef.attr()), "_noinline", &noinline).ok() &&
      noinline) {
    // The underlying mechanism that calls non-inlined functions uses
    // LocalExecutor, which interacts poorly with the LocalExecutor used by
    // tf2xla to translate the TF graph into XLA.  So we avoid this for now.
    //
    // TODO(b/36139787): Create a mechanism to set inlining hints.
    VLOG(2) << "Can't compile noinline function: " << fdef.DebugString();
    return false;
  }

  for (Node* node : fbody->graph->op_nodes()) {
    if (node->type_string() == "_Arg" || node->type_string() == "_Retval")
      continue;
    if (node->type_string() == "While") {
      // Handle functional While loop (not in open source build).
      return IsCompilableWhile(*node, jit_device_type, depth + 1, lib_runtime);
    }
    if (!HasXLAKernel(*node, jit_device_type) &&
        !IsCompilableCall(node->def(), jit_device_type, depth + 1,
                          lib_runtime)) {
      VLOG(2) << "Function marking failed: unsupported op " << node->name()
              << ": " << node->def().ShortDebugString();
      return false;
    }
  }
  VLOG(2) << "Function is compilable: " << call_def.op();
  return true;
}

// Returns the DeviceType corresponding to 'device'.
Status DeviceTypeOfDevice(const string& device, DeviceType* device_type) {
  DeviceNameUtils::ParsedName parsed;
  if (!DeviceNameUtils::ParseFullName(device, &parsed)) {
    return errors::Internal("Malformed assigned device '", device, "'");
  }
  *device_type = DeviceType(parsed.type);
  return Status::OK();
}

// Tests whether `node` has a DT_RESOURCE typed input or output.
bool HasResourceInputOrOutput(const Node& node) {
  return std::find(node.input_types().begin(), node.input_types().end(),
                   DT_RESOURCE) != node.input_types().end() ||
         std::find(node.output_types().begin(), node.output_types().end(),
                   DT_RESOURCE) != node.output_types().end();
}

Status FindCompilationCandidates(
    const Graph& graph, FunctionLibraryDefinition* flib_def, Env* env,
    const std::function<bool(const Node*, const DeviceType&)>& is_compilable_fn,
    std::unordered_set<Node*>* candidates) {
  OptimizerOptions opts;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(
      new ProcessFunctionLibraryRuntime(nullptr, env, TF_GRAPH_DEF_VERSION,
                                        flib_def, opts));
  FunctionLibraryRuntime* lib_runtime =
      pflr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);

  for (Node* node : graph.op_nodes()) {
    DeviceType device_type("");
    TF_RETURN_IF_ERROR(
        DeviceTypeOfDevice(node->assigned_device_name(), &device_type));

    if (is_compilable_fn && !is_compilable_fn(node, device_type)) continue;

    const XlaOpRegistry::DeviceRegistration* registration;
    CHECK(
        XlaOpRegistry::GetCompilationDevice(device_type.type(), &registration));
    DeviceType jit_device_type(registration->compilation_device_name);
    if (!HasXLAKernel(*node, jit_device_type) &&
        !IsCompilableCall(node->def(), jit_device_type, 0, lib_runtime)) {
      VLOG(2) << "Compilation rejected node: unsupported op " << node->name()
              << ": " << node->type_string();
      continue;
    }
    if (!registration->compile_resource_ops &&
        HasResourceInputOrOutput(*node)) {
      VLOG(2) << "Compilation rejected node: resource input/output "
              << node->name() << ": " << node->type_string();
      continue;
    }
    if (node->type_string() == "While" &&
        !IsCompilableWhile(*node, jit_device_type, 0, lib_runtime)) {
      continue;
    }
    candidates->insert(node);
  }
  return Status::OK();
}

struct Cluster {
  // Identifies the node that represents this cluster in the cycle detection
  // graph.
  int representative = -1;
};

// Returns a string describing how an edge from src to dst would
// create a cycle.
string DescribeCycle(const GraphCycles& cycles, const Graph& graph, int src,
                     int dst) {
  int32 max_path_size = graph.num_node_ids() + 1;
  std::vector<int32> path(max_path_size);
  int32 path_size = cycles.FindPath(dst, src, max_path_size, path.data());
  if (path_size == 0) {
    return "";
  }

  auto node_name = [&cycles, &graph](int node_id) {
    auto* node = graph.FindNodeId(node_id);
    if (node == nullptr) {
      return string("(null)");
    }
    return node->name();
  };

  string description;
  strings::StrAppend(&description, "Edge from ", node_name(src), " to ",
                     node_name(dst), " would create a cycle.\n");
  path.resize(path_size);
  for (int32 node_id : path) {
    string ascii_art;
    if (node_id == dst) {
      ascii_art = "+-> ";
    } else if (node_id != src) {
      ascii_art = "|   ";
    } else {
      ascii_art = "+-- ";
    }
    strings::StrAppend(&description, ascii_art, node_name(node_id), "\n");
  }
  return description;
}

}  // anonymous namespace

bool IsCompilable(FunctionLibraryRuntime* flr, const NodeDef& ndef) {
  Device* device = flr->device();
  const XlaOpRegistry::DeviceRegistration* registration;
  CHECK(XlaOpRegistry::GetCompilationDevice(device->device_type(),
                                            &registration));
  DeviceType jit_device_type(registration->compilation_device_name);
  return IsCompilableCall(ndef, jit_device_type, 0, flr);
}

Status MarkForCompilationPass::Run(
    const GraphOptimizationPassOptions& options) {
  // TODO(phawkins): precompute the "GetCompilationDevice" properties of each
  // device ahead of time.
  OptimizerOptions::GlobalJitLevel global_jit_level =
      options.session_options->config.graph_options()
          .optimizer_options()
          .global_jit_level();
  if (global_jit_level == OptimizerOptions::DEFAULT) {
    // To set compilation to be on by default, change the following line.
    global_jit_level = OptimizerOptions::OFF;
  }
  legacy_flags::MarkForCompilationPassFlags* flags =
      legacy_flags::GetMarkForCompilationPassFlags();
  if (flags->tf_xla_auto_jit == -1 ||
      (1 <= flags->tf_xla_auto_jit && flags->tf_xla_auto_jit <= 2)) {
    // If the flag tf_xla_auto_jit is a valid, non-zero setting, it overrides
    // the setting in ConfigProto.
    global_jit_level =
        static_cast<OptimizerOptions::GlobalJitLevel>(flags->tf_xla_auto_jit);
  }
  const FunctionLibraryDefinition* fld = options.flib_def;
  auto is_compilable = [global_jit_level, fld](const Node* node,
                                               const DeviceType& device_type) {
    const XlaOpRegistry::DeviceRegistration* registration;
    if (!XlaOpRegistry::GetCompilationDevice(device_type.type(),
                                             &registration)) {
      return false;
    }

    // Don't compile control trigger nodes. We won't preserve their deadness
    // semantics correctly, so it's safest not to compile them.
    if (node->IsControlTrigger()) return false;

    // If this device requires a JIT, we must say yes.
    if (registration->requires_compilation) return true;

    // If there is a _XlaCompile annotation, use its value.
    bool compile = false;
    Status status = GetNodeAttr(node->attrs(), kXlaCompileAttr, &compile);
    if (status.ok()) return compile;

    status = fld->GetAttr(*node, kXlaCompileAttr, &compile);
    if (status.ok()) return compile;

    // Otherwise use the value of global_jit_level.
    return registration->enable_jit_by_default && global_jit_level > 0;
  };
  return RunImpl(options, is_compilable);
}

// Is 'node' an operator that consumes only the shape of its input, not the
// data itself?
static bool IsShapeConsumerOp(const Node& node) {
  return node.type_string() == "Shape" || node.type_string() == "Rank" ||
         node.type_string() == "Size";
}

// Sequence number generator to ensure clusters have unique names.
static std::atomic<int64> cluster_sequence_num;

Status MarkForCompilationPass::RunImpl(
    const GraphOptimizationPassOptions& options,
    const std::function<bool(const Node*, const DeviceType&)>&
        is_compilable_fn) {
  VLOG(1) << "MarkForCompilationPass::Run";

  // Make sure that kernels have been registered on the JIT device.
  XlaOpRegistry::RegisterCompilationKernels();

  Graph* graph = options.graph->get();

  std::unordered_set<Node*> compilation_candidates;
  TF_RETURN_IF_ERROR(FindCompilationCandidates(
      *graph, options.flib_def,
      (options.session_options != nullptr) ? options.session_options->env
                                           : Env::Default(),
      is_compilable_fn, &compilation_candidates));

  GraphCycles cycles;
  for (int i = 0; i < graph->num_node_ids(); ++i) {
    // We rely on the node IDs in the cycle detection graph being consecutive
    // integers starting from 0.
    CHECK_EQ(i, cycles.NewNode());
  }

  // Compute the loop structure of the graph.
  std::vector<ControlFlowInfo> control_flow_info;
  TF_RETURN_IF_ERROR(BuildControlFlowInfo(graph, &control_flow_info));

  // The clustering code must avoid adding cycles to the graph to prevent
  // deadlock. However, the graph may contain loops, which would trigger the
  // cycle detection code. To handle loops, we alter the structure of the cycle
  // detection graph, disconnecting each loop from the enclosing graph.
  // Specifically, we:
  // * add a new "frame" node for each loop.
  // * replace edges to "Enter" nodes, and edges from "Exit" nodes with edges
  //   to/from the corresponding frame node. In essence, we collapse the loop
  //   into a single node for the purpose of cycle detection in the enclosing
  //   graph.
  // * the body of the loop should now be disconnected from the rest of the
  //   graph; we make it acyclic by breaking loop backedges (edges outgoing from
  //   "NextIteration" nodes.

  // Map from frame name strings to node IDs in the cycle detection graph.
  std::unordered_map<string, int> frame_nodes;

  // Get the cycle graph node ID for frame 'frame_name', or add one if none
  // exists.
  auto GetOrAddFrameNodeId = [&frame_nodes, &cycles](const string& frame_name) {
    int& frame_id = frame_nodes.emplace(frame_name, -1).first->second;
    if (frame_id < 0) {
      // The emplace succeeded; we have not allocated a frame node yet.
      frame_id = cycles.NewNode();
    }
    return frame_id;
  };

  for (Edge const* edge : graph->edges()) {
    if (edge->dst()->IsEnter()) {
      // Lift edges to an "Enter" node to the corresponding frame node.
      const string& frame_name =
          control_flow_info[edge->dst()->id()].frame_name;
      int dst = GetOrAddFrameNodeId(frame_name);
      if (!cycles.InsertEdge(edge->src()->id(), dst)) {
        return errors::Internal(
            "Cycle detected when adding enter->frame edge: ",
            DescribeCycle(cycles, *graph, edge->src()->id(), dst));
      }
      continue;
    }
    if (edge->src()->IsExit()) {
      // Lift edges from an "Exit" node to the corresponding frame node.
      const string& frame_name =
          control_flow_info[edge->src()->id()].frame_name;
      int src = GetOrAddFrameNodeId(frame_name);
      if (!cycles.InsertEdge(src, edge->dst()->id())) {
        return errors::Internal(
            "Cycle detected when adding frame->exit edge: ",
            DescribeCycle(cycles, *graph, src, edge->dst()->id()));
      }
      // Drop the original edge.
      continue;
    }
    if (edge->src()->IsNextIteration()) {
      // Break loop back-edges.
      continue;
    }
    if (!cycles.InsertEdge(edge->src()->id(), edge->dst()->id())) {
      // This should never happen. All cycles in the graph should contain
      // a control flow operator.
      return errors::Internal(
          "Found cycle in graph without control flow operator during XLA "
          "compilation: ",
          DescribeCycle(cycles, *graph, edge->src()->id(), edge->dst()->id()));
    }
  }

  // Each compilation candidate belongs to a cluster. The cluster's
  // representative
  // names the node in the 'cycles' graph that represents the cluster.
  std::vector<UnionFind<Cluster>> clusters(graph->num_node_ids());
  std::deque<UnionFind<Cluster>*> worklist;
  for (Node* node : compilation_candidates) {
    Cluster& cluster = clusters[node->id()].Get();
    cluster.representative = node->id();
    worklist.push_back(&clusters[node->id()]);
  }

  legacy_flags::MarkForCompilationPassFlags* flags =
      legacy_flags::GetMarkForCompilationPassFlags();

  // Repeatedly contract edges between clusters that are on the same device,
  // provided the contraction would not create a cycle.
  while (!worklist.empty()) {
    int from = worklist.front()->Get().representative;
    worklist.pop_front();

    Node* node_from = graph->FindNodeId(from);
    if (node_from->IsControlFlow()) {
      // Control flow nodes aren't compilation candidates and should never
      // appear.
      return errors::Internal(
          "Found control flow node in clustering worklist: ",
          node_from->type_string());
    }
    string from_scope;
    string to_scope;
    for (int to : cycles.Successors(from)) {
      if (to >= graph->num_node_ids()) {
        // Node is a "frame" node that is present only in the cycle detection
        // graph. No clustering is possible.
        continue;
      }
      Node* node_to = graph->FindNodeId(to);
      if (compilation_candidates.find(node_to) ==
          compilation_candidates.cend()) {
        continue;
      }
      if (node_from->assigned_device_name() !=
          node_to->assigned_device_name()) {
        continue;
      }
      // Look for an _XlaScope on both nodes.  If both nodes have a
      // scope and the scopes do not match, do not cluster along this
      // edge.  If even one of the nodes lacks an _XlaScope attribute,
      // then it is treated as a "bridge" and a cluster may be created
      // along it.  We may want to restrict this behavior to require
      // all nodes marked with _XlaCompile=true to also have a
      // _XlaScope property set (and raise an error otherwise); but
      // for now we don't do this.
      if (GetNodeAttr(node_from->attrs(), kXlaScopeAttr, &from_scope).ok() &&
          GetNodeAttr(node_to->attrs(), kXlaScopeAttr, &to_scope).ok() &&
          from_scope != to_scope) {
        continue;
      }

      // Ops that consume shapes cannot be the root of a cluster. This is an
      // optimization.
      if (clusters[from].Size() == 1 && IsShapeConsumerOp(*node_from)) {
        continue;
      }

      // Don't exceed the maximum cluster size.
      if (clusters[from].Size() + clusters[to].Size() >
          flags->tf_xla_max_cluster_size) {
        continue;
      }

      // If contracting the edge would create a cycle, bail out.
      // However, just because we can't merge the clusters now does not mean
      // we won't be able to merge them in the future.
      // e.g., if we have edges 1->2, 2->3 and 1->3, we cannot contract edge
      // 1->3. But if we first contract 1->2 then we can later contract 1->3.
      if (!cycles.ContractEdge(from, to)) continue;

      // Merge the clusters. ContractEdge uses 'from' as the number of the
      // merged node, so make sure 'from' is the chosen representative.
      clusters[from].Merge(&clusters[to]);

      worklist.push_back(&clusters[from]);
      break;
    }
  }

  // Count the number of elements in each cluster.
  std::vector<int> cluster_sizes(graph->num_node_ids());
  for (const Node* n : compilation_candidates) {
    int cluster = clusters[n->id()].Get().representative;
    cluster_sizes[cluster]++;
  }

  // Names for each cluster.
  std::unordered_map<int, string> cluster_names;

  // Mark clusters for compilation that:
  // * are placed on a device that requires compilation (an XlaDevice),
  // * are explicitly marked for compilation (_XlaCompile=true), or
  // * have more than flags->tf_xla_min_cluster_size elements (applicable only
  //   if compilation is enabled, otherwise there will be no such candidates).
  const int min_cluster_size = flags->tf_xla_min_cluster_size;
  for (Node* n : compilation_candidates) {
    int cluster = clusters[n->id()].Get().representative;

    // Compile if the user marked this node _XlaCompile=true
    bool compile_attr = false;
    bool marked_for_compilation = false;
    if (GetNodeAttr(n->attrs(), kXlaCompileAttr, &compile_attr).ok()) {
      marked_for_compilation = compile_attr;
    } else if (options.flib_def->GetAttr(*n, kXlaCompileAttr, &compile_attr)
                   .ok()) {
      marked_for_compilation = compile_attr;
    }

    // Compile if this operator is placed on a device that requires
    // compilation.
    DeviceType device_type("");
    TF_RETURN_IF_ERROR(
        DeviceTypeOfDevice(n->assigned_device_name(), &device_type));
    const XlaOpRegistry::DeviceRegistration* registration;
    XlaOpRegistry::GetCompilationDevice(device_type.type(), &registration);

    // Or compile if this is a cluster of >= min_cluster_size compilable
    // operators.
    if (cluster_sizes[cluster] >= min_cluster_size || marked_for_compilation ||
        registration->requires_compilation) {
      string& name = cluster_names[cluster];
      if (name.empty()) {
        name = strings::StrCat("cluster_", cluster_sequence_num++);
      }
      n->AddAttr(kXlaClusterAttr, name);
      VLOG(3) << "Assigning node " << n->name() << " to cluster " << name;
    }
  }

  if (flags->tf_xla_clustering_debug) {
    dump_graph::DumpGraphToFile("mark_for_compilation", **options.graph,
                                options.flib_def);
  }
  return Status::OK();
}

}  // namespace tensorflow

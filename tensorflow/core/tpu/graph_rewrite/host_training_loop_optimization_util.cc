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

#include "tensorflow/core/tpu/graph_rewrite/host_training_loop_optimization_util.h"

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/control_flow.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_pass_internal.h"
#include "tsl/platform/tstring.h"

namespace tensorflow {
namespace tpu {

namespace {

constexpr char kDefaultShardingValue[] = "";

const Edge* FindEdgeConnecting(const Node* src, const Node* dst) {
  for (const auto e : src->out_edges()) {
    if (e->dst()->name() == dst->name()) return &(*e);
  }
  return nullptr;
}

// Contains TPUExecute node and its DT_RESOURCE input nodes that
// correspond to model weights.
struct ExecuteNodeInfo {
  Node* execute_node;
  std::vector<const Edge*> var_inputs;
};

// Returns whether `node` is in `execute_nodes` or `(identity) -> execute`.
bool IsExecuteNodeOrIdentityToExecuteNode(
    const Graph& graph, const std::unordered_set<Node*>& loop_nodes,  // NOLINT
    const absl::flat_hash_set<Node*>& execute_nodes, Node* node) {
  if (execute_nodes.find(node) != execute_nodes.end()) return true;
  if (loop_nodes.find(node) == loop_nodes.end()) return false;
  if (node->IsNextIteration()) return true;
  if (!node->IsIdentity()) return false;

  for (const Edge* e : node->out_edges()) {
    if (e->IsControlEdge()) continue;

    Node* node = e->dst();
    if (!IsExecuteNodeOrIdentityToExecuteNode(graph, loop_nodes, execute_nodes,
                                              node)) {
      return false;
    }
  }

  return true;
}

// From input node to the TPUExecute op, finds the corresponding Enter node
// by searching/traversing nodes in below pattern of nodes:
// Enter ----> (identity) --->  While body input
// Returns nullptr if the Enter node is not found.
absl::StatusOr<Node*> FindEnterNodeFromTPUExecuteNodeInput(Node* input_node) {
  Node* node = input_node;
  while (node->IsIdentity()) {
    TF_RETURN_IF_ERROR(node->input_node(0, &node));
  }

  if (node->IsEnter()) {
    return node;
  }
  return nullptr;
}

absl::StatusOr<bool> ResourceOnlyUsedForTPUExecuteInLoop(
    const Graph& graph, const std::unordered_set<Node*>& loop_nodes,  // NOLINT
    const Node* enter_node, const absl::flat_hash_set<Node*> execute_nodes) {
  for (const Edge* output_edge : enter_node->out_edges()) {
    Node* output_node = output_edge->dst();
    if (output_edge->IsControlEdge() || output_node->IsExit()) continue;

    // If output node is not execute node, it must be output node
    // to the while loop body.
    if (!IsExecuteNodeOrIdentityToExecuteNode(graph, loop_nodes, execute_nodes,
                                              output_node)) {
      return false;
    }
  }
  return true;
}

// Given a TPUCompile node, find all TPUExecute nodes that executes the compiled
// program and its model weight variable inputs as well.
// TPUCompileMetadataProto of TPUCompile node must be reset to `new_metadata`
// if new reshard ops are added.
absl::Status ExtractExecuteNodeInfo(
    const Node* compile_node, const Graph& graph,
    const std::unordered_set<Node*>& loop_nodes,  // NOLINT
    std::vector<ExecuteNodeInfo>* execute_node_info,
    TPUCompileMetadataProto* new_metadata) {
  std::string metadata_string;
  TF_RETURN_IF_ERROR(
      GetNodeAttr(compile_node->attrs(), "metadata", &metadata_string));
  new_metadata->ParsePartialFromString(metadata_string);
  if (new_metadata->num_cores_per_replica() != 1) {
    // We do not support model parallelism yet.
    return absl::OkStatus();
  }

  execute_node_info->clear();
  for (Node* node : compile_node->out_nodes()) {
    if (node->type_string() == "TPUExecute") {
      execute_node_info->push_back({node});
    }
  }
  if (execute_node_info->empty()) {
    return absl::OkStatus();
  }
  TF_RET_CHECK(execute_node_info->size() == new_metadata->num_replicas())
      << "Number of replicas does not equal number of execute nodes: "
      << new_metadata->num_replicas() << " vs " << execute_node_info->size();
  DataTypeVector arg_types;
  TF_RETURN_IF_ERROR(GetNodeAttr((*execute_node_info)[0].execute_node->attrs(),
                                 "Targs", &arg_types));
  for (int64_t i = 0; i < arg_types.size(); ++i) {
    if (arg_types[i] != DT_RESOURCE) {
      continue;
    }
    const auto sharding_config = new_metadata->args(i).enable_xla_sharding();
    if (sharding_config != TPUCompileMetadataProto::Arg::TENTATIVE &&
        sharding_config != TPUCompileMetadataProto::Arg::ALLOWED) {
      continue;
    }
    std::vector<const Edge*> edges(execute_node_info->size());
    bool is_supported = true;
    std::unordered_map<Node*, absl::flat_hash_set<Node*>>
        enter_to_execute_nodes;
    for (int64_t j = 0; j < edges.size(); ++j) {
      auto execute = (*execute_node_info)[j].execute_node;
      TF_RETURN_IF_ERROR(execute->input_edge(i, &edges[j]));
      TF_RET_CHECK(edges[j]->src()->output_type(edges[j]->src_output()) ==
                   arg_types[i])
          << "Execute op has an unexpected input type.";
      // Traverse backwards to find the Enter node from which the input is
      // passed.
      // This makes sure that we are checking the usages of all potential
      // aliases of the input node as well.
      TF_ASSIGN_OR_RETURN(auto enter_node, FindEnterNodeFromTPUExecuteNodeInput(
                                               edges[j]->src()));
      if (enter_node == nullptr) {
        is_supported = false;
        enter_to_execute_nodes.clear();
        break;
      }
      enter_to_execute_nodes[enter_node].insert(edges[j]->dst());
    }

    for (const auto& it : enter_to_execute_nodes) {
      // Size of execute nodes should be either 1 (per-replica variables) or
      // num_replicas (distributed variables).
      if ((it.second.size() != 1) &&
          (it.second.size() != new_metadata->num_replicas())) {
        is_supported = false;
        break;
      }
      TF_ASSIGN_OR_RETURN(bool no_other_use,
                          ResourceOnlyUsedForTPUExecuteInLoop(
                              graph, loop_nodes, it.first, it.second));
      if (!no_other_use) {
        is_supported = false;
        break;
      }
    }

    // Add the variable input edges only when they are supported for all
    // executes.
    if (is_supported) {
      for (int64_t j = 0; j < edges.size(); ++j) {
        (*execute_node_info)[j].var_inputs.push_back(edges[j]);
      }
      new_metadata->mutable_args(i)->set_enable_xla_sharding(
          TPUCompileMetadataProto::Arg::ALLOWED);
    }
  }

  int64_t total = 0;
  for (const auto& a : new_metadata->args()) {
    if (a.enable_xla_sharding() == TPUCompileMetadataProto::Arg::ALLOWED) {
      total++;
    }
  }
  TF_RET_CHECK(total == (*execute_node_info)[0].var_inputs.size())
      << " total " << total << " var_inputs "
      << (*execute_node_info)[0].var_inputs.size();
  if (total == 0) {
    // We don't need to process anything if no input is added.
    execute_node_info->clear();
  }
  return absl::OkStatus();
}

bool IsTPUCompileOp(const Node& n) { return n.type_string() == "TPUCompile"; }

void FindTPUCompileNodes(
    const std::string* current_function_name,
    const AttrValueMap* current_function_attr,
    const std::unordered_map<std::string, WhileLoopFrame>& frames,
    std::vector<HostTrainingLoopInfo>* host_training_loops_info) {
  // Adds frames with no children (i.e., the innermost frames) to a worklist.
  std::deque<const WhileLoopFrame*> worklist;

  for (auto& frame : frames) {
    if (frame.second.num_children == 0) {
      worklist.push_back(&frame.second);
    }
  }

  // Check TPUCompile node from the innermost while loop to the outermost
  // while loop.
  while (!worklist.empty()) {
    const WhileLoopFrame* frame = worklist.front();
    worklist.pop_front();

    for (const auto& n : frame->nodes) {
      if (!IsTPUCompileOp(*n)) continue;

      HostTrainingLoopInfo host_training_loop_info;
      host_training_loop_info.compile_node_name = n->name();
      host_training_loop_info.loop_cond_node_name = frame->loop_cond->name();
      host_training_loop_info.while_loop_name = frame->name;

      for (const auto arg : frame->args) {
        LoopArgInfo arg_info;
        arg_info.enter_node_name = arg.enter->name();
        if (arg.exit) arg_info.exit_node_name = arg.exit->name();

        host_training_loop_info.loop_arguments.push_back(std::move(arg_info));
      }
      host_training_loop_info.loop_nodes = frame->nodes;

      if (current_function_name) {
        host_training_loop_info.encapsulating_function_name =
            *current_function_name;
      }
      if (current_function_attr) {
        host_training_loop_info.encapsulating_function_attrs =
            *current_function_attr;
      }

      host_training_loops_info->emplace_back(
          std::move(host_training_loop_info));
    }

    // If the parent has no remaining children, add it to the worklist.
    --frame->parent->num_children;
    if (frame->parent->num_children == 0) {
      worklist.push_back(frame->parent);
    }
  }
}

// From while loop cond node, finds all loop exit nodes by searching/traversing
// nodes in below pattern of nodes:
// LoopCond -----> Switch -----> Exit
std::vector<Node*> FindLoopExitNodes(const Node& loop_cond) {
  std::vector<Node*> loop_exit_nodes;
  for (const auto e_cond : loop_cond.out_edges()) {
    if (e_cond->IsControlEdge() || !e_cond->dst()->IsSwitch()) continue;
    auto switch_node = e_cond->dst();

    for (const auto e_switch : switch_node->out_edges()) {
      if (e_switch->IsControlEdge() || !e_switch->dst()->IsExit()) continue;

      loop_exit_nodes.push_back(e_switch->dst());
    }
  }
  return loop_exit_nodes;
}

// Returns or creates a node in that is executed before each loop iteration
// in the while loop.
// TODO(mdan): Inject this node between the Enter and Merge nodes instead.
// See AddNoOpAfterLastIteration for an example.
absl::Status GetOrCreateBeforeEachIterationNode(const Node& loop_cond_node,
                                                Graph* graph, Node** node_out) {
  Node* loop_switch_node = nullptr;
  for (auto n : loop_cond_node.out_nodes()) {
    if (n->IsSwitch()) {
      loop_switch_node = n;
      break;
    }
  }
  TF_RET_CHECK(loop_switch_node != nullptr)
      << "Unable to find any switch nodes.";

  // If while loop switch node already has a outgoing data to true brach
  // of the switch op, then reuse that node.
  for (const auto out_edge : loop_switch_node->out_edges()) {
    if (out_edge->src_output() == 1) {
      *node_out = out_edge->dst();
      return absl::OkStatus();
    }
  }

  // Create Identity node that represents execution at every loop iteration.
  NodeDef at_loop_iteration_nodedef;
  at_loop_iteration_nodedef.set_op("Identity");
  DataType dtype;
  TF_RETURN_IF_ERROR(GetNodeAttr(loop_switch_node->def(), "T", &dtype));

  AddNodeAttr("T", dtype, &at_loop_iteration_nodedef);
  at_loop_iteration_nodedef.set_name(graph->NewName(absl::StrCat(
      "TPUVariableReshard/before_iteration", "/_", internal::GetNodeId())));

  absl::Status status;
  TF_ASSIGN_OR_RETURN(Node * at_loop_iteration_node,
                      graph->AddNode(at_loop_iteration_nodedef));
  TF_RETURN_IF_ERROR(status);

  graph->AddEdge(loop_switch_node, 1, at_loop_iteration_node, 0);
  *node_out = at_loop_iteration_node;
  return absl::OkStatus();
}

// Injects a NoOp node in that is executed after the very last iteration
// of the while loop but before the while loop exit node.
// This node is positioned between the False output of all Switch nodes (
// meaning, it executes after the loop ended all its iterations) and their
// corresponding Exit nodes (meaning, before the loop finally completed).
// See the white paper for details:
// http://download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf
absl::Status AddNoOpAfterLastIteration(const Node& loop_cond_node, Graph* graph,
                                       Node** node_out) {
  NodeDef after_last_iteration;
  after_last_iteration.set_op("NoOp");

  after_last_iteration.set_name(graph->NewName(absl::StrCat(
      "TPUVariableReshard/after_last_iteration", "/_", internal::GetNodeId())));

  absl::Status status;
  Node* after_last_iteration_node =
      graph->AddNode(after_last_iteration, &status);
  TF_RETURN_IF_ERROR(status);

  for (auto switch_node : loop_cond_node.out_nodes()) {
    if (!switch_node->IsSwitch()) {
      continue;
    }

    NodeDef switch_exit;
    switch_exit.set_op("Identity");

    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(switch_node->def(), "T", &dtype));
    AddNodeAttr("T", dtype, &switch_exit);
    auto name = absl::StrCat("TPUVariableReshard/switch_exit/", "/_",
                             internal::GetNodeId());
    switch_exit.set_name(graph->NewName(name));
    // Introducing identity nodes risks a device copy, which isn't guaranteed
    // to be available for all types. Hence the colocation constraint.
    AddNodeAttr(kColocationAttrName,
                std::vector<std::string>{
                    absl::StrCat(kColocationGroupPrefix, switch_node->name())},
                &switch_exit);

    TF_ASSIGN_OR_RETURN(Node * after_switch_node, graph->AddNode(switch_exit));

    graph->AddEdge(switch_node, 0, after_switch_node, 0);
    graph->AddControlEdge(after_switch_node, after_last_iteration_node);

    for (const auto out_node : switch_node->out_nodes()) {
      if (out_node->IsExit()) {
        graph->AddControlEdge(after_last_iteration_node, out_node);
      }
    }
  }

  *node_out = after_last_iteration_node;
  return absl::OkStatus();
}

}  // namespace

absl::Status DetectHostTrainingLoop(
    const std::string* current_function_name,
    const AttrValueMap* current_function_attr,
    const FunctionLibraryDefinition* library, Graph* graph,
    FunctionLibraryRuntime* flr,
    std::vector<HostTrainingLoopInfo>* host_training_loops_info) {
  std::vector<AssociatedFunctionInfo> associated_function_list;
  for (const auto* n : graph->nodes()) {
    const auto associated_functions = GetAssociatedFunctions(*n, library);
    if (associated_functions.empty()) continue;

    associated_function_list.insert(associated_function_list.end(),
                                    associated_functions.begin(),
                                    associated_functions.end());
  }

  absl::Status ret_status = absl::OkStatus();
  for (const auto& function : associated_function_list) {
    if (function.type() != AssociatedFunctionInfo::kFunctionAttr) continue;

    // Convert the function to Graph.
    FunctionLibraryRuntime::Handle handle;
    TF_RETURN_IF_ERROR(flr->Instantiate(function.func_name(),
                                        AttrSlice(&function.attrs()), &handle));
    auto cleanup_handle = gtl::MakeCleanup([&]() {
      auto s = flr->ReleaseHandle(handle);
      if (!s.ok()) {
        ret_status.Update(s);
      }
    });
    const FunctionBody* body = flr->GetFunctionBody(handle);
    Graph* function_graph = body->graph;
    TF_RETURN_IF_ERROR(DetectHostTrainingLoop(
        &function.func_name(), &function.attrs(), library, function_graph, flr,
        host_training_loops_info));
  }

  // BuildControlFlowInfo() requires that the graph's source node is connected
  // to all source nodes in the graph. Many graphs violate this invariant.
  // As so, add edges to source/sink nodes so that this invariant is kept.
  FixupSourceAndSinkEdges(graph);
  std::vector<ControlFlowInfo> cf_info;
  TF_RETURN_IF_ERROR(
      BuildControlFlowInfo(graph, &cf_info, /*unreachable_nodes=*/nullptr));

  std::unordered_map<std::string, WhileLoopFrame> frames;
  TF_RETURN_IF_ERROR(ExtractWhileLoopFrames(cf_info, graph, &frames));
  FindTPUCompileNodes(current_function_name, current_function_attr, frames,
                      host_training_loops_info);
  return ret_status;
}

absl::Status AddReshardOp(Graph* graph,
                          const HostTrainingLoopInfo& host_loop_info) {
  const auto& compile_node_name = host_loop_info.compile_node_name;
  const auto node_name_map = graph->BuildNodeNameIndex();
  const auto node_it = node_name_map.find(compile_node_name);
  TF_RET_CHECK(node_it != node_name_map.end())
      << "Unable to find compile node : " << compile_node_name;

  const auto compile_node = node_it->second;
  std::vector<ExecuteNodeInfo> execute_nodes_info;

  absl::Status status;
  TPUCompileMetadataProto metadata;
  status =
      ExtractExecuteNodeInfo(compile_node, *graph, host_loop_info.loop_nodes,
                             &execute_nodes_info, &metadata);
  if (!status.ok()) {
    LOG(ERROR) << "Encountered error when trying to extract execute nodes, "
                  "skipping host loop optimization. Status: "
               << status;
    return absl::OkStatus();
  }

  if (execute_nodes_info.empty()) {
    return absl::OkStatus();
  }

  // Update the TPUCompileMetadata such that sharding config of the
  // sharded resource variable inputs is set to ALLOWED instead of
  // TENTATIVE.
  std::string new_metadata_string;
  metadata.SerializeToString(&new_metadata_string);
  compile_node->ClearAttr("metadata");
  compile_node->AddAttr("metadata", new_metadata_string);

  // Unsharding of the model weight variables must happen only at the very
  // last loop iteration. As so, add while loop condition predicate as an
  // input to the sharding switch node. If loop condition is true, we do not
  // unshard.
  const auto& cond_node_name = host_loop_info.loop_cond_node_name;
  auto loop_cond_node_it = node_name_map.find(cond_node_name);
  TF_RET_CHECK(loop_cond_node_it != node_name_map.end())
      << "Cannot find loop condition node : " << cond_node_name;
  auto* loop_condition_node = loop_cond_node_it->second;

  // In order to make sure that shard/unshard operations are invoked
  // at the start of every loop body and at the end of last iteration
  // of the loop, respectively, create a pair of guiding nodes, which
  // guaranteed to execute before each iteration and respectively after
  // all iterations.

  Node* after_last_iteration_node;
  TF_RETURN_IF_ERROR(AddNoOpAfterLastIteration(*loop_condition_node, graph,
                                               &after_last_iteration_node));

  Node* before_loop_iteration_node;
  TF_RETURN_IF_ERROR(GetOrCreateBeforeEachIterationNode(
      *loop_condition_node, graph, &before_loop_iteration_node));

  // Create const op that represents default sharding value
  // (i.e. no-op sharding).
  NodeDef default_sharding;
  default_sharding.set_op("Const");
  default_sharding.set_name(graph->NewName(absl::StrCat(
      "TPUVariableReshard/default_shard_state", "/_", internal::GetNodeId())));
  AddNodeAttr("dtype", DT_STRING, &default_sharding);

  Tensor t(DT_STRING, {3});
  t.vec<tsl::tstring>()(0) = kDefaultShardingValue;
  t.vec<tsl::tstring>()(1) = kDefaultShardingValue;
  t.vec<tsl::tstring>()(2) = kDefaultShardingValue;
  t.AsProtoTensorContent(
      (*default_sharding.mutable_attr())["value"].mutable_tensor());

  TF_ASSIGN_OR_RETURN(Node * default_sharding_node,
                      graph->AddNode(default_sharding));
  TF_RETURN_IF_ERROR(status);
  // Add control edge between loop condition to make sure that
  // default_sharding_node node is inside the while loop frame.
  graph->AddControlEdge(loop_condition_node, default_sharding_node);

  // Build a no-op node used to add control edges after unshard nodes.
  NodeDef after_unshard;
  after_unshard.set_op("NoOp");
  after_unshard.set_name(graph->NewName(absl::StrCat(
      "TPUVariableReshard/last_iteration", "/_", internal::GetNodeId())));
  TF_ASSIGN_OR_RETURN(auto after_unshard_node, graph->AddNode(after_unshard));

  for (auto info : execute_nodes_info) {
    auto execute_node = info.execute_node;
    // Create Reshard op that optionally shards model weight variables
    // prior to program execution.
    NodeDef reshard_node_def;
    reshard_node_def.set_name(graph->NewName(absl::StrCat(
        "TPUVariableReshard/reshard", "/_", internal::GetNodeId())));
    reshard_node_def.set_op("TPUReshardVariables");
    AddNodeAttr("N", static_cast<int>(info.var_inputs.size()),
                &reshard_node_def);
    TF_ASSIGN_OR_RETURN(Node * reshard_op_node,
                        graph->AddNode(reshard_node_def));

    reshard_op_node->set_assigned_device_name(
        execute_node->assigned_device_name());

    // Reshard op must execute at every loop iteration prior to
    // TPUExecute node.
    graph->AddControlEdge(before_loop_iteration_node, reshard_op_node);
    graph->AddControlEdge(reshard_op_node, execute_node);

    for (int i = 0; i < info.var_inputs.size(); ++i) {
      const auto variable_edge = info.var_inputs[i];
      graph->AddEdge(variable_edge->src(), variable_edge->src_output(),
                     reshard_op_node, i);
    }

    const int new_key_input = info.var_inputs.size();
    // Add program input edge from the compiler(i.e. compilation key).
    const auto compilation_key_edge =
        FindEdgeConnecting(compile_node, execute_node);
    graph->AddEdge(compile_node, compilation_key_edge->src_output(),
                   reshard_op_node, new_key_input);

    // Create VarHandleOp to store sharding state. Sharding state holds
    // std::string compilation key that identifies whether the graph is
    // re-compiled and the variables need to be sharded again.
    NodeDef var_handle_def;
    var_handle_def.set_op("VarHandleOp");
    var_handle_def.set_name(graph->NewName(absl::StrCat(
        "TPUVariableReshard/reshard_state", "/_", internal::GetNodeId())));
    AddNodeAttr("dtype", DT_STRING, &var_handle_def);
    AddNodeAttr("shape", TensorShape({}), &var_handle_def);
    TF_ASSIGN_OR_RETURN(Node * var_handle_node, graph->AddNode(var_handle_def));

    // Add control edge between `var_handle_def` node and while loop
    // loop condition so that `var_handle_def` is inside the same while loop
    // frame.
    // TODO(hongjunchoi): Consider adding control edge from another node--such
    // as input control node.
    graph->AddControlEdge(loop_condition_node, var_handle_node);

    // Connect data edge between var handle op and reshard op.
    const int format_state_input = new_key_input + 1;
    graph->AddEdge(var_handle_node, 0, reshard_op_node, format_state_input);

    // Create Reshard op that represents unsharding after TPUExecute.
    NodeDef unshard_node_def;
    unshard_node_def.set_name(graph->NewName(absl::StrCat(
        "TPUVariableReshard/unshard", "/_", internal::GetNodeId())));
    unshard_node_def.set_op("TPUReshardVariables");
    AddNodeAttr("N", static_cast<int>(info.var_inputs.size()),
                &unshard_node_def);
    TF_ASSIGN_OR_RETURN(Node * unshard_op_node,
                        graph->AddNode(unshard_node_def));

    unshard_op_node->set_assigned_device_name(
        execute_node->assigned_device_name());

    for (int i = 0; i < info.var_inputs.size(); ++i) {
      const auto variable_edge = info.var_inputs[i];
      // Connect model weight resource variables to unshard op. Since unshard op
      // must be only invoked after the very last loop iteration, for each while
      // loop inputs, we traverse backwards to find the switch node of the host
      // training loop and connect `output_false` field of the switch node with
      // unshard op.
      TF_ASSIGN_OR_RETURN(
          Node * enter_node,
          FindEnterNodeFromTPUExecuteNodeInput(variable_edge->src()));
      graph->AddEdge(enter_node, 0, unshard_op_node, i);
    }

    // Add control dependency before/after unshard node and the control nodes.
    graph->AddControlEdge(after_last_iteration_node, unshard_op_node);
    graph->AddControlEdge(unshard_op_node, after_unshard_node);

    graph->AddEdge(default_sharding_node, 0, unshard_op_node, new_key_input);

    // Add data edge from sharding state var handle op to unshard op.
    graph->AddEdge(var_handle_node, 0, unshard_op_node, format_state_input);
  }
  // Add control dependency from after_unshard_node to all exits nodes. This is
  // to make sure that the unshard ops will be executed as long as any of the
  // exits are used.
  for (auto exit : FindLoopExitNodes(*loop_condition_node)) {
    graph->AddControlEdge(after_unshard_node, exit);
  }
  return absl::OkStatus();
}

}  // namespace tpu
}  // namespace tensorflow

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/lower_functional_ops.h"

#include <string>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/device_propagation.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include "tensorflow/core/common_runtime/lower_case_op.h"
#include "tensorflow/core/common_runtime/lower_function_call_op.h"
#include "tensorflow/core/common_runtime/lower_if_op.h"
#include "tensorflow/core/common_runtime/lower_while_op.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

constexpr const char* const kLowerUsingSwitchMergeAttr =
    LowerFunctionalOpsConstants::kLowerUsingSwitchMergeAttr;
constexpr const char* const kLowerAsMultiDeviceFunctionAttr =
    LowerFunctionalOpsConstants::kLowerAsMultiDeviceFunctionAttr;

constexpr const char* const kTpuReplicateAttr = "_tpu_replicate";
constexpr const char* const kXlaClusterAttr = "_xla_compile_id";
constexpr const char* const kXlaMustCompileAttr = "_XlaMustCompile";

// Checks if boolean attribute is defined and it's value is 'true'.
bool CheckBoolAttr(const Node* n, absl::string_view attr_name) {
  bool match;
  bool found = TryGetNodeAttr(n->attrs(), attr_name, &match);
  return found && match;
}

// Checks if string attribute is defined and it's not empty.
bool CheckStringAttr(const Node* n, absl::string_view attr_name) {
  string match;
  bool found = TryGetNodeAttr(n->attrs(), attr_name, &match);
  return found && !match.empty();
}

bool LowerUsingSwitchMergeIsOn(const Node* n) {
  return CheckBoolAttr(n, kLowerUsingSwitchMergeAttr);
}

bool LowerAsMultiDeviceFunctionIsOn(const Node* n) {
  return CheckBoolAttr(n, kLowerAsMultiDeviceFunctionAttr);
}

bool MarkedForTpuCompilation(const Node* n) {
  return CheckStringAttr(n, kTpuReplicateAttr);
}

bool MarkedForXlaCompilation(const Node* n) {
  return CheckStringAttr(n, kXlaClusterAttr) ||
         CheckBoolAttr(n, kXlaMustCompileAttr);
}

bool HasArgsOrRetvals(const Graph& g) {
  for (const Node* n : g.op_nodes()) {
    if (n->IsArg() || n->IsRetval()) return true;
  }
  return false;
}

const absl::flat_hash_set<std::string>& DevicePropagationOpList() {
  // Control flow ops and Identity ops which are inserted by function call
  // inlining.
  static const auto op_list = new absl::flat_hash_set<std::string>(
      {"Identity", "IdentityN", "Enter", "Exit", "Switch", "Merge",
       "NextIteration"});
  return *op_list;
}

bool IsPropagatableDevice(absl::string_view device_string) {
  DeviceNameUtils::ParsedName device;
  return DeviceNameUtils::ParseFullName(device_string, &device) &&
         device.type == DEVICE_TPU;
}

}  // namespace

absl::Status LowerFunctionalOpsPass::Run(
    const GraphOptimizationPassOptions& options) {
  if (options.partition_graphs != nullptr) {
    return errors::Internal(
        "Lowering If/While ops should happen before partitioning.");
  }
  if (options.graph == nullptr) {
    return absl::OkStatus();
  }

  Graph* g = options.graph->get();
  if (g == nullptr) {
    return errors::Internal(
        "Lowering While op requires a graph to be available.");
  }

  FunctionLibraryDefinition* flib_def = options.flib_def;
  if (flib_def == nullptr) {
    return errors::Internal(
        "Lowering If op requires a FunctionLibraryDefinition to be available.");
  }

  // Lower function calls only if it's explicitly enabled in session options.
  const bool lower_function_calls =
      options.session_options && options.session_options->config.graph_options()
                                     .optimizer_options()
                                     .do_function_inlining();

  // If graph is a function instantiation, it will have `_Arg` and `_Retval`
  // nodes for input and output tensors. Otherwise it's unsafe to remove any of
  // the nodes, because they might be later used as fetches.
  //
  // When we do not keep lowered nodes fetchable, we still add a NoOp node to
  // the graph with the same name as lowered node, because it might be used as a
  // control output source, and it's currently not expressed in a graph.
  bool keep_lowered_nodes_fetchable = !HasArgsOrRetvals(*g);

  // We disable lowering control flow to switch/merge variants when requested,
  // and for the single-threaded executor and TFRT runtime, which does not
  // support it.
  const bool functional_control_flow =
      options.session_options &&
      (options.session_options->config.experimental().executor_type() ==
           "SINGLE_THREADED_EXECUTOR" ||
       options.session_options->config.experimental().use_tfrt() ||
       options.session_options->config.experimental()
           .disable_functional_ops_lowering());

  // Returns true if `node` will be used for XLA compilation.
  const auto used_by_xla = [](Node* node) -> bool {
    return MarkedForTpuCompilation(node) || MarkedForXlaCompilation(node);
  };

  // Returns true if control flow `node` should be lowered to Switch/Merge.
  const auto lower_control_flow = [&](Node* node) -> bool {
    return LowerUsingSwitchMergeIsOn(node) && !used_by_xla(node);
  };

  // Lower all If, Case, While ops that have the `kLowerUsingSwitchMergeAttr`
  // attr set and inline all function calls into the graph.
  // We start at `i` = 2 to skip the source and sink nodes.
  // Note that `g->num_node_ids()` may change in the for body if a matching If,
  // Case, While node is lowered. Since new graph nodes are always added to the
  // end of the list of nodes it is ensured that nested If/Case/While nodes will
  // be lowered as well.
  int num_node_ids_before_lowering = g->num_node_ids();
  for (int i = 2; i < g->num_node_ids(); ++i) {
    Node* n = g->FindNodeId(i);
    if (n == nullptr) continue;  // deleted node

    // Always lower function calls produced by lowering If/While nodes.
    if (IsFunctionCall(*flib_def, *n) && !used_by_xla(n) &&
        (lower_function_calls || LowerAsMultiDeviceFunctionIsOn(n))) {
      TF_RETURN_IF_ERROR(RewriteFunctionCallNode(n, g, *flib_def,
                                                 keep_lowered_nodes_fetchable));
      continue;
    }

    // If we are allowed to used function control flow, we do not need to check
    // for If/While/Case nodes in the graph.
    if (functional_control_flow) continue;

    if (n->IsIfNode() && lower_control_flow(n)) {
      TF_RETURN_IF_ERROR(RewriteIfNode(n, g, keep_lowered_nodes_fetchable));

    } else if (n->IsCaseNode() && lower_control_flow(n)) {
      TF_RETURN_IF_ERROR(RewriteCaseNode(n, g, keep_lowered_nodes_fetchable));

    } else if (n->IsWhileNode() && lower_control_flow(n)) {
      TF_RETURN_IF_ERROR(
          RewriteWhileNode(n, g, flib_def, keep_lowered_nodes_fetchable));

    } else {
      DCHECK(!lower_control_flow(n))
          << "Node " << FormatNodeForError(*n) << " of type "
          << n->type_string() << " has '"
          << LowerFunctionalOpsConstants::kLowerUsingSwitchMergeAttr
          << "' attr set but it does not support lowering.\n";
    }
  }

  // Propagates device assignments inside a function call to control flow ops
  // after function call is lowered, bcause If/Case/While node lowering happen
  // before function call lowering,
  PropagateDevices(
      [num_node_ids_before_lowering](const Node& n) {
        return DevicePropagationOpList().contains(n.type_string()) &&
               n.id() >= num_node_ids_before_lowering;  // Newly created nodes.
      },
      IsPropagatableDevice, g);

  return absl::OkStatus();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 10,
                      LowerFunctionalOpsPass);

}  // namespace tensorflow

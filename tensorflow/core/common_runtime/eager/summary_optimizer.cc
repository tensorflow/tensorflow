/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/summary_optimizer.h"

#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"

namespace tensorflow::summary_optimizer {
namespace {

constexpr char kDisableSummariesAtRuntime[] = "disable_summaries_at_runtime";
constexpr char kFlushSummaryWriter[] = "FlushSummaryWriter";
constexpr char kWriteSummary[] = "write_summary";
constexpr char kForwardFunctionName[] = "forward_function_name";
constexpr char kBackwardFunctionName[] = "backward_function_name";
constexpr char kEmptyString[] = "";

using summary_optimizer::internal::NormalizeEdgeName;
using ArgDef = OpDef::ArgDef;

void UpdateNestedFunctionName(NodeDef& ndef) {
  for (auto& [k, v] : *ndef.mutable_attr()) {
    if (v.has_func()) {
      v.mutable_func()->set_name(StrippedFunctionName(v.func().name()));
    } else if (v.list().func_size() > 0) {
      for (auto& func : *v.mutable_list()->mutable_func()) {
        func.set_name(StrippedFunctionName(func.name()));
      }
    }
  }
}

void PruneDeletedInputDeps(
    const absl::flat_hash_set<std::string>& nodes_to_keep, NodeDef& ndef) {
  auto inputs = ndef.input();
  ndef.clear_input();
  for (const std::string& input : inputs) {
    if (nodes_to_keep.contains(NormalizeEdgeName(input))) {
      ndef.add_input(input);
    }
  }
}

FunctionDef StripSummary(const FunctionDef& fdef_with_summaries) {
  FunctionDef fdef = fdef_with_summaries;
  fdef.mutable_signature()->set_name(
      StrippedFunctionName(fdef.signature().name()));
  auto nodes = fdef.node_def();
  fdef.clear_node_def();
  absl::flat_hash_set<std::string> nodes_to_keep;
  absl::c_transform(nodes, std::inserter(nodes_to_keep, nodes_to_keep.end()),
                    [](const NodeDef& node_def) { return node_def.name(); });
  absl::c_transform(fdef.signature().input_arg(),
                    std::inserter(nodes_to_keep, nodes_to_keep.end()),
                    [](const ArgDef& input_arg) { return input_arg.name(); });

  // Prune all nodes corresponding to `summary_ops`.
  for (const NodeDef& ndef : nodes) {
    // `FlushSummaryWriter` node indicates the final node in a function that
    // writes summaries.
    if (ndef.op() == kFlushSummaryWriter) nodes_to_keep.erase(ndef.name());

    // summary.write ops are created under a `write_summary` name scope.
    // Prune ops created internally for the summary.write operations.
    for (const auto& substr : absl::StrSplit(ndef.name(), '/')) {
      if (substr == kWriteSummary) {
        nodes_to_keep.erase(ndef.name());
        break;
      }
    }
  }

  // Update the FunctionDef to exclude the pruned nodes.
  for (NodeDef& ndef : nodes) {
    if (!nodes_to_keep.contains(ndef.name())) continue;
    PruneDeletedInputDeps(nodes_to_keep, ndef);
    UpdateNestedFunctionName(ndef);
    *fdef.add_node_def() = std::move(ndef);
  }

  // Prune out any control outputs that were used only for the summary_ops.
  auto control_ret = fdef.control_ret();
  fdef.clear_control_ret();
  for (const auto& [signature_node_name, node_name] : control_ret) {
    if (!nodes_to_keep.contains(NormalizeEdgeName(node_name))) continue;
    fdef.mutable_control_ret()->insert({signature_node_name, node_name});
  }

  // Prune out any summary_ops-related-control-nodes from the function's output
  // signature.
  auto control_outputs = fdef.signature().control_output();
  fdef.mutable_signature()->clear_control_output();
  for (const std::string& control_output : control_outputs) {
    if (!fdef.control_ret().contains(control_output)) continue;
    fdef.mutable_signature()->add_control_output(control_output);
  }

  for (auto& [k, v] : *fdef.mutable_attr()) {
    // Update the names of the forward and backward function names.
    if (k == kForwardFunctionName || k == kBackwardFunctionName) {
      v.set_s(StrippedFunctionName(v.s()));
    }
    // Disable summary stripping on functions that have already been stripped.
    if (k == kDisableSummariesAtRuntime) v.clear_list();
  }
  return fdef;
}

}  // namespace

namespace internal {

std::string NormalizeEdgeName(absl::string_view name) {
  // Control nodes begin with '^'.
  // If an edge_name is split by `:` it indicates which output of a node to
  // return. Since we only care about the node's name we can discard everything
  // following the first `:`.
  std::vector<std::string> edge_name =
      absl::StrSplit(name, absl::ByAnyChar("^:"));
  return edge_name[0].empty() ? edge_name[1] : edge_name[0];
}

}  // namespace internal

std::pair<absl::string_view, bool> GetDisableSummariesInputArg(
    const FunctionDef& fdef) {
  auto it = fdef.attr().find(kDisableSummariesAtRuntime);
  if (it == fdef.attr().end()) return {kEmptyString, false};
  if (it->second.has_list()) {
    const auto& list = it->second.list();
    if (list.s_size() == 1 && list.b_size() == 1) {
      return {list.s(0), list.b(0)};
    }
  }
  return {kEmptyString, false};
}

std::vector<FunctionDef> StripSummaries(const FunctionDef& fdef,
                                        const FunctionLibraryDefinition& flib) {
  std::vector<FunctionDef> results;
  if (GetDisableSummariesInputArg(fdef).first.empty()) return results;

  // Strip the summaries from the provided `fdef`.
  results.push_back(StripSummary(fdef));

  // Strip the summaries from all nested functions within `fdef`.
  FunctionLibraryDefinition reachable_library = flib.ReachableDefinitions(fdef);
  for (const std::string& fname : reachable_library.ListFunctionNames()) {
    auto* nested_fdef = flib.Find(fname);
    if (nested_fdef == nullptr) continue;
    results.push_back(StripSummary(*nested_fdef));
  }

  return results;
}

std::string StrippedFunctionName(absl::string_view fname) {
  return absl::StrCat(fname, "__instance__no_summaries");
}

}  // namespace tensorflow::summary_optimizer

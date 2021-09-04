/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/make_deterministic.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kInterleaveOp[] = "InterleaveDataset";
constexpr char kParallelInterleaveOp[] = "ParallelInterleaveDataset";
constexpr char kLegacyParallelInterleaveOp[] =
    "LegacyParallelInterleaveDatasetV2";

// List of stateful ops which do not introduce nondeterminism when put inside a
// ParallelInterleave dataset. All ops which do not mutate state can be put in
// this list, even if such ops read state. Note that random ops implicitly
// mutate state and cannot be put in this list.
// TODO(reedwm): Expand this list.
constexpr std::array<const char*, 7> kDeterministicStatefulOps = {
    "TextLineDataset", "FixedLengthRecordDataset",
    "TFRecordDataset", "TensorSliceDataset",
    "RangeDataset",    "SSTableDataset",
    "RecordIODataset",
};

bool IsParallelInterleave(const std::string& op) {
  return data::MatchesAnyVersion(kParallelInterleaveOp, op) ||
         op == kLegacyParallelInterleaveOp;
}

bool IsDeterministicStatefulOp(const std::string& op) {
  for (auto stateful_op : kDeterministicStatefulOps) {
    if (data::MatchesAnyVersion(stateful_op, op)) {
      return true;
    }
  }
  return false;
}

Status ConvertToNonParallelInterleave(const string& node_name,
                                      MutableGraphView* graph) {
  int index = graph_utils::FindGraphNodeWithName(node_name, *graph->graph());
  DCHECK_NE(index, -1) << "Failed to find node " << node_name
                       << " in the optimized graph.";
  NodeDef* interleave = graph->graph()->mutable_node(index);
  interleave->set_op(kInterleaveOp);

  auto Targuments = interleave->attr().find("Targuments");
  if (Targuments == interleave->attr().end()) {
    return errors::Internal(
        "Failed to find Targuments attribute for "
        "interleave node ",
        node_name);
  }
  const int num_inputs_after_rewrite =
      3 + Targuments->second.list().type_size();

  // ParallelInterleave ops take in more inputs than Interleave, so turn extra
  // inputs into control inputs. These extra inputs are for performance and are
  // safe to ignore.
  int inputs_processed = 0;
  for (int i = 0; i < interleave->input_size(); i++) {
    std::string input = interleave->input(i);
    if (IsControlInput(input)) {
      continue;
    }
    if (inputs_processed >= num_inputs_after_rewrite) {
      interleave->set_input(i, absl::StrCat("^", input));
    }
    inputs_processed++;
  }
  if (inputs_processed < num_inputs_after_rewrite) {
    return errors::Internal("Found only ", inputs_processed,
                            " inputs to interleave node ", node_name,
                            ", but expected to find at least ",
                            num_inputs_after_rewrite);
  }

  // Remove extra attributes not in Interleave.
  interleave->mutable_attr()->erase("deterministic");
  interleave->mutable_attr()->erase("sloppy");
  return Status::OK();
}

// Returns true if the ParallelInterleave with the given FunctionDef should be
// rewritten to Interleave. Recursively checks any function attributes of ops
// within the function. "functions_processed" is the list of functions already
// processed, so that the same function is not recursively checked twice.
bool FunctionMayIntroduceNondeterminism(
    const FunctionLibraryDefinition& library, const std::string& function_name,
    absl::flat_hash_set<std::string>* functions_processed) {
  if (functions_processed->contains(function_name)) {
    return false;
  }
  functions_processed->insert(function_name);
  const FunctionDef* function_def = library.Find(function_name);
  if (!function_def) {
    VLOG(2) << "Could not look up function " << function_name
            << " in FunctionLibraryDefinition, so rewriting interleave op to "
               "be safe";
    return true;
  }
  for (const NodeDef& node_def : function_def->node_def()) {
    const OpRegistrationData* op_reg_data = nullptr;
    Status s = library.LookUp(node_def.op(), &op_reg_data);
    if (!s.ok()) {
      VLOG(2) << "Could not look up op " << node_def.op()
              << " in FunctionLibraryDefinition, so rewriting interleave op to "
                 "be safe";
      return true;
    }
    bool is_function_op = op_reg_data->is_function_op;

    // Rewrite nondeterministic stateful ops. Function ops are skipped, since we
    // instead look at the ops within the function.
    if (function_utils::IsNodeStateful(library, node_def,
                                       /*skip_assert=*/false) &&
        !is_function_op && !IsStatefulPartitionedCall((node_def)) &&
        !IsDeterministicStatefulOp(node_def.op())) {
      VLOG(2) << "Will rewrite due to op: " << node_def.op();
      return true;
    }

    // Recursively check for nondeterminism in all function attributes.
    std::vector<std::string> attr_func_names;
    for (const auto& attr : node_def.attr()) {
      if (attr.second.has_func()) {
        attr_func_names.push_back(attr.second.func().name());
      }
      for (const auto& name_attr_list : attr.second.list().func()) {
        attr_func_names.push_back(name_attr_list.name());
      }
    }
    if (is_function_op) {
      attr_func_names.push_back(node_def.name());
    }
    for (const std::string& inner_function_name : attr_func_names) {
      if (FunctionMayIntroduceNondeterminism(library, inner_function_name,
                                             functions_processed)) {
        return true;
      }
    }
  }
  return false;
}

bool FunctionMayIntroduceNondeterminism(
    const FunctionLibraryDefinition& library,
    const std::string& function_name) {
  absl::flat_hash_set<std::string> functions_processed;
  return FunctionMayIntroduceNondeterminism(library, function_name,
                                            &functions_processed);
}

}  // namespace

Status MakeDeterministic::OptimizeAndCollectStats(Cluster* cluster,
                                                  const GrapplerItem& item,
                                                  GraphDef* output,
                                                  OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item.graph.library());

  for (const NodeDef& node : item.graph.node()) {
    if (!IsParallelInterleave(node.op())) {
      continue;
    }
    if (!FunctionMayIntroduceNondeterminism(
            function_library, node.attr().at("f").func().name())) {
      VLOG(1) << "Not rewriting node " << node.name() << " (" << node.op()
              << ") into a non-parallel InterleaveDataset";
      continue;
    }

    VLOG(1) << "Rewriting node " << node.name() << " (" << node.op()
            << ") into a non-parallel InterleaveDataset";
    TF_RETURN_IF_ERROR(ConvertToNonParallelInterleave(node.name(), &graph));
    stats->num_changes++;
  }

  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(MakeDeterministic, "make_deterministic");

}  // namespace grappler
}  // namespace tensorflow

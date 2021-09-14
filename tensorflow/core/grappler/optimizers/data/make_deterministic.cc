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
constexpr char kMapOp[] = "MapDataset";
constexpr char kParallelMapOp[] = "ParallelMapDataset";
constexpr char kMapAndBatchOp[] = "MapAndBatchDataset";

// List of stateful ops which do not introduce nondeterminism when put inside a
// parallel interleave or map dataset. All ops which do not mutate state can be
// put in this list, even if such ops read state. Note that random ops
// implicitly mutate state and cannot be put in this list.
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

bool IsParallelMap(const std::string& op) {
  return data::MatchesAnyVersion(kParallelMapOp, op);
}

bool IsMapAndBatch(const std::string& op) {
  return data::MatchesAnyVersion(kMapAndBatchOp, op);
}

bool IsDeterministicStatefulOp(const std::string& op) {
  for (auto stateful_op : kDeterministicStatefulOps) {
    if (data::MatchesAnyVersion(stateful_op, op)) {
      return true;
    }
  }
  return false;
}

NodeDef* GetMutableNode(const string& node_name, MutableGraphView* graph) {
  int index = graph_utils::FindGraphNodeWithName(node_name, *graph->graph());
  DCHECK_NE(index, -1) << "Failed to find node " << node_name
                       << " in the optimized graph.";
  return graph->graph()->mutable_node(index);
}

// Converts a ParallelInterleaveDataset or ParallelMapDataset to the equivalent
// non-parallel version, to make it deterministic.
Status ConvertToNonParallel(const string& node_name, MutableGraphView* graph) {
  NodeDef* node = GetMutableNode(node_name, graph);

  auto Targuments = node->attr().find("Targuments");
  if (Targuments == node->attr().end()) {
    return errors::Internal("Failed to find Targuments attribute for node ",
                            node_name);
  }

  int num_inputs_after_rewrite;
  if (IsParallelInterleave(node->op())) {
    node->set_op(kInterleaveOp);
    num_inputs_after_rewrite = 3 + Targuments->second.list().type_size();
  } else {
    DCHECK(IsParallelMap(node->op()));
    node->set_op(kMapOp);
    num_inputs_after_rewrite = 1 + Targuments->second.list().type_size();
  }

  // ParallelInterleave and ParallelMap ops take in more inputs than the
  // corresponding non-parallel versions, so turn extra inputs into control
  // inputs. These extra inputs are for performance and are safe to ignore.
  int inputs_processed = 0;
  for (int i = 0; i < node->input_size(); i++) {
    std::string input = node->input(i);
    if (IsControlInput(input)) {
      continue;
    }
    if (inputs_processed >= num_inputs_after_rewrite) {
      node->set_input(i, absl::StrCat("^", input));
    }
    inputs_processed++;
  }
  if (inputs_processed < num_inputs_after_rewrite) {
    return errors::Internal("Found only ", inputs_processed, " inputs to node ",
                            node_name, ", but expected to find at least ",
                            num_inputs_after_rewrite);
  }

  // Remove extra attributes not in Interleave or Map.
  node->mutable_attr()->erase("deterministic");
  node->mutable_attr()->erase("sloppy");
  return Status::OK();
}

Status ConvertMapAndBatch(const string& node_name, MutableGraphView* graph) {
  NodeDef* node = GetMutableNode(node_name, graph);
  auto Targuments = node->attr().find("Targuments");
  if (Targuments == node->attr().end()) {
    return errors::Internal("Failed to find Targuments attribute for node ",
                            node_name);
  }

  int num_parallel_calls_index = 2 + Targuments->second.list().type_size();
  node->add_input(absl::StrCat("^", node->input(num_parallel_calls_index)));
  NodeDef* tmp = graph_utils::AddScalarConstNode<int64_t>(1, graph);
  node->set_input(num_parallel_calls_index, tmp->name());
  return Status::OK();
}

// Returns true if the ParallelInterleave or ParallelMap with the given
// FunctionDef should be rewritten to the non-parallel version. Recursively
// checks any function attributes of ops within the function.
// "functions_processed" is the list of functions already processed, so that the
// same function is not recursively checked twice.
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
            << " in FunctionLibraryDefinition, so rewriting op to be safe";
    return true;
  }
  for (const NodeDef& node_def : function_def->node_def()) {
    const OpRegistrationData* op_reg_data = nullptr;
    Status s = library.LookUp(node_def.op(), &op_reg_data);
    if (!s.ok()) {
      VLOG(2) << "Could not look up op " << node_def.op()
              << " in FunctionLibraryDefinition, so rewriting op to be safe";
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
    if (graph_utils::HasSloppyAttr(node.op())) {
      NodeDef* mutable_node = GetMutableNode(node.name(), &graph);
      (*mutable_node->mutable_attr())["sloppy"].set_b(false);
      stats->num_changes++;
    }
    if (graph_utils::HasDeterministicAttr(node.op())) {
      NodeDef* mutable_node = GetMutableNode(node.name(), &graph);
      (*mutable_node->mutable_attr())["deterministic"].set_s("true");
      stats->num_changes++;
    }
    if (!IsParallelInterleave(node.op()) && !IsParallelMap(node.op()) &&
        !IsMapAndBatch(node.op())) {
      continue;
    }
    if (!FunctionMayIntroduceNondeterminism(
            function_library, node.attr().at("f").func().name())) {
      VLOG(1) << "Not rewriting node " << node.name() << " (" << node.op()
              << ") into the non-parallel version";
      continue;
    }

    if (IsMapAndBatch(node.op())) {
      VLOG(1) << "Changing num_parallel_calls attr of node " << node.name()
              << " (" << node.op() << ") to 1";
      TF_RETURN_IF_ERROR(ConvertMapAndBatch(node.name(), &graph));
    } else {
      VLOG(1) << "Rewriting node " << node.name() << " (" << node.op()
              << ") into the non-parallel version";
      TF_RETURN_IF_ERROR(ConvertToNonParallel(node.name(), &graph));
    }
    stats->num_changes++;
  }

  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(MakeDeterministic, "make_deterministic");

}  // namespace grappler
}  // namespace tensorflow

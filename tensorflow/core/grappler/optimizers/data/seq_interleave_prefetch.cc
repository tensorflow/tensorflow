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

#include "tensorflow/core/grappler/optimizers/data/seq_interleave_prefetch.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/errors.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kInterleaveDatasetOpName[] = "InterleaveDataset";
constexpr char kParallelInterleaveDatasetV2OpName[] =
    "ParallelInterleaveDatasetV2";
constexpr char kParallelInterleaveDatasetV3OpName[] =
    "ParallelInterleaveDatasetV3";
constexpr char kParallelInterleaveDatasetV4OpName[] =
    "ParallelInterleaveDatasetV4";
constexpr char kParallelInterleaveDatasetOpName[] = "ParallelInterleaveDataset";
constexpr char kPrefetchDatasetOpName[] = "PrefetchDataset";
constexpr char kDatasetStr[] = "Dataset";
constexpr char kConstOpName[] = "Const";
constexpr char kOutputShapes[] = "output_shapes";
constexpr char kOutputTypes[] = "output_types";
constexpr char kConstNodeOutputSuffix[] = ":output:0";
constexpr char kDatasetNodeOutputSuffix[] = ":handle:0";
constexpr char kDeterministicAttr[] = "deterministic";
constexpr char kFunctionAttr[] = "f";
constexpr char kDTypeAttr[] = "dtype";
constexpr char kValueAttr[] = "value";
constexpr char kTArgumentsAttr[] = "Targuments";
constexpr char kOutputTypesAttr[] = "output_types";
constexpr char kMetadataAttr[] = "metadata";
constexpr char kOutputShapesAttr[] = "output_shapes";
constexpr char kTOutputTypesAttr[] = "Toutput_types";
constexpr char kSeqInterleavePrefetchRewritePrefix[] =
    "inject/seq_interleave_prefetch_rewrite_";

//
// Steps involved in rewrite:
//
// For every deterministic parallel interleave node,
//    1. Create interleave node and set the `interleave_fn` function same as the
//    `interleave_fn` in parallel interleave node.
//          - Update fan outs in the top level graph.
//          - Delete parallel interleave nodes and its unused input nodes.
//    2. Create a prefetch node with 'input set to (input of Identity node in
//    FLD of the graph)
//          - From the signature of 'f', find the output node (Identity node).
//          - Find the input of this output node and set it as input of Prefetch
//            node.
//.         - Add prefetch and its input nodes to the FunctionDef.
//          - Update fan outs of prefetch node.
//

bool IsParallelInterleave(const std::string& op) {
  return data::MatchesAnyVersion(kParallelInterleaveDatasetOpName, op);
}

// Returns the number of inputs accepted by the parallel interleave op as per
// the version excluding the `other_arguments` input.
int GetNumInputsForParallelInterleaveOp(const std::string& op) {
  if (op == kParallelInterleaveDatasetV2OpName) {
    return 4;
  } else if (op == kParallelInterleaveDatasetV3OpName) {
    return 4;
  } else if (op == kParallelInterleaveDatasetV4OpName) {
    return 6;
  }
  return 0;
}

// Check if op type of `node` has "Dataset" suffix.
bool NodeOpHasDatasetSuffix(const NodeDef& node) {
  return absl::EndsWith(node.op(), kDatasetStr);
}

// Returns true if there is at least one function node with dataset op.
bool DatasetOpInFunction(const NodeDef& node, const FunctionDef* fn) {
  for (const auto& node : fn->node_def()) {
    if (NodeOpHasDatasetSuffix(node)) {
      return true;
    }
  }
  return false;
}

// A node is eligible for rewrite if it is a deterministic parallel interleave
// node and has a function node creating `Dataset`.
bool RewritePossibleForNode(const NodeDef& node,
                            const FunctionLibraryDefinition& fld) {
  auto is_deterministic_parallel_interleave_node = [&]() -> bool {
    if (!IsParallelInterleave(node.op())) return false;
    auto determinism_value = node.attr().find(kDeterministicAttr);
    return (determinism_value != node.attr().end()) &&
           (determinism_value->second.s() == "true");
  };

  if (node.attr().count(kFunctionAttr) == 0) return false;
  const FunctionDef* fn = fld.Find(node.attr().at(kFunctionAttr).func().name());

  if (fn == nullptr) return false;
  if (fn->signature().output_arg_size() != 1) return false;
  if (is_deterministic_parallel_interleave_node()) {
    return DatasetOpInFunction(node, fn);
  }

  return false;
}

NodeDef CreateBufferSizeNode(DataType dtype,
                             const std::function<void(TensorProto*)>& add_value,
                             MutableGraphView* graph, FunctionDef& fdef) {
  NodeDef node;
  node.set_op(kConstOpName);
  function_utils::SetUniqueFunctionNodeName(
      absl::StrCat(kSeqInterleavePrefetchRewritePrefix, "buffer_size"), &fdef,
      &node);

  (*node.mutable_attr())[kDTypeAttr].set_type(dtype);
  auto tensor = std::make_unique<tensorflow::TensorProto>();
  auto tensor_shape = std::make_unique<tensorflow::TensorShapeProto>();
  tensor->set_allocated_tensor_shape(tensor_shape.release());
  tensor->set_dtype(dtype);
  add_value(tensor.get());
  (*node.mutable_attr())[kValueAttr].set_allocated_tensor(tensor.release());

  return node;
}

absl::Status CreateAndAppendPrefetchNode(MutableGraphView* graph,
                                         FunctionDef& fdef) {
  auto get_last_dataset_op_node = [&]() -> const NodeDef* {
    // Find the input node of fdef's ret value.
    const auto& output_arg = fdef.signature().output_arg(0).name();
    const auto& ret_val = fdef.ret().at(output_arg);
    auto input = function_utils::FunctionDefTensorDesc(ret_val);
    // Walk from output to input and find the first eligible node.
    const NodeDef* dataset_op_node = nullptr;
    while (
        function_utils::ContainsFunctionNodeWithName(input.node_name, fdef)) {
      int idx = function_utils::FindFunctionNodeWithName(input.node_name, fdef);
      const NodeDef& node = fdef.node_def(idx);
      if (NodeOpHasDatasetSuffix(node)) {
        dataset_op_node = &node;
        break;
      }
      input = function_utils::FunctionDefTensorDesc(node.input(0));
    }
    return dataset_op_node;
  };

  // 1. Find the position for the `prefetch` node.
  const NodeDef* add_after = get_last_dataset_op_node();
  if (add_after == nullptr) {
    return errors::NotFound(
        "Could not find any dataset node to append `Prefetch` at its output in "
        "`seq_interleave_prefetch` rewrite");
  }

  // 2. Create prefetch node.
  NodeDef prefetch_node;
  prefetch_node.set_op(kPrefetchDatasetOpName);
  function_utils::SetUniqueFunctionNodeName(
      absl::StrCat(kSeqInterleavePrefetchRewritePrefix,
                   fdef.signature().name()),
      &fdef, &prefetch_node);

  // 3. Construct argument nodes.
  const auto input_dataset =
      absl::StrCat(add_after->name(), kDatasetNodeOutputSuffix);
  NodeDef buffer_size_node = CreateBufferSizeNode(
      DT_INT64,
      [](TensorProto* proto) { proto->add_int64_val(data::model::kAutotune); },
      graph, fdef);

  // 4. Add inputs to prefetch nodes.
  prefetch_node.add_input(input_dataset);
  prefetch_node.add_input(
      absl::StrCat(buffer_size_node.name(), kConstNodeOutputSuffix));

  // 5. Set other attributes of prefetch node.
  if (add_after->attr().count(kOutputShapes) > 0) {
    graph_utils::CopyAttribute(kOutputShapes, *add_after, &prefetch_node);
  } else {
    tensorflow::TensorShapeProto* shape =
        (*(prefetch_node.mutable_attr()))[kOutputShapes]
            .mutable_list()
            ->add_shape();
    shape->set_unknown_rank(true);
  }

  if (add_after->attr().count(kOutputTypes) > 0) {
    graph_utils::CopyAttribute(kOutputTypes, *add_after, &prefetch_node);
  } else if (add_after->attr().count(kTOutputTypesAttr) > 0) {
    (*(prefetch_node.mutable_attr()))[kOutputTypes] =
        add_after->attr().at(kTOutputTypesAttr);
  } else {
    (*(prefetch_node.mutable_attr()))[kOutputTypes].mutable_list()->add_type(
        tensorflow::DataType::DT_STRING);
  }

  // 6. Update fanouts.
  std::string old_input = input_dataset;
  std::string new_input =
      absl::StrCat(prefetch_node.name(), kDatasetNodeOutputSuffix);
  function_utils::ReplaceReferences(old_input, new_input, &fdef);

  // 7. Add `prefetch` and its argument nodes to `fdef`.
  *fdef.add_node_def() = std::move(prefetch_node);
  *fdef.add_node_def() = std::move(buffer_size_node);

  return absl::OkStatus();
}

absl::Status AddInterleaveNode(MutableGraphView* graph,
                               const NodeDef& parallel_interleave_node,
                               const std::string& interleave_map_func_name,
                               absl::flat_hash_set<string>& nodes_to_delete) {
  NodeDef interleave_node;
  interleave_node.set_op(kInterleaveDatasetOpName);
  graph_utils::SetUniqueGraphNodeName(
      absl::StrCat(kSeqInterleavePrefetchRewritePrefix,
                   parallel_interleave_node.name()),
      graph->graph(), &interleave_node);

  // Inputs to interleave node passed from parallel interleave node would
  // comprise of `input_dataset`, `other_arguments`, `cycle_length`, and
  // `block_length`.
  int num_other_args =
      parallel_interleave_node.input_size() -
      GetNumInputsForParallelInterleaveOp(parallel_interleave_node.op());
  int inputs_from_parallel_interleave = 1 /* input_dataset */ + num_other_args +
                                        1 /* cycle_length */ +
                                        1 /* block_length */;
  for (int i = 0; i < inputs_from_parallel_interleave; ++i) {
    interleave_node.add_input(parallel_interleave_node.input(i));
  }

  // Copy attributes.
  if (parallel_interleave_node.attr().contains(kTArgumentsAttr)) {
    graph_utils::CopyAttribute(kTArgumentsAttr, parallel_interleave_node,
                               &interleave_node);
  }
  if (parallel_interleave_node.attr().contains(kOutputTypesAttr)) {
    graph_utils::CopyAttribute(kOutputTypesAttr, parallel_interleave_node,
                               &interleave_node);
  }
  if (parallel_interleave_node.attr().contains(kOutputShapesAttr)) {
    graph_utils::CopyAttribute(kOutputShapesAttr, parallel_interleave_node,
                               &interleave_node);
  }
  if (parallel_interleave_node.attr().contains(kMetadataAttr)) {
    graph_utils::CopyAttribute(kMetadataAttr, parallel_interleave_node,
                               &interleave_node);
  }

  // Set the interleave function attr to the same function as in parallel
  // interleave.
  const auto& parallel_interleave_fn_attr =
      parallel_interleave_node.attr().at(kFunctionAttr);
  (*interleave_node.mutable_attr())[kFunctionAttr] =
      parallel_interleave_fn_attr;
  (*interleave_node.mutable_attr())[kFunctionAttr].mutable_func()->set_name(
      interleave_map_func_name);

  // Copy shapes and types attributes.
  graph_utils::CopyShapesAndTypesAttrs(parallel_interleave_node,
                                       &interleave_node);

  // Copy experimental types.
  *interleave_node.mutable_experimental_type() =
      parallel_interleave_node.experimental_type();

  // Add new node into graph and update edges
  NodeDef* new_node_graph = graph->AddNode(std::move(interleave_node));
  TF_RETURN_IF_ERROR(graph->UpdateFanouts(parallel_interleave_node.name(),
                                          new_node_graph->name()));

  // Delete the parallel interleave node.
  nodes_to_delete.insert(parallel_interleave_node.name());
  return absl::OkStatus();
}
}  // namespace

absl::Status SeqInterleavePrefetch::OptimizeAndCollectStats(
    Cluster* cluster, const GrapplerItem& item, GraphDef* output,
    OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;
  FunctionLibraryDefinition fld(OpRegistry::Global(), item.graph.library());

  for (const NodeDef& node : item.graph.node()) {
    if (!RewritePossibleForNode(node, fld)) continue;
    // Find the parallel_interleave_node's `map_func`.
    const FunctionDef* parallel_interleave_fn =
        fld.Find(node.attr().at("f").func().name());
    FunctionDef interleave_fn(*parallel_interleave_fn);
    interleave_fn.mutable_signature()->set_name(
        absl::StrCat(kSeqInterleavePrefetchRewritePrefix,
                     parallel_interleave_fn->signature().name()));
    // Replace the parallel interleave node with interleave.
    TF_RETURN_IF_ERROR(AddInterleaveNode(
        &graph, node, interleave_fn.signature().name(), nodes_to_delete));
    // Create and append the prefetch node to the interleave_fn.
    TF_RETURN_IF_ERROR(CreateAndAppendPrefetchNode(&graph, interleave_fn));
    // Replace the `parallel_interleave_fn` with `interleave_fn`.
    TF_RETURN_IF_ERROR(fld.ReplaceFunction(
        parallel_interleave_fn->signature().name(), interleave_fn));
    stats->num_changes++;
  }

  // Update the `FunctionDefLibrary` of the optimized graph.
  *output->mutable_library() = fld.ToProto();
  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));

  return absl::OkStatus();
}

REGISTER_GRAPH_OPTIMIZER_AS(SeqInterleavePrefetch, "seq_interleave_prefetch");

}  // namespace grappler
}  // namespace tensorflow

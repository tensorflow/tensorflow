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

#include "tensorflow/core/grappler/optimizers/data/map_vectorization.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/optimizers/data/vectorization_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kCastOp[] = "Cast";
constexpr char kRealDivOp[] = "RealDiv";
constexpr char kSubOp[] = "Sub";
constexpr char kMulOp[] = "Mul";
constexpr char kAddOp[] = "Add";
constexpr char kEqualOp[] = "Equal";
constexpr char kCeilOp[] = "Ceil";
constexpr char kBatchOp[] = "BatchDataset";
constexpr char kBatchV2Op[] = "BatchDatasetV2";
constexpr char kExperimentalMapAndBatchOp[] = "ExperimentalMapAndBatchDataset";
constexpr char kMapAndBatchOp[] = "MapAndBatchDataset";
constexpr char kMapOp[] = "MapDataset";
constexpr char kParallelMapOp[] = "ParallelMapDataset";
constexpr char kParallelMapV2Op[] = "ParallelMapDatasetV2";
constexpr char kChooseFastestOp[] = "ChooseFastestBranchDataset";
constexpr char kPrefetchOp[] = "PrefetchDataset";

// Returns a FunctionDef containing a MapDefun op that wraps the original
// function.
FunctionDef* CreateMapDefunWrapper(const NodeDef& map_node,
                                   const FunctionDef& orig_func,
                                   FunctionDefLibrary* library) {
  FunctionDef* vectorized_func = library->add_function();
  // Function inputs and outputs are the same as original, just
  // with different shapes.
  *vectorized_func->mutable_signature() = orig_func.signature();
  graph_utils::SetUniqueGraphFunctionName("naively_vectorized_fn", library,
                                          vectorized_func);

  // Add MapDefun node
  NodeDef* map_defun_node = vectorized_func->mutable_node_def()->Add();
  map_defun_node->set_op("MapDefun");
  function_utils::SetUniqueFunctionNodeName(map_defun_node->op(),
                                            vectorized_func, map_defun_node);

  // Set attrs and inputs
  for (const string& k : {"f", "output_types", "output_shapes"}) {
    // Function, output types and (unbatched) shapes are the same as the
    // original map node.
    graph_utils::CopyAttribute(k, map_node, map_defun_node);
  }

  // Note that the inputs to the function are either regular arguments (for
  // which the function is mapped across their 0th dimension) or captured inputs
  // (for which the function takes the argument wholesale). We can infer
  // the split between these arguments from the `map_node`'s attrs.
  // The Targuments attr on `map_node` corresponds to a list of types of
  // MapDataset's captured inputs.
  auto t_captured = map_node.attr().at("Targuments");

  // Get types of input arguments from original map function
  DataTypeVector t_args;  // Regular arguments
  for (const auto& input : vectorized_func->signature().input_arg()) {
    t_args.push_back(input.type());
    map_defun_node->add_input(input.name());
  }
  // Erase the captured arguments from Targuments
  t_args.erase(t_args.end() - t_captured.list().type_size(), t_args.end());
  AddNodeAttr("Targuments", t_args, map_defun_node);
  AddNodeAttr("Tcaptured", t_captured, map_defun_node);

  // Set return values to match output names
  string output_prefix = strings::StrCat(map_defun_node->name(), ":output:");
  for (size_t i = 0; i < vectorized_func->signature().output_arg_size(); ++i) {
    const auto& output_arg = vectorized_func->signature().output_arg(i);
    (*vectorized_func->mutable_ret())[output_arg.name()] =
        strings::StrCat(output_prefix, i);
  }

  return vectorized_func;
}

FunctionDef* AddVectorizedFunction(const NodeDef& map_node,
                                   const FunctionDef& orig_func,
                                   FunctionDefLibrary* library) {
  // Vectorizes orig_func naively by wrapping in a MapDefun op, then performing
  // efficient vectorization with VectorizeMapDefun.
  FunctionDef* vectorized_func =
      CreateMapDefunWrapper(map_node, orig_func, library);
  const NodeDef& map_defun_node = vectorized_func->node_def(0);
  DCHECK_EQ(map_defun_node.op(), "MapDefun");

  FunctionDef* result;
  Status s = vectorization_utils::VectorizeMapDefun(
      *vectorized_func, map_defun_node, library, &result);

  if (!s.ok()) {
    LOG(WARNING) << "VectorizeMapDefun failed. The function will only be "
                    "naively vectorized with MapDefun. Reason: "
                 << s;
    return vectorized_func;
  }
  return result;
}

bool IsOutputShapesFullyDefined(const NodeDef& node) {
  auto* shapes_attr = gtl::FindOrNull(node.attr(), "output_shapes");
  if (shapes_attr == nullptr) return false;
  const auto& shapes = shapes_attr->list().shape();

  for (const TensorShapeProto& shape : shapes) {
    if (shape.unknown_rank()) return false;
    for (const auto& dim : shape.dim()) {
      if (dim.size() == -1) {
        return false;
      }
    }
  }
  return true;
}

// Returns a mapping from input names to the [start, end) indices of the input
// in the node's input list.
Status GetInputMap(const NodeDef& node, NameRangeMap* result) {
  const OpRegistrationData* op_reg_data;  // Owned by global op registry
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUp(node.op(), &op_reg_data));

  return NameRangesForNode(node, op_reg_data->op_def, result,
                           /*outputs=*/nullptr);
}

Status CopyInputs(StringPiece input_name, const NameRangeMap& input_map,
                  const NodeDef& from, NodeDef* to) {
  const auto* range = gtl::FindOrNull(input_map, input_name);
  if (range == nullptr) {
    return errors::Internal(
        "Failed to copy inputs: did not find inputs with name: ", input_name,
        ", in node with name: ", from.name());
  }
  for (int i = range->first; i < range->second; ++i) {
    to->add_input(from.input(i));
  }

  return Status::OK();
}

Status GetInputNodeName(StringPiece input_name, const NameRangeMap& input_map,
                        const NodeDef& node, string* result) {
  const auto* range = gtl::FindOrNull(input_map, input_name);
  if (range == nullptr) {
    return errors::Internal(
        "Failed to get input node name: did not find input with name: ",
        input_name, ", in node with name: ", node.name());
  }
  if (range->second - range->first > 1) {
    return errors::Internal("Tried to get single input name for a list input.");
  }
  *result = node.input(range->first);
  return Status::OK();
}

Status AddNewBatchNode(const NodeDef& old_batch_node, const NodeDef& input_node,
                       const FunctionDef& vectorized_func,
                       MutableGraphView* graph, NodeDef** new_batch_node) {
  NodeDef batch_node;
  batch_node.set_op(old_batch_node.op() == kBatchOp ? kBatchOp : kBatchV2Op);
  graph_utils::SetUniqueGraphNodeName(batch_node.op(), graph->graph(),
                                      &batch_node);

  // Set the `input_dataset` input argument
  batch_node.add_input(input_node.name());

  NameRangeMap input_map;
  TF_RETURN_IF_ERROR(GetInputMap(old_batch_node, &input_map));

  // Set the `batch_size` input argument
  TF_RETURN_IF_ERROR(
      CopyInputs("batch_size", input_map, old_batch_node, &batch_node));

  // Set the `drop_remainder` input argument
  if (batch_node.op() != kBatchOp) {
    TF_RETURN_IF_ERROR(
        CopyInputs("drop_remainder", input_map, old_batch_node, &batch_node));
  }

  // Set attrs
  AttrValue output_types;
  for (const auto& input : vectorized_func.signature().input_arg()) {
    output_types.mutable_list()->add_type(input.type());
  }
  (*batch_node.mutable_attr())["output_types"] = output_types;

  // It is safe to assume that input_node has the "output_shapes" attr here,
  // because earlier we checked that the input node has fully defined output
  // shapes.
  auto& output_shapes_attr = (*batch_node.mutable_attr())["output_shapes"];
  const auto& input_shapes =
      input_node.attr().at("output_shapes").list().shape();

  int64 batch_size = -1;
  for (const auto& shape :
       old_batch_node.attr().at("output_shapes").list().shape()) {
    if (!shape.unknown_rank()) {
      batch_size = shape.dim(0).size();
      break;
    }
  }

  for (size_t i = 0; i < input_shapes.size(); ++i) {
    // Note: We already checked earlier that input shapes are all fully defined.
    TensorShapeProto* shape = output_shapes_attr.mutable_list()->add_shape();
    TensorShapeProto_Dim* dim = shape->add_dim();
    dim->set_size(batch_size);
    shape->MergeFrom(input_shapes.Get(i));
  }

  *new_batch_node = graph->AddNode(std::move(batch_node));
  return Status::OK();
}

Status AddNewMapNode(const NodeDef& old_map_node, const NodeDef& old_batch_node,
                     const NodeDef& new_batch_node,
                     const FunctionDef& vectorized_func,
                     MutableGraphView* graph, NodeDef** new_map_node) {
  NodeDef map_node;
  if (old_map_node.op() == kMapOp) {
    map_node.set_op(kMapOp);
  } else if (old_map_node.op() == kParallelMapOp) {
    map_node.set_op(kParallelMapOp);
  } else {
    map_node.set_op(kParallelMapV2Op);
  }
  graph_utils::SetUniqueGraphNodeName(map_node.op(), graph->graph(), &map_node);

  // Set the `input_dataset` input argument
  map_node.add_input(new_batch_node.name());

  NameRangeMap input_map;
  TF_RETURN_IF_ERROR(GetInputMap(old_map_node, &input_map));

  // Set the `other_arguments` input argument
  TF_RETURN_IF_ERROR(
      CopyInputs("other_arguments", input_map, old_map_node, &map_node));

  // Set the `num_parallel_calls` input argument
  if (map_node.op() == kParallelMapOp) {
    // `num_parallel_calls` = `kAutotune`
    // TODO(rachelim): Evaluate the performance of other potential
    // transformations to `num_parallel_calls`,
    // e.g. ceil(old num_parallel_calls // batch size)
    auto autotune_val = graph_utils::AddScalarConstNode(
        static_cast<int32>(data::model::kAutotune), graph);
    map_node.add_input(autotune_val->name());
  } else if (map_node.op() == kParallelMapV2Op) {
    auto autotune_val =
        graph_utils::AddScalarConstNode(data::model::kAutotune, graph);
    map_node.add_input(autotune_val->name());
  }

  // Set attrs
  graph_utils::CopyAttribute("Targuments", old_map_node, &map_node);
  auto& func_attr = (*map_node.mutable_attr())["f"];
  func_attr.mutable_func()->set_name(vectorized_func.signature().name());

  for (auto key : {"output_shapes", "output_types"}) {
    graph_utils::CopyAttribute(key, old_batch_node, &map_node);
  }

  (*map_node.mutable_attr())["use_inter_op_parallelism"].set_b(true);
  if (old_map_node.attr().contains("sloppy")) {
    graph_utils::CopyAttribute("sloppy", old_map_node, &map_node);
  }
  if (old_map_node.attr().contains("deterministic")) {
    graph_utils::CopyAttribute("deterministic", old_map_node, &map_node);
  }
  *new_map_node = graph->AddNode(std::move(map_node));
  return Status::OK();
}

Status AddNewPrefetchNode(const NodeDef& old_prefetch_node,
                          const NodeDef& old_batch_node,
                          const NodeDef& new_map_node, MutableGraphView* graph,
                          NodeDef** new_prefetch_node) {
  NodeDef prefetch_node;
  prefetch_node.set_op(kPrefetchOp);
  graph_utils::SetUniqueGraphNodeName(kPrefetchOp, graph->graph(),
                                      &prefetch_node);

  // `input_dataset`
  prefetch_node.add_input(new_map_node.name());

  // `buffer_size` == `kAutotune`
  // TODO(rachelim): Evaluate the performance of other potential transformations
  // to `buffer_size`, e.g. ceil(old buffer size // batch size)
  auto autotune_val = graph_utils::AddScalarConstNode(
      static_cast<int64>(data::model::kAutotune), graph);
  prefetch_node.add_input(autotune_val->name());

  for (const auto& key : {"output_shapes", "output_types"}) {
    graph_utils::CopyAttribute(key, new_map_node, &prefetch_node);
  }

  *new_prefetch_node = graph->AddNode(std::move(prefetch_node));
  return Status::OK();
}

Status AddBranch(gtl::ArraySlice<const NodeDef*> branch,
                 NodeDef* choose_fastest_node, DataTypeVector* t_arguments,
                 std::vector<NameAttrList>* branches,
                 std::vector<int>* other_arguments_lengths,
                 FunctionDefLibrary* library) {
  FunctionDef* branch_func = library->add_function();
  auto* signature = branch_func->mutable_signature();
  graph_utils::SetUniqueGraphFunctionName("branch", library, branch_func);

  // Input dataset.
  string prev_node_output = "args_0";
  auto* input_arg_0 = signature->add_input_arg();
  input_arg_0->set_name(prev_node_output);
  input_arg_0->set_type(DT_VARIANT);

  auto* output_arg = signature->add_output_arg();
  output_arg->set_name("output");
  output_arg->set_type(DT_VARIANT);

  int32 captured_arg_lengths = 0;

  // For each node in the branch, copy it to the function def. Add the
  // corresponding non-0th inputs as captured arguments, modifying the function
  // input signature, node input names, other_arguments_lengths, and t_arguments
  // accordingly.
  for (const NodeDef* node : branch) {
    // Copy the node to the function
    auto function_node = branch_func->add_node_def();
    *function_node = *node;
    function_utils::SetUniqueFunctionNodeName(node->name(), branch_func,
                                              function_node);
    function_node->clear_input();
    function_node->add_input(prev_node_output);

    // Every input besides the 0th (dataset) becomes a captured argument.
    int input_size = node->input_size();
    DataTypeVector input_types;
    const OpDef* op_def;
    TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(node->op(), &op_def));
    TF_RETURN_IF_ERROR(InputTypesForNode(*node, *op_def, &input_types));
    DCHECK_EQ(input_types.size(), input_size);

    for (int i = 1; i < input_size; ++i) {
      // Capture input in `other_arguments`
      choose_fastest_node->add_input(node->input(i));
      // Add type to function signature
      auto* input_arg = signature->add_input_arg();

      string input_arg_name = strings::StrCat(function_node->name(), "_", i);
      input_arg->set_name(input_arg_name);
      input_arg->set_type(input_types[i]);
      function_node->add_input(input_arg_name);
    }
    // Add to `Targuments`
    t_arguments->reserve(t_arguments->size() + input_types.size() - 1);
    t_arguments->insert(t_arguments->end(), input_types.begin() + 1,
                        input_types.end());
    captured_arg_lengths += input_size - 1;
    prev_node_output = strings::StrCat(function_node->name(), ":handle:0");
  }

  // Add to `other_arguments_lengths`
  other_arguments_lengths->push_back(captured_arg_lengths);
  (*branch_func->mutable_ret())["output"] = prev_node_output;

  // Add to `branches`
  NameAttrList func_attr;
  func_attr.set_name(branch_func->signature().name());
  branches->push_back(std::move(func_attr));
  return Status::OK();
}

Status AddNewChooseFastestNode(const NodeDef* input_dataset_node,
                               const string& ratio_numerator_name,
                               std::vector<const NodeDef*> original_branch,
                               std::vector<const NodeDef*> vectorized_branch,
                               MutableGraphView* graph,
                               FunctionDefLibrary* library,
                               NodeDef** new_choose_fastest_node) {
  NodeDef choose_fastest_node;
  choose_fastest_node.set_op(kChooseFastestOp);
  graph_utils::SetUniqueGraphNodeName(choose_fastest_node.op(), graph->graph(),
                                      &choose_fastest_node);

  // input_dataset
  choose_fastest_node.add_input(input_dataset_node->name());
  choose_fastest_node.add_input(ratio_numerator_name);
  // ratio_denominator == 1
  auto ratio_denominator =
      graph_utils::AddScalarConstNode(static_cast<int64>(1), graph);
  choose_fastest_node.add_input(ratio_denominator->name());

  DataTypeVector t_arguments;
  std::vector<NameAttrList> branches;
  std::vector<int32> other_arguments_lengths;
  // Branch 0: vectorized branch
  TF_RETURN_IF_ERROR(AddBranch(vectorized_branch, &choose_fastest_node,
                               &t_arguments, &branches,
                               &other_arguments_lengths, library));
  // Branch 1: original branch
  TF_RETURN_IF_ERROR(AddBranch(original_branch, &choose_fastest_node,
                               &t_arguments, &branches,
                               &other_arguments_lengths, library));

  DCHECK_EQ(t_arguments.size(), choose_fastest_node.input_size() - 3);
  DCHECK_EQ(branches.size(), other_arguments_lengths.size());

  AddNodeAttr("Targuments", t_arguments, &choose_fastest_node);
  AddNodeAttr("num_elements_per_branch", 100, &choose_fastest_node);
  AddNodeAttr("branches", branches, &choose_fastest_node);
  AddNodeAttr("other_arguments_lengths", other_arguments_lengths,
              &choose_fastest_node);

  for (auto key : {"output_shapes", "output_types"}) {
    graph_utils::CopyAttribute(key,
                               *vectorized_branch[vectorized_branch.size() - 1],
                               &choose_fastest_node);
  }

  *new_choose_fastest_node = graph->AddNode(std::move(choose_fastest_node));
  return Status::OK();
}

// Given an input pipeline graph and a query node, tries to match the node to
// the 'batch' node in a input_dataset->map->(optional prefetch->)batch pattern,
// or the 'map_and_batch' node in an input_dataset->map_and_batch pattern.
bool FindMapAndBatchPattern(const MutableGraphView& graph, const NodeDef& node,
                            const FunctionLibraryDefinition& function_library,
                            const NodeDef** batch_node_output,
                            const NodeDef** optional_prefetch_node_output,
                            const NodeDef** map_node_output,
                            const NodeDef** input_node_output,
                            const FunctionDef** map_fn_output) {
  const FunctionDef*& map_fn = *map_fn_output;
  const NodeDef*& batch_node = *batch_node_output;
  const NodeDef*& optional_prefetch_node = *optional_prefetch_node_output;
  const NodeDef*& map_node = *map_node_output;
  const NodeDef*& input_node = *input_node_output;

  if (node.op() == kMapAndBatchOp || node.op() == kExperimentalMapAndBatchOp) {
    batch_node = &node;
    map_node = &node;
  } else if (node.op() == kBatchOp || node.op() == kBatchV2Op) {
    batch_node = &node;
    auto tmp_input_node = graph_utils::GetInputNode(*batch_node, graph);
    if (tmp_input_node->op() == kPrefetchOp) {
      optional_prefetch_node = tmp_input_node;
      tmp_input_node = graph_utils::GetInputNode(*tmp_input_node, graph);
    }
    if (tmp_input_node->op() != kMapOp &&
        tmp_input_node->op() != kParallelMapOp &&
        tmp_input_node->op() != kParallelMapV2Op) {
      return false;
    }
    map_node = tmp_input_node;
    if (!IsOutputShapesFullyDefined(*map_node)) {
      // If any of the map func outputs have an unknown shape, don't
      // optimize, so that batching errors surface as before.
      VLOG(1) << "Cannot vectorize dataset.map().batch() because the map "
                 "dataset does not have fully defined output shapes.";
      return false;
    }
  } else {
    return false;
  }

  // Input to the map node
  input_node = graph_utils::GetInputNode(*map_node, graph);
  DCHECK_NE(input_node, nullptr);

  if (!IsOutputShapesFullyDefined(*input_node)) {
    // If any of the inputs have an unknown shape, don't optimize, since
    // inputs might not be batchable.
    VLOG(1) << "Cannot vectorize dataset.map().batch() because the input "
               "dataset does not have fully defined output shapes.";
    return false;
  }

  map_fn = function_library.Find(map_node->attr().at("f").func().name());

  if (function_utils::IsFunctionStateful(function_library, *map_fn)) {
    VLOG(1) << "Cannot vectorize dataset.map().batch() because the map "
               "function is stateful.";
    return false;
  }

  return true;
}

}  // namespace

Status MapVectorization::OptimizeAndCollectStats(Cluster* cluster,
                                                 const GrapplerItem& item,
                                                 GraphDef* output,
                                                 OptimizationStats* stats) {
  *output = item.graph;
  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;

  FunctionDefLibrary* library = output->mutable_library();

  for (const NodeDef& node : item.graph.node()) {
    FunctionLibraryDefinition function_library(OpRegistry::Global(), *library);
    const NodeDef* map_node;
    const NodeDef* optional_prefetch_node = nullptr;
    const NodeDef* batch_node;
    const NodeDef* input_node;
    const FunctionDef* map_func;
    if (!FindMapAndBatchPattern(graph, node, function_library, &batch_node,
                                &optional_prefetch_node, &map_node, &input_node,
                                &map_func)) {
      continue;
    }

    FunctionDef* vectorized_func =
        AddVectorizedFunction(*map_node, *map_func, library);
    CHECK_NOTNULL(vectorized_func);

    NodeDef* new_batch_node;
    TF_RETURN_IF_ERROR(AddNewBatchNode(
        *batch_node, *input_node, *vectorized_func, &graph, &new_batch_node));

    NodeDef* new_map_node;
    TF_RETURN_IF_ERROR(AddNewMapNode(*map_node, *batch_node, *new_batch_node,
                                     *vectorized_func, &graph, &new_map_node));

    NodeDef* optional_new_prefetch_node = nullptr;
    if (optional_prefetch_node) {
      // If the original pipeline was .map().prefetch().batch(), the new
      // pipeline is .batch().map().prefetch()
      TF_RETURN_IF_ERROR(AddNewPrefetchNode(*optional_prefetch_node,
                                            *batch_node, *new_map_node, &graph,
                                            &optional_new_prefetch_node));
    }

    std::vector<const NodeDef*> vectorized_branch(
        {new_batch_node, new_map_node});

    std::vector<const NodeDef*> original_branch({map_node});
    if (optional_prefetch_node) {
      original_branch.push_back(optional_prefetch_node);
      vectorized_branch.push_back(optional_new_prefetch_node);
    }
    if (batch_node->op() != kMapAndBatchOp &&
        batch_node->op() != kExperimentalMapAndBatchOp) {
      original_branch.push_back(batch_node);
    }

    // Mark the original nodes for deletion.
    for (const auto& n : original_branch) {
      nodes_to_delete.insert(n->name());
    }

    if (use_choose_fastest_) {
      // Optionally, use ChooseFastestBranch node to mitigate potential
      // regressions caused by vectorization.
      for (const auto& n : vectorized_branch) {
        // Mark the vectorized nodes for deletion, since they will be added in
        // the choose fastest dataset branch function separately.
        nodes_to_delete.insert(n->name());
      }
      NodeDef* new_choose_fastest_node;
      TF_RETURN_IF_ERROR(AddNewChooseFastestNode(
          input_node, /*ratio_numerator_name=*/new_batch_node->input(1),
          std::move(original_branch), std::move(vectorized_branch), &graph,
          library, &new_choose_fastest_node));
      // Make output of Batch point to ChooseFastest instead.
      TF_RETURN_IF_ERROR(graph.UpdateFanouts(batch_node->name(),
                                             new_choose_fastest_node->name()));

    } else {
      // Make output of Batch point to the new Map (or Prefetch) node instead.
      TF_RETURN_IF_ERROR(graph.UpdateFanouts(
          batch_node->name(), optional_new_prefetch_node
                                  ? optional_new_prefetch_node->name()
                                  : new_map_node->name()));
    }

    TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
    stats->num_changes++;
  }
  return Status::OK();
}

void MapVectorization::Feedback(Cluster* cluster, const GrapplerItem& item,
                                const GraphDef& optimize_output,
                                double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(MapVectorization, "map_vectorization");

}  // namespace grappler
}  // namespace tensorflow

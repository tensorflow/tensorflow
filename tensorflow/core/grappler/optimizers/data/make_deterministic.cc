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

#include <algorithm>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/optimizers/data/split_utils.h"
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
constexpr char kParallelMapOpV2[] = "ParallelMapDatasetV2";
constexpr char kMapAndBatchOp[] = "MapAndBatchDataset";
constexpr char kBatchOp[] = "BatchDataset";
constexpr char kBatchV2Op[] = "BatchDatasetV2";
constexpr char kParallelBatchOp[] = "ParallelBatchDataset";
constexpr char kPrefetchOp[] = "PrefetchDataset";

// List of stateful ops which do not introduce nondeterminism when run as part
// of a Dataset function, e.g. within an InterleaveDataset's function. These are
// stateful dataset ops which do not read or modify TensorFlow state. Stateful
// ops not in this list can introduce nondeterminism, either due to the fact
// they are run in parallel (e.g. in a MapDataset with num_parallel_calls > 1)
// or because they can run asynchronously (e.g. a PrefetchDataset can cause ops
// in a MapDataset to run at the same time as ops outside a dataset).
//
// Ops in this list are allowed to read from files, as we do not make any
// guarantees on determinism if files are modified while a dataset is running.
// TODO(reedwm): Expand this list.
constexpr std::array<const char*, 9> kDeterministicStatefulOps = {
    "TextLineDataset", "FixedLengthRecordDataset", "TFRecordDataset",
    "TensorSliceDataset", "RangeDataset", "SSTableDataset", "RecordIODataset",
    // Because Print and Assert are on this list, the order of Print and Assert
    // ops may not be deterministic. This is acceptable, as it doesn't affect
    // model outputs or weights or other numeric values.
    "Print", "Assert"};

// List of stateful ops which do not introduce nondeterminism when run
// asynchronously as part of a Dataset function, but may introduce
// nondeterminism when run in parallel. All legacy random ops can be put on this
// list, since the state in internal to the op itself, and so there is no risk
// of ops outside the dataset reading or modifying the state.
constexpr std::array<const char*, 13> kDeterministicStatefulOpsWhenAsync = {
    "RandomUniform",
    "RandomUniformInt",
    "RandomStandardNormal",
    "ParameterizedTruncatedNormal",
    "TruncatedNormal",
    "RandomShuffle",
    "Multinomial",
    "RandomGamma",
    "RandomGammaGrad",
    "RandomPoisson",
    "RandomCrop",
    "SampleDistortedBoundingBox",
    "SampleDistortedBoundingBoxV2"};

bool IsDeterministicWhenRunInParallel(const std::string& stateful_op) {
  for (auto op_in_array : kDeterministicStatefulOps) {
    if (data::MatchesAnyVersion(op_in_array, stateful_op)) {
      return true;
    }
  }
  return false;
}

bool IsDeterministicWhenRunAsynchronously(const std::string& stateful_op) {
  for (auto op_in_array : kDeterministicStatefulOps) {
    if (data::MatchesAnyVersion(op_in_array, stateful_op)) {
      return true;
    }
  }
  for (auto op_in_array : kDeterministicStatefulOpsWhenAsync) {
    if (data::MatchesAnyVersion(op_in_array, stateful_op)) {
      return true;
    }
  }
  return false;
}

bool IsParallelInterleave(const std::string& op) {
  return data::MatchesAnyVersion(kParallelInterleaveOp, op) ||
         op == kLegacyParallelInterleaveOp;
}

bool IsParallelMap(const std::string& op) {
  return data::MatchesAnyVersion(kParallelMapOp, op);
}

bool IsParallelBatch(const std::string& op) {
  return data::MatchesAnyVersion(kParallelBatchOp, op);
}

bool IsMapAndBatch(const std::string& op) {
  return data::MatchesAnyVersion(kMapAndBatchOp, op);
}

bool IsPrefetch(const std::string& op) {
  return data::MatchesAnyVersion(kPrefetchOp, op);
}

// Returns whether the op is a dataset op which runs a function multiple times
// in parallel.
bool IntroducesFunctionParallelism(const std::string& op) {
  return IsParallelInterleave(op) || IsParallelMap(op) || IsMapAndBatch(op);
}

// Returns whether the op is a dataset op which can cause functions in the input
// pipeline to run asynchronously.
bool IntroducesAsynchrony(const std::string& op) {
  // Currently, every op that introduces parallelism also introduces
  // asynchrony.
  return IntroducesFunctionParallelism(op) || IsPrefetch(op) ||
         IsParallelBatch(op);
}

// Returns map from node name to NodeDef in a function.
absl::flat_hash_map<absl::string_view, const NodeDef*> NameToNode(
    const FunctionDef& function) {
  absl::flat_hash_map<absl::string_view, const NodeDef*> name_to_node;
  for (const NodeDef& node : function.node_def()) {
    name_to_node.insert({node.name(), &node});
  }
  return name_to_node;
}

NodeDef* GetMutableNode(const string& node_name, MutableGraphView* graph) {
  int index = graph_utils::FindGraphNodeWithName(node_name, *graph->graph());
  DCHECK_NE(index, -1) << "Failed to find node " << node_name
                       << " in the optimized graph.";
  return graph->graph()->mutable_node(index);
}

// Converts a ParallelInterleaveDataset or ParallelMapDataset to the equivalent
// non-parallel version, to make it deterministic.
Status ConvertMapOrInterleave(const string& node_name,
                              MutableGraphView* graph) {
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
  return absl::OkStatus();
}

// Returns all transitive dependencies of a set of nodes, including the nodes
// themselves.
absl::flat_hash_set<absl::string_view> GetAllTransitiveDependencies(
    const FunctionDef& function_def,
    const absl::flat_hash_set<absl::string_view>& nodes) {
  std::vector<absl::string_view> nodes_to_process;
  std::copy(nodes.begin(), nodes.end(), std::back_inserter(nodes_to_process));

  absl::flat_hash_map<absl::string_view, const NodeDef*> name_to_node =
      NameToNode(function_def);
  absl::flat_hash_set<absl::string_view> dependencies;
  while (!nodes_to_process.empty()) {
    absl::string_view node_name = nodes_to_process.back();
    nodes_to_process.pop_back();
    if (dependencies.contains(node_name)) {
      continue;
    }
    dependencies.insert(node_name);
    auto iter = name_to_node.find(node_name);
    if (iter == name_to_node.end()) {
      // If the node doesn't exist, the function is malformed, so just ignore
      // the node for now.
      continue;
    }
    for (absl::string_view inp : iter->second->input()) {
      absl::string_view inp_node = inp.substr(0, inp.find(':'));
      if (inp_node.at(0) == '^') {
        inp_node = inp_node.substr(1);
      }
      // Input may be an argument instead of a node, so explicitly check if name
      // is in name_to_node.
      if (name_to_node.contains(inp_node)) {
        nodes_to_process.push_back(inp_node);
      }
    }
  }
  return dependencies;
}

// Makes a ParallelMapV2 op deterministic by splitting it into separate Map and
// ParallelMapV2 ops, or a MapAndBatch op deterministic by splitting it into
// separate Map and MapAndBatch ops. All the nondeterministic nodes and their
// dependencies are moved to the Map node.
Status SplitMap(
    const FunctionLibraryDefinition& library, const string& map_node_name,
    MutableGraphView* graph,
    const absl::flat_hash_set<absl::string_view>& nondeterministic_nodes) {
  NodeDef* map_node = GetMutableNode(map_node_name, graph);
  NameAttrList func = map_node->attr().at("f").func();
  const FunctionDef* function_def = library.Find(func.name());
  if (!function_def) {
    return errors::Internal("Could not look up function ", func.name(),
                            " in FunctionLibraryDefinition");
  }

  absl::flat_hash_set<absl::string_view> nodes_to_move =
      GetAllTransitiveDependencies(*function_def, nondeterministic_nodes);

  VLOG(2) << "Will move nodes to nonparallel function: "
          << absl::StrJoin(nodes_to_move, ", ");

  int64_t num_captured_arguments =
      map_node->attr().find("Targuments")->second.list().type_size();

  TF_ASSIGN_OR_RETURN(
      split_utils::SplitResults split_results,
      split_utils::SplitFunction(*function_def, nodes_to_move,
                                 num_captured_arguments, library));

  if (split_results.first_function_output_types.empty()) {
    // Map datasets require there to be at least one output.
    return errors::Unimplemented(
        "The case where the first function has no outputs is unimplemented.");
  }

  bool is_map_and_batch = map_node->op() == kMapAndBatchOp;

  NodeDef* first_map_node_ptr;
  {
    NodeDef first_map_node;
    graph_utils::SetUniqueGraphNodeName(
        strings::StrCat("make_deterministic_sequential_map/", map_node->name()),
        graph->graph(), &first_map_node);
    first_map_node.set_op(kMapOp);
    int num_control_deps = NumControlInputs(*map_node);
    // ParallelMap and MapAndBatch nodes have "num_extra_inputs" more inputs
    // than Map. All inputs are copied to the Map node, but the
    // "num_extra_inputs" inputs are converted to control dependencies.
    int num_extra_inputs = is_map_and_batch ? 3 : 1;
    int control_deps_index = map_node->input_size() - num_control_deps;
    int extra_inputs_index = control_deps_index - num_extra_inputs;
    for (int i = 0; i < extra_inputs_index; i++) {
      // Copy inputs that are also inputs to Map
      DCHECK(!IsControlInput(map_node->input(i)));
      first_map_node.add_input(map_node->input(i));
    }
    for (int i = extra_inputs_index; i < control_deps_index; i++) {
      // Copy the extra inputs, converting them to control dependencies
      DCHECK(!IsControlInput(map_node->input(i)));
      first_map_node.add_input(absl::StrCat("^", map_node->input(i)));
    }
    for (int i = control_deps_index; i < map_node->input_size(); i++) {
      // Copy the control dependencies
      DCHECK(IsControlInput(map_node->input(i)));
      first_map_node.add_input(map_node->input(i));
    }

    NameAttrList* name_attr_list =
        (*first_map_node.mutable_attr())["f"].mutable_func();
    // TODO(reedwm): Set attrs?
    name_attr_list->set_name(split_results.first_function.signature().name());

    graph_utils::CopyAttribute("Targuments", *map_node, &first_map_node);
    for (auto key : {"use_inter_op_parallelism", "preserve_cardinality"}) {
      if (gtl::FindOrNull(map_node->attr(), key)) {
        graph_utils::CopyAttribute(key, *map_node, &first_map_node);
      }
    }
    AddNodeAttr("output_types", split_results.first_function_output_types,
                &first_map_node);
    TensorShapeProto unknown_shape;
    unknown_shape.set_unknown_rank(true);
    std::vector<TensorShapeProto> output_shapes(
        split_results.first_function_output_types.size(), unknown_shape);
    AddNodeAttr("output_shapes", output_shapes, &first_map_node);
    first_map_node_ptr = graph->AddNode(std::move(first_map_node));
  }

  NodeDef* second_map_node_ptr;
  {
    NodeDef second_map_node;
    string node_name =
        map_node->op() == kMapAndBatchOp ? "map_and_batch" : "parallel_map";
    graph_utils::SetUniqueGraphNodeName(
        strings::StrCat("make_deterministic_parallel_", node_name, "/",
                        map_node->name()),
        graph->graph(), &second_map_node);
    second_map_node.set_op(map_node->op());
    second_map_node.add_input(first_map_node_ptr->name());
    for (int i = 1; i < map_node->input_size(); i++) {
      second_map_node.add_input(map_node->input(i));
    }
    NameAttrList* name_attr_list =
        (*second_map_node.mutable_attr())["f"].mutable_func();
    // TODO(reedwm): Set attrs?
    name_attr_list->set_name(split_results.second_function.signature().name());
    graph_utils::CopyAttribute("Targuments", *map_node, &second_map_node);
    graph_utils::CopyAttribute("output_types", *map_node, &second_map_node);
    graph_utils::CopyAttribute("output_shapes", *map_node, &second_map_node);
    if (!is_map_and_batch) {
      AddNodeAttr("deterministic", "true", &second_map_node);
    }
    for (auto key : {"use_inter_op_parallelism", "preserve_cardinality"}) {
      if (gtl::FindOrNull(map_node->attr(), key)) {
        graph_utils::CopyAttribute(key, *map_node, &second_map_node);
      }
    }
    second_map_node_ptr = graph->AddNode(std::move(second_map_node));
  }

  TF_RETURN_IF_ERROR(
      graph->UpdateFanouts(map_node->name(), second_map_node_ptr->name()));
  *graph->graph()->mutable_library()->mutable_function()->Add() =
      split_results.first_function;
  *graph->graph()->mutable_library()->mutable_function()->Add() =
      split_results.second_function;
  return absl::OkStatus();
}

// Converts a ParallalBatch dataset to a Batch dataset, to make it
// deterministic.
Status ConvertBatch(const string& node_name, MutableGraphView* graph) {
  NodeDef* node = GetMutableNode(node_name, graph);
  node->set_op(kBatchV2Op);
  std::string num_parallel_calls_input = node->input(2);
  node->set_input(2, node->input(3));
  node->set_input(3, absl::StrCat("^", num_parallel_calls_input));
  node->mutable_attr()->erase("deterministic");
  return absl::OkStatus();
}

// Convert a MapAndBatch node to a separate Map node and Batch node, to make it
// deterministic. Caller should delete the MapAndBatch node afterwards.
// TODO(reedwm): Handle 'metadata' attribute. Currently the Map node and Batch
// node will have an empty 'metadata' attribute.
Status ConvertMapAndBatch(const string& node_name, MutableGraphView* graph) {
  int index = graph_utils::FindGraphNodeWithName(node_name, *graph->graph());
  DCHECK_NE(index, -1) << "Failed to find node " << node_name
                       << " in the optimized graph.";
  const NodeDef& orig_node = graph->graph()->node(index);

  auto Targuments = orig_node.attr().find("Targuments");
  if (Targuments == orig_node.attr().end()) {
    return errors::Internal("Failed to find Targuments attribute for node ",
                            node_name);
  }

  // Create map node
  NodeDef new_map_node;
  new_map_node.set_op(kMapOp);
  graph_utils::SetUniqueGraphNodeName(kMapOp, graph->graph(), &new_map_node);
  int num_map_inputs = 1 + Targuments->second.list().type_size();
  for (int i = 0; i < num_map_inputs; i++) {
    new_map_node.add_input(orig_node.input(i));
  }
  for (int i = num_map_inputs; i < orig_node.input_size(); i++) {
    if (IsControlInput(orig_node.input(i))) {
      new_map_node.add_input(orig_node.input(i));
    } else {
      new_map_node.add_input(absl::StrCat("^", orig_node.input(i)));
    }
  }
  for (auto key : {"f", "Targuments", "output_types"}) {
    graph_utils::CopyAttribute(key, orig_node, &new_map_node);
  }
  for (auto key : {"preserve_cardinality"}) {
    if (gtl::FindOrNull(new_map_node.attr(), key)) {
      graph_utils::CopyAttribute(key, orig_node, &new_map_node);
    }
  }
  auto orig_output_shapes = orig_node.attr().find("output_shapes");
  if (orig_output_shapes == orig_node.attr().end()) {
    return errors::Internal("Failed to find output_shapes attribute for node ",
                            node_name);
  }

  // Set "output_shapes" attr of Map to be "output_shapes" of MapAndBatch with
  // the leading dimension removed for each shape.
  AttrValue& map_output_shapes =
      (*new_map_node.mutable_attr())["output_shapes"];
  for (const TensorShapeProto& orig_shape :
       orig_output_shapes->second.list().shape()) {
    TensorShapeProto* new_shape = map_output_shapes.mutable_list()->add_shape();
    if (orig_shape.unknown_rank()) {
      new_shape->set_unknown_rank(true);
    } else if (orig_shape.dim_size() == 0) {
      return errors::Internal(
          "Output shape of MapAndBatch node cannot be scalar");
    } else {
      for (int i = 1; i < orig_shape.dim_size(); i++) {
        *new_shape->add_dim() = orig_shape.dim(i);
      }
    }
  }

  // Create batch node
  NodeDef new_batch_node;
  new_batch_node.set_op(kBatchV2Op);
  graph_utils::SetUniqueGraphNodeName(kBatchOp, graph->graph(),
                                      &new_batch_node);
  new_batch_node.add_input(new_map_node.name());
  new_batch_node.add_input(orig_node.input(num_map_inputs));  // batch_size
  new_batch_node.add_input(
      orig_node.input(num_map_inputs + 2));  // drop_remainder
  graph_utils::CopyShapesAndTypesAttrs(orig_node, &new_batch_node);

  graph->AddNode(std::move(new_map_node));
  NodeDef* graph_batch_node = graph->AddNode(std::move(new_batch_node));
  TF_RETURN_IF_ERROR(
      graph->UpdateFanouts(orig_node.name(), graph_batch_node->name()));
  return absl::OkStatus();
}

// Change the buffer_size of a Prefetch node to zero, effectively disabling it,
// to make it deterministic.
Status ConvertPrefetch(const string& node_name, MutableGraphView* graph) {
  NodeDef* node = GetMutableNode(node_name, graph);
  constexpr int buffer_size_index = 1;
  node->add_input(absl::StrCat("^", node->input(buffer_size_index)));
  NodeDef* tmp = graph_utils::AddScalarConstNode<int64_t>(0, graph);
  node->set_input(buffer_size_index, tmp->name());
  return absl::OkStatus();
}

// The two ways nondeterminism can occur in an input pipeline when there are
// stateful ops.
enum class NondeterminismType { PARALLELISM, ASYNCHRONY };

// Returns whether the stateful op is deterministic if run in parallel or
// asynchronously.
bool IsDeterministicStatefulOp(NondeterminismType type,
                               const std::string& stateful_op) {
  return type == NondeterminismType::PARALLELISM
             ? IsDeterministicWhenRunInParallel(stateful_op)
             : IsDeterministicWhenRunAsynchronously(stateful_op);
}

// Defined below. Mutually recursive with FunctionMayIntroduceNondeterminism.
bool FunctionNodeMayIntroduceNondeterminism(
    const FunctionLibraryDefinition& library, const NodeDef& node_def,
    NondeterminismType nondeterminism_type,
    absl::flat_hash_set<std::string>* functions_processed);

// Returns true if the function may introduce nondeterminism. Depending on
// 'nondeterminism_type', either checks if nondeterminism can occur when the
// function is run several times in parallel or when run asynchronously.
// Recursively checks any function attributes of ops within the function.
// "functions_processed" is the list of functions already processed, so that the
// same function is not recursively checked twice. If not null, nodes causing
// nondeterminism will be added to "nondeterministic_nodes".
bool FunctionMayIntroduceNondeterminism(
    const FunctionLibraryDefinition& library, const std::string& function_name,
    NondeterminismType nondeterminism_type,
    absl::flat_hash_set<std::string>* functions_processed,
    absl::flat_hash_set<absl::string_view>* nondeterministic_nodes) {
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
  bool found = false;
  for (const NodeDef& node_def : function_def->node_def()) {
    bool nondeterministic = FunctionNodeMayIntroduceNondeterminism(
        library, node_def, nondeterminism_type, functions_processed);
    if (nondeterministic) {
      if (nondeterministic_nodes) {
        nondeterministic_nodes->insert(node_def.name());
        found = true;
      } else {
        return true;
      }
    }
  }
  return found;
}

bool FunctionMayIntroduceNondeterminism(
    const FunctionLibraryDefinition& library, const std::string& function_name,
    NondeterminismType nondeterminism_type) {
  absl::flat_hash_set<string> functions_processed;
  return FunctionMayIntroduceNondeterminism(library, function_name,
                                            nondeterminism_type,
                                            &functions_processed, nullptr);
}

// Returns true if the given NodeDef inside a function may cause nondeterminism.
bool FunctionNodeMayIntroduceNondeterminism(
    const FunctionLibraryDefinition& library, const NodeDef& node_def,
    NondeterminismType nondeterminism_type,
    absl::flat_hash_set<std::string>* functions_processed) {
  const OpRegistrationData* op_reg_data = nullptr;
  Status s = library.LookUp(node_def.op(), &op_reg_data);
  if (!s.ok()) {
    VLOG(2) << "Could not look up op " << node_def.op()
            << " in FunctionLibraryDefinition, so rewriting op to be safe";
    return true;
  }
  bool is_function_op = op_reg_data->is_function_op;

  bool is_stateful = false;
  if (!is_function_op) {
    const OpDef* op_def;
    s = OpRegistry::Global()->LookUpOpDef(node_def.op(), &op_def);
    if (!s.ok()) {
      VLOG(2) << "Could not look up op " << node_def.op()
              << " in OpRegistry, so rewriting op to be safe";
      return true;
    }
    is_stateful = op_def->is_stateful();
  }

  // Rewrite nondeterministic stateful ops. Function ops and If/While ops are
  // skipped, since we instead look at the ops within the function(s).
  if (is_stateful && !IsStatefulPartitionedCall((node_def)) &&
      !IsIf(node_def) && !IsWhile(node_def) &&
      !IsDeterministicStatefulOp(nondeterminism_type, node_def.op())) {
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
    attr_func_names.push_back(node_def.op());
  }
  for (const std::string& inner_function_name : attr_func_names) {
    if (FunctionMayIntroduceNondeterminism(library, inner_function_name,
                                           nondeterminism_type,
                                           functions_processed, nullptr)) {
      return true;
    }
  }
  return false;
}

// Returns true if "node" is a dataset node whose function can introduce
// nondeterminism when run asynchronously.
bool NodeMayIntroduceNondeterminismWhenAsync(
    const FunctionLibraryDefinition& library, const NodeDef& node) {
  const OpDef* op_def;
  Status s = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);
  if (s.code() == error::NOT_FOUND) {
    return false;
  } else if (!s.ok()) {
    return true;
  }
  if (data::DatasetOpKernel::IsDatasetOp(*op_def)) {
    std::vector<std::string> attr_func_names;
    for (const auto& attr : node.attr()) {
      if (attr.second.has_func()) {
        attr_func_names.push_back(attr.second.func().name());
      }
      for (const auto& name_attr_list : attr.second.list().func()) {
        attr_func_names.push_back(name_attr_list.name());
      }
    }
    for (const std::string& inner_function_name : attr_func_names) {
      if (FunctionMayIntroduceNondeterminism(library, inner_function_name,
                                             NondeterminismType::ASYNCHRONY)) {
        return true;
      }
    }
  }
  return false;
}

// Returns true if the graph has any dataset node whose function can introduce
// nondeterminism when run asynchronously.
bool GraphMayHaveAsyncNondeterminism(const FunctionLibraryDefinition& library,
                                     const GraphDef& graph) {
  for (const NodeDef& node : graph.node()) {
    if (NodeMayIntroduceNondeterminismWhenAsync(library, node)) {
      return true;
    }
  }
  for (const string& function_name : library.ListFunctionNames()) {
    const FunctionDef* function_def = library.Find(function_name);
    CHECK(function_def);  // Crash Ok
    for (const NodeDef& node : function_def->node_def()) {
      if (NodeMayIntroduceNondeterminismWhenAsync(library, node)) {
        return true;
      }
    }
  }
  return false;
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
  absl::flat_hash_set<string> nodes_to_delete;
  bool remove_async_nodes =
      GraphMayHaveAsyncNondeterminism(function_library, item.graph);

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

    bool rewrite_due_to_async =
        IntroducesAsynchrony(node.op()) && remove_async_nodes;
    absl::flat_hash_set<std::string> functions_processed;
    absl::flat_hash_set<absl::string_view> nondeterministic_nodes;
    bool rewrite_due_to_parallelism =
        IntroducesFunctionParallelism(node.op()) &&
        FunctionMayIntroduceNondeterminism(
            function_library, node.attr().at("f").func().name(),
            NondeterminismType::PARALLELISM, &functions_processed,
            &nondeterministic_nodes);
    if (!rewrite_due_to_async && !rewrite_due_to_parallelism) {
      continue;
    }

    VLOG(1) << "Rewriting node " << node.name() << " (" << node.op()
            << ") because it introduces nondeterminism through "
            << (rewrite_due_to_async ? "asynchrony" : "parallelism");

    bool maybe_can_split =
        !rewrite_due_to_async &&
        (node.op() == kParallelMapOpV2 || IsMapAndBatch(node.op()));
    if (maybe_can_split) {
      Status s = SplitMap(function_library, node.name(), &graph,
                          nondeterministic_nodes);
      if (s.ok()) {
        VLOG(1) << "Split node " << node.name() << " (" << node.op()
                << ") into two map nodes: a nonparallel version and a "
                   "parallel version.";
        nodes_to_delete.insert(node.name());
        continue;
      } else if (s.code() == error::UNIMPLEMENTED) {
        // If splitting the function is unimplemented, instead convert the node
        // to a nonparallel version below.
        VLOG(1) << "Could not move stateful ops to their own function, so will "
                   "convert node "
                << node.name()
                << " to a nonparallel version instead. Reason: " << s;
      } else {
        return s;
      }
    }

    if (IsPrefetch(node.op())) {
      TF_RETURN_IF_ERROR(ConvertPrefetch(node.name(), &graph));
    } else if (IsMapAndBatch(node.op())) {
      TF_RETURN_IF_ERROR(ConvertMapAndBatch(node.name(), &graph));
      nodes_to_delete.insert(node.name());
    } else if (IsParallelBatch(node.op())) {
      TF_RETURN_IF_ERROR(ConvertBatch(node.name(), &graph));
    } else {
      DCHECK(IsParallelInterleave(node.op()) || IsParallelMap(node.op()));
      TF_RETURN_IF_ERROR(ConvertMapOrInterleave(node.name(), &graph));
    }
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return absl::OkStatus();
}

REGISTER_GRAPH_OPTIMIZER_AS(MakeDeterministic, "make_deterministic");

}  // namespace grappler
}  // namespace tensorflow

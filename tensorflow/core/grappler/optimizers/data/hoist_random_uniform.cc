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

#include "tensorflow/core/grappler/optimizers/data/hoist_random_uniform.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {

NodeDef MakeStatelessMap(const NodeDef& map_node, const NodeDef& zip_node,
                         const FunctionDef& stateless_function,
                         MutableGraphView* graph) {
  NodeDef stateless_map;
  graph_utils::SetUniqueGraphNodeName("stateless_map", graph->graph(),
                                      &stateless_map);

  stateless_map.set_op("MapDataset");
  stateless_map.add_input(zip_node.name());
  // Add placeholders.
  for (int i = 1; i < map_node.input_size(); i++)
    stateless_map.add_input(map_node.input(i));

  auto attr = map_node.attr().at("f");
  *attr.mutable_func()->mutable_name() = stateless_function.signature().name();
  *attr.mutable_func()->mutable_attr() = stateless_function.attr();
  (*stateless_map.mutable_attr())["f"] = std::move(attr);

  graph_utils::CopyAttribute("Targuments", map_node, &stateless_map);
  for (auto key : {"output_shapes", "output_types"})
    graph_utils::CopyAttribute(key, map_node, &stateless_map);

  if (const auto* attr =
          gtl::FindOrNull(map_node.attr(), "use_inter_op_parallelism"))
    (*stateless_map.mutable_attr())["use_inter_op_parallelism"] = *attr;

  return stateless_map;
}

NodeDef MakeRandomDataset(const NodeDef& random_uniform_node,
                          MutableGraphView* graph) {
  NodeDef random_dataset;
  random_dataset.set_op("ExperimentalRandomDataset");
  graph_utils::SetUniqueGraphNodeName("RandomDataset", graph->graph(),
                                      &random_dataset);

  const auto* seed = graph_utils::AddScalarConstNode<int64>(
      random_uniform_node.attr().at("seed").i(), graph);
  const auto* seed2 = graph_utils::AddScalarConstNode<int64>(
      random_uniform_node.attr().at("seed2").i(), graph);

  random_dataset.add_input(seed->name());
  random_dataset.add_input(seed2->name());

  (*random_dataset.mutable_attr())["output_shapes"].mutable_list()->add_shape();
  (*random_dataset.mutable_attr())["output_types"].mutable_list()->add_type(
      DT_INT64);

  return random_dataset;
}

NodeDef MakeBatchTwo(const NodeDef& random_dataset, MutableGraphView* graph) {
  NodeDef batch_dataset;
  batch_dataset.set_op("BatchDatasetV2");
  graph_utils::SetUniqueGraphNodeName("pair_of_random", graph->graph(),
                                      &batch_dataset);
  const auto* batch_size = graph_utils::AddScalarConstNode<int64>(2, graph);
  const auto* drop_reminder = graph_utils::AddScalarConstNode(false, graph);
  batch_dataset.add_input(random_dataset.name());
  batch_dataset.add_input(batch_size->name());
  batch_dataset.add_input(drop_reminder->name());

  (*batch_dataset.mutable_attr())["output_shapes"]
      .mutable_list()
      ->add_shape()
      ->mutable_dim()
      ->Add()
      ->set_size(-1);
  (*batch_dataset.mutable_attr())["output_types"].mutable_list()->add_type(
      DT_INT64);

  return batch_dataset;
}

NodeDef MakeZipNode(const NodeDef& first_node, const NodeDef& second_node,
                    MutableGraphView* graph) {
  NodeDef zip_node;
  graph_utils::SetUniqueGraphNodeName("zip_with_random", graph->graph(),
                                      &zip_node);

  zip_node.set_op("ZipDataset");
  zip_node.add_input(first_node.name());
  zip_node.add_input(second_node.name());

  for (auto key : {"output_shapes", "output_types"})
    graph_utils::ConcatAttributeList(key, first_node, second_node, &zip_node);

  (*zip_node.mutable_attr())["N"].set_i(2);

  return zip_node;
}

// We need to insert our argument before the placeholders, which are the last
// arguments.
OpDef_ArgDef* InsertSeedArgument(OpDef* signature, int num_placeholders) {
  int new_argument_idx = signature->input_arg_size() - num_placeholders;
  signature->add_input_arg();
  for (int i = signature->input_arg_size() - 1; i > new_argument_idx; i--) {
    signature->mutable_input_arg()->SwapElements(i - 1, i);
  }
  auto* seed_arg = signature->mutable_input_arg(new_argument_idx);
  seed_arg->set_name(strings::StrCat("seed_arg", new_argument_idx));
  seed_arg->set_type(DT_INT64);

  return seed_arg;
}

// Make function that uses `StatelessRandomUniform` instead of `RandomUniform`
// to make it less statefull.  The function can still be stateful, but in when
// other stateful ops are e.g. `Assert`, then it will be parallelizable.
const FunctionDef* MakeLessStatefulFunction(const FunctionDef& map_function,
                                            bool is_stateful,
                                            int num_placeholders,
                                            FunctionDefLibrary* library) {
  FunctionDef* stateless_function = library->add_function();
  *stateless_function = map_function;
  if (is_stateful)
    stateless_function->mutable_signature()->set_is_stateful(is_stateful);
  graph_utils::SetUniqueGraphFunctionName("stateless_function", library,
                                          stateless_function);

  auto* seed_arg = InsertSeedArgument(stateless_function->mutable_signature(),
                                      num_placeholders);

  auto* const random_uniform = stateless_function->mutable_node_def(
      function_utils::FindFunctionNodeWithOp("RandomUniform",
                                             *stateless_function));

  // Replace RandomUniform node with StatelessRandomUniform.
  random_uniform->set_op("StatelessRandomUniform");
  random_uniform->add_input(seed_arg->name());
  (*random_uniform->mutable_attr())["Tseed"].set_type(DT_INT64);
  random_uniform->mutable_attr()->erase("seed");
  random_uniform->mutable_attr()->erase("seed2");

  return stateless_function;
}
// This function returns true if function is stateful and has single
// RandomUniform op and no other stateful ops except Assert.
// `is_stateful_after_hoisting` is set to true if RandomUniform is the only
// stateful op and hoisting can be performed.
bool CanHoistRandomUniform(const FunctionDef& map_function,
                           const FunctionLibraryDefinition& library,
                           bool* is_stateful_after_hoisting,
                           const NodeDef** random_uniform_op) {
  if (!map_function.signature().is_stateful()) return false;
  *is_stateful_after_hoisting = true;

  bool have_other_stateful_ops = false;

  for (const auto& node : map_function.node_def()) {
    const OpDef* op_def;
    TF_CHECK_OK(library.LookUpOpDef(node.op(), &op_def));
    // Skip stateless nodes and assert, as it does not actually have a state.
    if (!op_def->is_stateful()) continue;

    if (op_def->name() == "Assert") {
      have_other_stateful_ops = true;
      continue;
    }

    // TODO(prazek): For now we only handle RandomUniform, we should handle
    // RandomUniformInt as well.
    if (op_def->name() != "RandomUniform") return false;

    // TODO(prazek): For now we can only hoist single RandomUniform.
    if (*random_uniform_op != nullptr) return false;

    *random_uniform_op = &node;
  }

  if (!have_other_stateful_ops) *is_stateful_after_hoisting = false;

  // Have we found single RandomUniform?
  return *random_uniform_op != nullptr;
}

int NumberOfPlaceholders(const NodeDef& map_node) {
  // First input of MapDataset is the argument to the function.  Rest of the
  // inputs are placeholders.
  return map_node.input_size() - 1;
}

}  // namespace

Status HoistRandomUniform::OptimizeAndCollectStats(Cluster* cluster,
                                                   const GrapplerItem& item,
                                                   GraphDef* output,
                                                   OptimizationStats* stats) {
  *output = item.graph;

  MutableGraphView graph(output);
  absl::flat_hash_set<string> nodes_to_delete;
  FunctionLibraryDefinition function_library(OpRegistry::Global(),
                                             item.graph.library());

  auto get_map_node = [](const NodeDef& node) -> const NodeDef* {
    // TODO(prazek): we could also handle ParallelMapDataset and
    // MapAndBatchDataset.
    if (node.op() == "MapDataset") return &node;
    return nullptr;
  };

  for (const NodeDef& node : item.graph.node()) {
    const NodeDef* map_node = get_map_node(node);
    if (!map_node) continue;

    const auto& fun = map_node->attr().at("f");
    const FunctionDef* func = function_library.Find(fun.func().name());

    const NodeDef* random_uniform_op = nullptr;
    bool is_stateful_after_hoisting = true;
    if (!CanHoistRandomUniform(*func, function_library,
                               &is_stateful_after_hoisting, &random_uniform_op))
      continue;
    const auto* random_seed_dataset =
        graph.AddNode(MakeRandomDataset(*random_uniform_op, &graph));

    const auto* batch_dataset =
        graph.AddNode(MakeBatchTwo(*random_seed_dataset, &graph));

    const NodeDef& parent_node = *graph_utils::GetInputNode(*map_node, graph);

    const auto* zip_node =
        graph.AddNode(MakeZipNode(parent_node, *batch_dataset, &graph));

    const auto* stateless_func = MakeLessStatefulFunction(
        *func, is_stateful_after_hoisting, NumberOfPlaceholders(*map_node),
        output->mutable_library());

    const auto* stateless_map = graph.AddNode(
        MakeStatelessMap(*map_node, *zip_node, *stateless_func, &graph));

    TF_RETURN_IF_ERROR(
        graph.UpdateFanouts(map_node->name(), stateless_map->name()));

    // TODO(b/116285210): we could also remove map functions from library if
    // they are not used anymore.
    nodes_to_delete.insert(map_node->name());
    stats->num_changes++;
  }

  TF_RETURN_IF_ERROR(graph.DeleteNodes(nodes_to_delete));
  return Status::OK();
}

void HoistRandomUniform::Feedback(Cluster* cluster, const GrapplerItem& item,
                                  const GraphDef& optimize_output,
                                  double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(HoistRandomUniform, "hoist_random_uniform");

}  // namespace grappler
}  // namespace tensorflow

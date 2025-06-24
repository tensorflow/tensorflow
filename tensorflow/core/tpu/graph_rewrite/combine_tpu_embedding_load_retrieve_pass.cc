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

#include "tensorflow/core/tpu/graph_rewrite/combine_tpu_embedding_load_retrieve_pass.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/tpu/optimization_parameters.pb.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_configuration.pb.h"
#include "tensorflow/core/tpu/graph_rewrite/tpu_embedding_rewrite_pass_utils.h"
#include "tensorflow/core/tpu/ops/tpu_embedding_ops.h"
#include "tensorflow/core/tpu/tpu_embedding_optimization_parameters_utils.h"

namespace tensorflow {

namespace {

using ValueCase = AttrValue::ValueCase;
using TableNameToIntegerMap = absl::flat_hash_map<std::string, int>;

// Returns names of the load and retrieve node of all optimizers supported
// by TPUEmbedding.
absl::flat_hash_set<std::string> GetLoadRetrieveNodeNames() {
  std::vector<std::string> load_names =
      GetPerTableLoadOptimizationParametersOps();
  std::vector<std::string> retrieve_names =
      GetPerTableRetrieveOptimizationParametersOps();
  absl::flat_hash_set<std::string> nodes(load_names.begin(), load_names.end());
  nodes.insert(retrieve_names.begin(), retrieve_names.end());
  return nodes;
}

// Gets TPUEmbeddingConfiguration proto from embedding ops.
absl::Status GetTPUEmbeddingConfiguration(
    Graph* graph,
    tensorflow::tpu::TPUEmbeddingConfiguration* tpu_embedding_config,
    std::string* tpu_embedding_config_str) {
  bool have_config = false;
  const absl::flat_hash_set<std::string> load_retrieve_nodes =
      GetLoadRetrieveNodeNames();
  for (Node* n : graph->nodes()) {
    const auto& node_name = n->op_def().name();
    if (n->IsOp() && (load_retrieve_nodes.contains(node_name) ||
                      node_name == "XlaRecvTPUEmbeddingActivations" ||
                      node_name == "XlaSendTPUEmbeddingGradients" ||
                      node_name == "ConfigureTPUEmbedding")) {
      std::string test_tpu_embedding_config_str;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(n->def(), "config", &test_tpu_embedding_config_str));
      if (test_tpu_embedding_config_str.empty()) {
        if (load_retrieve_nodes.contains(node_name)) {
          continue;
        } else if (node_name == "ConfigureTPUEmbedding") {
          return errors::InvalidArgument(
              "ConfigureTPUEmbedding used but no configuration provided");
        }
      }
      if (have_config) {
        TF_RET_CHECK(test_tpu_embedding_config_str ==
                     *tpu_embedding_config_str);
      } else {
        have_config = true;
        *tpu_embedding_config_str = test_tpu_embedding_config_str;
        TF_RET_CHECK(
            tpu_embedding_config->ParseFromString(*tpu_embedding_config_str));
      }
    }
  }
  if (!have_config) {
    return errors::InvalidArgument("No TPU embedding config provided");
  }
  return absl::OkStatus();
}

// Validates that all of the table names are distinct and non-empty.
absl::Status ValidateEmbeddingTableNames(
    const tensorflow::tpu::TPUEmbeddingConfiguration& tpu_embedding_config) {
  // Map from table names to first occurrences.
  TableNameToIntegerMap table_name_map;
  for (int table_id = 0;
       table_id < tpu_embedding_config.table_descriptor_size(); ++table_id) {
    const auto& table = tpu_embedding_config.table_descriptor(table_id);
    const std::string& name = table.name();
    if (name.empty()) {
      return errors::InvalidArgument(
          absl::StrFormat("Table %d has empty name string.", table_id));
    }
    bool inserted = gtl::InsertIfNotPresent(&table_name_map, name, table_id);
    if (!inserted) {
      return errors::InvalidArgument(
          absl::StrFormat("Tables %d and %d have the same name '%s'.",
                          table_name_map[name], table_id, name.c_str()));
    }
  }
  return absl::OkStatus();
}

// Gets single-table load-TPUEmbedding-parameter nodes in the graph.
absl::flat_hash_set<Node*> GetLoadNodes(Graph* graph) {
  const auto load_op_names = GetPerTableLoadOptimizationParametersOps();
  // Determines whether this node is a per-table load-parameters op.
  auto is_load_op = [&load_op_names](Node* n) {
    return n->IsOp() && std::find(load_op_names.begin(), load_op_names.end(),
                                  n->op_def().name()) != load_op_names.end();
  };
  absl::flat_hash_set<Node*> result;
  for (Node* n : graph->nodes()) {
    if (is_load_op(n)) {
      result.insert(n);
    }
  }
  return result;
}

// Gets single-table retrieve-TPUEmbedding-parameter nodes in the graph.
absl::flat_hash_set<Node*> GetRetrieveNodes(Graph* graph) {
  const auto retrieve_op_names = GetPerTableRetrieveOptimizationParametersOps();
  // Determines whether this node is a per-table retrieve-parameters op.
  auto is_retrieve_op = [&retrieve_op_names](Node* n) {
    return n->IsOp() &&
           std::find(retrieve_op_names.begin(), retrieve_op_names.end(),
                     n->op_def().name()) != retrieve_op_names.end();
  };
  absl::flat_hash_set<Node*> result;
  for (Node* n : graph->nodes()) {
    if (is_retrieve_op(n)) {
      result.insert(n);
    }
  }
  return result;
}

// Gets load or retrieve parameters nodes, plus some other related information,
// from those candidate nodes found on a given device. There is required to be
// exactly one such node for each table in the embedding layer configuration.
//
// Parameters:
//   candidate_nodes: Load or retrieve nodes; all must have the correct device
//     and shard_id values.
//   tpu_embedding_config: Embedding layer configuration to use.
//   node_name_format: Format (printf-style) for expected node names, with %s
//     replaced by optimization algorithm name; must come from a trusted source.
//   nodes_per_table: Used to return a node for each table mentioned in
//     tpu_embedding_config; (*nodes_per_table)[i] will contain the candidate
//     node with table_id == i or table_name == table i's name.
//   is_debug_load_retrieve_node: Used to return a bool for each table that
//     tells whether a GradAccumDebug op was used for it; those ops allow the
//     user to explicitly save and restore otherwise-hidden gradient
//     accumulation buffers.
//   num_shards: Used to return number of shards from first matching node
//     (checked for consistency with others).
//
// Returns: Status (OK if successful; otherwise, an error status).
//
absl::Status GetLoadOrRetrieveNodesByTable(
    const absl::flat_hash_set<Node*>& candidate_nodes,
    const tensorflow::tpu::TPUEmbeddingConfiguration& tpu_embedding_config,
    const TableNameToIntegerMap& table_name_to_id_map,
    const std::string& operation_str, std::vector<Node*>* nodes_per_table,
    std::vector<bool>* is_debug_load_retrieve_node, int* num_shards) {
  TF_RET_CHECK(!candidate_nodes.empty());
  TF_RETURN_IF_ERROR(
      GetNodeAttr((*candidate_nodes.begin())->def(), "num_shards", num_shards));
  for (Node* n : candidate_nodes) {
    // Check that all tables have the same shard count.
    int test_num_shards;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "num_shards", &test_num_shards));
    TF_RET_CHECK(*num_shards == test_num_shards);
  }
  const int num_tables = tpu_embedding_config.table_descriptor_size();
  nodes_per_table->clear();
  nodes_per_table->resize(num_tables, nullptr);
  is_debug_load_retrieve_node->clear();
  is_debug_load_retrieve_node->resize(num_tables, false);
  for (Node* n : candidate_nodes) {
    int table_id;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "table_id", &table_id));
    std::string table_name;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "table_name", &table_name));
    if (table_id < 0 && table_name.empty()) {
      return errors::InvalidArgument(
          "Neither table_id nor table_name attribute specified in node " +
          n->name());
    }
    if (table_id >= 0 && !table_name.empty()) {
      return errors::InvalidArgument(
          "Both table_id and table_name attributes specified in node " +
          n->name());
    }
    if (!table_name.empty()) {
      if (!table_name_to_id_map.contains(table_name)) {
        return errors::InvalidArgument(
            "Table name attribute refers to non-existent table '" + table_name +
            "' in node " + n->name());
      }
      table_id = table_name_to_id_map.at(table_name);
    }
    if (table_id < 0 || table_id >= num_tables) {
      return errors::InvalidArgument(
          "table_id attribute out of range in node " + n->name());
    }
    if ((*nodes_per_table)[table_id] != nullptr) {
      return errors::AlreadyExists("Found duplicate table_id caused by op " +
                                   n->name());
    }
    const auto alg = tpu_embedding_config.table_descriptor(table_id)
                         .optimization_parameters()
                         .parameters_case();
    const std::string alg_name = tpu::GetOptimizationAlgorithmName(alg);
    const std::string expected_op_name =
        absl::StrCat(operation_str, "TPUEmbedding", alg_name, "Parameters");
    const std::string expected_op_name_debug =
        absl::StrCat(expected_op_name, "GradAccumDebug");
    if (n->op_def().name() != expected_op_name &&
        n->op_def().name() != expected_op_name_debug) {
      return errors::InvalidArgument(
          absl::StrFormat("Node %s has op type %s instead of the name %s or %s "
                          "expected from the embedding layer configuration",
                          n->name(), n->op_def().name(), expected_op_name,
                          expected_op_name_debug));
    }
    (*nodes_per_table)[table_id] = n;
    (*is_debug_load_retrieve_node)[table_id] =
        (n->op_def().name() == expected_op_name_debug);
  }
  for (int table_id = 0; table_id < num_tables; ++table_id) {
    if ((*nodes_per_table)[table_id] == nullptr) {
      return errors::NotFound(absl::StrFormat(
          "Did not find per-table load or retrieve op for table '%s' ID(%d)",
          tpu_embedding_config.table_descriptor(table_id).name(), table_id));
    }
  }
  return absl::OkStatus();
}

// Pair of a node and an input or output number used to record edge endpoints.
struct Port {
  Node* node;
  int port_index;
};

using LoadCombinedParametersType =
    std::array<std::vector<Port>, (tpu::kMaxAuxiliaryParameterCount + 1)>;

// Computes an array of ports containing the source of each table/parameter
// combination. Fills in any unused ports with {nullptr, 0}.
absl::Status CombinePerTableParametersForLoad(
    const absl::flat_hash_set<Node*>& load_nodes,
    std::vector<Node*>* nodes_per_table,
    std::vector<bool>* is_debug_load_retrieve_node,
    const tensorflow::tpu::TPUEmbeddingConfiguration& tpu_embedding_config,
    const TableNameToIntegerMap& table_name_to_id_map,
    LoadCombinedParametersType* combined_inputs, int* num_shards) {
  TF_RETURN_IF_ERROR(GetLoadOrRetrieveNodesByTable(
      load_nodes, tpu_embedding_config, table_name_to_id_map, "Load",
      nodes_per_table, is_debug_load_retrieve_node, num_shards));
  const int num_tables = tpu_embedding_config.table_descriptor_size();
  // Set dummy values for unused tensor inputs/outputs.
  for (auto& v : *combined_inputs) {
    v.clear();
    v.resize(num_tables, Port{nullptr, 0});
  }
  for (int table_id = 0; table_id < num_tables; ++table_id) {
    Node* neighbor = (*nodes_per_table)[table_id];
    CHECK_NE(neighbor, nullptr);  // Crash OK
    const auto& opt_params = tpu_embedding_config.table_descriptor(table_id)
                                 .optimization_parameters();

    std::vector<tpu::StateVariableSpecification> state_variable_specs;
    absl::Status status = tpu::GetOptimizationAlgorithmStateVariables(
        opt_params, &state_variable_specs);

    if (!status.ok()) {
      return status;
    }

    CHECK_LE(state_variable_specs.size(),  // Crash OK
             tpu::kMaxAuxiliaryParameterCount + 1);
    for (int parameter_num = 0;
         parameter_num < tpu::kMaxAuxiliaryParameterCount + 1;
         ++parameter_num) {
      (*combined_inputs)[parameter_num][table_id] = Port{nullptr, 0};
    }
    for (const auto& e : neighbor->in_edges()) {
      if (!e->IsControlEdge()) {
        TF_RET_CHECK(e->dst_input() >= 0);
        TF_RET_CHECK(e->dst_input() < state_variable_specs.size());
        TF_RET_CHECK(state_variable_specs[e->dst_input()].has_user_defined() ||
                     (*is_debug_load_retrieve_node)[table_id]);
        TF_RET_CHECK((*combined_inputs)[e->dst_input()][table_id].node ==
                     nullptr);
        (*combined_inputs)[e->dst_input()][table_id] =
            Port{e->src(), e->src_output()};
      }
    }
    for (int parameter_num = 0;
         parameter_num < tpu::kMaxAuxiliaryParameterCount + 1;
         ++parameter_num) {
      const auto node = (*combined_inputs)[parameter_num][table_id].node;
      if (parameter_num < state_variable_specs.size() &&
          (state_variable_specs[parameter_num].has_user_defined() ||
           (*is_debug_load_retrieve_node)[table_id])) {
        if (node == nullptr) {
          return errors::InvalidArgument(absl::StrFormat(
              "Found missing parameter in slot %d of table %s.", parameter_num,
              tpu_embedding_config.table_descriptor(table_id).name()));
        }
      } else {
        if (node != nullptr) {
          return errors::InvalidArgument(absl::StrFormat(
              "Found extra parameter in slot %d of table %s.", parameter_num,
              tpu_embedding_config.table_descriptor(table_id).name()));
        }
      }
    }
  }
  return absl::OkStatus();
}

// Removes edges between individual load/retrieve nodes that are added by
// tf.function.
void RemoveEdgesBetweenIndividualNodes(
    Graph* graph, const std::vector<std::string>& ops,
    const absl::flat_hash_set<Node*>& nodes) {
  const absl::flat_hash_set<std::string> ops_names(ops.begin(), ops.end());
  std::vector<const Edge*> edges_to_remove;
  for (const auto* node : nodes) {
    for (const Edge* edge : node->out_edges()) {
      // We are expecting only control edges to appear in the EdgeSet
      // `node->out_edges()`, and thus only those are being removed.
      if (ops_names.contains(edge->dst()->op_def().name())) {
        edges_to_remove.push_back(edge);
      }
    }
  }
  // Separate for-loop because EdgeSet does not allow mutation
  // while iterating.
  for (const auto edge : edges_to_remove) {
    graph->RemoveControlEdge(edge);
  }
}

}  // namespace

absl::Status CombineTPUEmbeddingLoadRetrievePass::Run(
    const GraphOptimizationPassOptions& options) {
  VLOG(2) << "Starting CombineTPUEmbeddingLoadRetrievePass";
  Graph* graph = options.graph->get();

  tensorflow::tpu::TPUEmbeddingConfiguration tpu_embedding_config;
  std::string tpu_embedding_config_str;
  const absl::Status tpu_embedding_config_error = GetTPUEmbeddingConfiguration(
      graph, &tpu_embedding_config, &tpu_embedding_config_str);
  TF_RETURN_IF_ERROR(ValidateEmbeddingTableNames(tpu_embedding_config));

  TableNameToIntegerMap table_name_to_id_map;
  for (int table_id = 0;
       table_id < tpu_embedding_config.table_descriptor_size(); ++table_id) {
    const auto& table = tpu_embedding_config.table_descriptor(table_id);
    const std::string& name = table.name();
    // Duplicates should be prevented by ValidateEmbeddingTableNames.

    TF_RET_CHECK(
        gtl::InsertIfNotPresent(&table_name_to_id_map, name, table_id));
  }

  absl::flat_hash_set<Node*> load_nodes = GetLoadNodes(graph);
  absl::flat_hash_set<Node*> retrieve_nodes = GetRetrieveNodes(graph);
  // Remove control edges between individual load/retrieve ops that are
  // added by tf.function. tf.function auto-inserts dependencies between these
  // nodes. These edges would create cycles in the graph once the edges
  // between individual load/retrieve ops and the combined ops are added.
  RemoveEdgesBetweenIndividualNodes(
      graph, GetPerTableLoadOptimizationParametersOps(), load_nodes);
  RemoveEdgesBetweenIndividualNodes(
      graph, GetPerTableRetrieveOptimizationParametersOps(), retrieve_nodes);

  absl::flat_hash_map<int, std::string> load_devices;
  absl::flat_hash_map<int, std::string> retrieve_devices;
  if (!load_nodes.empty() || !retrieve_nodes.empty()) {
    // Only check for a config if there are load or retrieve nodes.
    TF_RETURN_IF_ERROR(tpu_embedding_config_error);
  }
  for (Node* n : load_nodes) {
    int shard_id;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "shard_id", &shard_id));
    auto it = load_devices.find(shard_id);
    if (it != load_devices.end()) {
      if (n->def().device() != it->second) {
        return errors::InvalidArgument(absl::StrFormat(
            "Mismatched device name in load parameter op for shard %d: found "
            "%s and conflicting %s in node %s",
            shard_id, it->second, n->def().device(), n->name()));
      }
      TF_RET_CHECK(n->def().device() == it->second);
    } else {
      load_devices.insert(std::make_pair(shard_id, n->def().device()));
    }
  }
  for (Node* n : retrieve_nodes) {
    int shard_id;
    TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "shard_id", &shard_id));
    auto it = retrieve_devices.find(shard_id);
    if (it != retrieve_devices.end()) {
      if (n->def().device() != it->second) {
        return errors::InvalidArgument(absl::StrFormat(
            "Mismatched device name in retrieve parameter op for shard %d: "
            "found %s and conflicting %s in node %s",
            shard_id, it->second, n->def().device(), n->name()));
      }
    } else {
      retrieve_devices.insert(std::make_pair(shard_id, n->def().device()));
    }
  }

  std::vector<std::vector<tpu::StateVariableSpecification>>
      state_variable_specs_by_table(
          tpu_embedding_config.table_descriptor_size());
  for (int table_id = 0;
       table_id < tpu_embedding_config.table_descriptor_size(); ++table_id) {
    const auto& opt_params = tpu_embedding_config.table_descriptor(table_id)
                                 .optimization_parameters();
    TF_RETURN_IF_ERROR(tpu::GetOptimizationAlgorithmStateVariables(
        opt_params, &state_variable_specs_by_table[table_id]));
  }

  int num_combined_nodes_added = 0;

  for (const auto& shard_and_device : load_devices) {
    const int shard_id = shard_and_device.first;
    const std::string& device = shard_and_device.second;
    VLOG(2) << "Doing transformation for load device " << device << " shard "
            << shard_id;
    absl::flat_hash_set<Node*> load_nodes_for_shard;
    for (Node* n : load_nodes) {
      TF_RET_CHECK(n->IsOp());
      int shard_id_to_filter;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(n->def(), "shard_id", &shard_id_to_filter));
      if (shard_id_to_filter != shard_id) continue;
      load_nodes_for_shard.insert(n);
    }
    std::vector<Node*> nodes_per_table;
    std::vector<bool> is_debug_load_retrieve_node;
    LoadCombinedParametersType combined_parameters;
    int num_shards;
    TF_RETURN_IF_ERROR(CombinePerTableParametersForLoad(
        load_nodes_for_shard, &nodes_per_table, &is_debug_load_retrieve_node,
        tpu_embedding_config, table_name_to_id_map, &combined_parameters,
        &num_shards));
    NodeDef new_load_node_def;
    new_load_node_def.set_name(graph->NewName("LoadAllTPUEmbeddingParameters"));
    new_load_node_def.set_op("LoadAllTPUEmbeddingParameters");
    new_load_node_def.set_device(device);
    (*new_load_node_def.mutable_attr())["NumTables"].set_i(
        tpu_embedding_config.table_descriptor_size());
    (*new_load_node_def.mutable_attr())["config"].set_s(
        tpu_embedding_config_str);
    (*new_load_node_def.mutable_attr())["num_shards"].set_i(num_shards);
    (*new_load_node_def.mutable_attr())["shard_id"].set_i(shard_id);
    std::vector<Port> new_node_inputs;
    for (int parameter_num = 0;
         parameter_num < tpu::kMaxAuxiliaryParameterCount + 1;
         ++parameter_num) {
      for (int table_id = 0;
           table_id < tpu_embedding_config.table_descriptor_size();
           ++table_id) {
        const Port& output_of_input_node =
            combined_parameters[parameter_num][table_id];
        auto make_const_node = [&](absl::string_view name,
                                   TensorProto const_tensor,
                                   Node** constant_node) {
          return NodeBuilder(name, "Const")
              .Attr("dtype", DT_FLOAT)
              .Attr("value", const_tensor)
              .Device(device)
              .Finalize(graph, constant_node);
        };
        VLOG(2) << "Load for table " << table_id << " parameter "
                << parameter_num << ": "
                << (parameter_num >=
                            state_variable_specs_by_table[table_id].size()
                        ? "empty"
                        : state_variable_specs_by_table[table_id][parameter_num]
                              .ShortDebugString());
        if (parameter_num >= state_variable_specs_by_table[table_id].size()) {
          VLOG(2) << "Creating empty node";
          TF_RET_CHECK(output_of_input_node.node == nullptr);
          const std::string const_node_name =
              graph->NewName("EmptyAuxiliaryParameter");
          TensorProto const_tensor =
              tensor::CreateTensorProto<float>({}, {0, 0});
          Node* constant_node;
          TF_RETURN_IF_ERROR(
              make_const_node(const_node_name, const_tensor, &constant_node));
          *new_load_node_def.add_input() = constant_node->name();
          new_node_inputs.push_back(Port{constant_node, 0});
        } else if (!is_debug_load_retrieve_node[table_id] &&
                   state_variable_specs_by_table[table_id][parameter_num]
                       .has_fill_with_constant()) {
          TF_RET_CHECK(output_of_input_node.node == nullptr);
          const std::string const_node_name =
              graph->NewName("AuxiliaryParameterFillValue");
          const float tensor_initial_value = static_cast<float>(
              state_variable_specs_by_table[table_id][parameter_num]
                  .fill_with_constant()
                  .initial_value());
          VLOG(2) << "Creating aux fill value " << tensor_initial_value;
          TensorProto const_tensor =
              tensor::CreateTensorProto<float>({tensor_initial_value}, {});
          Node* constant_node;
          TF_RETURN_IF_ERROR(
              make_const_node(const_node_name, const_tensor, &constant_node));
          Node* shape_of_params;
          VLOG(2) << "Getting shape from "
                  << combined_parameters[0][table_id].node->name() << ":"
                  << combined_parameters[0][table_id].port_index;
          TF_RETURN_IF_ERROR(
              NodeBuilder(graph->NewName("ParamsShape"), "Shape")
                  .Input(combined_parameters[0][table_id].node,
                         combined_parameters[0][table_id].port_index)
                  .Device(device)
                  .Finalize(graph, &shape_of_params));
          Node* fill_node;
          TF_RETURN_IF_ERROR(
              NodeBuilder(graph->NewName("FilledAuxiliaryParameter"), "Fill")
                  .Attr("T", DT_FLOAT)
                  .Input(shape_of_params, 0)
                  .Input(constant_node, 0)
                  .Device(device)
                  .Finalize(graph, &fill_node));
          *new_load_node_def.add_input() = fill_node->name();
          new_node_inputs.push_back(Port{fill_node, 0});
        } else {
          TF_RET_CHECK(output_of_input_node.node != nullptr);
          const std::string new_input_str =
              absl::StrFormat("%s:%d", output_of_input_node.node->name(),
                              output_of_input_node.port_index);
          VLOG(2) << "Using input " << new_input_str;
          *new_load_node_def.add_input() = new_input_str;
          new_node_inputs.push_back(output_of_input_node);
        }
      }
    }
    Node* new_load_node;
    TF_RETURN_IF_ERROR(AddNode(new_load_node_def, &new_load_node, graph));
    new_load_node->set_assigned_device_name(
        (*load_nodes_for_shard.begin())->assigned_device_name());
    new_load_node->set_assigned_device_name_index(
        (*load_nodes_for_shard.begin())->assigned_device_name_index());

    // Copy control flow edges over.
    for (Node* n : load_nodes_for_shard) {
      for (const Edge* e : n->in_edges()) {
        if (e->IsControlEdge()) {
          graph->AddControlEdge(e->src(), new_load_node);
        }
      }
      for (const Edge* e : n->out_edges()) {
        if (e->IsControlEdge()) {
          graph->AddControlEdge(new_load_node, e->dst());
        }
      }
    }

    // Copy data flow edges over.
    for (int input_num = 0; input_num < new_node_inputs.size(); ++input_num) {
      const Port& input = new_node_inputs[input_num];
      CHECK_NE(input.node, nullptr);  // Crash OK
      graph->AddEdge(input.node, input.port_index, new_load_node, input_num);
    }

    // Replace previous load nodes with NoOp nodes in case they are referred
    // to from a "fetches" argument to session.run.
    for (Node* old_single_load_op : nodes_per_table) {
      CHECK_NE(old_single_load_op, nullptr);  // Crash OK
      load_nodes.erase(old_single_load_op);
      NodeDef new_no_op_def = old_single_load_op->def();
      ChangeToNoOp(&new_no_op_def);
      new_no_op_def.clear_input();
      new_no_op_def.clear_attr();
      Node* new_no_op;
      TF_RETURN_IF_ERROR(
          ReplaceNode(new_no_op_def, old_single_load_op, &new_no_op, graph));
      // Remove incoming non-control edges to new node.
      std::vector<const Edge*> edges_to_remove;
      for (const Edge* e : new_no_op->in_edges()) {
        if (!e->IsControlEdge()) {
          edges_to_remove.push_back(e);
        }
      }
      for (const Edge* e : edges_to_remove) {
        graph->RemoveEdge(e);
      }
      graph->AddControlEdge(new_load_node, new_no_op);
    }
    ++num_combined_nodes_added;
  }

  for (const auto& shard_and_device : retrieve_devices) {
    const int shard_id = shard_and_device.first;
    const std::string& device = shard_and_device.second;
    VLOG(2) << "Doing transformation for retrieve device " << device
            << " shard " << shard_id;
    absl::flat_hash_set<Node*> retrieve_nodes_for_shard;
    for (Node* n : retrieve_nodes) {
      TF_RET_CHECK(n->IsOp());
      int shard_id_to_filter;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(n->def(), "shard_id", &shard_id_to_filter));
      if (shard_id_to_filter != shard_id) continue;
      retrieve_nodes_for_shard.insert(n);
    }
    std::vector<Node*> nodes_per_table;
    std::vector<bool> is_debug_load_retrieve_node;
    int num_shards;
    TF_RETURN_IF_ERROR(GetLoadOrRetrieveNodesByTable(
        retrieve_nodes_for_shard, tpu_embedding_config, table_name_to_id_map,
        "Retrieve", &nodes_per_table, &is_debug_load_retrieve_node,
        &num_shards));
    NodeDef new_retrieve_node_def;
    new_retrieve_node_def.set_name(
        graph->NewName("RetrieveAllTPUEmbeddingParameters"));
    new_retrieve_node_def.set_op("RetrieveAllTPUEmbeddingParameters");
    new_retrieve_node_def.set_device(device);
    (*new_retrieve_node_def.mutable_attr())["NumTables"].set_i(
        tpu_embedding_config.table_descriptor_size());
    (*new_retrieve_node_def.mutable_attr())["config"].set_s(
        tpu_embedding_config_str);
    (*new_retrieve_node_def.mutable_attr())["num_shards"].set_i(num_shards);
    (*new_retrieve_node_def.mutable_attr())["shard_id"].set_i(shard_id);
    Node* new_retrieve_node;
    TF_RETURN_IF_ERROR(
        AddNode(new_retrieve_node_def, &new_retrieve_node, graph));
    // Copy over incoming control flow edges (outgoing ones will be handled by
    // the data dependencies from the newly created IdentityN nodes and copying
    // the control dependencies from the per-table retrieve ops to the
    // corresponding IdentityN ops).
    {
      absl::flat_hash_set<Node*> control_edge_sources;
      for (Node* n : retrieve_nodes_for_shard) {
        for (const Edge* e : n->in_edges()) {
          if (e->IsControlEdge()) {
            control_edge_sources.insert(e->src());
          }
        }
      }
      for (auto n : control_edge_sources) {
        graph->AddControlEdge(n, new_retrieve_node);
      }
    }
    for (int output_table_id = 0;
         output_table_id < tpu_embedding_config.table_descriptor_size();
         ++output_table_id) {
      Node* output_chunk = nodes_per_table[output_table_id];
      retrieve_nodes.erase(output_chunk);
      NodeDef identity_n_def;
      // Need to preserve old name since there are references to it.
      identity_n_def.set_name(output_chunk->name());
      identity_n_def.set_op("IdentityN");
      identity_n_def.set_device(device);
      auto* types = (*identity_n_def.mutable_attr())["T"].mutable_list();
      for (int parameter_num = 0;
           parameter_num <
           state_variable_specs_by_table[output_table_id].size();
           ++parameter_num) {
        if (state_variable_specs_by_table[output_table_id][parameter_num]
                .has_user_defined() ||
            is_debug_load_retrieve_node[output_table_id]) {
          types->add_type(DT_FLOAT);
        }
      }
      for (int parameter_num = 0;
           parameter_num <
           state_variable_specs_by_table[output_table_id].size();
           ++parameter_num) {
        if (state_variable_specs_by_table[output_table_id][parameter_num]
                .has_user_defined() ||
            is_debug_load_retrieve_node[output_table_id]) {
          const std::string new_input_str = absl::StrFormat(
              "%s:%d", new_retrieve_node->name(),
              parameter_num * tpu_embedding_config.table_descriptor_size() +
                  output_table_id);
          *identity_n_def.add_input() = new_input_str;
        }
      }
      Node* identity_n_node;
      TF_RETURN_IF_ERROR(
          ReplaceNode(identity_n_def, output_chunk, &identity_n_node, graph));
      for (int parameter_num = 0;
           parameter_num <
           state_variable_specs_by_table[output_table_id].size();
           ++parameter_num) {
        if (state_variable_specs_by_table[output_table_id][parameter_num]
                .has_user_defined() ||
            is_debug_load_retrieve_node[output_table_id]) {
          graph->AddEdge(
              new_retrieve_node,
              parameter_num * tpu_embedding_config.table_descriptor_size() +
                  output_table_id,
              identity_n_node, parameter_num);
        }
      }
    }
    ++num_combined_nodes_added;
  }

  VLOG(2) << "Generated " << num_combined_nodes_added
          << " combined load or retrieve nodes.";

  return absl::OkStatus();
}

}  // namespace tensorflow

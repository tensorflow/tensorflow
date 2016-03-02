/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/simple_graph_execution_state.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/simple_placer.h"
#include "tensorflow/core/distributed_runtime/process_util.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/dot.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tensorflow/core/util/util.h"

namespace tensorflow {

SimpleGraphExecutionState::SimpleGraphExecutionState(
    const OpRegistryInterface* ops,
    const SimpleGraphExecutionStateOptions& options)
    : ops_(ops),
      device_set_(options.device_set),
      session_options_(options.session_options),
      base_(nullptr),
      placed_(nullptr) {
  // TODO(mrry): Publish placement visualizations or handle the log
  // placement option.
}

SimpleGraphExecutionState::~SimpleGraphExecutionState() {
  mutex_lock l(mu_);
  delete base_;
  delete placed_;
}

Status SimpleGraphExecutionState::Create(GraphDef* graph_def) {
  if (original_graph_def_.node_size() > 0) {
    return errors::InvalidArgument(
        "Cannot call Create on SimpleGraphExecutionState twice");
  }

  original_graph_def_.Swap(graph_def);
  VLOG(2) << "Incoming def: " << original_graph_def_.DebugString();
  return AddDefaultAttrsToGraphDef(&original_graph_def_, *ops_, 0);
}

Status SimpleGraphExecutionState::Extend(
    const GraphDef& extension_def, SimpleGraphExecutionState** out) const {
  std::unordered_set<string> new_names;
  // 1. Build an index of the new node names.
  for (const NodeDef& node : extension_def.node()) {
    new_names.insert(node.name());
  }

  // 2. Add the non-duplicates from the old graph to the new graph.
  //    Return an error if the same node name appears in both the
  //    old graph and the extension.
  GraphDef gdef;
  for (const NodeDef& node : original_graph_def_.node()) {
    if (new_names.count(node.name()) == 0) {
      *gdef.add_node() = node;
    } else {
      return errors::InvalidArgument(tensorflow::strings::Printf(
          "GraphDef argument to Extend includes node '%s', which was created "
          "by a previous call to Create or Extend in this session.",
          node.name().c_str()));
    }
  }

  int old_node_size = gdef.node_size();
  gdef.mutable_node()->MergeFrom(extension_def.node());
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&gdef, *ops_, old_node_size));

  // 3. Add the extension.
  SimpleGraphExecutionStateOptions combined_options;
  combined_options.device_set = device_set_;

  SimpleGraphExecutionState* new_execution_state =
      new SimpleGraphExecutionState(ops_, combined_options);
  Status new_execution_state_status = new_execution_state->Create(&gdef);
  if (!new_execution_state_status.ok()) {
    delete new_execution_state;
    return new_execution_state_status;
  }
  *out = new_execution_state;

  // Ensure that any state created in the precursor is accessible in the
  // new graph.
  {
    mutex_lock l(mu_);
    for (const auto& placement : stateful_placements_) {
      (*out)->stateful_placements_.insert(placement);
    }
  }

  // TODO(mrry): This is likely to be used for non-throughput-sensitive
  // interactive workloads, but in future we may want to transfer other
  // parts of the placement and/or cost model.
  return Status::OK();
}

Status SimpleGraphExecutionState::InitBaseGraph() {
  std::unique_ptr<Graph> new_base(new Graph(ops_));
  GraphConstructorOptions opts;
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(opts, original_graph_def_, new_base.get()));
  for (const Node* n : new_base->nodes()) {
    VLOG(2) << "Mapping " << n->name() << " to " << n->cost_id();
    node_name_to_cost_id_map_[n->name()] = n->cost_id();
  }

  Status status = PreliminaryPlace(*new_base);
  if (!status.ok()) {
    node_name_to_cost_id_map_.clear();
    return status;
  }
  base_ = new_base.release();
  return Status::OK();
}

Status SimpleGraphExecutionState::GlobalNodeDefByName(const string& name,
                                                      NodeDef* out) {
  NodeNameToCostIdMap::const_iterator iter =
      node_name_to_cost_id_map_.find(name);
  if (iter != node_name_to_cost_id_map_.end()) {
    mutex_lock l(mu_);  // could use reader lock
    const Node* node = placed_->FindNodeId(iter->second);
    if (node) {
      *out = node->def();
      return Status::OK();
    }
  }
  return errors::NotFound("Node name: ", name);
}

Status SimpleGraphExecutionState::PreliminaryPlace(const Graph& base) {
  VLOG(1) << "PreliminaryPlace";
  Graph* ng = new Graph(ops_);

  CopyGraph(base, ng);
  Status status = DoPlace(ng);
  if (!status.ok()) {
    delete ng;
  } else {
    delete placed_;
    placed_ = ng;
    FreezeStatefulNodes(true /*is_prelim*/);
  }
  return status;
}

void SimpleGraphExecutionState::FreezeStatefulNodes(bool is_prelim) {
  if (is_prelim) {
    // During the preliminary placement every stateful Node got placed
    // somewhere, and we need to remember where, so it doesn't move.
    for (Node* n : placed_->nodes()) {
      if (n->op_def().is_stateful()) {
        stateful_placements_[n->name()] = n->assigned_device_name();
      }
    }
  } else {
    // During later placements it's possible for new stateful nodes to
    // appear. They are noticed while we're pinning the pre-existing
    // stateful nodes to their prior positions, and after they've been
    // placed this function is entered to record their placements.
    for (Node* n : missing_stateful_placements_) {
      CHECK(n->op_def().is_stateful());
      stateful_placements_[n->name()] = n->assigned_device_name();
    }
    missing_stateful_placements_.clear();
  }
}

void SimpleGraphExecutionState::PlaceStatefulNodes(Graph* graph) {
  for (Node* n : graph->nodes()) {
    if (n->op_def().is_stateful()) {
      PlaceMap::const_iterator iter = stateful_placements_.find(n->name());
      if (iter == stateful_placements_.end()) {
        // NOTE(tucker): I don't understand why this can occur.  So far,
        // I've only seen it in eval instances, started from a checkpoint.
        missing_stateful_placements_.push_back(n);
      } else {
        n->set_assigned_device_name(iter->second);
      }
    }
  }
}

Status SimpleGraphExecutionState::DoPlace(Graph* graph) {
  Status status;
  // TODO(mrry): Port other placement algorithms from whitepaper.
  return SimplePlacement(graph);
}

Status SimpleGraphExecutionState::BuildGraph(const BuildGraphOptions& options,
                                             ClientGraph** out) {
  VLOG(1) << "BuildGraph";
  mutex_lock l(mu_);
  // Lazily initialize the base graph.
  if (base_ == nullptr) {
    TF_RETURN_IF_ERROR(InitBaseGraph());
  }

  if (!base_ || !placed_) {
    return ::tensorflow::errors::Internal(
        "There was a problem building the graph.");
  }

  std::unique_ptr<ClientGraph> cgraph(new ClientGraph(ops_));
  CopyGraph(*placed_, &cgraph->graph);

  // Extract the subset of the graph that needs to be run, adding feed/fetch
  // ops as needed.
  TF_RETURN_IF_ERROR(subgraph::RewriteGraphForExecution(
      &cgraph->graph, options.feed_endpoints, options.fetch_endpoints,
      options.target_nodes, device_set_->client_device()->attributes()));

  // Copy the extracted graph in order to make its node ids dense,
  // since the local CostModel used to record its stats is sized by
  // the largest node id.
  {
    std::unique_ptr<ClientGraph> dense_copy(new ClientGraph(ops_));
    CopyGraph(cgraph->graph, &dense_copy->graph);
    cgraph = std::move(dense_copy);
  }

  // TODO(vrv): We should check invariants of the graph here.

  *out = cgraph.release();

  return Status::OK();
}

Status SimpleGraphExecutionState::DeviceIsCompatible(
    Node* n, const Device* device) const {
  if (!n->def().device().empty()) {
    DeviceNameUtils::ParsedName pname;
    if (!DeviceNameUtils::ParseFullName(n->def().device(), &pname)) {
      return AttachDef(
          errors::InvalidArgument("Malformed device specification '",
                                  n->def().device(), "'"),
          n->def());
    }
    std::vector<Device*> devices;
    device_set_->FindMatchingDevices(pname, &devices);
    for (auto d : devices) {
      if (d == device) {
        return Status::OK();
      }
    }

    return AttachDef(
        errors::InvalidArgument(
            "Specified device '", n->def().device(),
            "' not compatible with device of ref connection: ", device->name()),
        n->def());
  }
  return Status::OK();
}

Status SimpleGraphExecutionState::SimplePlacement(Graph* graph) {
  SimplePlacer placer(graph, device_set_, &node_name_to_cost_id_map_,
                      session_options_);
  // TODO(mrry): Consider making the SimplePlacer cancelable.
  return placer.Run();
}

}  // namespace tensorflow

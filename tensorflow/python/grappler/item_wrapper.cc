/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = pybind11;

class ColocationGroups {
 public:
  void Group(const std::string& x, const std::string& y) {
    Rep* x_root = Find(x);
    Rep* y_root = Find(y);

    // x and y are already in the same set
    if (x_root == y_root) {
      return;
    }
    // x and y are not in same set, so we merge them
    // Use the occasion to strengthen what we know about the handle by merging
    // the information about the 2 subsets.
    if (x_root->rank < y_root->rank) {
      x_root->parent = y_root;
    } else if (x_root->rank > y_root->rank) {
      y_root->parent = x_root;
    } else {
      // Arbitrarily make one root the new parent
      y_root->parent = x_root;
      x_root->rank = x_root->rank + 1;
    }
  }

  void ExtractGroups(std::vector<std::vector<std::string>>* groups) {
    groups->reserve(nodes_.size());
    std::unordered_map<const Rep*, int> group_ids;
    for (const auto& rep : nodes_) {
      Rep* r = Find(rep.first);
      auto it = group_ids.find(r);
      std::vector<std::string>* g;
      if (it == group_ids.end()) {
        int id = group_ids.size();
        group_ids[r] = id;
        groups->resize(id + 1);
        g = &groups->back();
      } else {
        int id = it->second;
        g = &((*groups)[id]);
      }
      g->push_back(rep.first);
    }
  }

 private:
  struct Rep {
    // Parent in the tree used to encode the set.
    Rep* parent;
    // Rank in the tree, used to figure out how to compress the path to the root
    // of the tree.
    int rank;
    // The node.
    std::string value;
  };

  Rep* Find(const std::string& n) {
    auto it = nodes_.find(n);
    if (it == nodes_.end()) {
      // This is the first time we process this handle, create an entry for it.
      Rep* node = new Rep;
      node->parent = node;
      node->rank = 0;
      node->value = n;
      nodes_[n] = node;
      return node;
    }
    // Return the representative for the set, which is the root of the tree.
    // Apply path compression to speedup future queries.
    Rep* node = it->second;
    Rep* root = node->parent;
    while (root != root->parent) {
      root = root->parent;
    }
    while (node->parent != root) {
      Rep* next = node->parent;
      node->parent = root;
      node = next;
    }
    return root;
  }

  std::unordered_map<std::string, Rep*> nodes_;
};

PYBIND11_MAKE_OPAQUE(tensorflow::grappler::GrapplerItem);

PYBIND11_MODULE(_pywrap_tf_item, m) {
  py::class_<tensorflow::grappler::GrapplerItem> grappler_item(
      m, "tensorflow::grappler::GrapplerItem");

  m.def("TF_NewItem",
        [](const py::bytes& serialized_metagraph, bool ignore_colocation,
           bool ignore_user_placement) -> tensorflow::grappler::GrapplerItem* {
          tensorflow::MetaGraphDef metagraph;
          if (!metagraph.ParseFromString(std::string(serialized_metagraph))) {
            throw std::invalid_argument(
                "The MetaGraphDef could not be parsed as a valid protocol "
                "buffer");
          }
          if (metagraph.collection_def().count("train_op") == 0) {
            MaybeRaiseRegisteredFromStatus(tensorflow::errors::InvalidArgument(
                "train_op not specified in the metagraph"));
          }

          tensorflow::grappler::ItemConfig cfg;
          cfg.ignore_user_placement = ignore_user_placement;
          cfg.ignore_colocation = ignore_colocation;
          std::unique_ptr<tensorflow::grappler::GrapplerItem> item =
              tensorflow::grappler::GrapplerItemFromMetaGraphDef(
                  "item", metagraph, cfg);
          if (item == nullptr) {
            MaybeRaiseRegisteredFromStatus(
                tensorflow::errors::InvalidArgument("Invalid metagraph"));
          }
          return item.release();
        });

  m.def("TF_IdentifyImportantOps",
        [](tensorflow::grappler::GrapplerItem* item,
           bool sort_topologically) -> std::vector<std::string> {
          std::vector<const tensorflow::NodeDef*> main_ops =
              item->MainOpsFanin();
          std::vector<const tensorflow::NodeDef*> enqueue_ops =
              item->EnqueueOpsFanin();
          std::unordered_set<std::string> op_names;
          for (auto op : main_ops) {
            op_names.insert(op->name());
          }
          for (auto op : enqueue_ops) {
            op_names.insert(op->name());
          }

          std::vector<std::string> ops;
          if (sort_topologically) {
            tensorflow::GraphDef subgraph;
            for (const tensorflow::NodeDef& node : item->graph.node()) {
              if (op_names.find(node.name()) != op_names.end()) {
                *subgraph.add_node() = node;
              }
            }
            tensorflow::MaybeRaiseFromStatus(
                tensorflow::grappler::TopologicalSort(&subgraph));
            for (const tensorflow::NodeDef& node : subgraph.node()) {
              ops.push_back(node.name());
            }
          } else {
            for (const auto& op_name : op_names) {
              ops.push_back(op_name);
            }
          }
          return ops;
        });

  m.def("TF_GetOpProperties",
        [](tensorflow::grappler::GrapplerItem* item)
            -> std::unordered_map<std::string, std::vector<py::bytes>> {
          tensorflow::grappler::GraphProperties properties(*item);
          tensorflow::MaybeRaiseFromStatus(properties.InferStatically(false));

          std::unordered_map<std::string, std::vector<py::bytes>> props;
          for (const auto& node : item->graph.node()) {
            const std::string& node_name = node.name();
            const std::vector<tensorflow::OpInfo::TensorProperties>&
                output_props = properties.GetOutputProperties(node_name);

            std::vector<py::bytes> prop;
            prop.reserve(output_props.size());
            for (const auto& output_prop : output_props) {
              prop.push_back(output_prop.SerializeAsString());
            }
            props[node_name] = prop;
          }
          return props;
        });

  m.def("TF_GetColocationGroups",
        [](tensorflow::grappler::GrapplerItem* item)
            -> std::vector<std::vector<std::string>> {
          ColocationGroups groupings;
          tensorflow::OpRegistry* registry = tensorflow::OpRegistry::Global();
          for (const auto& node : item->graph.node()) {
            const tensorflow::OpDef* op_def;
            if (!registry->LookUpOpDef(node.op(), &op_def).ok()) {
              continue;
            }
            tensorflow::NameRangeMap inputs;
            tensorflow::NameRangeMap outputs;
            if (!tensorflow::NameRangesForNode(node, *op_def, &inputs, &outputs)
                     .ok()) {
              continue;
            }
            for (const auto& arg : op_def->input_arg()) {
              if (!arg.is_ref()) {
                continue;
              }
              const auto& range = inputs[arg.name()];
              for (int i = range.first; i < range.second; ++i) {
                groupings.Group(node.name(),
                                tensorflow::grappler::NodeName(node.input(i)));
              }
            }
          }

          std::vector<std::vector<std::string>> groups;
          groupings.ExtractGroups(&groups);
          return groups;
        });
}

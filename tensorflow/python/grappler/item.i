/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

%include <std_shared_ptr.i>
%typemap(in) const tensorflow::MetaGraphDef& (tensorflow::MetaGraphDef temp) {
  char* c_string;
  Py_ssize_t py_size;
  if (PyBytes_AsStringAndSize($input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    SWIG_fail;
  }

  if (!temp.ParseFromString(string(c_string, py_size))) {
    PyErr_SetString(
        PyExc_TypeError,
        "The MetaGraphDef could not be parsed as a valid protocol buffer");
    SWIG_fail;
  }
  $1 = &temp;
}

// Wrap the item into an object that swig can manipulate. This ensures it will call the object
// destructor upon garbage collection instead of leaking memory.
struct GItem {
  std::shared_ptr<tensorflow::grappler::GrapplerItem> item_;
};


%{
#include <unordered_set>
#include <map>
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"

// Provide the implementation fo the GItem struct here.
struct GItem {
  GItem() {}
  GItem(tensorflow::grappler::GrapplerItem* item) : item_(item) {}

  tensorflow::grappler::GrapplerItem* operator->() const {
    return item_.get();
  }
  const tensorflow::grappler::GrapplerItem& operator*() const {
    return *item_.get();
  }
  bool is_none() const {
    return item_.get() == nullptr;
  }
  std::shared_ptr<tensorflow::grappler::GrapplerItem> item_;
};

static GItem TF_NewItem(
    const tensorflow::MetaGraphDef& meta_graph, bool ignore_colocation,
    bool ignore_user_placement, TF_Status* out_status) {
  if (meta_graph.collection_def().count("train_op") == 0) {
    tensorflow::Set_TF_Status_from_Status(
        out_status,
        tensorflow::errors::InvalidArgument("train_op not specified in the metagraph"));
    return nullptr;
  }

  tensorflow::grappler::ItemConfig cfg;
  cfg.ignore_user_placement = ignore_user_placement;
  cfg.ignore_colocation = ignore_colocation;
  std::unique_ptr<tensorflow::grappler::GrapplerItem> item =
      tensorflow::grappler::GrapplerItemFromMetaGraphDef("item", meta_graph, cfg);
  if (!item) {
    tensorflow::Set_TF_Status_from_Status(
        out_status,
        tensorflow::errors::InvalidArgument("Invalid metagraph"));
    return nullptr;
  }
  tensorflow::Set_TF_Status_from_Status(out_status, tensorflow::Status::OK());
  return GItem(item.release());
}

static PyObject* TF_IdentifyImportantOps(GItem item, bool sort_topologically,
                                                   TF_Status* status) {
  if (item.is_none()) {
    Py_RETURN_NONE;
  }

  std::vector<const tensorflow::NodeDef*> main_ops = item->MainOpsFanin();
  std::vector<const tensorflow::NodeDef*> enqueue_ops = item->EnqueueOpsFanin();
  std::unordered_set<string> op_names;
  for (auto op : main_ops) {
    op_names.insert(op->name());
  }
  for (auto op : enqueue_ops) {
    op_names.insert(op->name());
  }

  std::vector<string> ops;
  if (sort_topologically) {
    tensorflow::GraphDef subgraph;
    for (const tensorflow::NodeDef& node : item->graph.node()) {
      if (op_names.find(node.name()) != op_names.end()) {
        *subgraph.add_node() = node;
      }
    }
    tensorflow::Status s = tensorflow::grappler::TopologicalSort(&subgraph);
    tensorflow::Set_TF_Status_from_Status(status, s);
    for (const tensorflow::NodeDef& node : subgraph.node()) {
      ops.push_back(node.name());
    }
  }
  else {
    for (const auto& op_name : op_names) {
      ops.push_back(op_name);
    }
  }

  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* result = PyList_New(ops.size());
  for (int i = 0; i < ops.size(); ++i) {
    PyList_SetItem(result, i, PyString_FromString(ops[i].c_str()));
  }
  PyGILState_Release(gstate);
  return result;
}

static PyObject* TF_GetOpProperties(GItem item) {
  if (item.is_none()) {
    Py_RETURN_NONE;
  }
  tensorflow::grappler::GraphProperties properties(*item);
  tensorflow::Status status = properties.InferStatically(false);
  if (!status.ok()) {
    Py_RETURN_NONE;
  }

  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* props = PyDict_New();
  for (const auto& node : item->graph.node()) {
    const string& node_name = node.name();
    const std::vector<tensorflow::OpInfo::TensorProperties>& output_props =
        properties.GetOutputProperties(node_name);

    PyObject* prop = PyList_New(output_props.size());
    for (int i = 0; i < output_props.size(); ++i) {
      string output_prop_str = output_props[i].SerializeAsString();
      PyObject* output_prop = PyBytes_FromStringAndSize(output_prop_str.data(),
                                                        output_prop_str.size());
      PyList_SetItem(prop, i, output_prop);
    }
    CHECK_EQ(0, PyDict_SetItem(props, PyString_FromString(node_name.c_str()), prop));
  }
  PyGILState_Release(gstate);
  return props;
}

class ColocationGroups {
public:
  void Group(const string& x, const string& y) {
    Rep* x_root = Find(x);
    Rep* y_root = Find(y);

    // x and y are already in the same set
    if (x_root == y_root) {
      return;
    }
    // x and y are not in same set, so we merge them
    // Use the occasion to strengthen what we know about the handle by merging the
    // information about the 2 subsets.
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

  void ExtractGroups(std::vector<std::vector<string>>* groups) {
    groups->reserve(nodes_.size());
    std::unordered_map<const Rep*, int> group_ids;
    for (const auto& rep : nodes_) {
      Rep* r = Find(rep.first);
      auto it = group_ids.find(r);
      std::vector<string>* g;
      if (it == group_ids.end()) {
        int id = group_ids.size();
        group_ids[r] = id;
        groups->resize(id+1);
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
    string value;
  };

  Rep* Find(const string& n) {
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
    // Return the representative for the set, which is the root of the tree. Apply
    // path compression to speedup future queries.
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

  std::unordered_map<string, Rep*> nodes_;
};

static PyObject* TF_GetColocationGroups(GItem item) {
  if (item.is_none()) {
    Py_RETURN_NONE;
  }
  ColocationGroups groupings;
  tensorflow::OpRegistry* registry = tensorflow::OpRegistry::Global();
  for (const auto& node : item->graph.node()) {
    const tensorflow::OpDef* op_def;
    tensorflow::Status s = registry->LookUpOpDef(node.op(), &op_def);
    if (!s.ok()) {
      continue;
    }
    tensorflow::NameRangeMap inputs;
    tensorflow::NameRangeMap outputs;
    s = tensorflow::NameRangesForNode(node, *op_def, &inputs, &outputs);
    if (!s.ok()) {
      continue;
    }
    int i = 0;
    for (const auto& arg : op_def->input_arg()) {
      if (!arg.is_ref()) {
        continue;
      }
      const auto& range = inputs[arg.name()];
      for (int i = range.first; i < range.second; ++i) {
        string input = tensorflow::grappler::NodeName(node.input(i));
        groupings.Group(node.name(), input);
      }
    }
  }

  std::vector<std::vector<string>> groups;
  groupings.ExtractGroups(&groups);

  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject* result = PyList_New(groups.size());
  for (int i = 0; i < groups.size(); ++i) {
    const std::vector<string>& group = groups[i];
    PyObject* g = PyTuple_New(group.size());
    for (int j = 0; j < group.size(); ++j) {
      const string& node_name = group[j];
      PyTuple_SetItem(g, j, PyString_FromString(node_name.c_str()));
    }
    PyList_SetItem(result, i, g);
  }
  PyGILState_Release(gstate);
  return result;
}

%}


// Wrap these functions.
static GItem TF_NewItem(
    const tensorflow::MetaGraphDef& meta_graph, bool ignore_colocation,
    bool ignore_user_placement, TF_Status* out_status);
static PyObject* TF_IdentifyImportantOps(GItem item, bool sort_topologically,
                                         TF_Status* status);
static PyObject* TF_GetOpProperties(GItem item);
static PyObject* TF_GetColocationGroups(GItem item);

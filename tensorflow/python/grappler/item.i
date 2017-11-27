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
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
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

static std::vector<string> TF_IdentifyImportantOps(GItem item) {
  if (item.is_none()) {
    return {};
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
  for (const auto& op_name : op_names) {
    ops.push_back(op_name);
  }

  return ops;
}

static PyObject* TF_GetOpProperties(GItem item) {
  if (item.is_none()) {
    Py_RETURN_NONE;
  }
  tensorflow::grappler::GraphProperties properties(*item);
  tensorflow::Status status = properties.InferStatically();
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

%}


// Wrap these functions.
static GItem TF_NewItem(
    const tensorflow::MetaGraphDef& meta_graph, bool ignore_colocation,
    bool ignore_user_placement, TF_Status* out_status);
static std::vector<string> TF_IdentifyImportantOps(GItem item);
static PyObject* TF_GetOpProperties(GItem item);

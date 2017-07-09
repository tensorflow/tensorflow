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


%include "tensorflow/python/platform/base.i"

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

%typemap(in) const tensorflow::RewriterConfig& (
    tensorflow::RewriterConfig temp) {
  char* c_string;
  Py_ssize_t py_size;
  if (PyBytes_AsStringAndSize($input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    SWIG_fail;
  }

  if (!temp.ParseFromString(string(c_string, py_size))) {
    PyErr_SetString(
        PyExc_TypeError,
        "The RewriterConfig could not be parsed as a valid protocol buffer");
    SWIG_fail;
  }
  $1 = &temp;
}

%{
  #include <memory>
  #include "tensorflow/c/tf_status_helper.h"
  #include "tensorflow/core/lib/core/status.h"
  #include "tensorflow/core/framework/graph.pb.h"
  #include "tensorflow/core/grappler/grappler_item.h"
  #include "tensorflow/core/grappler/grappler_item_builder.h"
  #include "tensorflow/core/grappler/clusters/virtual_cluster.h"
  #include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
  #include "tensorflow/core/protobuf/meta_graph.pb.h"
  #include "tensorflow/core/protobuf/rewriter_config.pb.h"

PyObject* TF_OptimizeGraph(
      const tensorflow::RewriterConfig& rewriter_config,
      const tensorflow::MetaGraphDef& metagraph,
      const string& graph_id, TF_Status* out_status) {
    tensorflow::grappler::ItemConfig item_config;
    item_config.inline_functions = false;
    item_config.apply_optimizations = false;
    std::unique_ptr<tensorflow::grappler::GrapplerItem> grappler_item =
        tensorflow::grappler::GrapplerItemFromMetaGraphDef(graph_id, metagraph, item_config);
    std::unordered_map<string, tensorflow::DeviceProperties> device_map;
    tensorflow::grappler::VirtualCluster cluster(device_map);
    tensorflow::GraphDef out_graph;
    tensorflow::Status status = tensorflow::grappler::RunMetaOptimizer(
        *grappler_item, rewriter_config, &cluster, &out_graph);
    tensorflow::Set_TF_Status_from_Status(out_status, status);
    string out_graph_str = out_graph.SerializeAsString();
    PyObject* ret = PyBytes_FromStringAndSize(out_graph_str.data(),
                                              out_graph_str.size());
    return ret;
  }
%}


// Wrap this function
PyObject* TF_OptimizeGraph(
    const tensorflow::RewriterConfig& rewriter_config,
    const tensorflow::MetaGraphDef& metagraph,
    const string& graph_id, TF_Status* out_status);




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

#ifndef TENSORFLOW_CONTRIB_XLA_TF_GRAPH_XLA_TF_GRAPH_UTIL_H_
#define TENSORFLOW_CONTRIB_XLA_TF_GRAPH_XLA_TF_GRAPH_UTIL_H_

#include <unordered_map>

#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace xla_tf_graph {

// A set of utilities to handle xla computation requests.
// These utilities help developers leverage existing tools to work with
// xla computations, also provide a way to support TensorFlow ops by
// implementing xla computations so that they can do experiments on their
// specialized environments.

// A structure to represent typed attributes of TensorFlow graph node.
// This structure contains op specific attributes as members so that
// we can treat them explicitly.
struct XlaNode {
  // Unique node name
  string name;
  // Op type of xla computation
  string op_type;
  // List of pair of unique id and port of input node.
  // We store this value instead
  // of node name in order not to wait for all XlaNodes to be constructed.
  std::vector<std::tuple<int64, int>> input_ids;
  // Oputput shapes
  std::vector<TensorShape> output_shapes;
  // Output data types
  std::vector<DataType> output_data_types;

  //---------------------------
  // Op specific attributes
  // #xla::OpRequest::kBinaryOpRequest
  std::vector<int64> broadcast_dimensions;
};

// Convert a tf graph to a xla session module
xla::StatusOr<std::unique_ptr<xla::SessionModule>>
ConvertTfGraphToXlaSessionModule(const std::vector<XlaCompiler::Argument>& args,
                                 std::unique_ptr<Graph> graph);

// Convert a xla session module to a map to XlaNode from unique id
xla::StatusOr<std::unordered_map<int64, XlaNode>>
ConvertXlaSessionModuleToXlaNodes(const xla::SessionModule& session_module);

}  // namespace xla_tf_graph
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_XLA_TF_GRAPH_XLA_TF_GRAPH_UTIL_H_

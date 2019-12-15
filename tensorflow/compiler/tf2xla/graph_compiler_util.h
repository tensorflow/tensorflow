/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_TF2XLA_GRAPH_COMPILER_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_GRAPH_COMPILER_UTIL_H_

#include <unordered_map>

#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace tensorflow {

// Fills in xla_args from the corresponding _Arg nodes in the graph.
Status CreateXlaArgs(const Graph& graph,
                     std::vector<XlaCompiler::Argument>* xla_args);

// Populate xla_args and xla_aliases for the given XLA config.
void PopulateXlaArgsAndXlaAlias(
    const tf2xla::Config& config, std::vector<XlaCompiler::Argument>* xla_args,
    std::vector<xla::XlaBuilder::InputOutputAlias>* xla_aliases);

// InitGraph creates a graph based on the graph_def, that may then be converted
// to an xla::XlaComputation via ConvertGraphToXla.
//
// The graph is rewritten with _Arg and _Retval nodes, representing the inputs
// and outputs of the function that will be compiled.  Each feed id causes a new
// _Arg node to be created, where we first collect all existing edges pointing
// from the named node's output index, and then rewrite them to point from that
// _Arg node instead.  Each fetch id causes a new _Retval node to be created,
// with a new edge pointing from the named node's output index to that _Retval
// node.
Status InitGraph(const GraphDef& graph_def, const tf2xla::Config& config,
                 std::unique_ptr<Graph>* graph);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_GRAPH_COMPILER_UTIL_H_

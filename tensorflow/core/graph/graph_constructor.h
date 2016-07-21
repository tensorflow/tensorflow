/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_GRAPH_GRAPH_CONSTRUCTOR_H_
#define TENSORFLOW_GRAPH_GRAPH_CONSTRUCTOR_H_

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// Options specific to constant folding optimizations.
struct ConstantFoldingOptions {
  // If "consider" is not a nullptr, then only constant fold a node "n" if
  // consider(n) returns true.
  std::function<bool(const Node*)> consider = nullptr;
};

// Construct a graph *g out of a GraphDef gdef. Returns non-OK on
// error, in which case *g is left in an incomplete state.
struct GraphConstructorOptions {
  GraphConstructorOptions();
  explicit GraphConstructorOptions(const OptimizerOptions& opts);

  // If true, allows internal ops in the GraphDef.
  bool allow_internal_ops = false;

  // If true, the graph def is expected to have fully specified
  // devices for all nodes. A node in the resulting graph "g" has the
  // device name set accordingly.
  //
  // TODO(zhifengc): if possible, consider removing this option.
  bool expect_device_spec = false;

  // If true, perform common subexpression elimination on the graph.
  // TODO(jeff): Turn this default to true?
  bool optimizer_do_cse = false;

  // If "optimizer_do_cse" is true and "cse_consider_function" is
  // not nullptr, then only consider nodes for CSE for which
  // "cse_consider_function(node)" returns true.
  std::function<bool(const Node*)> cse_consider_function = nullptr;

  // If true, perform constant folding on the graph.
  bool optimizer_do_constant_folding = false;

  ConstantFoldingOptions constant_folding_opts;
};
extern Status ConvertGraphDefToGraph(const GraphConstructorOptions& opts,
                                     const GraphDef& gdef, Graph* g);

// Make a copy of "src" into "*dest".
//
// REQUIRES: "*dest" is a freshly allocated graph without any nodes or edges
// other than the implicit Source/Sink nodes.
extern void CopyGraph(const Graph& src, Graph* dest);

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_GRAPH_CONSTRUCTOR_H_

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

// Convert a tf graph to a xla session module
xla::StatusOr<std::unique_ptr<xla::SessionModule>>
ConvertTfGraphToXlaSessionModule(const std::vector<XlaCompiler::Argument>& args,
                                 std::unique_ptr<Graph> graph);

}  // namespace xla_tf_graph
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_XLA_TF_GRAPH_XLA_TF_GRAPH_UTIL_H_

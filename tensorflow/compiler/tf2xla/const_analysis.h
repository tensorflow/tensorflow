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

#ifndef TENSORFLOW_COMPILER_TF2XLA_CONST_ANALYSIS_H_
#define TENSORFLOW_COMPILER_TF2XLA_CONST_ANALYSIS_H_

#include <vector>

#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Backwards dataflow analysis that finds nodes in a graph that must be
// compile-time constants for us to be able to lower the graph to XLA.
//
// The indices of the arguments to `graph` that must be constant are returned in
// `compile_time_const_arg_indices`, if `compile_time_const_arg_indices` is not
// null.
//
// The ids of the nodes in `graph` that must be constant are returned in
// `compile_time_const_nodes`, if `compile_time_const_nodes` is not null.
//
// Only propagate const-ness along edges for which `edge_filter` returns true.
Status BackwardsConstAnalysis(
    const Graph& g, std::vector<bool>* compile_time_const_arg_indices,
    std::vector<bool>* compile_time_const_nodes,
    FunctionLibraryRuntime* flib_runtime,
    std::function<bool(const Edge&)> edge_filter = [](const Edge& e) {
      return true;
    });

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_CONST_ANALYSIS_H_

/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_OPTIMIZER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_OPTIMIZER_H_

#include "tensorflow/core/framework/config.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class GraphOptimizer {
 public:
  GraphOptimizer(const OptimizerOptions& opts);
  ~GraphOptimizer();

  // Applies optimization passes specified in 'opts' to 'graph'.
  // Maybe replace *graph with a new graph object.
  void Optimize(FunctionLibraryRuntime* runtime, Graph** graph);

 private:
  OptimizerOptions opts_;

  TF_DISALLOW_COPY_AND_ASSIGN(GraphOptimizer);
};

}  // end namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_OPTIMIZER_H_

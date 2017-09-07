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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_OPTIMIZER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_OPTIMIZER_H_

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class GraphOptimizer {
 public:
  GraphOptimizer(const OptimizerOptions& opts);
  ~GraphOptimizer();

  // Applies optimization passes specified in 'opts' to 'graph'.
  // Maybe replace *graph with a new graph object.  'device' is device
  // on which the 'graph' will execute. It's passed to the optimizers
  // so that they can respect constraints if any, that should be
  // respected. If shape_map is not null it maps from nodes in graph
  // to partially-known shapes of their outputs, and may be used,
  // e.g., in the constant folding pass.
  void Optimize(
      FunctionLibraryRuntime* runtime, Env* env, Device* device,
      std::unique_ptr<Graph>* graph,
      const std::unordered_map<const Node*, std::vector<PartialTensorShape>>*
          shape_map);

 private:
  OptimizerOptions opts_;

  TF_DISALLOW_COPY_AND_ASSIGN(GraphOptimizer);
};

}  // end namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_OPTIMIZER_H_

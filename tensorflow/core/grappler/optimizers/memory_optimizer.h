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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_MEMORY_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_MEMORY_OPTIMIZER_H_

#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

// Swap tensors in and out of device memory.
class MemoryOptimizer : public GraphOptimizer {
 public:
  // optimization_level: Controls the level of autonomy for the memory
  //   optimizer. See RewriterConfig::memory_optimization.
  // recomputation_targets_name_scope: Name scope for potential outputs of
  //   recomputations. See
  //   RewriterConfig::memory_optimizer_target_node_name_scope.
  explicit MemoryOptimizer(
      RewriterConfig::MemOptType optimization_level,
      double per_process_gpu_memory_fraction = 1.0,
      const string& recomputation_targets_name_scope = "gradients/")
      : optimization_level_(optimization_level),
        per_process_gpu_memory_fraction_(per_process_gpu_memory_fraction),
        recomputation_targets_name_scope_(recomputation_targets_name_scope) {}
  ~MemoryOptimizer() override {}

  string name() const override { return "memory_optimizer"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* pruned_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& pruned_graph, double result) override;

 private:
  RewriterConfig::MemOptType optimization_level_;
  double per_process_gpu_memory_fraction_;
  string recomputation_targets_name_scope_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_MEMORY_OPTIMIZER_H_

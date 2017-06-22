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

#ifndef TENSORFLOW_GRAPPLER_OPTIMIZERS_MEMORY_OPTIMIZER_H_
#define TENSORFLOW_GRAPPLER_OPTIMIZERS_MEMORY_OPTIMIZER_H_

#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

// Swap tensors in and out of device memory.
class MemoryOptimizer : public GraphOptimizer {
 public:
  explicit MemoryOptimizer(RewriterConfig::MemOptType optimization_level)
      : optimization_level_(optimization_level) {}
  ~MemoryOptimizer() override {}

  string name() const override { return "memory_optimizer"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* pruned_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& pruned_graph, double result) override;

 private:
  RewriterConfig::MemOptType optimization_level_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_OPTIMIZERS_MEMORY_OPTIMIZER_H_

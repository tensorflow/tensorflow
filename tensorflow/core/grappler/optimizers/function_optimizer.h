/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUNCTION_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUNCTION_OPTIMIZER_H_

#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

// Remap TensorFlow subgraphs onto alternative operations or collection of
// operations to make the overall graph more efficient.
class FunctionOptimizer : public GraphOptimizer {
 public:
  explicit FunctionOptimizer(RewriterConfig::Toggle opt_level,
                             bool lower_control_flow)
      : opt_level_(opt_level), lower_control_flow_(lower_control_flow) {}
  ~FunctionOptimizer() override = default;

  string name() const override { return "function_optimizer"; };

  bool UsesFunctionLibrary() const override { return true; }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override;

 private:
  friend class FunctionOptimizerTest;

  // Runs a single function optimizer pass over the `graph`. All nodes that are
  // not function calls will be copied from the `graph` to the
  // `optimized_graph`. Function call nodes inlined or specialized, and
  // instantiated function body or specialized function call nodes will be added
  // to the `optimized_graph`.
  Status RunFunctionOptimizerPass(const GrapplerItem& item,
                                  GraphDef* optimized_graph) const;

  RewriterConfig::Toggle opt_level_;
  bool lower_control_flow_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FUNCTION_OPTIMIZER_H_

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_REMAPPER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_REMAPPER_H_

#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

// Optimize TF computations by remapping subgraphs/nodes onto other subgraphs or
// nodes to decrease the amount of operations needed to perform a computation.
class Remapper : public GraphOptimizer {
 public:
  explicit Remapper(RewriterConfig::Toggle opt_level,
                    bool xla_auto_clustering_on = false)
      : opt_level_(opt_level),
        xla_auto_clustering_on_(xla_auto_clustering_on) {}

  ~Remapper() override {}

  string name() const override { return "remapper"; };

  bool UsesFunctionLibrary() const override { return false; }

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

 private:
  RewriterConfig::Toggle opt_level_;
  bool xla_auto_clustering_on_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_REMAPPER_H_

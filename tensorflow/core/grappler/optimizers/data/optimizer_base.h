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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_OPTIMIZER_BASE_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_OPTIMIZER_BASE_H_

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"

namespace tensorflow {
namespace grappler {

// A base class for tf.data optimizers.
class TFDataOptimizerBase : public CustomGraphOptimizer {
 public:
  struct OptimizationStats {
    // Identifies the number of independent graph changes for an optimization.
    int64 num_changes = 0;
  };

  TFDataOptimizerBase() = default;
  ~TFDataOptimizerBase() override = default;

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) final;

  virtual Status OptimizeAndCollectStats(Cluster* cluster,
                                         const GrapplerItem& item,
                                         GraphDef* output,
                                         OptimizationStats* stats) = 0;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_OPTIMIZER_BASE_H_

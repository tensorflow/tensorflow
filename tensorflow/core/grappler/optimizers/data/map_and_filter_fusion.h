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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_MAP_AND_FILTER_FUSION_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_MAP_AND_FILTER_FUSION_H_

#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {

// This transformation fuses map and filter operations by moving computation of
// filter predicate to MapDataset, which as a result produces an extra boolean
// component. We filter by the boolean component, then project it away.
//
// In symbols, we transform map(x -> f(x)).filter(f(x) -> p(f(x))) into
// map(x -> f(x), p(f(x))).filter(f(x), p(f(x)) -> p(f(x))).map(f(x), p(f(x))
// -> f(x)). This is more efficient because the latter filter and map operations
// can be performed short-circuit, so only the first map requires an executor
// invocation.
class MapAndFilterFusion : public TFDataOptimizerBase {
 public:
  MapAndFilterFusion() = default;
  ~MapAndFilterFusion() override = default;

  string name() const override { return "map_and_filter_fusion"; };

  bool UsesFunctionLibrary() const override { return false; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return absl::OkStatus();
  }

  Status OptimizeAndCollectStats(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output,
                                 OptimizationStats* stats) override;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_MAP_AND_FILTER_FUSION_H_

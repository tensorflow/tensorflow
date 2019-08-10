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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_MAP_VECTORIZATION_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_MAP_VECTORIZATION_H_

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {

// This optimizer rewrites dataset.map(map_fn, ...).batch(...) and
// dataset.apply(tf.data.experimental.map_and_batch(map_fn, ...)) patterns in an
// input pipeline. It vectorizes the map_fn, such that this segment can be
// rewritten as dataset.batch().map(vectorized_map_fn). This is more performant
// when the map_fn is cheap, because it amortizes the cost of running a map
// function over a larger batch.
//
// From:
//      input --> map --> batch --> output
//              (or map_and_batch)
//
// To:
//      input --> batch --> map --> output
//
// If the "ChooseFastest" configuration is enabled, it adds a
// ChooseFastestBranch dataset node to pick between the original map->batch
// branch and the vectorized batch->map branch.
//
class MapVectorization : public TFDataOptimizerBase {
 public:
  MapVectorization() = default;
  ~MapVectorization() override = default;

  string name() const override { return "map_vectorization"; };

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    if (!config) return Status::OK();

    const string& choose_fastest_param =
        config->parameter_map().at("use_choose_fastest").s();
    if (choose_fastest_param == "true") {
      use_choose_fastest_ = true;
    } else if (choose_fastest_param == "false") {
      use_choose_fastest_ = false;
    } else {
      return errors::Internal(
          "Received an invalid value for parameter \"use_choose_fastest\"",
          choose_fastest_param);
    }
    return Status::OK();
  }

  Status OptimizeAndCollectStats(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output,
                                 OptimizationStats* stats) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override;

 private:
  bool use_choose_fastest_ = false;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_MAP_VECTORIZATION_H_

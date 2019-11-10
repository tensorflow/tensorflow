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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_HOIST_RANDOM_UNIFORM_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_HOIST_RANDOM_UNIFORM_H_

#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {

// This optimization hoists instances of `random_uniform` out of a function
// with the aim of making it stateless.  It creates a new function that takes a
// random seed as an extra argument and uses `stateless_random_uniform` instead
// of `random_uniform` to make it stateless.
// It also creates RandomDataset(seed).batch(2), which is zipped with old input
// to the map.  The batching in RandomDataset is because we need 2 seeds for
// `stateless_random_uniform`.
// TODO(prazek): for now only `RandomUniform` is handled, but we could handle
// `RandomUniformInt` similarly.
class HoistRandomUniform : public TFDataOptimizerBase {
 public:
  HoistRandomUniform() = default;
  ~HoistRandomUniform() override = default;

  string name() const override { return "hoist_random_uniform"; };

  bool UsesFunctionLibrary() const override { return true; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return Status::OK();
  }

  Status OptimizeAndCollectStats(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output,
                                 OptimizationStats* stats) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_HOIST_RANDOM_UNIFORM_H_

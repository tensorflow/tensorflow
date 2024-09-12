/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_MAKE_DETERMINISTIC_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_MAKE_DETERMINISTIC_H_

#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {

// Removes sources on nondeterminism from dataset ops. Nondeterminism can occur
// in the follow ways, each which this pass addresses:
//
// 1. The datasets ParallelInterleave, ParallelMap, and MapAndBatch can
//    introduce nondeterminism by running a function multiple times in parallel.
//    Specifically, if the function can mutate state, it is potentially
//    nondeterministic. In such cases, this pass converts such dataset ops to a
//    non-parallel version. As a performance optimization, in certain cases this
//    pass will instead move nondeterministic ops to a separate non-parallel Map
//    op, so that most of the ops can still run in parallel.
//
// 2. Certain datasets, such as Prefetch, can introduce asynchrony by running a
//    dataset iterator in a background thread while ops outside the dataset are
//    also running. This can introduce nondeterminism if the input pipeline has
//    certain stateful ops. Other than Prefetch, datasets with a
//    `num_parallel_calls` argument also introduce asynchrony, which includes
//    the parallel datasets mentioned in (1) above.
//
//    This pass modifies nodes to remove asynchrony when there are any datasets
//    in the graph with problematic stateful ops. This is done by converting
//    parallel ops into non-parallel versions, as in (1), and by removing
//    Prefetch nodes. Unlike (1), legacy random ops such as RandomUniform are
//    not problematic despite being stateful, as if the op is within a dataset's
//    function, ops outside the dataset cannot access the state. Also unlike
//    (1), nondeterministic ops are never moved to a separate Map op, since
//    doing so would not remove asynchrony.
//
// 3. Nondeterminism occurs if an op has a "deterministic" attribute that is
//    false or a "sloppy" attribute that is true. This pass changes such
//    attributes to be deterministic.
class MakeDeterministic : public TFDataOptimizerBase {
 public:
  MakeDeterministic() = default;
  ~MakeDeterministic() override = default;

  string name() const override { return "make_deterministic"; };

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

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_MAKE_DETERMINISTIC_H_

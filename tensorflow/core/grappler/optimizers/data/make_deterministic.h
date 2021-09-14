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

constexpr char kAutotune[] = "autotune";

// Removes sources on nondeterminism from dataset ops. In particular, this pass
// does the following:
//   1. Transforms ParallelInterleave and ParallelMap datasets into Interleave
//      and Map datasets respectively, if the interleave/map function can
//      introduce nondeterminism when run in parallel. Specifically, if the
//      function can mutate state, it is considered nondeterministic. A
//      MapAndBatch dataset instead has num_parallel_calls set to 1 if it can
//      introduce non-determinism since there is no non-parallel version.
//   2. Sets the "deterministic" attribute to true and "sloppy" attribute to
//      False on dataset ops which have such attributes. Note step (1) is still
//      needed, as even when the "deterministic" attribute is true,
//      nondeterminism can occur if the interleave/map function mutates state.
//
// NOTE: ParallelMap datasets are often rewritten to the non-parallel version,
// as map functions which distort images typically use random ops (which are
// stateful). Unfortunately, this rewrite usually causes a large performance
// penalty in such cases by forcing the map function to run in serial.
//
// TODO(reedwm): Avoid serial execution of stateful functions that contain
// random ops.
class MakeDeterministic : public TFDataOptimizerBase {
 public:
  MakeDeterministic() = default;
  ~MakeDeterministic() override = default;

  string name() const override { return "make_deterministic"; };

  bool UsesFunctionLibrary() const override { return false; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return Status::OK();
  }

  Status OptimizeAndCollectStats(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output,
                                 OptimizationStats* stats) override;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_MAKE_DETERMINISTIC_H_

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_DISABLE_INTRA_OP_PARALLELISM_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_DISABLE_INTRA_OP_PARALLELISM_H_

#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {

// This optimization sets intra-op parallelism to be 1.
class DisableIntraOpParallelism : public TFDataOptimizerBase {
 public:
  DisableIntraOpParallelism() = default;
  ~DisableIntraOpParallelism() override = default;

  string name() const override { return "disable_intra_op_parallelism"; };

  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return absl::OkStatus();
  }

  absl::Status OptimizeAndCollectStats(Cluster* cluster,
                                       const GrapplerItem& item,
                                       GraphDef* output,
                                       OptimizationStats* stats) override;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_DISABLE_INTRA_OP_PARALLELISM_H_

/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_SEQ_INTERLEAVE_PREFETCH_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_SEQ_INTERLEAVE_PREFETCH_H_

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {

// This optimization replaces parallel interleave with sequential interleave and
// adds `prefetch(AUTOTUNE)` after the user defined map function in interleave.
class SeqInterleavePrefetch : public TFDataOptimizerBase {
 public:
  SeqInterleavePrefetch() = default;
  ~SeqInterleavePrefetch() override = default;

  std::string name() const override { return "seq_interleave_prefetch"; };

  // The SeqInterleavePrefetch optimizer requires access to the function
  // library.
  bool UsesFunctionLibrary() const override { return true; }

  absl::Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return absl::OkStatus();
  }

  absl::Status OptimizeAndCollectStats(Cluster* cluster,
                                       const GrapplerItem& item,
                                       GraphDef* output,
                                       OptimizationStats* stats) override;

 protected:
  bool autotune_ = true;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_SEQ_INTERLEAVE_PREFETCH_H_

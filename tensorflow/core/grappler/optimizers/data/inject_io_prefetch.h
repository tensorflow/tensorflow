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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_INJECT_IO_PREFETCH_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_INJECT_IO_PREFETCH_H_

#include <string>

#include "absl/status/status.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

class InjectIoPrefetch : public TFDataOptimizerBase {
 public:
  InjectIoPrefetch() = default;
  ~InjectIoPrefetch() override = default;

  std::string name() const override { return "inject_io_prefetch"; };

  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override;

  absl::Status OptimizeAndCollectStats(Cluster* cluster,
                                       const GrapplerItem& item,
                                       GraphDef* output,
                                       OptimizationStats* stats) override;

 protected:
  bool autotune_ = true;
};

class InjectIoPrefetchEligible : public InjectIoPrefetch {
 public:
  std::string name() const override { return "inject_io_prefetch_eligible"; };

  absl::Status OptimizeAndCollectStats(Cluster* cluster,
                                       const GrapplerItem& item,
                                       GraphDef* output,
                                       OptimizationStats* stats) override;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_INJECT_IO_PREFETCH_H_

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

#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

#include "tensorflow/core/framework/metrics.h"

namespace tensorflow {
namespace grappler {

absl::Status TFDataOptimizerBase::Optimize(Cluster* cluster,
                                           const GrapplerItem& item,
                                           GraphDef* output) {
  OptimizationStats stats;
  absl::Status s = OptimizeAndCollectStats(cluster, item, output, &stats);
  if (s.ok() && stats.num_changes > 0) {
    metrics::RecordTFDataOptimization(name(), stats.num_changes);
  }
  return s;
}

}  // namespace grappler
}  // namespace tensorflow

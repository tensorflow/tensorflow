/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_AUTO_SHARD_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_AUTO_SHARD_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {

class AutoShard : public TFDataOptimizerBase {
 public:
  AutoShard() = default;
  ~AutoShard() override = default;

  string name() const override { return "tf_auto_shard"; }

  bool UsesFunctionLibrary() const override { return true; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override;

  Status OptimizeAndCollectStats(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output,
                                 OptimizationStats* stats) override;

 private:
  int64_t num_workers_;
  int64_t num_replicas_;
  int64_t index_;
  tensorflow::data::AutoShardPolicy auto_shard_policy_;
};

// For testing only
namespace internal {
bool IsEligibleRewriteBatchSize(const NodeDef& sink_node,
                                const MutableGraphView& graph,
                                std::vector<std::string>* ineligible_reason);
}

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_AUTO_SHARD_H_

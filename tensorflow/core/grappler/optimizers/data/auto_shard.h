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

#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {

// AutoShard takes a Dataset graph and tries to insert a shard node
// automatically before a ReaderDataset (e.g. a CSVDataset or a TFRecordDataset)
// such that the dataset is sharded without any modifications to the original
// dataset-based input pipeline.
class AutoShard : public TFDataOptimizerBase {
 public:
  AutoShard() = default;
  ~AutoShard() override = default;

  string name() const override { return "tf_auto_shard"; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override;

  Status OptimizeAndCollectStats(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output,
                                 OptimizationStats* stats) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override {}

 private:
  int64 num_workers_;
  int64 index_;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_AUTO_SHARD_H_

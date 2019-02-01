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
#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_REBATCH_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_REBATCH_H_

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"

namespace tensorflow {
namespace grappler {

// This optimizer changes the batch size of the output dataset by dividing the
// current batch size by parameter `num_workers`. Currently, this works only
// for very simple pipelines with a single BatchDatasetV2 transformation.
//
// TODO(rohanj): Extend this logic to correctly handle any input pipeline that
// uses core tf.data APIs + MapAndBatch.
class RebatchOptimizer : public CustomGraphOptimizer {
 public:
  RebatchOptimizer() = default;
  ~RebatchOptimizer() override = default;

  string name() const override { return "tf_data_rebatcher"; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override;

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override;

 private:
  int64 num_workers_;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_REBATCH_H_

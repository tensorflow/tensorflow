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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_META_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_META_OPTIMIZER_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"

namespace tensorflow {
namespace grappler {

// This optimizer performs tf.data-specific optimizations by invoking
// other optimizers.
class TFDataMetaOptimizer : public CustomGraphOptimizer {
 public:
  TFDataMetaOptimizer() = default;
  ~TFDataMetaOptimizer() override = default;

  string name() const override { return "tf_data_meta_optimizer"; };

  bool UsesFunctionLibrary() const override { return true; }

  absl::Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override;

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* output) override;

 private:
  absl::flat_hash_map<string, std::unique_ptr<GraphOptimizer>>
      enabled_optimizers_;

  // Applies an optimization with the specified name on `item`, and stores
  // the result in `item.graph`
  absl::Status ApplyOptimization(const string& name, Cluster* cluster,
                                 GrapplerItem* item) const;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_META_OPTIMIZER_H_

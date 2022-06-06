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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_SLACK_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_SLACK_H_

#include "absl/strings/numbers.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {

// This optimization sets the slack attr of the terminal PrefetchDataset node in
// an input pipeline.
class Slack : public TFDataOptimizerBase {
 public:
  Slack() = default;
  ~Slack() override = default;

  string name() const override { return "slack"; };

  bool UsesFunctionLibrary() const override { return false; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    if (!config) return errors::InvalidArgument("Config parameter required.");

    const string& slack_period_param =
        config->parameter_map().at("slack_period").s();
    if (!absl::SimpleAtoi(slack_period_param, &slack_period_)) {
      return errors::InvalidArgument("Invalid `slack_period` parameter: ",
                                     slack_period_param);
    }
    return OkStatus();
  }

  Status OptimizeAndCollectStats(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output,
                                 OptimizationStats* stats) override;

 private:
  int64_t slack_period_ = -1;

  Status RecursivelyHandleOp(const MutableGraphView& graph,
                             NodeDef* dataset_node);
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_SLACK_H_

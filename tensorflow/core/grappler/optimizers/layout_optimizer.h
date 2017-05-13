/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_GRAPPLER_OPTIMIZERS_LAYOUT_OPTIMIZER_H_
#define TENSORFLOW_GRAPPLER_OPTIMIZERS_LAYOUT_OPTIMIZER_H_

#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"

namespace tensorflow {
namespace grappler {

// Convert the NHWC layout to NCHW for Conv-related ops on GPUs.
class LayoutOptimizer : public GraphOptimizer {
 public:
  LayoutOptimizer() {}
  ~LayoutOptimizer() override {}

  string name() const override { return "layout"; };

  // This is for testing only.
  void set_num_gpus(int num_gpus) { num_gpus_ = num_gpus; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override;

 private:
  int num_gpus_ = 0;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_OPTIMIZERS_LAYOUT_OPTIMIZER_H_

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_LAYOUT_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_LAYOUT_OPTIMIZER_H_

#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"

namespace tensorflow {
namespace grappler {
// Convert the NHWC layout to NCHW for Conv-related ops on GPUs.
class LayoutOptimizer : public GraphOptimizer {
 public:
  LayoutOptimizer() {}
  ~LayoutOptimizer() override {}

  string name() const override { return "layout"; };

  bool UsesFunctionLibrary() const override { return false; }

  struct TuningConfig {
    // If true, do not use the NHWC GEMM implementation. When filter size is
    // one or filter size is equal to input image size,
    // the NHWC implementation of Conv2D, Conv2DBackpropInput, and
    // Conv2DBackpropFilter will use a specialized GEMM implementation, which is
    // usually faster than the NCHW implementation. The downside is that this
    // might result in more non-cancellable layout conversion nodes (implemented
    // by the Transpose op).
    bool no_gemm;
  };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;

 private:
  std::unique_ptr<VirtualPlacer> virtual_placer_;
  std::unordered_set<string> nodes_to_preserve_;
  Status Tune(const GrapplerItem& item, const GraphProperties& graph_properties,
              const TuningConfig& config, GraphDef* output);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_LAYOUT_OPTIMIZER_H_

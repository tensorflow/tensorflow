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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DEBUG_STRIPPER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DEBUG_STRIPPER_H_

#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"

namespace tensorflow {
namespace grappler {

// DebugStripper strips off debug-related nodes (e.g.
// Assert, CheckNumerics, Print) from the graph.
class DebugStripper : public GraphOptimizer {
 public:
  DebugStripper() {}
  ~DebugStripper() override {}

  string name() const override { return "debug_stripper"; };

  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* output) override;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DEBUG_STRIPPER_H_

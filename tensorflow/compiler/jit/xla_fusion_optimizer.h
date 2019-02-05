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

#ifndef TENSORFLOW_COMPILER_JIT_XLA_FUSION_OPTIMIZER_H_
#define TENSORFLOW_COMPILER_JIT_XLA_FUSION_OPTIMIZER_H_

#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"

namespace tensorflow {

// Optimizes graphs by fusing ops where possible, resulting in more efficient
// execution.
class XlaFusionOptimizer : public grappler::CustomGraphOptimizer {
 public:
  XlaFusionOptimizer() {}
  ~XlaFusionOptimizer() override {}

  Status Init(
      const RewriterConfig_CustomGraphOptimizer* config = nullptr) override {
    return Status::OK();
  }

  string name() const override { return "xla-fusion"; };

  Status Optimize(grappler::Cluster* cluster,
                  const grappler::GrapplerItem& item,
                  GraphDef* output) override;

  void Feedback(grappler::Cluster* cluster, const grappler::GrapplerItem& item,
                const GraphDef& optimize_output, double result) override {
    // Nothing to do for XlaFusionOptimizer.
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_XLA_FUSION_OPTIMIZER_H_

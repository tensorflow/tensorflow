/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_MLIR_GRAPPLER_GRAPPLER_HOOK_H_
#define TENSORFLOW_CORE_MLIR_GRAPPLER_GRAPPLER_HOOK_H_

#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"

namespace mlir {
namespace tfg {

// This class implements a grappler optimizer wrapping a pipeline of passes
// implemented with TFG.
class TfgGrapplerOptimizer : public tensorflow::grappler::GraphOptimizer {
 public:
  explicit TfgGrapplerOptimizer(const std::string& pass_pipeline);

  std::string name() const override {
    return "tfg_optimizer{" + pass_pipeline_ + "}";
  };
  bool UsesFunctionLibrary() const override { return true; }

  tensorflow::Status Optimize(tensorflow::grappler::Cluster* cluster,
                              const tensorflow::grappler::GrapplerItem& item,
                              tensorflow::GraphDef* optimized_graph) override;

 private:
  std::string pass_pipeline_;
};

}  // end namespace tfg
}  // end namespace mlir

#endif  // TENSORFLOW_CORE_MLIR_GRAPPLER_GRAPPLER_HOOK_H_

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

#include <functional>
#include <string>

#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"

namespace mlir {
class PassManager;

namespace tfg {

// Constructs the default TFG pass pipeline.
void DefaultGrapplerPipeline(PassManager& mgr);

// A function that builds the TFG pass pipeline.
using TFGPassPipelineBuilder = std::function<void(PassManager& pm)>;

// This class implements a Grappler optimizer wrapping a pipeline of passes
// implemented with TFG.
class TFGGrapplerOptimizer : public tensorflow::grappler::GraphOptimizer {
 public:
  // Constructs a TFG optimizer using the provided pipeline builder.
  explicit TFGGrapplerOptimizer(TFGPassPipelineBuilder builder);
  // Explicit destructor to defer instantiation of Impl.
  ~TFGGrapplerOptimizer() override;

  // Constructs a name for the optimizer using the registered passes.
  std::string name() const override;
  // The TFG optimizer requires access to the function library.
  bool UsesFunctionLibrary() const override { return true; }

  // Runs the optimizer on the GraphDef. The optimizer converts the GraphDef to
  // TFG using the importer, runs the passes on the MLIR, and exports back to
  // GraphDef. The result is stored in `optimized_graph`.
  tensorflow::Status Optimize(tensorflow::grappler::Cluster* cluster,
                              const tensorflow::grappler::GrapplerItem& item,
                              tensorflow::GraphDef* optimized_graph) override;

 private:
  // Hide the implementation details.
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // end namespace tfg
}  // end namespace mlir

#endif  // TENSORFLOW_CORE_MLIR_GRAPPLER_GRAPPLER_HOOK_H_

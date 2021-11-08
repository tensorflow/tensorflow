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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_FORWARD_TYPE_INFERENCE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_FORWARD_TYPE_INFERENCE_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {

// Run a very basic forward type inference on the graph. Forward type inference
// simply propagates type information from inputs to outputs, until reaching
// stability.
//
// The pass is designed to run as a graph diffusion process, refining type
// information until it reaches a fixed point. However, the current
// implementation is a simplification that only ensures that:
//   1. each node is visited at least once
//   2. a successful update of a node's type ID prevents future visits
//   3. each node is visited at most once
//
// If needed, we can drop rule #3 and change rule #2 to consider an update to
// be any deep type change (rather than just the type ID).
//
// The state of the diffusion process is the NodeDef.experimental_full_type
// field, while the diffusion function is the node's corresponding
// OpRegistrationData.fwd_type_fn function.
//
// TODO(mdan): Use a regular union-based algorithm instead?
class ForwardTypeInferencePass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_FORWARD_TYPE_INFERENCE_H_

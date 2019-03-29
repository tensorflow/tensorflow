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
#ifndef TENSORFLOW_COMPILER_JIT_INTRODUCE_FLOATING_POINT_JITTER_PASS_H_
#define TENSORFLOW_COMPILER_JIT_INTRODUCE_FLOATING_POINT_JITTER_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {
// A debug-only pass that introduces error into outputs of specific TF nodes.
// This can be used to check the sensitivity of a TF graph to floating point
// rounding differences.
//
// This pass is controlled by TF_XLA_FLAGS.  Please see
// IntroduceFloatingPointJitterPassFlags for information on how to use this.
class IntroduceFloatingPointJitterPass : public GraphOptimizationPass {
 public:
  IntroduceFloatingPointJitterPass() = default;

  Status Run(const GraphOptimizationPassOptions& options) override;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_INTRODUCE_FLOATING_POINT_JITTER_PASS_H_

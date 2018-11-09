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

#ifndef TENSORFLOW_COMPILER_JIT_BUILD_XLA_OPS_PASS_H_
#define TENSORFLOW_COMPILER_JIT_BUILD_XLA_OPS_PASS_H_

#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Adds _XlaCompile and _XlaRun operations to the TF graph that compiles and
// executes (using XLA) TF function calls marked with "_XlaCompiledKernel".
class BuildXlaOpsPass : public GraphOptimizationPass {
 public:
  // If enable_lazy_compilation is not nullopt then *enable_lazy_compilation
  // overrides --tf_xla_enable_lazy_compilation flag in deciding whether lazy
  // compilation is enabled.
  explicit BuildXlaOpsPass(
      absl::optional<bool> enable_lazy_compilation = absl::nullopt)
      : enable_lazy_compilation_(enable_lazy_compilation) {}

  Status Run(const GraphOptimizationPassOptions& options) override;

 private:
  absl::optional<bool> enable_lazy_compilation_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_BUILD_XLA_OPS_PASS_H_

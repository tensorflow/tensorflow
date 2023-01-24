/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_INT32_FULLTYPE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_INT32_FULLTYPE_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// An optimization (graph rewrite) pass to automatically set TFT_SHAPE_TENSOR
// full type information annotations for all int32 tensors, creating or
// modifying existing full type information as needed. This allows placement
// mechanisms using full type information to always place int32 on host.
class Int32Fulltype {
 public:
  // Creates an instance of the algorithm that sets TFT_SHAPE_TENSOR full
  // type information for all int32 tensors in the given Graph "graph".
  explicit Int32Fulltype(Graph* graph);

  ~Int32Fulltype();

  // For each node in this graph that outputs int32 tensors, set full
  // type information such that the int32 tensors use TFT_SHAPE_TENSOR.
  //
  // This method is not thread-safe.
  // Run() may be invoked at most once.
  Status Run();
  Status Run(const GraphOptimizationPassOptions& options);

 private:
  Graph* const graph_;  // Not owned.

  TF_DISALLOW_COPY_AND_ASSIGN(Int32Fulltype);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_INT32_FULLTYPE_H_

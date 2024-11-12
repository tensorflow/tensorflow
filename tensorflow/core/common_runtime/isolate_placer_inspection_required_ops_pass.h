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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_ISOLATE_PLACER_INSPECTION_REQUIRED_OPS_PASS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_ISOLATE_PLACER_INSPECTION_REQUIRED_OPS_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"

namespace tensorflow {
// Adds Identities for each input/output of function-calling ops.
//
// For example, the following graph calling a function on inputs `a` and `b`
// and producing output `y` will be rewritted to include identities on all
// edges:
//
//      a             b
//      |             |
//      v             v
//    f (PartitionedCallOp)
//         |
//         v
//         y
//
// is transformed to
//
//      a             b
//      |             |
//  a_f (Identity)   a_f (Identity)
//      |             |
//      v             v
//    f (PartitionedCallOp)
//         |
//      f_y (Identity)
//         |
//         v
//         y
//
// This pass is currently needed to simplify correctly placing the nodes
// producing inputs for as well as consuming output from function-calling ops.
//
// This pass should also help to implement replacing PartitionedCallOp with
// component function calls (to avoid copying input/output tensors), if we get
// to it.
class IsolatePlacerInspectionRequiredOpsPass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_ISOLATE_PLACER_INSPECTION_REQUIRED_OPS_PASS_H_

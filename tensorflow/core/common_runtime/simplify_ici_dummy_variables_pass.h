/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SIMPLIFY_ICI_DUMMY_VARIABLES_PASS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SIMPLIFY_ICI_DUMMY_VARIABLES_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/platform/status.h"

// Create new dummy zero variables to TPUExecute Op for ICI
// weight distribution, which is a critical feature in TF2/Min. The new dummy
// zero variables will be put on the same task as the TPUExecute Op. The old
// dummy zero variables will be removed afterwards.
//
// For example, in the following graph, the inputs to TPUExecute Op are on
// task:0, after the pass, the dummy zero variables will be put on task:2.
// which is the same as the TPUExecute.
//
// The graph before pass is:
//
//   node {name: "const0", op: "Const"}
//   node {name: "const1", op: "Const"}
//   node {name: "fill0", op: "Fill", input: "const1", input: "const0"}
//   node {name: "Identity0", op: "Identity", input: "fill0",
//     device: "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"
//     attr {
//       key: "_ici_weight_distribution_mlir_bridge_marker", value {b: true}
//     }
//   }
//   node {name: "const2", op: "Const"}
//   node {name: "const3", op: "Const"}
//   node {name: "fill1", op: "Fill", input: "const2", input: "const3"}
//   node {name: "identity1", op: "Identity", input: "fill1"
//     device: "/job:tpu_host_worker/replica:0/task:0/device:CPU:0"
//     attr {
//       key: "_ici_weight_distribution_mlir_bridge_marker", value {b: true}
//     }
//   }
//   node {name: "const4", op: "Const"}
//   node {name: "split0", op: "Split", input: "const4", input: "identity1"
//     attr {
//       key: "_ici_weight_distribution_mlir_bridge_marker"
//       value {b: true}
//     }
//   }
//   node {name: "TPUExecute0", op: "TPUExecute"
//     input: "identity0", input: "split0:1"
//     device: "/job:worker/replica:0/task:2/device:TPU:0"
//     attr {
//      key: "_parallel_execution_ids"
//      value {s: "r0:1,p0:2"}
//     }
//   }
//
// The graph after pass is:
//
//   node {name: "const0_dummy", op: "Const",
//     device: "/job:tpu_host_worker/replica:0/task:2/device:CPU:0"
//   }
//   node {name: "const1_dummy", op: "Const",
//     device: "/job:tpu_host_worker/replica:0/task:2/device:CPU:0"
//   }
//   node {name: "fill0_dummy", op: "Fill",
//     input: "const1_dummy", input: "const0_dummy",
//     device: "/job:tpu_host_worker/replica:0/task:2/device:CPU:0"
//   }
//   node {name: "const2_dummy", op: "Const",
//     device: "/job:tpu_host_worker/replica:0/task:2/device:CPU:0"
//   }
//   node {name: "const3_dummy", op: "Const",
//     device: "/job:tpu_host_worker/replica:0/task:2/device:CPU:0"
//   }
//   node {name: "fill1_dummy", op: "Fill",
//     input: "const2_dummy", input: "const3_dummy",
//     device: "/job:tpu_host_worker/replica:0/task:2/device:CPU:0"
//   }
//   node {name: "TPUExecute0", op: "TPUExecute"
//     input: "fill0_dummy", input: "fill1_dummy"
//     device: "/job:worker/replica:0/task:2/device:TPU:0"
//     attr {
//      key: "_parallel_execution_ids"
//      value {s: "r0:1,p0:2"}
//     }
//   }

namespace tensorflow {

// This pass will simplify the dummy variables for ICI weight distribution.
// The dummy variables will be put on the same task as the TPUExecute Op.
class SimplifyIciDummyVariablesPass : public GraphOptimizationPass {
 public:
  absl::Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SIMPLIFY_ICI_DUMMY_VARIABLES_PASS_H_

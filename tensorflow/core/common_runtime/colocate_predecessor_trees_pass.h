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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COLOCATE_PREDECESSOR_TREES_PASS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COLOCATE_PREDECESSOR_TREES_PASS_H_

#include "tensorflow/core/common_runtime/optimization_registry.h"

// TODO(b/344910755): Use the marker in Fill op to find the identity op. This
// makes the heuristic more straightforward.
// Colocate a tree of unplaced nodes with its placed Identity node. Identify a
// dangling tree of ops whose Identify nodes are assigned but rest of ops are
// not assigned. Then it should colocate the rest of the ops.
//
// For example, the graph before pass is:
//
//   node {
//     name: "const0"
//     op: "Const"
//   }
//   node {
//     name: "const1"
//     op: "Const"
//   }
//   node {
//     name: "fill0"
//     op: "Fill"
//     input: "const1"
//     input: "const0"
//   }
//   node {
//     name: "id0"
//     op: "Identity"
//     input: "fill0"
//     device: "/job:worker/replica:0/task:2/device:CPU:0"
//   }
//   node {
//     name: "id1"
//     op: "Identity"
//     input: "fill0"
//     device: "/job:worker/replica:0/task:2/device:CPU:0"
//   }
//
// The graph after pass is:
//
//   node {
//     name: "const0"
//     op: "Const"
//     attr {
//       key: "_class"
//       value {
//         list {
//           s: "loc:@id0"
//         }
//       }
//     }
//   }
//   node {
//     name: "const1"
//     op: "Const"
//     attr {
//       key: "_class"
//       value {
//         list {
//           s: "loc:@id0"
//         }
//       }
//     }
//   }
//   node {
//     name: "fill0"
//     op: "Fill"
//     input: "const1"
//     input: "const0"
//     attr {
//       key: "_class"
//       value {
//         list {
//           s: "loc:@id0"
//         }
//       }
//     }
//   }
//   node {
//     name: "id0"
//     op: "Identity"
//     input: "fill0"
//     device: "/job:worker/replica:0/task:2/device:CPU:0"
//     attr {
//       key: "_class"
//       value {
//         list {
//           s: "loc:@id0"
//         }
//       }
//     }
//   }
//   node {
//     name: "id1"
//     op: "Identity"
//     input: "fill0"
//     device: "/job:worker/replica:0/task:2/device:CPU:0"
//     attr {
//       key: "_class"
//       value {
//         list {
//           s: "loc:@id0"
//         }
//       }
//     }
//   }

namespace tensorflow {

// This pass can place each tree of unassigned nodes with its Identity nodes,
// when the Identity nodes are already assigned to a device. Placement is
// instructed here with the colocation class attribute _class. This is a good
// heuristic because it reduces number of cut edges and tends to load balance.
class ColocatePredecessorTreesPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COLOCATE_PREDECESSOR_TREES_PASS_H_

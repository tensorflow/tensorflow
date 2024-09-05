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
#ifndef TENSORFLOW_CORE_TFRT_UTILS_GRAPH_PARTITION_H_
#define TENSORFLOW_CORE_TFRT_UTILS_GRAPH_PARTITION_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tfrt_stub {

// Inserts send/recv ops to `graph` if nodes are assigned to multiple devices.
// Specifically, nodes on the same device will be wrapped in a function and
// invoked by a PartitionedCall op. All PartitionedCall ops are connected to a
// StatefulPartitionedCall op (which behaves as a 'stateful IdentityN') to
// protect them from being pruned in the subsequent MLIR lowering passes
// (b/232026253).
//
// The following shows a simple example of using this method.
//
// The original graph has four nodes that are placed on different devices.
//
//        ----->  op1(host)  ------
//       /                         \
//   input(host)               output(host)
//       \                         /
//        -----> op2(device) ------
//
// Calling this method will return the following graph, where `op1` is wrapped
// in the function invoked by `PartitionedCall_1`, and `op2` is wrapped in the
// function invoked by `PartitionedCall_2`. Both of them have a data dependency
// with the `StatefulPartitionedCall` op.
//
//   input ---> PartitionedCall_1 ----
//                                    \
//                         StatefulPartitionedCall ---> output
//                                    /
//              PartitionedCall_2 ----
//
absl::StatusOr<std::unique_ptr<Graph>> InsertTransferOps(
    const std::string& graph_func_name, const DeviceSet& device_set,
    const Device* host_device, const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& control_outputs,
    std::unique_ptr<Graph> graph);

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_UTILS_GRAPH_PARTITION_H_

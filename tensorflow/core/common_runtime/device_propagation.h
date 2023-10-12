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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_PROPAGATION_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_PROPAGATION_H_

#include <functional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace tensorflow {

namespace device_propagation {

typedef std::function<bool(StringPiece)> DeviceFilter;
typedef std::function<bool(const Node&)> NodeFilter;
}  // namespace device_propagation

// Propagates device assignments from a certain types of nodes to their outputs
// to avoid unnecessary D2H or H2D copies.
// If an node satisfies the following conditions, it will be placed on the same
// device as its inputs:
//   (1) The node can accept device update (`node_filter` returns true).
//   (2) The node itself has no requested or assigned devices.
//   (3) The source nodes of this node's input edges, except for edges that are
//   "LoopCond->Switch" or "Enter->Merge", are all placed on the same device.
//   (4) The device can be propagated (`device_filter` returns true)
void PropagateDevices(const device_propagation::NodeFilter& node_filter,
                      const device_propagation::DeviceFilter& device_filter,
                      Graph* graph);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_PROPAGATION_H_

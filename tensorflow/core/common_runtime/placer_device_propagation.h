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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PLACER_DEVICE_PROPAGATION_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PLACER_DEVICE_PROPAGATION_H_

#include <functional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace tensorflow {

typedef std::function<bool(StringPiece)> IsPropagatableDeviceFn;

// Propagates device assignments from a certain types of nodes to their outputs
// to avoid unnecessary D2H or H2D copies.
// If an op satisfies the following conditions, it will be placed on the same
// device as its inputs:
//   (1) The op can be placed on the device (in the `target_ops`)
//   (2) The op itself has no requested or assigned devices.
//   (3) All the data inputs of this op are placed on the same device and the
//       device type can be propagated (checked by `is_propagatable`)
//       There are exceptions like the NextIterations input of Switch node can
//       be placed on CPU as it is just a boolean.
void PropagateDevices(const absl::flat_hash_set<std::string>& target_ops,
                      IsPropagatableDeviceFn is_propagatable, Graph* graph);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PLACER_DEVICE_PROPAGATION_H_

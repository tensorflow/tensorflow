/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PLACER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PLACER_H_

#include <string>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

// A placement algorithm that assigns the nodes of the given Graph to
// devices the given DeviceSet, respecting the following constraints:
//
// 1. Existing device assignments remain unchanged.
// 2. Requested (partial or complete) device specifications given by device name
//    for each node are granted.
// 3. Nodes connected by edges of a reference type are colocated on
//    the same device.
// 4. Given nodes "A" and "B", if node "B" has a colocation group
//    "@loc:A", nodes "A" and "B" will be colocated on the same device.
//
// The implementation builds a constraint graph with the same set of
// nodes, and edges that represent colocation constraints between
// nodes.  Each connected component in the resulting constraint graph
// is then assigned to a set of valid devices.
//
// Run() will finally assign the device to each node given the list of
// possible devices.
//
// TODO(mrry): "Soft" constraints, such as "place node 'x' as close as
// possible to node 'y' while respecting the other constraints"?
// TODO(mrry): Create a common interface for this and the other
// placement algorithms so that they may be injected into the graph
// builder.
class Placer {
 public:
  // Creates an instance of the Placer algorithm for the given
  // Graph "graph" (nodes in which may or may not be assigned) on the
  // given DeviceSet "devices".
  // "function_name" should be set to the name of the function whose body is
  // represented by "graph". If "graph" is not representing a function body,
  // "function_name" should be empty.
  //
  // If non-null, default_local_device is used where possible as a placement for
  // nodes which do not have a device specified, ahead of other devices which
  // would otherwise be higher priority. default_local_device should be on the
  // local host so that its FLR is directly accessible by the current process.
  //
  // The "graph", "devices", and "default_local_device" pointer arguments are
  // borrowed by this Placer, and must outlive it.
  Placer(Graph* graph, const string& function_name,
         const FunctionLibraryDefinition* flib_def, const DeviceSet* devices,
         const Device* default_local_device, bool allow_soft_placement,
         bool log_device_placement);
  Placer(Graph* graph, const string& function_name,
         const FunctionLibraryDefinition* flib_def, const DeviceSet* devices);
  Placer(Graph* graph, const string& function_name,
         const FunctionLibraryDefinition* flib_def, const DeviceSet* devices,
         const Device* default_local_device);

  ~Placer();

  // Assigns each node in this Placer's graph to a device in its
  // set of devices.
  //
  // This method is not thread-safe.
  // Run() may be invoked at most once.
  Status Run();

 private:
  // Returns true if the device type of 'candidate_device_name' is
  // found in 'devices'.
  bool CanAssignToDevice(const string& candidate_device_name,
                         const std::vector<Device*>& devices) const;

  Graph* const graph_;  // Not owned.
  const string function_name_;
  const FunctionLibraryDefinition* const flib_def_;  // Not owned.
  const DeviceSet* const devices_;                   // Not owned.
  const Device* default_local_device_;               // Not owned.
  const bool allow_soft_placement_;
  const bool log_device_placement_;

  TF_DISALLOW_COPY_AND_ASSIGN(Placer);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PLACER_H_

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_INSPECTING_PLACER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_INSPECTING_PLACER_H_

#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/placer_inspection_required_ops_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

// TODO(iga): Convert this struct into a class to ensure invariants between
// device names, i.e.
//  DeviceNameUtils::IsSpecification(resource_device_name,
//                                   requested_device_name)
// PossibleDevices does not contain assigned_device_name because we don't
// assign devices to nested functions.
struct PossibleDevices {
  // The same as Member::requested_device_name_ in colocation_graph.cc.
  DeviceNameUtils::ParsedName requested_device_name;

  // The same as Member::resource_device_name_ in colocation_graph.cc.
  DeviceNameUtils::ParsedName resource_device_name;

  // A device type outside of this set will not be supported by some
  // internal op.
  PrioritizedDeviceTypeVector device_types;
};

// A struct for communicating constraints on devices that can
// be chosen for inputs and outputs of an op requiring deep placer inspection.
struct IOColocationGroups {
  // input_groups[i] contains the group id that i'th input belongs to.
  // List inputs are not supported.
  std::vector<int> input_groups;
  // output_groups[i] contains the group id that i'th output belongs to.
  // List inputs are not supported.
  std::vector<int> output_groups;
  // group_devices[i] contains possible devices for group with id i.
  std::vector<PossibleDevices> group_devices;

  string DebugString() const;
};

class InspectingPlacer {
 public:
  // graph and device_set must not be null and must outlive this
  // InspectingPlacer. default_device can be null. If not, must outlive this.
  // TODO(iga): Add a "stack trace" to detect recursion and improve log
  // messages. Currently, we will enter an infinite loop for recursive
  // functions.
  InspectingPlacer(const FunctionStack& stack,
                   const FunctionLibraryDefinition* flib_def,
                   const DeviceSet* device_set, const Device* default_device,
                   bool allow_soft_placement, bool log_device_placement);

  // `node` must be
  // PlacerInspectionRequiredOpsChecker::IsPlacerInspectionRequired.
  absl::Status ComputeIOColocationGroups(const Node& node,
                                         IOColocationGroups* groups);

 private:
  const FunctionStack stack_;
  const FunctionLibraryDefinition& flib_def_;
  const DeviceSet& device_set_;
  const Device* default_device_;
  const bool allow_soft_placement_;
  const bool log_device_placement_;

  InspectingPlacer(const InspectingPlacer&) = delete;
  void operator=(const InspectingPlacer&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_INSPECTING_PLACER_H_

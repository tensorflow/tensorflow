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

#ifndef TENSORFLOW_GRAPPLER_COSTS_UTILS_H_
#define TENSORFLOW_GRAPPLER_COSTS_UTILS_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {

// Returns a vector of InputProperties for 'node'. The vector will contain one
// entry for each input of 'node'.
// For each node in the graph, the 'name_to_cost' map stores a pointer to the
// corresponding cost graph node indexed by node name. The 'name_to_node' maps a
// node name to its node definition.
std::vector<OpInfo::TensorProperties> FindInputFeatures(
    const NodeDef& node,
    const std::unordered_map<string, const CostGraphDef::Node*>& name_to_cost,
    const std::unordered_map<string, const NodeDef*>& name_to_node);

// Returns the DeviceProperties of the device on which 'node' runs.
DeviceProperties GetDeviceInfo(const CostGraphDef::Node& node);
DeviceProperties GetDeviceInfo(const string& device_str);

// Builds the OpInfo proto for node, given all nodes in the graph, the node's
// device and its input properties which are typically built by shape inference
// or calling FindInputFeatures.
OpInfo BuildOpInfo(
    const NodeDef& node, const string& device_str,
    const std::unordered_map<string, const NodeDef*>& name_to_node,
    const std::vector<OpInfo::TensorProperties>& inputs);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_COSTS_UTILS_H_

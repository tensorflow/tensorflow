/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Helper functions for TPU rewrite passes.

#ifndef TENSORFLOW_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_REWRITE_HELPERS_H_
#define TENSORFLOW_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_REWRITE_HELPERS_H_

#include "xla/status_macros.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class DistributedTPURewriteHelpers {
 public:
  // Given a user-assigned device string, system_spec_string, parse it into
  // system_spec. Verify that the device type is either TPU_SYSTEM or
  // unassigned, and in the latter case set it to TPU_SYSTEM:0. Having set the
  // type, verify that the spec matches a unique device in device_set, and
  // return that device in system_device. The normal use case is for
  // system_spec_string to identify the TPU_SYSTEM on replica 0, task 0 of the
  // job that contains the TPU hardware.
  // TODO(b/110910013): Possibly remove the tpu system device.
  static Status GetSystemDevice(const string& system_spec_string,
                                const DeviceSet& device_set,
                                DeviceNameUtils::ParsedName* system_spec,
                                Device** system_device);

  // Given a parsed system spec (e.g., the one returned above from
  // GetSystemDeviceName), return in host_devices the TPU_SYSTEM:0 device on
  // every host in the spec's job. If the spec does not include an explicit job,
  // "localhost" is used.  Returns an error if system_spec matches devices from
  // a multiple jobs or replicas.
  static Status GetHostSystemDevices(
      const DeviceNameUtils::ParsedName& system_spec,
      const DeviceSet& device_set, std::vector<Device*>* host_system_devices);

  // Given a parsed system spec (e.g., the one returned above from
  // GetSystemDeviceName), sets `*tpu_devices` to a per-host vector of the TPU
  // devices on every host in the spec's job. If the spec does not include an
  // explicit job, "localhost" is used. Sets `*num_tpus_per_host` to the number
  // of TPU devices in each host, and verifies that each host in the job has
  // the same number of TPU devices.
  // Returns an error if system_spec matches devices from a multiple jobs or
  // replicas.
  static Status GetTPUDevices(const DeviceNameUtils::ParsedName& system_spec,
                              const DeviceSet& device_set,
                              int* num_tpus_per_host,
                              std::vector<std::vector<Device*>>* tpu_devices);

  // Perform 'action' on every node in 'graph' of type
  // 'node_type'. This function is designed for use with configuration
  // Ops that have no inputs or outputs. The arguments passed to 'action' are:
  // 'configuration_node_name': the name of the node that matched
  // 'configuration_device_name': the name of the device that the
  // matching node is placed on
  // 'host_devices': the set of TPU_SYSTEM devices on hosts with TPUs that are
  // in the same system as the node that matched.
  // 'input_dependencies': the set of nodes that have control edges to
  // the matching node.
  // 'output_dependencies': the set of output port, destination node, input port
  // triples that have edges from the matching node. Input port is
  // Graph::kControlSlot for a control edge.
  // 'graph': the graph being mutated.
  struct OutputDependency {
    int src_output;
    Node* dst;
    int dst_input;
  };
  static Status ForConfigurationNodeMatchingType(
      const string& node_type, Graph* graph, const DeviceSet& device_set,
      const std::function<
          Status(const NodeDef& configuration_node_def,
                 const string& configuration_device_name,
                 const std::vector<Device*>& host_devices,
                 const std::vector<Node*>& input_dependencies,
                 const std::vector<OutputDependency>& output_dependencies,
                 Graph* graph)>& action);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_GRAPH_REWRITE_DISTRIBUTED_TPU_REWRITE_HELPERS_H_

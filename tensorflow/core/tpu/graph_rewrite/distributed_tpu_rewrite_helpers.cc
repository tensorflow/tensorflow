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

#include "tensorflow/core/tpu/graph_rewrite/distributed_tpu_rewrite_helpers.h"

#include <algorithm>
#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "xla/status_macros.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tsl/platform/errors.h"

namespace tensorflow {

// LINT.IfChange
absl::Status DistributedTPURewriteHelpers::GetSystemDevice(
    const string& system_spec_string, const DeviceSet& device_set,
    DeviceNameUtils::ParsedName* system_spec, Device** system_device) {
  if (!DeviceNameUtils::ParseFullName(system_spec_string, system_spec)) {
    system_spec->Clear();
  }

  // Callers may have relied on an Op only being registered on TPU_SYSTEM
  // devices to ensure the Op is placed there. Augment the device spec to make
  // the device type explicit.
  if (!system_spec->has_type || system_spec->type != DEVICE_TPU_SYSTEM) {
    system_spec->type = DEVICE_TPU_SYSTEM;
    system_spec->has_type = true;
    system_spec->id = 0;
    system_spec->has_id = true;
  }

  std::vector<Device*> system_devices;
  device_set.FindMatchingDevices(*system_spec, &system_devices);
  if (system_devices.empty()) {
    if (system_spec_string.empty()) {
      return errors::InvalidArgument(
          "No TPU_SYSTEM device found. Please ensure that you're connected to "
          "a host with a TPU_SYSTEM device.");
    }
    return errors::InvalidArgument("No matching devices found for '",
                                   system_spec_string, "'");
  } else if (system_devices.size() > 1) {
    // Validate that all system devices are part of the same job.
    std::unordered_set<string> job_names;
    for (auto device : system_devices) {
      const auto& parsed_name = device->parsed_name();
      TF_RET_CHECK(parsed_name.has_job);
      job_names.insert(parsed_name.job);
    }
    if (job_names.size() > 1) {
      return errors::InvalidArgument(
          "System devices cannot be part "
          "of multiple different jobs.  Found: ",
          absl::StrJoin(job_names, ","));
    }

    // Identify the lexicographically first device from the list of
    // valid TPU SYSTEM devices, so that every process in the same
    // 'cluster' definition uses the same system device.
    std::sort(system_devices.begin(), system_devices.end(),
              [](Device* i, Device* j) {
                auto i_name = i->parsed_name();
                auto j_name = j->parsed_name();
                if (i_name.replica != j_name.replica) {
                  return i_name.replica < j_name.replica;
                }
                return i_name.task < j_name.task;
              });
  }

  *system_device = system_devices[0];
  if (!DeviceNameUtils::ParseFullName((*system_device)->name(), system_spec)) {
    return errors::InvalidArgument("Unable to re-parse system device name ",
                                   (*system_device)->name(),
                                   " as a device spec.");
  }
  return absl::OkStatus();
}
// LINT.ThenChange(//tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.cc)

// LINT.IfChange
absl::Status DistributedTPURewriteHelpers::GetHostSystemDevices(
    const DeviceNameUtils::ParsedName& system_spec, const DeviceSet& device_set,
    std::vector<Device*>* host_system_devices) {
  DeviceNameUtils::ParsedName host_spec;
  if (system_spec.has_job) {
    // The system Op has been explicitly assigned to a job, so we want
    // all the hosts in that job.
    CHECK(DeviceNameUtils::ParseFullName(
        strings::StrCat("/job:", system_spec.job, "/device:", DEVICE_TPU_SYSTEM,
                        ":0"),
        &host_spec));
  } else {
    // The system Op has not been explicitly assigned to a
    // job, so take all hosts in the system. There will be a runtime
    // error if some of those hosts don't contain TPU devices.
    CHECK(DeviceNameUtils::ParseFullName(
        strings::StrCat("/device:", DEVICE_TPU_SYSTEM, ":0"), &host_spec));
  }
  device_set.FindMatchingDevices(host_spec, host_system_devices);

  TF_RET_CHECK(!host_system_devices->empty())
      << "No hosts found matching device spec "
      << DeviceNameUtils::ParsedNameToString(host_spec);

  // Check that all the devices belong to the same job.
  TF_RET_CHECK((*host_system_devices)[0]->parsed_name().has_job);
  const string& job_name = (*host_system_devices)[0]->parsed_name().job;
  int replica = (*host_system_devices)[0]->parsed_name().replica;
  for (const auto host_device : *host_system_devices) {
    const auto& parsed_name = host_device->parsed_name();
    TF_RET_CHECK(parsed_name.has_job);
    if (parsed_name.job != job_name) {
      return errors::InvalidArgument(
          "All TPU host devices must be in the same job");
    }
    TF_RET_CHECK(parsed_name.has_replica);
    if (parsed_name.replica != replica) {
      return errors::InvalidArgument(
          "All TPU host devices must be in the same replica");
    }
  }

  // Sort the devices by replica and then task.
  std::sort(host_system_devices->begin(), host_system_devices->end(),
            [](Device* i, Device* j) {
              auto i_name = i->parsed_name();
              auto j_name = j->parsed_name();
              return i_name.task < j_name.task;
            });
  return absl::OkStatus();
}
// LINT.ThenChange(//tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.cc)

// LINT.IfChange
absl::Status DistributedTPURewriteHelpers::GetTPUDevices(
    const DeviceNameUtils::ParsedName& system_spec, const DeviceSet& device_set,
    int* num_tpus_per_host, std::vector<std::vector<Device*>>* tpu_devices) {
  // GetHostSystemDevices returns the CPU device on each host that is
  // going to be used for executing TPU code.
  std::vector<Device*> host_system_devices;
  TF_RETURN_IF_ERROR(DistributedTPURewriteHelpers::GetHostSystemDevices(
      system_spec, device_set, &host_system_devices));

  // Enumerate all the physical devices. Enumerate devices on task 0,
  // then task 1, etc.
  std::sort(host_system_devices.begin(), host_system_devices.end(),
            [](Device* i, Device* j) {
              return i->parsed_name().task < j->parsed_name().task;
            });

  *num_tpus_per_host = 0;
  tpu_devices->clear();
  tpu_devices->reserve(host_system_devices.size());
  for (const auto device : host_system_devices) {
    // Make a copy of the parsed name because we are going to change it.
    DeviceNameUtils::ParsedName device_spec = device->parsed_name();
    device_spec.has_type = true;
    device_spec.type = "TPU";
    // Enumerate all the available TPUs.
    device_spec.has_id = false;
    std::vector<Device*> host_tpu_devices;
    device_set.FindMatchingDevices(device_spec, &host_tpu_devices);
    // Sort the devices by device id.
    std::sort(host_tpu_devices.begin(), host_tpu_devices.end(),
              [](Device* i, Device* j) {
                return i->parsed_name().id < j->parsed_name().id;
              });
    if (tpu_devices->empty()) {
      // First iteration: set *num_tpus_per_host to the number of TPUs on the
      // first host.
      *num_tpus_per_host = host_tpu_devices.size();
    } else if (*num_tpus_per_host != host_tpu_devices.size()) {
      // Subsequent iterations: check the number of TPUs match the number on
      // the first host.
      return errors::InvalidArgument(
          "Mismatched number of TPU devices in cluster ", *num_tpus_per_host,
          " vs. ", host_tpu_devices.size());
    }
    tpu_devices->push_back(std::move(host_tpu_devices));
  }
  return absl::OkStatus();
}
// LINT.ThenChange(//tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.cc)

absl::Status DistributedTPURewriteHelpers::ForConfigurationNodeMatchingType(
    const string& node_type, Graph* graph, const DeviceSet& device_set,
    const std::function<
        absl::Status(const NodeDef& configuration_node_def,
                     const string& configuration_device_name,
                     const std::vector<Device*>& host_devices,
                     const std::vector<Node*>& input_dependencies,
                     const std::vector<OutputDependency>& output_dependencies,
                     Graph* graph)>& action) {
  // Find all the matching nodes before mutating the graph.
  std::vector<Node*> nodes;
  for (Node* node : graph->nodes()) {
    if (node->type_string() == node_type) {
      nodes.push_back(node);
    }
  }

  for (Node* node : nodes) {
    string spec_string = node->requested_device();
    DeviceNameUtils::ParsedName spec;
    Device* device;
    TF_RETURN_IF_ERROR(
        GetSystemDevice(spec_string, device_set, &spec, &device));
    const string& device_name = device->name();

    std::vector<Device*> host_devices;
    TF_RETURN_IF_ERROR(GetHostSystemDevices(spec, device_set, &host_devices));

    std::vector<Node*> input_dependencies;
    for (const Edge* edge : node->in_edges()) {
      // Config ops have no inputs, so all edges must be control edges.
      CHECK(edge->IsControlEdge());
      input_dependencies.push_back(edge->src());
    }
    std::vector<OutputDependency> output_dependencies;
    for (const Edge* edge : node->out_edges()) {
      OutputDependency dep;
      dep.src_output = edge->src_output();
      dep.dst = edge->dst();
      dep.dst_input = edge->dst_input();
      output_dependencies.push_back(dep);
    }
    NodeDef node_def = node->def();

    // Remove the node now so we can insert a new node with the same
    // name inside the action.
    graph->RemoveNode(node);

    TF_RETURN_IF_ERROR(action(node_def, device_name, host_devices,
                              input_dependencies, output_dependencies, graph));
  }

  return absl::OkStatus();
}

}  // namespace tensorflow

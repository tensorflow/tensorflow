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
#include "tensorflow/core/tpu/tpu_global_init.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"
#include "tensorflow/core/tfrt/common/pjrt_client_factory_registry.h"
#include "tensorflow/core/tfrt/common/pjrt_util.h"
#include "tensorflow/core/tpu/tpu_init_mode.h"

namespace tensorflow {

namespace {

ABSL_CONST_INIT static absl::Mutex global_init_tpu_mutex(absl::kConstInit);
static tpu::TopologyProto* global_tpu_topology
    ABSL_GUARDED_BY(global_init_tpu_mutex) = nullptr;

constexpr char kTaskSpec[] = "/job:localhost/replica:0/task:0";

}  // namespace

absl::Status BuildTopologyProto(xla::PjRtClient* client,
                                tpu::TopologyProto* topology) {
  int max_x = -1, max_y = -1, max_z = -1;
  int max_core = -1;
  absl::flat_hash_set<int> task_indices;

  for (xla::PjRtDevice* device : client->devices()) {
    auto coords_it = device->description().Attributes().find("coords");
    auto core_it = device->description().Attributes().find("core_on_chip");
    if (coords_it == device->description().Attributes().end() ||
        core_it == device->description().Attributes().end()) {
      return absl::InternalError(
          "TPU device missing coords or core_on_chip attributes");
    }
    const auto& coords = std::get<std::vector<int64_t>>(coords_it->second);
    int64_t core = std::get<int64_t>(core_it->second);

    max_x = std::max(max_x, static_cast<int>(coords[0]));
    max_y = std::max(max_y, static_cast<int>(coords[1]));
    max_z = std::max(max_z, static_cast<int>(coords[2]));
    max_core = std::max(max_core, static_cast<int>(core));

    task_indices.insert(device->process_index());
  }

  if (max_x == -1) {
    return absl::InternalError("No TPU devices found to build topology");
  }

  int num_tasks = task_indices.size();
  topology->set_num_tasks(num_tasks);

  topology->add_mesh_shape(max_x + 1);
  topology->add_mesh_shape(max_y + 1);
  topology->add_mesh_shape(max_z + 1);
  topology->add_mesh_shape(max_core + 1);

  std::vector<int> sorted_tasks(task_indices.begin(), task_indices.end());
  std::sort(sorted_tasks.begin(), sorted_tasks.end());

  absl::flat_hash_map<int, std::vector<xla::PjRtDevice*>> task_to_devices;
  for (xla::PjRtDevice* device : client->devices()) {
    task_to_devices[device->process_index()].push_back(device);
  }

  for (auto& [task, devices] : task_to_devices) {
    std::sort(devices.begin(), devices.end(),
              [](const xla::PjRtDevice* a, const xla::PjRtDevice* b) {
                return a->global_device_id() < b->global_device_id();
              });
  }

  int num_tpu_devices_per_task = 0;
  if (!sorted_tasks.empty()) {
    num_tpu_devices_per_task = task_to_devices[sorted_tasks[0]].size();
  }
  topology->set_num_tpu_devices_per_task(num_tpu_devices_per_task);

  for (int task_id : sorted_tasks) {
    const auto& devices = task_to_devices[task_id];
    if (devices.size() != num_tpu_devices_per_task) {
      return absl::InternalError(
          "Asymmetric TPU topology (different number of devices per task)");
    }
    for (const xla::PjRtDevice* device : devices) {
      const auto& coords = std::get<std::vector<int64_t>>(
          device->description().Attributes().at("coords"));
      int64_t core = std::get<int64_t>(
          device->description().Attributes().at("core_on_chip"));
      topology->add_device_coordinates(coords[0]);
      topology->add_device_coordinates(coords[1]);
      topology->add_device_coordinates(coords[2]);
      topology->add_device_coordinates(core);
    }
  }

  topology->mutable_tpu_hardware_feature()->set_embedding_feature(
      tpu::TPUHardwareFeature::V2);

  return absl::OkStatus();
}

// NOTE: Session would have been the obvious first choice to run the graph
// here, but instead we use a GraphRunner because Session creates a global
// EigenThreadPool based on the SessionOptions it receives the first time it
// runs. This means that we need to create the right options and pass it to this
// API to make it work correctly. We felt it was an onerous restriction to place
// on the API, so we went with the current approach.
absl::Status InitializeTPUSystemGlobally(Env* env,
                                         tpu::TopologyProto* tpu_topology) {
  absl::MutexLock lock(global_init_tpu_mutex);
  if (global_tpu_topology != nullptr) {
    *tpu_topology = *global_tpu_topology;
    return absl::OkStatus();
  }

  auto obtained_pjrt_client = GetPjRtClient(DeviceType(DEVICE_TPU));
  if (!obtained_pjrt_client.ok() &&
      obtained_pjrt_client.status().code() == absl::StatusCode::kNotFound) {
    VLOG(1) << "PJRT client not found for TPU, creating it...";
    xla::PjrtClientFactoryOptions options;
    auto created_client_or =
        xla::PjrtClientFactoryRegistry::Get().GetPjrtClient(
            DeviceType(DEVICE_TPU), options);
    if (!created_client_or.ok()) {
      return absl::InternalError(
          absl::StrCat("Failed to create PJRT client for TPU: ",
                       created_client_or.status().message()));
    }
    VLOG(1) << "PJRT client created successfully for TPU.";
    TF_RETURN_IF_ERROR(SetPjRtClientInTFGlobalResourceManager(
        DeviceType(DEVICE_TPU), std::move(*created_client_or)));
    obtained_pjrt_client = GetPjRtClient(DeviceType(DEVICE_TPU));
  }
  TF_RETURN_IF_ERROR(obtained_pjrt_client.status());

  LOG(INFO) << "Using PJRT for global TPU initialization";
  TF_RETURN_IF_ERROR(SetTPUInitMode(TPUInitMode::kGlobal));

  global_tpu_topology = new tpu::TopologyProto();
  TF_RETURN_IF_ERROR(
      BuildTopologyProto(*obtained_pjrt_client, global_tpu_topology));
  *tpu_topology = *global_tpu_topology;
  return absl::OkStatus();
}

absl::Status InitializeTPUSystemGlobally() {
  tensorflow::tpu::TopologyProto tpu_topology;
  return InitializeTPUSystemGlobally(tensorflow::Env::Default(), &tpu_topology);
}

}  // namespace tensorflow

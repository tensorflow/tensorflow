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

#include "tensorflow/compiler/xla/service/computation_placer.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/host/host_platform_id.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"

using absl::StrAppend;
using absl::StrCat;

namespace xla {

StatusOr<DeviceAssignment::LogicalID> DeviceAssignment::LogicalIdForDevice(
    GlobalDeviceId device_id) const {
  absl::optional<DeviceAssignment::LogicalID> logical_id;
  for (int r = 0; r < replica_count(); ++r) {
    for (int c = 0; c < computation_count(); ++c) {
      if ((*this)(r, c) == device_id.value()) {
        if (logical_id.has_value()) {
          return InternalError(
              "Device %d appears twice in DeviceAssignment: %s",
              device_id.value(), ToString());
        }
        logical_id.emplace(DeviceAssignment::LogicalID{r, c});
      }
    }
  }
  if (logical_id.has_value()) {
    return *logical_id;
  } else {
    return InternalError("Device %d doesn't appear in DeviceAssignment: %s",
                         device_id.value(), ToString());
  }
}

StatusOr<int> DeviceAssignment::ReplicaIdForDevice(
    GlobalDeviceId device_id) const {
  TF_ASSIGN_OR_RETURN(const LogicalID logical_id,
                      LogicalIdForDevice(device_id));
  return logical_id.replica_id;
}

Status DeviceAssignment::Serialize(DeviceAssignmentProto* proto) const {
  proto->set_replica_count(replica_count());
  proto->set_computation_count(computation_count());
  for (int computation = 0; computation < computation_count(); ++computation) {
    DeviceAssignmentProto::ComputationDevice* computation_device =
        proto->add_computation_devices();
    for (int replica = 0; replica < replica_count(); ++replica) {
      computation_device->add_replica_device_ids((*this)(replica, computation));
    }
  }
  return Status::OK();
}

/* static */ StatusOr<std::unique_ptr<DeviceAssignment>>
DeviceAssignment::Deserialize(const DeviceAssignmentProto& proto) {
  TF_RET_CHECK(proto.computation_devices_size() == proto.computation_count());
  if (proto.replica_count() <= 0 || proto.computation_count() <= 0) {
    return InvalidArgument(
        "Invalid device assignment topology: replica_count=%d, "
        "computation_count=%d",
        proto.replica_count(), proto.computation_count());
  }
  auto assignment = absl::make_unique<DeviceAssignment>(
      proto.replica_count(), proto.computation_count());
  for (int computation = 0; computation < proto.computation_count();
       ++computation) {
    const auto& computation_device = proto.computation_devices(computation);
    TF_RET_CHECK(computation_device.replica_device_ids_size() ==
                 proto.replica_count());
    for (int replica = 0; replica < proto.replica_count(); ++replica) {
      (*assignment)(replica, computation) =
          computation_device.replica_device_ids(replica);
    }
  }
  return std::move(assignment);
}

std::string DeviceAssignment::ToString() const {
  std::string output = StrCat("Computations: ", computation_count(),
                              " Replicas: ", replica_count(), "\n");
  for (int computation = 0; computation < computation_count(); ++computation) {
    StrAppend(&output, "Computation ", computation, ": ");
    for (int replica = 0; replica < replica_count(); ++replica) {
      StrAppend(&output, operator()(replica, computation), " ");
    }
    StrAppend(&output, "\n");
  }
  return output;
}

StatusOr<int> ComputationPlacer::DeviceId(int replica, int computation,
                                          int replica_count,
                                          int computation_count) {
  TF_RET_CHECK(replica < replica_count);
  TF_RET_CHECK(computation < computation_count);

  return computation * replica_count + replica;
}

StatusOr<DeviceAssignment> ComputationPlacer::AssignDevices(
    int replica_count, int computation_count) {
  DeviceAssignment assignment(replica_count, computation_count);
  for (int replica = 0; replica < replica_count; ++replica) {
    for (int computation = 0; computation < computation_count; ++computation) {
      TF_ASSIGN_OR_RETURN(
          int device_id,
          DeviceId(replica, computation, replica_count, computation_count));
      assignment(replica, computation) = device_id;
    }
  }
  return std::move(assignment);
}

/* static */ void ComputationPlacer::RegisterComputationPlacer(
    se::Platform::Id platform_id,
    ComputationPlacerCreationFunction creation_function) {
  tensorflow::mutex_lock lock(
      ComputationPlacer::platform_computation_placer_mutex_);
  auto* computation_placers = GetPlatformComputationPlacers();
  CHECK(computation_placers->find(platform_id) == computation_placers->end());
  (*computation_placers)[platform_id].creation_function = creation_function;
}

/* static */ StatusOr<ComputationPlacer*> ComputationPlacer::GetForPlatform(
    const se::Platform* platform) {
  tensorflow::mutex_lock lock(
      ComputationPlacer::platform_computation_placer_mutex_);
  auto* computation_placers = GetPlatformComputationPlacers();

  auto it = computation_placers->find(platform->id());
  if (it == computation_placers->end()) {
    return NotFound(
        "could not find registered computation placer for platform %s -- check "
        "target linkage",
        platform->Name());
  }

  if (it->second.placer == nullptr) {
    // Lazily create the computation placer the first time it is needed.
    it->second.placer = (*it->second.creation_function)();
  }

  return it->second.placer.get();
}

/* static */ tensorflow::mutex
    ComputationPlacer::platform_computation_placer_mutex_(
        tensorflow::LINKER_INITIALIZED);

/* static */ std::map<se::Platform::Id, ComputationPlacer::State>*
ComputationPlacer::GetPlatformComputationPlacers() {
  static auto* r = new std::map<se::Platform::Id, ComputationPlacer::State>;
  return r;
}

}  // namespace xla

static std::unique_ptr<xla::ComputationPlacer> CreateComputationPlacer() {
  return absl::make_unique<xla::ComputationPlacer>();
}

static bool InitModule() {
  xla::ComputationPlacer::RegisterComputationPlacer(
      stream_executor::host::kHostPlatformId, &CreateComputationPlacer);
  xla::ComputationPlacer::RegisterComputationPlacer(
      stream_executor::cuda::kCudaPlatformId, &CreateComputationPlacer);
  xla::ComputationPlacer::RegisterComputationPlacer(
      stream_executor::rocm::kROCmPlatformId, &CreateComputationPlacer);
  return true;
}
static bool module_initialized = InitModule();

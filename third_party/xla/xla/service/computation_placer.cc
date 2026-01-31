/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/computation_placer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/runtime/device_id.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

using absl::StrAppend;

namespace xla {

absl::StatusOr<DeviceAssignment::LogicalID>
DeviceAssignment::LogicalIdForDevice(GlobalDeviceId device_id) const {
  std::optional<LogicalID> res;
  int64_t id = device_id.value();
  for (int r = 0; r < replica_count(); ++r) {
    for (int c = 0; c < computation_count(); ++c) {
      if (operator()(r, c) == device_id.value()) {
        if (res.has_value()) {
          return Internal("Device %d not unique in %v", id, *this);
        }
        res = LogicalID{r, c};
      }
    }
  }
  if (!res.has_value()) {
    return Internal("Device %d doesn't appear in %v", id, *this);
  }
  return res.value();
}

absl::StatusOr<int> DeviceAssignment::ReplicaIdForDevice(
    GlobalDeviceId device_id) const {
  TF_ASSIGN_OR_RETURN(const LogicalID logical_id,
                      LogicalIdForDevice(device_id));
  return logical_id.replica_id;
}

absl::StatusOr<int> DeviceAssignment::PartitionIdForDevice(
    GlobalDeviceId device_id) const {
  TF_ASSIGN_OR_RETURN(const LogicalID logical_id,
                      LogicalIdForDevice(device_id));
  return logical_id.computation_id;
}

absl::flat_hash_map<GlobalDeviceId, DeviceAssignment::LogicalID>
DeviceAssignment::GetDeviceToLogicalIdMap() const {
  absl::flat_hash_map<GlobalDeviceId, DeviceAssignment::LogicalID>
      device_to_logical_id;
  for (int r = 0; r < replica_count(); ++r) {
    for (int c = 0; c < computation_count(); ++c) {
      GlobalDeviceId device_id((*this)(r, c));
      device_to_logical_id[device_id] = DeviceAssignment::LogicalID{r, c};
    }
  }
  return device_to_logical_id;
}

void DeviceAssignment::Serialize(DeviceAssignmentProto* proto) const {
  proto->set_replica_count(replica_count());
  proto->set_computation_count(computation_count());
  for (int computation = 0; computation < computation_count(); ++computation) {
    DeviceAssignmentProto::ComputationDevice* computation_device =
        proto->add_computation_devices();
    for (int replica = 0; replica < replica_count(); ++replica) {
      computation_device->add_replica_device_ids((*this)(replica, computation));
    }
  }
}
namespace {
#define RET_CHECK_ARG(condition) \
  if (!(condition)) return absl::InvalidArgumentError(#condition);
}  // namespace

/* static */ absl::StatusOr<std::unique_ptr<DeviceAssignment>>
DeviceAssignment::Deserialize(const DeviceAssignmentProto& proto) {
  RET_CHECK_ARG(proto.computation_devices_size() == proto.computation_count());
  RET_CHECK_ARG(proto.replica_count() > 0);
  RET_CHECK_ARG(proto.computation_count() > 0);
  auto da = std::make_unique<DeviceAssignment>(proto.replica_count(),
                                               proto.computation_count());
  for (int comp_id = 0; comp_id < proto.computation_count(); ++comp_id) {
    const auto& comp = proto.computation_devices(comp_id);
    RET_CHECK_ARG(comp.replica_device_ids_size() == proto.replica_count());
    for (int replica = 0; replica < proto.replica_count(); ++replica) {
      (*da)(replica, comp_id) = comp.replica_device_ids(replica);
    }
  }
  return std::move(da);
}

std::string DeviceAssignment::ToString() const {
  std::string output = absl::StrFormat(
      "DeviceAssignment{replica_count=%d, computation_count=%d,",
      replica_count(), computation_count());
  for (int computation = 0; computation < computation_count(); ++computation) {
    StrAppend(&output, " Computation", computation, "{");
    for (int replica = 0; replica < replica_count(); ++replica) {
      if (replica > 0) {
        StrAppend(&output, " ");
      }
      int device_id = operator()(replica, computation);
      StrAppend(&output, device_id);
    }
    StrAppend(&output, "}");
  }
  StrAppend(&output, "}");
  return output;
}

absl::StatusOr<DeviceAssignment> ComputationPlacer::AssignDevices(
    int replica_count, int computation_count) {
  DeviceAssignment assignment(replica_count, computation_count);
  for (int replica = 0; replica < replica_count; ++replica) {
    for (int computation = 0; computation < computation_count; ++computation) {
      assignment(replica, computation) = computation * replica_count + replica;
    }
  }
  return assignment;
}

namespace {
absl::Mutex placer_mutex(absl::kConstInit);

// State kept for each kind of ComputationPlacer. Registration functions set
// up creation_function, and then we use that to lazily create "placer" the
// first time GetForPlatform is invoked for a particular id.
struct PlacerState {
  std::unique_ptr<ComputationPlacer> placer;
  ComputationPlacer::CreationFunction creation_function;
};

// Platform id (pointer) to ComputationPlacer with creation function.
using PlacerFactoryMap = absl::flat_hash_map<se::Platform::Id, PlacerState>;

PlacerFactoryMap& GetPlatformComputationPlacers() {
  static PlacerFactoryMap* const r = new PlacerFactoryMap;
  return *r;
}
}  // namespace

/* static */
void ComputationPlacer::RegisterComputationPlacer(
    se::Platform::Id id, CreationFunction creation_function) {
  absl::MutexLock lock(placer_mutex);
  PlacerFactoryMap& placers = GetPlatformComputationPlacers();
  if (placers.find(id) != placers.end()) {
    LOG(WARNING) << "Computation placer creation function is already "
                    "registered for this platform";
  }
  placers[id].creation_function = creation_function;
}

/* static */
absl::StatusOr<ComputationPlacer*> ComputationPlacer::GetForPlatform(
    const se::Platform* platform) {
  absl::MutexLock lock(placer_mutex);
  PlacerFactoryMap& placers = GetPlatformComputationPlacers();

  auto it = placers.find(platform->id());
  if (it == placers.end()) {
    return NotFound(
        "Could not find registered computation placer for platform %s",
        platform->Name());
  }

  PlacerState& state = it->second;
  if (state.placer == nullptr) {
    // Lazily create the computation placer the first time it is needed.
    state.placer = state.creation_function();
  }
  return state.placer.get();
}

}  // namespace xla

namespace {
// registering default computation placer factory for common platforms.
std::unique_ptr<xla::ComputationPlacer> DefaultComputationPlacer() {
  return std::make_unique<xla::ComputationPlacer>();
}

bool InitModule() {
  xla::ComputationPlacer::RegisterComputationPlacer(
      stream_executor::host::kHostPlatformId, DefaultComputationPlacer);
  xla::ComputationPlacer::RegisterComputationPlacer(
      stream_executor::cuda::kCudaPlatformId, DefaultComputationPlacer);
  xla::ComputationPlacer::RegisterComputationPlacer(
      stream_executor::rocm::kROCmPlatformId, DefaultComputationPlacer);
  xla::ComputationPlacer::RegisterComputationPlacer(
      stream_executor::sycl::kSyclPlatformId, DefaultComputationPlacer);
  return true;
}

bool module_initialized = InitModule();
}  // namespace

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

#ifndef XLA_SERVICE_COMPUTATION_PLACER_H_
#define XLA_SERVICE_COMPUTATION_PLACER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "xla/array2d.h"
#include "xla/service/global_device_id.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/platform.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status.h"

namespace xla {

// Class that represents the device assignment for a set of XLA replicated
// computations. For R replicas and C computations, R * C devices are required
// execute the computation in parallel. The assigned device ids can be accessed
// by assignment(replica, computation).
class DeviceAssignment : public Array2D<int64_t> {
 public:
  DeviceAssignment() {}
  DeviceAssignment(int replica_count, int computation_count)
      : Array2D<int64_t>(replica_count, computation_count, -1) {
    CHECK_GT(replica_count, 0);
    CHECK_GT(computation_count, 0);
  }

  int replica_count() const { return height(); }
  int computation_count() const { return width(); }

  // The logical ID of a device is its (replica ID, computation ID) pair.
  struct LogicalID {
    int replica_id;
    int computation_id;
  };

  // Finds the (replica ID, computation ID) pair for the given device.
  absl::StatusOr<LogicalID> LogicalIdForDevice(GlobalDeviceId device_id) const;
  // Finds the replica ID for the given device.
  absl::StatusOr<int> ReplicaIdForDevice(GlobalDeviceId device_id) const;
  // Returns a map from device ID to logical ID. Querying this map is much more
  // efficient than `LogicalIdForDevice` if queried repeatedly.
  absl::flat_hash_map<GlobalDeviceId, LogicalID> GetDeviceToLogicalIdMap()
      const;

  // Protocol buffer serialization and deserialization.
  absl::Status Serialize(DeviceAssignmentProto* proto) const;

  // Return a std::unique_ptr<DeviceAssignment> instead of a DeviceAssignment
  // directly because one of the supported TF platforms (mac) does not compile
  // due to a absl::StatusOr of an incomplete type (DeviceAssignment).
  static absl::StatusOr<std::unique_ptr<DeviceAssignment>> Deserialize(
      const DeviceAssignmentProto& proto);

  std::string ToString() const;
};

// A generic implementation of the XLA computation placer, which assigns device
// ids to a set of replicated computations.
class ComputationPlacer {
 public:
  ComputationPlacer() {}
  virtual ~ComputationPlacer() {}

  // Returns the device id assigned to the given replica and computation
  // instance for [replica_count x computation_count] setup. The returned device
  // id must match the assignment from PlaceReplicatedComputation().
  virtual absl::StatusOr<int> DeviceId(int replica, int computation,
                                       int replica_count,
                                       int computation_count);

  // Returns the device ids assigned to a set of replicated computations, given
  // the number of replicas and the number of computations.
  virtual absl::StatusOr<DeviceAssignment> AssignDevices(int replica_count,
                                                         int computation_count);

  using ComputationPlacerCreationFunction =
      std::unique_ptr<ComputationPlacer> (*)();

  // Registers a computation placer creation function for a particular platform.
  static void RegisterComputationPlacer(
      se::Platform::Id platform_id,
      ComputationPlacerCreationFunction creation_function);

  // Returns the computation placer singleton pointer if it is available for the
  // given platform, or an error status if it is not.
  static absl::StatusOr<ComputationPlacer*> GetForPlatform(
      const se::Platform* platform);

 private:
  // The mutex that guards the platform-to-computation placer map.
  static absl::Mutex platform_computation_placer_mutex_;

  // State kept for each kind of ComputationPlacer. Registration functions set
  // up creation_function, and then we use that to lazily create "placer" the
  // first time GetForPlatform is invoked for a particular id.
  struct State {
    std::unique_ptr<ComputationPlacer> placer;
    ComputationPlacerCreationFunction creation_function = nullptr;
  };

  // Map from platform kind to computation placer singleton.
  static std::map<se::Platform::Id, State>* GetPlatformComputationPlacers();

  ComputationPlacer(const ComputationPlacer&) = delete;
  ComputationPlacer& operator=(const ComputationPlacer&) = delete;
};

}  // namespace xla

#endif  // XLA_SERVICE_COMPUTATION_PLACER_H_

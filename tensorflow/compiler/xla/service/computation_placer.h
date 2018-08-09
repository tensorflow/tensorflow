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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COMPUTATION_PLACER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COMPUTATION_PLACER_H_

#include <map>
#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Class that represents the device assignment for a set of XLA replicated
// computations. For R replicas and C computations, R * C devices are required
// execute the computation in parallel. The assigned device ids can be accessed
// by assignment(replica, computation).
class DeviceAssignment : public Array2D<int> {
 public:
  DeviceAssignment() {}
  DeviceAssignment(int replica_count, int computation_count)
      : Array2D<int>(replica_count, computation_count, -1) {
    CHECK_GT(replica_count, 0);
    CHECK_GT(computation_count, 0);
  }

  int replica_count() const { return height(); }
  int computation_count() const { return width(); }

  // Protocol buffer serialization and deserialization.
  Status Serialize(DeviceAssignmentProto* proto) const;

  // Return a std::unique_ptr<DeviceAssignment> instead of a DeviceAssignment
  // directly because one of the supported TF platforms (mac) does not compile
  // due to a StatusOr of an incomplete type (DeviceAssignment).
  static StatusOr<std::unique_ptr<DeviceAssignment>> Deserialize(
      const DeviceAssignmentProto& proto);

  string ToString() const;
};

// A generic implementation of the XLA computation placer, which assigns device
// ids to a set of replicated computations.
class ComputationPlacer {
 public:
  ComputationPlacer() {}
  virtual ~ComputationPlacer() {}

  // Returns the device id assigned to the given replica and computation
  // instance for [replica_count x computation_count] setup. The returned device
  // id must match the assignement from PlaceReplicatedComputation().
  virtual StatusOr<int> DeviceId(int replica, int computation,
                                 int replica_count, int computation_count);

  // Returns the device ids assigned to a set of replicated computations, given
  // the number of replicas and the number of computations.
  virtual StatusOr<DeviceAssignment> AssignDevices(int replica_count,
                                                   int computation_count);

  using ComputationPlacerCreationFunction =
      std::unique_ptr<ComputationPlacer> (*)();

  // Registers a computation placer creation function for a particular platform.
  static void RegisterComputationPlacer(
      se::Platform::Id platform_id,
      ComputationPlacerCreationFunction creation_function);

  // Returns the computation placer singleton pointer if it is available for the
  // given platform, or an error status if it is not.
  static StatusOr<ComputationPlacer*> GetForPlatform(
      const se::Platform* platform);

 private:
  // The mutex that guards the platform-to-computation placer map.
  static tensorflow::mutex platform_computation_placer_mutex_;

  // State kept for each kind of ComputationPlacer. Registration functions set
  // up creation_function, and then we use that to lazily create "placer" the
  // first time GetForPlatform is invoked for a particular id.
  struct State {
    std::unique_ptr<ComputationPlacer> placer;
    ComputationPlacerCreationFunction creation_function = nullptr;
  };

  // Map from platform kind to computation placer singleton.
  static std::map<se::Platform::Id, State>* GetPlatformComputationPlacers();

  se::Platform::Id platform_id_;

  TF_DISALLOW_COPY_AND_ASSIGN(ComputationPlacer);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COMPUTATION_PLACER_H_

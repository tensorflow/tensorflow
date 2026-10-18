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

#include <functional>
#include <memory>

#include "absl/status/statusor.h"
#include "xla/service/device_assignment.h"
#include "xla/stream_executor/platform_id.h"

namespace xla {

// A generic implementation of the XLA computation placer, which assigns device
// ids to a set of replicated computations.
class ComputationPlacer {
 public:
  ComputationPlacer() = default;
  virtual ~ComputationPlacer() = default;

  // Returns the device ids assigned to a set of replicated computations, given
  // the number of replicas and the number of computations.
  virtual absl::StatusOr<DeviceAssignment> AssignDevices(int replica_count,
                                                         int computation_count);

  using CreationFunction = std::function<std::unique_ptr<ComputationPlacer>()>;

  // Registers a computation placer creation function for a particular platform.
  static void RegisterComputationPlacer(se::PlatformId platform_id,
                                        CreationFunction creation_function);

  // Returns the computation placer singleton pointer registered for the given
  // platform, or the default computation placer if none is registered.
  static ComputationPlacer* GetForPlatform(se::PlatformId platform_id);

 private:
  ComputationPlacer(const ComputationPlacer&) = delete;
  ComputationPlacer& operator=(const ComputationPlacer&) = delete;
};

}  // namespace xla

#endif  // XLA_SERVICE_COMPUTATION_PLACER_H_

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

#include <memory>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/device_assignment.h"
#include "xla/stream_executor/platform_id.h"

namespace xla {

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
using PlacerFactoryMap = absl::flat_hash_map<se::PlatformId, PlacerState>;

PlacerFactoryMap& GetPlatformComputationPlacers() {
  static PlacerFactoryMap* const r = new PlacerFactoryMap;
  return *r;
}

ComputationPlacer* GetDefaultComputationPlacer() {
  static auto* const default_placer = new ComputationPlacer;
  return default_placer;
}
}  // namespace

/* static */
void ComputationPlacer::RegisterComputationPlacer(
    se::PlatformId id, CreationFunction creation_function) {
  absl::MutexLock lock(placer_mutex);
  PlacerFactoryMap& placers = GetPlatformComputationPlacers();
  if (placers.find(id) != placers.end()) {
    LOG(WARNING) << "Computation placer creation function is already "
                    "registered for this platform";
  }
  placers[id].creation_function = creation_function;
}

/* static */
ComputationPlacer* ComputationPlacer::GetForPlatform(
    se::PlatformId platform_id) {
  absl::MutexLock lock(placer_mutex);
  PlacerFactoryMap& placers = GetPlatformComputationPlacers();

  auto it = placers.find(platform_id);
  if (it == placers.end()) {
    return GetDefaultComputationPlacer();
  }

  PlacerState& state = it->second;
  if (state.placer == nullptr) {
    // Lazily create the computation placer the first time it is needed.
    state.placer = state.creation_function();
  }
  return state.placer ? state.placer.get() : GetDefaultComputationPlacer();
}

}  // namespace xla

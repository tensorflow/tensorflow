/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/collectives/cpu_collectives.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {

CpuCollectives* CpuCollectives::Default() {
  absl::StatusOr<Collectives*> collectives =
      CollectivesRegistry::Default("host");
  CHECK_OK(collectives) << "Failed to get CPU collectives";  // Crash OK

  if (auto* cpu_collectives = absl::down_cast<CpuCollectives*>(*collectives)) {
    return cpu_collectives;
  }

  LOG(FATAL) << "Unsupported collectives implementation for CPU";
}

absl::StatusOr<const CpuCollectives::Device*> CpuCollectives::TryCast(
    const Collectives::Device* device) {
  if (auto* cpu_device = absl::down_cast<const Device*>(device)) {
    return cpu_device;
  }
  return InvalidArgument("Collectives device is not a CPU device");
}

absl::StatusOr<const CpuCollectives::Executor*> CpuCollectives::TryCast(
    const Communicator::Executor* executor) {
  if (auto* cpu_executor = absl::down_cast<const Executor*>(executor)) {
    return cpu_executor;
  }
  return InvalidArgument("Collectives executor is not a CPU executor");
}

CpuCollectives::Executor::Executor(RendezvousKey rendezvous_key,
                                   absl::Duration timeout)
    : rendezvous_key_(rendezvous_key), timeout_(timeout) {}

}  // namespace xla::cpu

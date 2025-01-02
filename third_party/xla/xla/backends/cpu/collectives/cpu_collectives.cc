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
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {

CpuCollectives* CpuCollectives::Default() {
  absl::StatusOr<Collectives*> collectives =
      CollectivesRegistry::Default("host");
  CHECK_OK(collectives) << "Failed to get CPU collectives";  // Crash OK

  if (auto* cpu_collectives = tsl::down_cast<CpuCollectives*>(*collectives)) {
    return cpu_collectives;
  }

  LOG(FATAL) << "Unsupported collectives implementation for CPU";
}

}  // namespace xla::cpu

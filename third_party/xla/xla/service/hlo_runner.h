/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_RUNNER_H_
#define XLA_SERVICE_HLO_RUNNER_H_

#include <memory>
#include <utility>

#include "xla/service/hlo_runner_legacy.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"

namespace xla {

// HloRunner exists as an alias of HloRunnerLegacy, for use with legacy
// HloRunner targets. All targets that use HloRunner will be migrated to
// HloRunnerPjRt. Migration candidates are all dependencies of HloRunner.
// Depending on HloRunnerLegacy directly excludes a target from this group.
//
// GPU targets that cannot be migrated to HloRunnerPjRt should use
// HloRunnerLegacy directly.
class HloRunner : public HloRunnerLegacy {
 public:
  explicit HloRunner(
      se::Platform* platform, int intra_op_parallelism_threads = -1,
      std::unique_ptr<se::DeviceAddressAllocator> allocator = nullptr)
      : HloRunnerLegacy(platform, intra_op_parallelism_threads,
                        std::move(allocator)) {}
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_RUNNER_H_

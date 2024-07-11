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
#ifndef XLA_SERVICE_HOST_OFFLOADING_PREPARE_H_
#define XLA_SERVICE_HOST_OFFLOADING_PREPARE_H_

#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Host compute offload requires that all inputs be in device memory if they're
// temporaries. If they're streamed inputs, they can either be device or host
// memory.
class HostOffloadingPrepare : public HloModulePass {
 public:
  ~HostOffloadingPrepare() override = default;

  absl::string_view name() const override { return "host-offloading-prepare"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<bool> RemoveSurroundingMoveCustomCalls(
      HloInstruction* async_start);
};

}  // namespace xla

#endif  // XLA_SERVICE_HOST_OFFLOADING_PREPARE_H_

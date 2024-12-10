/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_SCHEDULE_POSTPROCESSING_H_
#define XLA_SERVICE_GPU_TRANSFORMS_SCHEDULE_POSTPROCESSING_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Amends a schedule result with the needed information to support a runtime
// implementation. Currently, this pass refines attribute
// no_parallel_custom_call for asynchronous collective operations to support
// runtime optimization, such as skipping rendezvous of all participating
// threads for NCCL collective operations. In particular, it sets the attribute
// value for Collective-start operations with is_sync=false; it also keeps the
// attribute value untouch for the operations with is_sync=true and for P2P
// operations, assumming the runtime won't use those values.
//
class SchedulePostprocessing : public HloModulePass {
 public:
  absl::string_view name() const override { return "schedule-postprocessing"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_SCHEDULE_POSTPROCESSING_H_

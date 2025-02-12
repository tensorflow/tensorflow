/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_ALIAS_PASSTHROUGH_PARAMS_H_
#define XLA_SERVICE_GPU_TRANSFORMS_ALIAS_PASSTHROUGH_PARAMS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// This pass aliases input and output buffers that are associated with a
// parameter that is passed through to the module root unmodified.
//
// This pass assumes that parameters and the root use unnested shapes, which is
// the case for XLA:GPU.
//
// This pass must run prior to copy insertion.
class AliasPassthroughParams : public HloModulePass {
 public:
  AliasPassthroughParams() = default;
  ~AliasPassthroughParams() override = default;
  absl::string_view name() const override { return "alias_passthrough_params"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_ALIAS_PASSTHROUGH_PARAMS_H_

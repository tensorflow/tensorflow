/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_SCALAR_CONSTANT_SINKER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_SCALAR_CONSTANT_SINKER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Sinks scalar constants into fusions. Normally, such constants are always
// fused (see priority_fusion), but it is possible for post-fusion passes to
// create new unfused scalar constants. This is common in particular for passes
// that modify while loops, e.g. by peeling them. The induction variable then
// typically becomes an unfused scalar constant.
class ScalarConstantSinker : public HloModulePass {
 public:
  absl::string_view name() const override { return "scalar-constant-sinker"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_SCALAR_CONSTANT_SINKER_H_

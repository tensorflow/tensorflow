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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_SPLITK_REWRITER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_SPLITK_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Rewrites dot instructions that don't fully utilize cores but have a long K
// dimension. For such dots, the input tensors are split along the K dimension
// (forming a new batch dimension) and the resulting dot is reduced along the
// new batch dimension.
class SplitkRewriter : public HloModulePass {
 public:
  explicit SplitkRewriter(se::DeviceDescription device_description)
      : device_description_(device_description) {}

 private:
  absl::string_view name() const override { return "splitk-rewriter"; }
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  se::DeviceDescription device_description_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_SPLITK_REWRITER_H_

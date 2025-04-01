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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_RAGGED_ALL_TO_ALL_DECOMPOSER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_RAGGED_ALL_TO_ALL_DECOMPOSER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrites a `ragged-all-to-all` as a regular `all-to-all`.
//
// A ragged tensor is converted into a dense representation by slicing each
// ragged row from the input and padding with zeros. Then, `all-to-all` is
// performed on the dense representation to exchange rows between replicas.
// Finally, the dense representation is converted back to ragged using
// `dynamic-update-slice` and filling padded values with zero.
//
// This pass is intended as a temporary solution to unblock end-to-end
// integration of `ragged-all-to-all` on GPU, to serve as a reference
// implementation and help with writing integration tests.
//
// TODO(b/379629619): Remove this pass once `ragged-all-to-all` is implemented
// natively on GPU with NCCL.
class RaggedAllToAllDecomposer : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "ragged-all-to-all-decomposer";
  }

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_RAGGED_ALL_TO_ALL_DECOMPOSER_H_

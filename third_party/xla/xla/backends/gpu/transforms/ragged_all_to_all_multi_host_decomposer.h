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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_RAGGED_ALL_TO_ALL_MULTI_HOST_DECOMPOSER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_RAGGED_ALL_TO_ALL_MULTI_HOST_DECOMPOSER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrites a `ragged-all-to-all` into inter-host and intra-host parts.
class RaggedAllToAllMultiHostDecomposer : public HloModulePass {
 public:
  explicit RaggedAllToAllMultiHostDecomposer(int fast_interconnect_slice_size)
      : fast_interconnect_slice_size_(fast_interconnect_slice_size) {}

  absl::string_view name() const override {
    return "ragged-all-to-all-multi-host-decomposer";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  int64_t fast_interconnect_slice_size_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_RAGGED_ALL_TO_ALL_MULTI_HOST_DECOMPOSER_H_

/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_SIMPLIFY_PADDING_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_SIMPLIFY_PADDING_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// Simplifies or eliminates padding introduced by CudnnPadForConvolutions.
//
// If a convolution's weights are padded with 0s in the output feature
// dimension, then the convolution output will contain 0s in corresponding
// features. If these zero features are sliced off, and then padded back with 0s
// (e.g. as input to another convolution), this slice+pad sequence may be
// redundant. This pass simplifies such slice+pad sequences by merging the slice
// into the pad. We then rely on algsimp to remove the pad if it's a nop.
class CudnnSimplifyPadding : public HloModulePass {
 public:
  CudnnSimplifyPadding() = default;

  absl::string_view name() const override { return "cudnn_simplify_padding"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_SIMPLIFY_PADDING_H_

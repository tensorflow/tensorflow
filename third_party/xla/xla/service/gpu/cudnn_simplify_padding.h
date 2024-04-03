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

#ifndef XLA_SERVICE_GPU_CUDNN_SIMPLIFY_PADDING_H_
#define XLA_SERVICE_GPU_CUDNN_SIMPLIFY_PADDING_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla::gpu {

// Simplifies or eliminates padding introduced by CudnnPadForConvolutions and
// CudnnVectorizeConvolutions.
//
// CudnnVectorizeConvolutions will generate code that does the following.
//  - pad input and output features to a multiple of 32 (or 4),
//  - reshape input from [N,C,H,W] to [N,C/32,H,W,32] and reshape kernel from
//    [I,O,H,W] to [I/32,32,O,H,W],
//  - run the conv,
//  - reshape output from [N,C/32,H,W,32] to [N,C,H,W], and finally
//  - slice off the padding on the C channel.
//
// But if this is followed by another convolution (very common), then the slice
// is immediately followed by another pad. This may be redundant; we know that
// the trailing channels sliced off from the first conv are 0.
//
// Ideally we can eliminate the whole reshape+slice+pad+reshape sequence between
// the two convolutions.
//
// Specifically, this pass tries to merge the slice at the end of the sequence
// above into the pad from the next convolution (when we can prove that the
// sliced-off elements are all 0). We then rely on algsimp to remove the pad if
// it's a nop and then to merge and eliminate the remaining reshapes.
//
// This pass should run after CudnnVectorizeConvolutions and there should be no
// simplification passes in between that modify the reshape-transpose-reshape
// introduced by int8x32 convolution filter reordering.
class CudnnSimplifyPadding : public HloModulePass {
 public:
  CudnnSimplifyPadding() = default;

  absl::string_view name() const override { return "cudnn_simplify_padding"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_CUDNN_SIMPLIFY_PADDING_H_

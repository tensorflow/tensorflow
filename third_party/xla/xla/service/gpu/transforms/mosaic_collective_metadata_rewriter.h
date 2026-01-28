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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_MOSAIC_COLLECTIVE_METADATA_REWRITER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_MOSAIC_COLLECTIVE_METADATA_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// TODO(b/478802454): Remove this pass once after the bug is fixed.
// Marks the collective metadata buffer used in the mosaic custom call
// as the unified memory, so it can be available from both CPU and GPU sides.
class MosaicCollectiveMetadataRewriter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "mosaic_collective_metadata_rewriter";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // end namespace xla
#endif  // XLA_SERVICE_GPU_TRANSFORMS_MOSAIC_COLLECTIVE_METADATA_REWRITER_H_

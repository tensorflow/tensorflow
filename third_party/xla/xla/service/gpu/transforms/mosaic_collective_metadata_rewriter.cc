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

#include "xla/service/gpu/transforms/mosaic_collective_metadata_rewriter.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace {

class MosaicCollectiveMetadataRewriterVisitor : public DfsHloRewriteVisitor {
  absl::Status HandleCustomCall(HloInstruction* hlo) override {
    if (hlo->custom_call_target() != "mosaic_gpu_v2" ||
        !hlo->has_backend_config() ||
        !absl::StrContains(hlo->raw_backend_config_string(),
                           "uses_xla_collective_metadata")) {
      return absl::OkStatus();
    }

    Shape* shape = hlo->mutable_shape();
    if (shape->IsTuple()) {
      shape->mutable_tuple_shapes()->back().mutable_layout()->set_memory_space(
          (int)xla::gpu::MemorySpaceColor::kUnified);
    } else {
      shape->mutable_layout()->set_memory_space(
          (int)xla::gpu::MemorySpaceColor::kUnified);
    }
    HloInstruction* new_custom_call =
        hlo->AddInstruction(hlo->CloneWithNewShape(*shape));
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, new_custom_call));
    return absl::OkStatus();
  }
};

}  // end namespace

absl::StatusOr<bool> MosaicCollectiveMetadataRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return MosaicCollectiveMetadataRewriterVisitor{}.RunOnModule(
      module, execution_threads);
}

}  // end namespace xla

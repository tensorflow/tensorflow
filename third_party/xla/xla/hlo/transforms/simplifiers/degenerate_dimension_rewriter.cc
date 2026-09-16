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

#include "xla/hlo/transforms/simplifiers/degenerate_dimension_rewriter.h"

#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/simplifiers/hlo_rewrite_utils.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

absl::StatusOr<bool> DegenerateDimensionRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const auto& computation :
       module->MakeNonfusionComputations(execution_threads)) {
    HloComputation::CachingPostOrder cpo(computation);

    for (HloInstruction* hlo : cpo.PostOrder()) {
      if (!hlo->shape().IsArray()) {
        continue;
      }
      if (!ShapeUtil::HasDegenerateDimensions(hlo->shape())) {
        continue;
      }
      if (IsTrivialElementwise(*hlo)) {
        Shape shape_to_use = ShapeUtil::DropDegenerateDimensions(hlo->shape());

        absl::flat_hash_set<HloInstruction*> finds;
        if (std::optional<Shape> target_shape =
                FindElementwiseSubgraphSurroundedByReshapesAndBroadcasts(
                    hlo, shape_to_use, &finds)) {
          absl::flat_hash_map<HloInstruction*, HloInstruction*> replacements;
          ABSL_ASSIGN_OR_RETURN(
              HloInstruction * reshaped_hlo,
              MakeReshapeHlo(
                  hlo->shape(),
                  ReplaceElementwiseGroupSurroundedByReshapesAndBroadcasts(
                      shape_to_use, hlo, computation, &replacements)));
          ABSL_RETURN_IF_ERROR(computation->ReplaceInstruction(hlo, reshaped_hlo));
          cpo.RecordChange(true);
          changed = true;
        }
      }
    }
  }
  return changed;
}

}  // namespace xla

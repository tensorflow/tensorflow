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

#include "xla/codegen/tiling/tiled_hlo_schedule.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {

namespace {

bool IsDotLike(const HloInstruction* hlo) {
  return hlo->opcode() == HloOpcode::kDot ||
         hlo->opcode() == HloOpcode::kScaledDot;
}

// Given a parameter mapping, produces its "input" (or indexing) space, i.e.,
// for each parameter, the length of the dimension it abstracts over.
std::vector<int64_t> InputSpaceForParameterMapping(
    const TilingSpecification::ParameterMapping& parameter_mapping) {
  int64_t num_parameters = absl::c_accumulate(
      parameter_mapping, 0,
      [](int64_t sum,
         const TilingSpecification::InstructionAndNumTilingParameters&
             mapping) { return sum + mapping.num_tiling_parameters; });
  std::vector<int64_t> input_space;
  input_space.reserve(num_parameters);

  for (const auto& [hlo, num_parameters] : parameter_mapping) {
    // TODO(b/419026602): handle reductions.
    if (IsDotLike(hlo)) {
      auto contracting_dimensions =
          hlo->dot_dimension_numbers().lhs_contracting_dimensions();
      // First, we need to add the contracting dimensions of the `dot`
      // instruction to the input space.
      for (int64_t contracting_dimension : contracting_dimensions) {
        input_space.push_back(
            hlo->operand(0)->shape().dimensions(contracting_dimension));
      }
      int64_t num_contracting_dimensions = contracting_dimensions.size();
      // Optionally, we also add the output dimensions of the `dot` instruction,
      // if they are actual parameters.
      if (num_parameters != num_contracting_dimensions) {
        CHECK_EQ(num_parameters,
                 num_contracting_dimensions + hlo->shape().dimensions().size());
        for (int64_t output_dimension : hlo->shape().dimensions()) {
          input_space.push_back(output_dimension);
        }
      }
      continue;
    }

    CHECK_EQ(hlo->shape().dimensions().size(), num_parameters);
    for (int64_t dimension : hlo->shape().dimensions()) {
      input_space.push_back(dimension);
    }
  }

  return input_space;
}
}  // namespace

absl::StatusOr<IndexingMap> MajorToMinorTiledHloSchedule::RootSchedule(
    const HloInstruction* root,
    const TilingSpecification::ParameterMapping& parameter_mapping,
    mlir::MLIRContext* ctx) const {
  std::vector<int64_t> input_space =
      InputSpaceForParameterMapping(parameter_mapping);
  int64_t num_output_parameters = root->shape().dimensions().size();

  std::vector<mlir::AffineExpr> result_exprs;
  result_exprs.reserve(num_output_parameters);

  int64_t dim_offset = 0;
  for (const auto& [hlo, num_tiling_parameters] : parameter_mapping) {
    if (hlo != root) {
      dim_offset += num_tiling_parameters;
      continue;
    }
    int64_t num_hidden_parameters =
        num_tiling_parameters - num_output_parameters;
    for (int64_t parameter_index = num_hidden_parameters;
         parameter_index < num_tiling_parameters; ++parameter_index) {
      result_exprs.push_back(
          mlir::getAffineDimExpr(dim_offset + parameter_index, ctx));
    }
    CHECK_EQ(result_exprs.size(), num_output_parameters);

    mlir::AffineMap affine_map = mlir::AffineMap::get(
        input_space.size(), /*symbolCount=*/0, result_exprs, ctx);

    return IndexingMap::FromTensorSizes(affine_map, std::move(input_space),
                                        /*symbol_upper_bounds=*/{});
  }
  return absl::NotFoundError(absl::StrCat(
      "No mapping found for root instruction: ", root->ToString()));
}

}  // namespace xla

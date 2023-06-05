/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/dot_dimension_merger.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

namespace {

// Decrease dimension numbers that are >= `start` by `shift`. Copy the other
// ones unmodified.
std::vector<int64_t> ShiftDimensions(absl::Span<const int64_t> dimensions,
                                     const int64_t start, const int64_t shift) {
  std::vector<int64_t> new_dimensions;
  new_dimensions.reserve(dimensions.size());
  for (const int64_t i : dimensions) {
    if (i < start) {
      new_dimensions.push_back(i);
    } else {
      new_dimensions.push_back(i - shift);
    }
  }
  return new_dimensions;
}

// Merge all batch dimensions into logically first one for both operands.
class BatchDimensionMerger : public DfsHloRewriteVisitor {
 public:
  Status HandleDot(HloInstruction* dot) override {
    const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
    const Shape& lhs_shape = dot->operand(0)->shape();
    const Shape& rhs_shape = dot->operand(1)->shape();
    CHECK_EQ(dnums.lhs_batch_dimensions_size(),
             dnums.rhs_batch_dimensions_size());
    const int64_t batch_dimension_count = dnums.lhs_batch_dimensions_size();

    if (batch_dimension_count < 2 ||
        // Logical consecutiveness is required only to simplify the code.
        !DistinctNumbersAreConsecutiveIfSorted(dnums.lhs_batch_dimensions()) ||
        !DistinctNumbersAreConsecutiveIfSorted(dnums.rhs_batch_dimensions()) ||
        !absl::c_is_sorted(dnums.lhs_batch_dimensions()) ||
        !absl::c_is_sorted(dnums.rhs_batch_dimensions()) ||
        !LayoutUtil::AreDimensionsConsecutive(lhs_shape.layout(),
                                              dnums.lhs_batch_dimensions()) ||
        !LayoutUtil::AreDimensionsConsecutive(rhs_shape.layout(),
                                              dnums.rhs_batch_dimensions())) {
      return OkStatus();
    }

    // Index of logically first original batch dimension and the only kept one.
    const int64_t lhs_batch_dimension =
        *absl::c_min_element(dnums.lhs_batch_dimensions());
    const int64_t rhs_batch_dimension =
        *absl::c_min_element(dnums.rhs_batch_dimensions());

    int64_t batch_size = 1;
    for (const int64_t dimension_number : dnums.lhs_batch_dimensions()) {
      batch_size *= lhs_shape.dimensions(dimension_number);
    }

    // Sizes of new dimensions of the operand where batch dimensions are merged
    // into batch_dimension. Non-batch dimensions keep their sizes and order.
    auto operand_merged_dimensions = [&](Shape shape, int batch_dimension) {
      std::vector<int64_t> dimensions;
      dimensions.reserve(shape.rank() + 1 - batch_dimension_count);
      for (int i = 0; i < batch_dimension; ++i) {
        dimensions.push_back(shape.dimensions(i));
      }
      dimensions.push_back(batch_size);
      for (int i = batch_dimension + batch_dimension_count; i < shape.rank();
           ++i) {
        dimensions.push_back(shape.dimensions(i));
      }
      return dimensions;
    };

    std::vector<int64_t> lhs_reshape_dimensions =
        operand_merged_dimensions(lhs_shape, lhs_batch_dimension);
    std::vector<int64_t> rhs_reshape_dimensions =
        operand_merged_dimensions(rhs_shape, rhs_batch_dimension);

    DotDimensionNumbers new_dot_dimension_numbers;
    new_dot_dimension_numbers.add_lhs_batch_dimensions(lhs_batch_dimension);
    new_dot_dimension_numbers.add_rhs_batch_dimensions(rhs_batch_dimension);

    // Dimensions past the batch ones get shifted down.
    {
      const std::vector<int64_t> shifted_contracting_dimensions =
          ShiftDimensions(dnums.lhs_contracting_dimensions(),
                          lhs_batch_dimension, batch_dimension_count - 1);
      new_dot_dimension_numbers.mutable_lhs_contracting_dimensions()->Assign(
          shifted_contracting_dimensions.begin(),
          shifted_contracting_dimensions.end());
    }
    {
      const std::vector<int64_t> shifted_contracting_dimensions =
          ShiftDimensions(dnums.rhs_contracting_dimensions(),
                          rhs_batch_dimension, batch_dimension_count - 1);
      new_dot_dimension_numbers.mutable_rhs_contracting_dimensions()->Assign(
          shifted_contracting_dimensions.begin(),
          shifted_contracting_dimensions.end());
    }

    std::vector<int64_t> new_dot_output_dimensions;
    new_dot_output_dimensions.reserve(dot->shape().rank() + 1 -
                                      batch_dimension_count);
    new_dot_output_dimensions.push_back(batch_size);
    for (int i = batch_dimension_count; i < dot->shape().rank(); ++i) {
      new_dot_output_dimensions.push_back(dot->shape().dimensions(i));
    }

    TF_ASSIGN_OR_RETURN(
        HloInstruction * reshaped_lhs,
        MakeReshapeHlo(ShapeUtil::MakeShape(lhs_shape.element_type(),
                                            lhs_reshape_dimensions),
                       dot->mutable_operand(0)));

    TF_ASSIGN_OR_RETURN(
        HloInstruction * reshaped_rhs,
        MakeReshapeHlo(ShapeUtil::MakeShape(rhs_shape.element_type(),
                                            rhs_reshape_dimensions),
                       dot->mutable_operand(1)));

    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_dot,
        MakeDotHlo(reshaped_lhs, reshaped_rhs, new_dot_dimension_numbers,
                   dot->precision_config(), dot->shape().element_type(),
                   &dot->metadata()));
    dot->SetupDerivedInstruction(new_dot);

    std::unique_ptr<HloInstruction> out_reshape =
        HloInstruction::CreateReshape(dot->shape(), new_dot);
    return ReplaceWithNewInstruction(dot, std::move(out_reshape));
  }
};

}  // namespace

StatusOr<bool> DotDimensionMerger::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return BatchDimensionMerger().RunOnModule(module, execution_threads);
}

}  // namespace xla

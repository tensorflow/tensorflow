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

#include "xla/hlo/transforms/simplifiers/reduce_window_util.h"

#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace reduce_window_util {

Shape ShapeAtIndex(const Shape& shape, const ShapeIndex& shape_index) {
  if (shape_index.empty()) {
    return shape;
  }
  CHECK_EQ(shape_index.size(), 1);
  return ShapeUtil::GetTupleElementShape(shape, shape_index.back());
}

HloInstruction* GetAtIndex(HloInstruction* hlo, const ShapeIndex& shape_index) {
  if (shape_index.empty()) {
    return hlo;
  }
  CHECK_EQ(shape_index.size(), 1);
  return hlo->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
      ShapeAtIndex(hlo->shape(), shape_index), hlo, shape_index.back()));
}

absl::Status Replace1DReduceWindowWithReshape(
    HloReduceWindowInstruction* reduce_window) {
  VLOG(2) << "Converting R1 reduce window: " << reduce_window->ToString();

  std::vector<Shape> r2_output_shapes;
  ShapeUtil::ForEachSubshape(
      reduce_window->shape(),
      [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (!ShapeUtil::IsLeafIndex(reduce_window->shape(), shape_index)) {
          return;
        }
        Shape r2_output_shape = subshape;
        ShapeUtil::AppendMajorDimension(1, &r2_output_shape);
        ShapeUtil::UpdateElementSizeInBits(&r2_output_shape,
                                           /*pack_subbyte_types=*/true);
        r2_output_shapes.push_back(r2_output_shape);

        VLOG(2) << "ReduceWindowRewriter: Converting R2 result to R1: "
                << ShapeUtil::HumanStringWithLayout(r2_output_shape);
      });

  Window r2_window = reduce_window->window();
  WindowDimension* dim = r2_window.add_dimensions();
  dim->set_size(1);
  dim->set_stride(1);
  dim->set_base_dilation(1);
  dim->set_window_dilation(1);

  std::vector<HloInstruction*> r2_operands;
  for (HloInstruction* operand : reduce_window->inputs()) {
    Shape r2_input_shape = operand->shape();
    ShapeUtil::AppendMajorDimension(1, &r2_input_shape);
    ShapeUtil::UpdateElementSizeInBits(&r2_input_shape,
                                       /*pack_subbyte_types=*/true);

    VLOG(2) << "ReduceWindowRewriter: Converting R1 operand to R2: "
            << ShapeUtil::HumanStringWithLayout(r2_input_shape);
    HloInstruction* r2_operand = operand->parent()->AddInstruction(
        HloInstruction::CreateReshape(r2_input_shape, operand));
    VLOG(2) << "R2 new operand: " << r2_operand->ToString();
    r2_operands.push_back(r2_operand);
  }
  HloInstruction* new_reduce_window = reduce_window->parent()->AddInstruction(
      HloInstruction::CreateReduceWindow(
          reduce_window->shape().IsTuple()
              ? ShapeUtil::MakeTupleShape(r2_output_shapes)
              : r2_output_shapes[0],
          r2_operands, reduce_window->init_values(), r2_window,
          reduce_window->to_apply()));

  VLOG(2) << "R2 resulting reduce window: " << new_reduce_window->ToString();

  std::vector<HloInstruction*> final_reshapes;
  ShapeUtil::ForEachSubshape(
      reduce_window->shape(),
      [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (!ShapeUtil::IsLeafIndex(reduce_window->shape(), shape_index)) {
          return;
        }
        HloInstruction* final_reshape =
            new_reduce_window->parent()->AddInstruction(
                HloInstruction::CreateReshape(
                    subshape, GetAtIndex(new_reduce_window, shape_index)));
        final_reshapes.push_back(final_reshape);
      });
  HloInstruction* result;
  if (reduce_window->shape().IsTuple()) {
    result = new_reduce_window->parent()->AddInstruction(
        HloInstruction::CreateTuple(final_reshapes));
  } else {
    CHECK_EQ(final_reshapes.size(), 1);
    result = final_reshapes[0];
  }
  TF_RETURN_IF_ERROR(reduce_window->ReplaceAllUsesWith(result));
  TF_RETURN_IF_ERROR(
      new_reduce_window->parent()->RemoveInstruction(reduce_window));

  return absl::OkStatus();
}

}  // namespace reduce_window_util
}  // namespace xla

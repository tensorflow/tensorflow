/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/sub_byte_normalization.h"

#include <cstdint>

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/layout.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace xla {

namespace {

// Updates the layout by setting element_size_in_bits to the appropriate value.
// Returns true if the layout was changed.
bool UpdateLayout(Layout* layout, PrimitiveType type,
                  SubByteNormalization::Mode mode) {
  auto set_element_size = [layout](int64_t element_size) {
    if (layout->element_size_in_bits() != element_size) {
      layout->set_element_size_in_bits(element_size);
      return true;
    }
    return false;
  };

  switch (mode) {
    case SubByteNormalization::REMOVE_ELEMENT_SIZE:
      return set_element_size(0);
    case SubByteNormalization::SET_ELEMENT_SIZE:
      if (primitive_util::Is4BitType(type)) {
        return set_element_size(4);
      } else {
        return set_element_size(0);
      }
  }
}

// Updates the shape by setting set_element_size_in_bits on the shape's layout.
// Returns true if a layout was changed.
bool UpdateShape(Shape* shape, SubByteNormalization::Mode mode) {
  if (shape->IsTuple()) {
    bool changed = false;
    for (int idx = 0; idx < shape->tuple_shapes_size(); ++idx) {
      changed |= UpdateShape(shape->mutable_tuple_shapes(idx), mode);
    }
    return changed;
  }
  if (shape->IsArray() && shape->has_layout()) {
    return UpdateLayout(shape->mutable_layout(), shape->element_type(), mode);
  }
  return false;
}

// Sets element_size_in_bits on a ShapeLayout's layout. Returns true if the
// layout was changed.
bool ProcessInputOrOutputLayout(ShapeLayout* shape_layout,
                                SubByteNormalization::Mode mode) {
  Shape shape = shape_layout->shape();
  bool changed = UpdateShape(&shape, mode);
  if (changed) {
    TF_CHECK_OK(shape_layout->CopyLayoutFromShape(shape));
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> SubByteNormalization::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  FunctionVisitor visitor([&](HloInstruction* hlo) -> Status {
    auto* shape = hlo->mutable_shape();
    changed |= UpdateShape(shape, mode_);
    return OkStatus();
  });
  for (HloComputation* computation : module->computations()) {
    // We rewrite all computations instead of non-fusion computations, despite
    // element_size_in_bits within fusions being meaningless, because HloVerfier
    // checks for the correct use of element_size_in_bits even in fusion
    // computations.
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  }
  auto* computation_layout = module->mutable_entry_computation_layout();
  for (int param_no = 0; param_no < computation_layout->parameter_count();
       ++param_no) {
    auto* shape_layout = computation_layout->mutable_parameter_layout(param_no);
    changed |= ProcessInputOrOutputLayout(shape_layout, mode_);
  }
  auto* output_layout = computation_layout->mutable_result_layout();
  changed |= ProcessInputOrOutputLayout(output_layout, mode_);
  if (changed) {
    XLA_VLOG_LINES(2, "SubByteNormalization::Run() modified hlo_module:\n" +
                          module->ToString());
  }
  return changed;
}

}  // namespace xla

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

#include "tensorflow/compiler/xla/service/sub_byte_normalization.h"

#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/tsl/platform/errors.h"

namespace xla {

namespace {

bool RemoveInt4SizeFromShape(Shape* shape) {
  if (shape->IsTuple()) {
    bool changed = false;
    for (int idx = 0; idx < shape->tuple_shapes_size(); ++idx) {
      changed |= RemoveInt4SizeFromShape(shape->mutable_tuple_shapes(idx));
    }
    return changed;
  }
  if (shape->IsArray()) {
    const int64_t element_size_in_bits = shape->layout().element_size_in_bits();
    if (element_size_in_bits != 0 && element_size_in_bits < 8) {
      shape->mutable_layout()->set_element_size_in_bits(0);
      return true;
    }
  }
  return false;
}

}  // namespace

StatusOr<bool> SubByteNormalization::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  FunctionVisitor visitor([&](HloInstruction* hlo) -> Status {
    auto* shape = hlo->mutable_shape();
    changed |= RemoveInt4SizeFromShape(shape);
    return OkStatus();
  });
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  }
  auto* computation_layout = module->mutable_entry_computation_layout();
  for (int param_no = 0; param_no < computation_layout->parameter_count();
       ++param_no) {
    auto* shape_layout = computation_layout->mutable_parameter_layout(param_no);
    if (shape_layout->LayoutIsSet() && shape_layout->shape().IsArray()) {
      Layout layout = shape_layout->layout();
      const int64_t element_size_in_bits = layout.element_size_in_bits();
      if (element_size_in_bits != 0 && element_size_in_bits < 8) {
        layout.set_element_size_in_bits(0);
        shape_layout->ResetLayout(layout);
        changed = true;
      }
    }
  }
  if (changed) {
    XLA_VLOG_LINES(2, "SubByteNormalization::Run() modified hlo_module:\n" +
                          module->ToString());
  }
  return changed;
}

}  // namespace xla

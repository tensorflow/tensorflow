/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/scan_expander.h"

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/call_inliner.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

absl::StatusOr<HloComputation*> BuildConditionComputation(
    HloScanInstruction* scan, const Shape& loop_state_shape) {
  TF_ASSIGN_OR_RETURN(int64_t scan_dim_size, scan->GetScanDimSize());

  HloComputation::Builder builder(absl::StrCat(scan->name(), "_condition"));
  auto* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "loop_state"));
  auto* index =
      builder.AddInstruction(HloInstruction::CreateGetTupleElement(param, 0));
  auto* limit = builder.AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::CreateR0<int64_t>(scan_dim_size)));
  builder.AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, {}), index, limit, ComparisonDirection::kLt));
  return scan->parent()->parent()->AddEmbeddedComputation(builder.Build());
}

std::vector<HloInstruction*> GetTupleElements(HloComputation::Builder* builder,
                                              HloInstruction* tuple,
                                              int64_t start_index,
                                              int64_t count) {
  std::vector<HloInstruction*> elements;
  elements.reserve(count);
  for (int64_t i = 0; i < count; ++i) {
    elements.push_back(builder->AddInstruction(
        HloInstruction::CreateGetTupleElement(tuple, start_index + i)));
  }
  return elements;
}

std::vector<HloInstruction*> SliceInputs(
    HloComputation::Builder* builder,
    const std::vector<HloInstruction*>& current_inputs,
    HloInstruction* access_index, int64_t scan_dim) {
  std::vector<HloInstruction*> sliced_inputs;
  sliced_inputs.reserve(current_inputs.size());
  for (HloInstruction* input : current_inputs) {
    const Shape& input_shape = input->shape();

    // DynamicSlice require start indices for all dimensions.
    std::vector<HloInstruction*> start_indices(
        input_shape.dimensions().size(),
        builder->AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(0))));
    start_indices[scan_dim] = access_index;

    std::vector<int64_t> slice_sizes(input_shape.dimensions().begin(),
                                     input_shape.dimensions().end());
    slice_sizes[scan_dim] = 1;

    auto* slice = builder->AddInstruction(HloInstruction::CreateDynamicSlice(
        ShapeUtil::MakeShape(input_shape.element_type(), slice_sizes), input,
        start_indices, slice_sizes));

    // Reshape to remove scan dimension
    Shape sliced_shape = ShapeUtil::DeleteDimension(scan_dim, input_shape);
    sliced_inputs.push_back(builder->AddInstruction(
        HloInstruction::CreateReshape(sliced_shape, slice)));
  }
  return sliced_inputs;
}

std::vector<HloInstruction*> UpdateOutputArrays(
    HloComputation::Builder* builder,
    const std::vector<HloInstruction*>& current_outputs,
    const std::vector<HloInstruction*>& new_output_elems,
    HloInstruction* access_index, int64_t scan_dim) {
  std::vector<HloInstruction*> updated_output_arrays;
  updated_output_arrays.reserve(current_outputs.size());
  for (int64_t i = 0; i < current_outputs.size(); ++i) {
    auto* new_output_elem = new_output_elems[i];
    auto* current_output_array = current_outputs[i];

    // Reshape element to add scan dimension (size 1)
    const Shape& output_array_shape = current_output_array->shape();
    std::vector<int64_t> expanded_dims(output_array_shape.dimensions().begin(),
                                       output_array_shape.dimensions().end());
    expanded_dims[scan_dim] = 1;

    auto* expanded_elem = builder->AddInstruction(HloInstruction::CreateReshape(
        ShapeUtil::MakeShape(new_output_elem->shape().element_type(),
                             expanded_dims),
        new_output_elem));

    // Dynamic Update Slice
    std::vector<HloInstruction*> start_indices(
        output_array_shape.dimensions().size(),
        builder->AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(0))));
    start_indices[scan_dim] = access_index;

    updated_output_arrays.push_back(
        builder->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            output_array_shape, current_output_array, expanded_elem,
            start_indices)));
  }
  return updated_output_arrays;
}

absl::StatusOr<HloComputation*> BuildBodyComputation(
    HloScanInstruction* scan, const Shape& loop_state_shape) {
  int64_t scan_dim = scan->scan_dimension();
  int64_t num_carries = scan->num_carries();
  auto inputs = scan->inputs();
  int64_t num_inputs = inputs.size();
  int64_t num_outputs;
  if (scan->shape().IsTuple()) {
    num_outputs = scan->shape().tuple_shapes().size() - num_carries;
  } else {
    num_outputs = 1 - num_carries;
  }
  TF_ASSIGN_OR_RETURN(int64_t scan_dim_size, scan->GetScanDimSize());
  Shape scalar_shape = ShapeUtil::MakeShape(S64, {});

  HloComputation::Builder builder(absl::StrCat(scan->name(), "_body"));
  auto* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, loop_state_shape, "loop_state"));

  // Extract state
  auto* index =
      builder.AddInstruction(HloInstruction::CreateGetTupleElement(param, 0));
  auto current_accs = GetTupleElements(&builder, param, 1, num_carries);
  auto current_inputs =
      GetTupleElements(&builder, param, 1 + num_carries, num_inputs);
  auto current_outputs = GetTupleElements(
      &builder, param, 1 + num_carries + num_inputs, num_outputs);

  // Calculate access index
  HloInstruction* access_index = index;
  if (scan->is_reverse()) {
    auto* limit = builder.AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int64_t>(scan_dim_size)));
    auto* one = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(1)));
    auto* limit_minus_one = builder.AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kSubtract, limit, one));
    access_index = builder.AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kSubtract, limit_minus_one, index));
  }

  // Prepare to_apply arguments
  std::vector<HloInstruction*> to_apply_args =
      SliceInputs(&builder, current_inputs, access_index, scan_dim);
  to_apply_args.insert(to_apply_args.end(), current_accs.begin(),
                       current_accs.end());

  // Call to_apply
  auto* call = builder.AddInstruction(
      HloInstruction::CreateCall(scan->to_apply()->root_instruction()->shape(),
                                 to_apply_args, scan->to_apply()));

  // Extract results from call
  // tuple structure: (outputs..., new_accs...)
  std::vector<HloInstruction*> new_outputs;
  std::vector<HloInstruction*> new_accs;
  if (call->shape().IsTuple()) {
    new_outputs = GetTupleElements(&builder, call, 0, num_outputs);
    new_accs = GetTupleElements(&builder, call, num_outputs, num_carries);
  } else {
    if (num_outputs == 1) {
      new_outputs.push_back(call);
    } else if (num_carries == 1) {
      new_accs.push_back(call);
    }
  }

  // Update output arrays
  auto updated_output_arrays = UpdateOutputArrays(
      &builder, current_outputs, new_outputs, access_index, scan_dim);

  // Increment index
  auto* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(1)));
  auto* next_index = builder.AddInstruction(
      HloInstruction::CreateBinary(scalar_shape, HloOpcode::kAdd, index, one));

  // Form next loop state tuple
  std::vector<HloInstruction*> next_tuple_elements;
  next_tuple_elements.reserve(1 + num_carries + num_inputs + num_outputs);
  next_tuple_elements.push_back(next_index);
  next_tuple_elements.insert(next_tuple_elements.end(), new_accs.begin(),
                             new_accs.end());
  next_tuple_elements.insert(next_tuple_elements.end(), current_inputs.begin(),
                             current_inputs.end());
  next_tuple_elements.insert(next_tuple_elements.end(),
                             updated_output_arrays.begin(),
                             updated_output_arrays.end());

  builder.AddInstruction(HloInstruction::CreateTuple(next_tuple_elements));

  HloComputation* body_computation =
      scan->parent()->parent()->AddEmbeddedComputation(builder.Build());

  // Inline the call instruction within body_computation
  TF_RETURN_IF_ERROR(CallInliner::Inline(call).status());
  return body_computation;
}

}  // namespace

bool ScanExpander::InstructionMatchesPattern(HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kScan;
}

absl::StatusOr<HloInstruction*> ScanExpander::ExpandInstruction(
    HloInstruction* instruction) {
  auto* scan = Cast<HloScanInstruction>(instruction);
  auto inputs = scan->inputs();
  auto inits = scan->inits();
  int64_t num_carries = scan->num_carries();
  int64_t num_inputs = inputs.size();
  int64_t num_outputs;
  if (scan->shape().IsTuple()) {
    num_outputs = scan->shape().tuple_shapes().size() - num_carries;
  } else {
    num_outputs = 1 - num_carries;
  }

  Shape scalar_shape = ShapeUtil::MakeShape(S64, {});

  // The loop state is a tuple: (index, accumulators..., inputs..., outputs...)
  std::vector<Shape> loop_state_shapes;
  loop_state_shapes.push_back(scalar_shape);  // index
  for (auto* init : inits) {
    loop_state_shapes.push_back(init->shape());
  }
  for (auto* input : inputs) {
    loop_state_shapes.push_back(input->shape());
  }
  // Outputs match scan output shapes (the first num_outputs elements)
  for (int64_t i = 0; i < num_outputs; ++i) {
    if (scan->shape().IsTuple()) {
      loop_state_shapes.push_back(
          ShapeUtil::GetTupleElementShape(scan->shape(), i));
    } else {
      loop_state_shapes.push_back(scan->shape());
    }
  }
  Shape loop_state_shape = ShapeUtil::MakeTupleShape(loop_state_shapes);

  TF_ASSIGN_OR_RETURN(HloComputation * condition_computation,
                      BuildConditionComputation(scan, loop_state_shape));

  TF_ASSIGN_OR_RETURN(HloComputation * body_computation,
                      BuildBodyComputation(scan, loop_state_shape));

  // 3. Build Init Loop State
  std::vector<HloInstruction*> init_values;
  init_values.push_back(scan->parent()->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(0))));
  for (auto* init : inits) {
    init_values.push_back(init);
  }
  for (auto* input : inputs) {
    init_values.push_back(input);
  }

  // Initialize output arrays (zeros or garbage)
  // We can broadcast a constant 0 of the element type.
  for (int64_t i = 0; i < num_outputs; ++i) {
    Shape output_shape;
    if (scan->shape().IsTuple()) {
      output_shape = ShapeUtil::GetTupleElementShape(scan->shape(), i);
    } else {
      output_shape = scan->shape();
    }
    HloInstruction* zero =
        scan->parent()->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::Zero(output_shape.element_type())));
    init_values.push_back(scan->parent()->AddInstruction(
        HloInstruction::CreateBroadcast(output_shape, zero, {})));
  }

  auto* init_loop_state =
      scan->parent()->AddInstruction(HloInstruction::CreateTuple(init_values));

  // 4. Create While Loop
  auto* while_loop = scan->parent()->AddInstruction(
      HloInstruction::CreateWhile(loop_state_shape, condition_computation,
                                  body_computation, init_loop_state));

  // 5. Extract results
  std::vector<HloInstruction*> final_results;

  // Scan output: (output_arrays..., final_carries...)
  // Loop output: (index, final_carries..., inputs..., output_arrays...)

  // Extract output_arrays
  final_results.reserve(num_outputs);
  for (int64_t i = 0; i < num_outputs; ++i) {
    final_results.push_back(
        scan->parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
            while_loop, 1 + num_carries + num_inputs + i)));
  }

  // Extract final_carries
  for (int64_t i = 0; i < num_carries; ++i) {
    final_results.push_back(scan->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(while_loop, 1 + i)));
  }

  HloInstruction* result_tuple = scan->parent()->AddInstruction(
      HloInstruction::CreateTuple(final_results));
  if (scan->shape().IsTuple()) {
    return result_tuple;
  }
  return scan->parent()->AddInstruction(
      HloInstruction::CreateGetTupleElement(result_tuple, 0));
}

}  // namespace xla

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/tuple_util.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"

namespace xla {

/*static*/ HloInstruction* TupleUtil::ExtractPrefix(HloInstruction* input_tuple,
                                                    int64 elements) {
  CHECK(input_tuple->shape().IsTuple());

  HloComputation* computation = input_tuple->parent();
  const Shape& input_shape = input_tuple->shape();

  std::vector<HloInstruction*> tuple_elements;
  tuple_elements.reserve(elements);
  for (int i = 0; i < elements; i++) {
    tuple_elements.push_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            input_shape.tuple_shapes(i), input_tuple, i)));
  }

  return computation->AddInstruction(
      HloInstruction::CreateTuple(tuple_elements));
}

/*static*/ HloInstruction* TupleUtil::AppendSuffix(
    HloInstruction* input_tuple,
    absl::Span<HloInstruction* const> trailing_values) {
  CHECK(input_tuple->shape().IsTuple());

  HloComputation* computation = input_tuple->parent();
  const Shape& input_shape = input_tuple->shape();
  std::vector<HloInstruction*> tuple_elements;
  tuple_elements.reserve(input_shape.tuple_shapes_size());
  for (int i = 0; i < input_shape.tuple_shapes_size(); i++) {
    tuple_elements.push_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            input_shape.tuple_shapes(i), input_tuple, i)));
  }
  tuple_elements.insert(tuple_elements.end(), trailing_values.begin(),
                        trailing_values.end());
  return computation->AddInstruction(
      HloInstruction::CreateTuple(tuple_elements));
}

}  // namespace xla

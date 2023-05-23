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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_TUPLE_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_TUPLE_UTIL_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"

namespace xla {
class TupleUtil {
 public:
  // Generates HLO instructions to get a prefix tuple from `input_tuple` (which
  // must be of tuple shape) of length `elements`.  Returns the root of the
  // graph of instructions generated.
  //
  // The instructions are generated into the computation containing
  // `input_tuple`.
  static HloInstruction* ExtractPrefix(HloInstruction* input_tuple,
                                       int64_t elements);

  // Generates HLO instructions to create a tuple that consists of the values in
  // `trailing_values` appended to `input_tuple` (which must be of tuple shape).
  // Returns the root of the graph of instructions generated.
  //
  // The instructions are generated into the computation containing
  // `input_tuple`.
  static HloInstruction* AppendSuffix(
      HloInstruction* input_tuple,
      absl::Span<HloInstruction* const> trailing_values);

  // Generates HLO instructions that duplicates the tuple by inserting
  // get-tuple-elements and a new tuple instruction. Returns the root of the
  // graph of instructions generated.
  static HloInstruction* Duplicate(HloInstruction* input_tuple) {
    return ExtractPrefix(input_tuple, input_tuple->shape().tuple_shapes_size());
  }

  // Descend to the shape_index element of the tuple and replace that with
  // new_instruction. If the replacement instruction has a different shape than
  // the old one, we insert a bitcast if insert_bitcast_if_different_shape is
  // set to true.
  static StatusOr<HloInstruction*> ReplaceTupleWith(
      HloInstruction* new_instruction, HloInstruction* tuple,
      ShapeIndex shape_index, bool insert_bitcast_if_different_shape = true);

  // Recursively create kGetTupleElement instructions if the defining position
  // shape is not an array. Returns the new instruction that has array shape.
  static HloInstruction* AddGetTupleElements(const HloPosition& position);
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_TUPLE_UTIL_H_

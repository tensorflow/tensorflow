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

#include "tensorflow/compiler/xla/service/sparse_util.h"

#include <algorithm>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/layout_util.h"

namespace xla {

/*static*/ bool SparseUtil::HasSparseInOut(HloInstruction* instruction) {
  // Tests sparse operands.
  if (std::any_of(instruction->operands().begin(),
                  instruction->operands().end(), [](HloInstruction* operand) {
                    return LayoutUtil::IsSparse(operand->shape().layout());
                  })) {
    return true;
  }
  // Tests sparse result.
  return LayoutUtil::IsSparse(instruction->shape().layout());
}

}  // namespace xla

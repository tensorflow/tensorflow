/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/logical_buffer.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

LogicalBuffer::LogicalBuffer(HloInstruction* instruction,
                             const ShapeIndex& index, Id id)
    : BufferValue(instruction, index, id),
      instruction_(instruction),
      index_(index) {}

std::string LogicalBuffer::ToString() const {
  std::string color_string;
  if (has_color()) {
    color_string = absl::StrCat(" @", color());
  }
  return absl::StrCat(instruction_->name(), "[", absl::StrJoin(index_, ","),
                      "](#", id(), color_string, ")");
}

}  // namespace xla

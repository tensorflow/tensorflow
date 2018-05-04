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

#include "tensorflow/compiler/xla/service/buffer_value.h"

#include <iosfwd>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

BufferValue::BufferValue(HloInstruction* instruction, const ShapeIndex& index,
                         Id id)
    : id_(id) {
  const Shape& shape = ShapeUtil::GetSubshape(instruction->shape(), index);
  is_array_ = ShapeUtil::IsArray(shape);
  is_tuple_ = ShapeUtil::IsTuple(shape);
}

BufferValue::~BufferValue() {}

std::ostream& operator<<(std::ostream& out, const BufferValue& buffer) {
  out << buffer.ToString();
  return out;
}

/*static*/ LogicalBufferProto::Location BufferValue::ToLocationProto(
    const HloInstruction& instruction, const ShapeIndex& index) {
  LogicalBufferProto::Location proto;
  proto.set_computation_name(instruction.parent()->name());
  proto.set_instruction_name(instruction.name());
  for (const int64 index_entry : index) {
    proto.add_shape_index(index_entry);
  }
  return proto;
}

LogicalBufferProto BufferValue::ToProto(const SizeFunction& size_fn) const {
  LogicalBufferProto proto;
  proto.set_id(id());
  proto.set_size(size_fn(*this));
  LogicalBufferProto::Location proto_location =
      ToLocationProto(*instruction(), index());
  proto.mutable_defined_at()->Swap(&proto_location);
  if (has_color()) {
    proto.set_color(color().value());
  }
  return proto;
}

}  // namespace xla

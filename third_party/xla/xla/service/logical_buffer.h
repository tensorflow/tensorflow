/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LOGICAL_BUFFER_H_
#define XLA_SERVICE_LOGICAL_BUFFER_H_

#include <string>

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_value.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/gtl/int_type.h"

namespace xla {

// TuplePointsToAnalysis uses this subclass of BufferValue.
class LogicalBuffer : public BufferValue {
 public:
  LogicalBuffer(HloInstruction* instruction, const ShapeIndex& index, Id id);

  // Return the instruction that defines the buffer.
  HloInstruction* instruction() const override { return instruction_; }

  // Return the index within the output of the instruction where the buffer is
  // defined. Index used defined as in ShapeUtil::GetSubshape()
  const ShapeIndex& index() const override { return index_; }

  // Return the shape of the buffer. This reference points into the shape field
  // of the instruction defining the buffer.  Therefore, the returned shape will
  // contain the layout of instruction, if any.
  const Shape& shape() const override {
    return ShapeUtil::GetSubshape(instruction_->shape(), index_);
  }

  std::string ToString() const override;

 private:
  HloInstruction* instruction_;
  ShapeIndex index_;
};

}  // namespace xla

#endif  // XLA_SERVICE_LOGICAL_BUFFER_H_

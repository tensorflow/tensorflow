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

#include <ostream>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {

string LogicalBuffer::ToString() const {
  return tensorflow::strings::StrCat(instruction_->name(), "[",
                                     tensorflow::str_util::Join(index_, ","),
                                     "](#", id_, ")");
}

std::ostream& operator<<(std::ostream& out, const LogicalBuffer& buffer) {
  out << buffer.ToString();
  return out;
}

}  // namespace xla

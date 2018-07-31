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

#include "tensorflow/compiler/xla/service/source_map_util.h"

#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace source_map_util {
namespace {

Status InvalidParameterArgumentV(const OpMetadata& op_metadata,
                                 const char* format, va_list args) {
  string message;
  tensorflow::strings::Appendv(&message, format, args);
  if (!op_metadata.source_file().empty()) {
    tensorflow::strings::Appendf(&message, " (%s:%d)",
                                 op_metadata.source_file().c_str(),
                                 op_metadata.source_line());
  }
  return InvalidArgument("%s", message.c_str());
}

}  // namespace

Status InvalidParameterArgument(const OpMetadata& op_metadata,
                                const char* format, ...) {
  va_list args;
  va_start(args, format);
  Status result = InvalidParameterArgumentV(op_metadata, format, args);
  va_end(args);
  return result;
}

Status InvalidParameterArgument(Executable* executable, int parameter_number,
                                const char* format, ...) {
  va_list args;
  va_start(args, format);
  if (executable != nullptr && executable->has_module()) {
    const HloModule& module = executable->module();
    const HloComputation& computation = *module.entry_computation();
    HloInstruction* param = computation.parameter_instruction(parameter_number);
    const OpMetadata& metadata = param->metadata();
    Status result = InvalidParameterArgumentV(metadata, format, args);
    va_end(args);
    return result;
  }
  Status result = InvalidArgumentV(format, args);
  va_end(args);
  return result;
}

}  // namespace source_map_util
}  // namespace xla

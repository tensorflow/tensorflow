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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_PARSER_HLO_PARSER_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_PARSER_HLO_PARSER_H_

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tools/parser/hlo_lexer.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace tools {

// The api of the hlo parser. Given a string in the HloModule::ToString()
// format, parses the string and creates a HloModule with the given config.
StatusOr<std::unique_ptr<HloModule>> Parse(tensorflow::StringPiece str,
                                           const HloModuleConfig& config);

// The api of the hlo parser. Given a string in the HloModule::ToString()
// format, parses the string and creates a HloModule with default config.
StatusOr<std::unique_ptr<HloModule>> Parse(tensorflow::StringPiece str);

}  // namespace tools
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_PARSER_HLO_PARSER_H_

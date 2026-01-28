/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_SERIALIZATION_UTILS_H_
#define XLA_PYTHON_IFRT_IR_SERIALIZATION_UTILS_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/compiled_ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"

namespace xla {
namespace ifrt {

// Serializes an IFRT executable into a string.
absl::StatusOr<std::string> SerializeIfrtIrExecutable(
    std::shared_ptr<CompiledIfrtIrProgram> program);

struct DeserializedIfrtIRProgram {
  std::unique_ptr<xla::ifrt::IfrtIRProgram> program;
  std::unique_ptr<xla::ifrt::IfrtIRCompileOptions> compile_options;
};

// Deserializes an IFRT executable and compile options from a string.
absl::StatusOr<DeserializedIfrtIRProgram> DeserializeIfrtIrExecutable(
    xla::ifrt::Client* client, absl::string_view serialized,
    std::unique_ptr<xla::ifrt::DeserializeIfrtIRProgramOptions> options);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_SERIALIZATION_UTILS_H_

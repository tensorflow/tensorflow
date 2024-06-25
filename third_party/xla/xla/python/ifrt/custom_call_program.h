/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_CUSTOM_CALL_PROGRAM_H_
#define XLA_PYTHON_IFRT_CUSTOM_CALL_PROGRAM_H_

#include <string>
#include <utility>
#include <vector>

#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/program.h"

namespace xla {
namespace ifrt {

// Wraps a custom call program that expresses a runtime-specific execution.
struct CustomCallProgram
    : public llvm::RTTIExtends<CustomCallProgram, Program> {
  // Specification for a single array. The sharding of all input and output
  // specs must use only the devices in `devices`.
  CustomCallProgram(std::string type, std::string name,
                    std::string serialized_program_text, DeviceList devices,
                    std::vector<ArraySpec> input_specs,
                    std::vector<ArraySpec> output_specs)
      : type(std::move(type)),
        name(std::move(name)),
        serialized_program_text(std::move(serialized_program_text)),
        devices(std::move(devices)),
        input_specs(std::move(input_specs)),
        output_specs(std::move(output_specs)) {}
  ~CustomCallProgram() override = default;

  // Type of this custom call program recognized by IFRT implementations. It
  // indicates what this program represents, e.g., a runtime-specific feature or
  // a pickled Python function.
  std::string type;

  // Name of this program. Used for debugging.
  std::string name;

  // Serialized custom call program. The interpretation of the program text
  // depends `type`.
  std::string serialized_program_text;

  // List of devices to compile and run the custom call program on.
  DeviceList devices;

  // Specification for input and output arrays. The custom call program must
  // expect to receive input arrays and return output arrays both following the
  // specification.
  std::vector<ArraySpec> input_specs;
  std::vector<ArraySpec> output_specs;

  static char ID;  // NOLINT
};

// Compile options for a custom call program. It is currently empty because
// the custom call program does not use any other runtime objects for
// compilation.
struct CustomCallCompileOptions
    : llvm::RTTIExtends<CustomCallCompileOptions, CompileOptions> {
  CustomCallCompileOptions() = default;
  ~CustomCallCompileOptions() override = default;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_CUSTOM_CALL_PROGRAM_H_

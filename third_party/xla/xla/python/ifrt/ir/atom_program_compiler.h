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

#ifndef XLA_PYTHON_IFRT_IR_ATOM_PROGRAM_COMPILER_H_
#define XLA_PYTHON_IFRT_IR_ATOM_PROGRAM_COMPILER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {

// Loaded executable and unique name for a compiled atom program.
struct AtomProgramCompileResult {
  std::string name;
  std::shared_ptr<LoadedExecutable> executable;
};

using AtomExecutableMap =
    absl::flat_hash_map<std::string, std::shared_ptr<LoadedExecutable>>;

class AtomProgramCompiler {
 public:
  virtual ~AtomProgramCompiler() = default;

  // Delegates the compilation of an atom XLA program.
  // `options` uses logical device id in the main mlir module.
  virtual absl::StatusOr<AtomProgramCompileResult> CompileXla(
      std::unique_ptr<HloProgram> computation, xla::CompileOptions options) = 0;

  // Delegates the compilation of an MPMD reshard program.
  virtual absl::StatusOr<AtomProgramCompileResult> CompileMpmdReshard(
      std::vector<DType> dtypes, std::vector<Shape> shapes,
      std::vector<IfrtArrayType> in_array_types,
      std::vector<IfrtArrayType> out_array_types) = 0;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_ATOM_PROGRAM_COMPILER_H_

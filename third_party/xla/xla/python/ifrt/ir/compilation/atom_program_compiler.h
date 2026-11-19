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

#ifndef XLA_PYTHON_IFRT_IR_COMPILATION_ATOM_PROGRAM_COMPILER_H_
#define XLA_PYTHON_IFRT_IR_COMPILATION_ATOM_PROGRAM_COMPILER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/shape.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// Compiled executable and load options for an atom program that will be
// loaded in a subsequent `Client::Load` call.
struct CompiledAtomProgram {
  absl_nonnull ExecutableRef executable;
  absl_nonnull std::unique_ptr<const LoadOptions> load_options;
};

using AtomExecutableMap = absl::flat_hash_map<std::string, LoadedExecutableRef>;

using AtomExecutableFutureMap =
    absl::flat_hash_map<std::string, tsl::Future<LoadedExecutableRef>>;

using CompiledAtomProgramMap = absl::flat_hash_map<
    std::string, absl_nonnull std::shared_ptr<const CompiledAtomProgram>>;

using CompiledAtomProgramFutureMap = absl::flat_hash_map<
    std::string,
    tsl::Future<absl_nonnull std::shared_ptr<const CompiledAtomProgram>>>;

class AtomProgramCompiler {
 public:
  virtual ~AtomProgramCompiler() = default;

  // Delegates compilation and loading of an atom XLA program.
  // `options` uses logical device id in the main mlir module.
  virtual tsl::Future<LoadedExecutableRef> CompileAndLoadXla(
      std::unique_ptr<HloProgram> computation, xla::CompileOptions options,
      std::vector<tsl::RCReference<LoadedHostCallback>>
          loaded_host_callbacks) = 0;

  // Delegates the compilation of an atom XLA program without loading it.
  // `options` uses logical device id in the main mlir module.
  virtual tsl::Future<absl_nonnull std::shared_ptr<const CompiledAtomProgram>>
  CompileXla(std::unique_ptr<HloProgram> computation,
             xla::CompileOptions options,
             std::vector<tsl::RCReference<LoadedHostCallback>>
                 loaded_host_callbacks) = 0;

  // Delegates compilation and loading of an MPMD reshard program.
  virtual tsl::Future<LoadedExecutableRef> CompileAndLoadMpmdReshard(
      std::vector<DType> dtypes, std::vector<Shape> shapes,
      std::vector<IfrtArrayType> in_array_types,
      std::vector<IfrtArrayType> out_array_types) = 0;

  // Delegates the compilation of an MPMD reshard program without loading it.
  virtual tsl::Future<absl_nonnull std::shared_ptr<const CompiledAtomProgram>>
  CompileMpmdReshard(std::vector<DType> dtypes, std::vector<Shape> shapes,
                     std::vector<IfrtArrayType> in_array_types,
                     std::vector<IfrtArrayType> out_array_types) = 0;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_COMPILATION_ATOM_PROGRAM_COMPILER_H_

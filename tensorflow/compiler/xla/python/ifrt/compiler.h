/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_COMPILER_H_

#include <memory>
#include <optional>
#include <utility>

#include "absl/strings/string_view.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "tensorflow/compiler/xla/python/ifrt/executable.h"
#include "tensorflow/compiler/xla/python/ifrt/serdes.h"

namespace xla {
namespace ifrt {

// Abstract options for compiling an MLIR module and load it as
// `LoadedExecutable`. Ideally, compile options should be present in the MLIR
// module being compiled to help static checking and completeness. This option
// structure is to express legacy compilation options that are not included in
// the MLIR module.
// TODO(hyeontaek): Make an new `LoadOptions` that is specific for loading.
// TODO(hyeontaek): Add `Serialize()`.
struct CompileOptions : llvm::RTTIExtends<CompileOptions, Serializable> {
  static char ID;  // NOLINT
};

// Abstract options for deserializing an `Executable` and load it as
// `LoadedExecutable`. This option structure is to express legacy compilation
// options that are not included in the MLIR module.
// TODO(hyeontaek): Make an new `LoadOptions` that is specific for loading.
struct DeserializeOptions
    : llvm::RTTIExtends<DeserializeOptions, llvm::RTTIRoot> {
  static char ID;  // NOLINT
};

// Represents a compiler that creates an `Executable` that can run a computation
// on devices.
//
// TODO(hyeontaek): All `Compiler` methods should take target information such
// as "Platform" or "Topology" that is not tied to a real hardware allocation,
// and return unloaded objects only. `Client` should take over the role of
// loading of compiled objects into the target low-level runtime and hardware to
// ready the them for execution. This will enable ahead-of-time compilation,
// better separation between compilation, loading, and serialization and
// deserialization.
class Compiler : public llvm::RTTIExtends<Compiler, llvm::RTTIRoot> {
 public:
  // Compiles `mlir_module` and returns a `LoadedExecutable`.
  // TODO(hyeontaek): Move executable loading to `Client`.
  virtual StatusOr<std::unique_ptr<LoadedExecutable>> Compile(
      mlir::ModuleOp mlir_module, std::unique_ptr<CompileOptions> options) = 0;

  // Deserializes a serialized executable as produced by
  // `LoadedExecutable::Serialize()`. The compatibility of `serialized` is
  // implementation specific.
  // TODO(hyeontaek): Move executable loading to `Client`.
  virtual StatusOr<std::unique_ptr<LoadedExecutable>>
  DeserializeLoadedExecutable(absl::string_view serialized,
                              std::unique_ptr<DeserializeOptions> options) = 0;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_COMPILER_H_

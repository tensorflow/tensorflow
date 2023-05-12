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

namespace xla {
namespace ifrt {

// TODO(hyeontaek): Generalize `xla::CompileOptions`.
struct CompileOptions : llvm::RTTIExtends<CompileOptions, llvm::RTTIRoot> {
  CompileOptions() = default;
  explicit CompileOptions(xla::CompileOptions xla_options)
      : xla_options(std::move(xla_options)) {}

  xla::CompileOptions xla_options;

  static char ID;  // NOLINT
};

// Represents a compiler that creates an `Executable` that can run a computation
// on devices.
class Compiler : public llvm::RTTIExtends<Compiler, llvm::RTTIRoot> {
 public:
  // Compiles `mlir_module` and returns an `Executable`.

  // TODO(hyeontaek): Introduce `Platform`/`Topology` and return `Executable`
  // instead of `LoadedExecutable`. This will factor out the loading portion of
  // the compilation, enabling ahead-of-time compilation.
  virtual StatusOr<std::unique_ptr<LoadedExecutable>> Compile(
      mlir::ModuleOp mlir_module, std::unique_ptr<CompileOptions> options) = 0;

  // Deserializes a serialized executable as produced by
  // `LoadedExecutable::Serialize()`. The compatibility of `serialized` is
  // implementation specific.
  virtual StatusOr<std::unique_ptr<LoadedExecutable>>
  DeserializeLoadedExecutable(absl::string_view serialized,
                              std::optional<xla::CompileOptions> options) = 0;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_COMPILER_H_

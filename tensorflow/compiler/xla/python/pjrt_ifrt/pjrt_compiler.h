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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_COMPILER_H_

#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/python/ifrt/compiler.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

class PjRtClient;

class PjRtCompiler final : public llvm::RTTIExtends<PjRtCompiler, Compiler> {
 public:
  explicit PjRtCompiler(PjRtClient* client) : client_(client) {}
  ~PjRtCompiler() override = default;

  StatusOr<std::unique_ptr<LoadedExecutable>> Compile(
      mlir::ModuleOp mlir_module, CompileOptions options) override;

  StatusOr<std::unique_ptr<LoadedExecutable>> DeserializeLoadedExecutable(
      absl::string_view serialized, CompileOptions options) override;

  static char ID;  // NOLINT

 private:
  PjRtClient* client_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_COMPILER_H_

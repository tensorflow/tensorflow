/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_GPU_SE_GPU_PJRT_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_GPU_SE_GPU_PJRT_COMPILER_H_

#include <memory>

#include "tensorflow/compiler/xla/pjrt/pjrt_compiler.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"

namespace xla {
// Implements the interfaces that are needed for the registered compiler.
// TODO(b/285385306): current implementation purely relies on the `client`
// Compile() functions and ignores the `topology` parameter.
class StreamExecutorGpuCompiler : public PjRtCompiler {
 public:
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, const XlaComputation& computation,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      CompileOptions options, mlir::ModuleOp module,
      const PjRtTopologyDescription& topology, PjRtClient* client) override;
};
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_PJRT_GPU_SE_GPU_PJRT_COMPILER_H_

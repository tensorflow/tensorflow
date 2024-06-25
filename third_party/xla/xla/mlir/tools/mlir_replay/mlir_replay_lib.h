/* Copyright 2022 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_MLIR_TOOLS_MLIR_REPLAY_MLIR_REPLAY_LIB_H_
#define XLA_MLIR_TOOLS_MLIR_REPLAY_MLIR_REPLAY_LIB_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_replay/public/execution_trace.pb.h"
#include "xla/service/hlo.pb.h"

namespace mlir {
namespace interpreter {

// Runs the given IR on the inputs from `snapshot` and returns the result.
absl::StatusOr<SmallVector<InterpreterValue>> Run(
    MLIRContext& context, const std::string& mlir_ir,
    const xla::HloSnapshot& snapshot, ExecutionTrace* trace,
    const std::vector<std::string>& entry);

}  // namespace interpreter
}  // namespace mlir

#endif  // XLA_MLIR_TOOLS_MLIR_REPLAY_MLIR_REPLAY_LIB_H_

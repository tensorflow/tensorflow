// Copyright 2021 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pass_utils.h"

#include <string>
#include <utility>

#include "mlir/Pass/PassManager.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/xla/mlir/transforms/gpu/passes.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

Status ConvertLmhloToJitRt(mlir::ModuleOp module,
                           mlir::StringRef entry_function_name,
                           llvm::ArrayRef<int64_t> buffer_sizes,
                           xla::gpu::ThunkSequence* thunk_sequence) {
  if (!module) {
    return errors::FailedPrecondition("No MLIR module to lower.");
  }
  mlir::PassManager pm(module.getContext(),
                       mlir::PassManager::Nesting::Implicit);

  tensorflow::applyTensorflowAndCLOptions(pm);

  xla::gpu::populateXlaGpuRuntimePasses(pm, thunk_sequence);

  if (pm.run(module).failed()) {
    return errors::Internal(
        "Failed to lower LMHLO to Gpu runtime custom calls.");
  }

  return OkStatus();
}

}  // namespace tensorflow

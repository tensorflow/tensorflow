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

#include "mlir/Pass/PassManager.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_gpu_binary.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_tfrt_gpu.h"
#include "tensorflow/core/platform/errors.h"
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/wrapper.h"  // from @tf_runtime

namespace tensorflow {

Status ConvertLmhloToTfrtGpuWithBinary(mlir::ModuleOp module,
                                       mlir::StringRef entry_function_name,
                                       llvm::ArrayRef<int64_t> buffer_sizes) {
  if (!module) {
    return errors::FailedPrecondition("No MLIR module to lower.");
  }
  mlir::PassManager pm(module.getContext(),
                       mlir::PassManager::Nesting::Implicit);
  tensorflow::applyTensorflowAndCLOptions(pm);
  pm.addPass(tensorflow::createConvertLmhloToGpuBinaryPass());
  populateLmhloToTfrtGpuPasses(pm);

  if (pm.run(module).failed()) {
    return errors::Internal(
        "Failed to lower LMHLO to TFRT Dialect with gpu kernels.");
  }

  tfrt::gpu::setEntryPoint(module, tfrt::gpu::wrapper::Platform::CUDA,
                           entry_function_name, buffer_sizes);
  return Status::OK();
}

}  // namespace tensorflow

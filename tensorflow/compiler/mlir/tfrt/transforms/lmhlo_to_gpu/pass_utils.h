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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LMHLO_TO_GPU_PASS_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LMHLO_TO_GPU_PASS_UTILS_H_

#include "mlir/IR/BuiltinOps.h"
#include "tensorflow/core/platform/status.h"

namespace xla::gpu {
class ThunkSequence;
}  // namespace xla::gpu

namespace tensorflow {

// Runs the lowering pipeline to convert the given LMHLO module to a JitRt
// module, with a Gpu runtime custom calls to drive the device code execution.
Status ConvertLmhloToJitRt(mlir::ModuleOp module,
                           mlir::StringRef entry_function_name,
                           llvm::ArrayRef<int64_t> buffer_sizes,
                           xla::gpu::ThunkSequence* thunk_sequence);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LMHLO_TO_GPU_PASS_UTILS_H_

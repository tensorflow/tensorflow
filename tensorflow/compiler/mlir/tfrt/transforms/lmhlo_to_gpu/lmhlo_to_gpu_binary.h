// Copyright 2020 The TensorFlow Runtime Authors
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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LMHLO_TO_GPU_LMHLO_TO_GPU_BINARY_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LMHLO_TO_GPU_LMHLO_TO_GPU_BINARY_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace tensorflow {

// Creates a pass that lowers lmhlo.fusion ops to a gpu.module with a binary
// device code attribute plus a gpu.launch_func.
std::unique_ptr<mlir::Pass> createConvertLmhloToGpuBinaryPass();

void registerConvertLmhloToGpuBinaryPass();

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSFORMS_LMHLO_TO_GPU_LMHLO_TO_GPU_BINARY_H_

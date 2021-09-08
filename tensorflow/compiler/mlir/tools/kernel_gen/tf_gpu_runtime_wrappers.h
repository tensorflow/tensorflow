/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_GPU_RUNTIME_WRAPPERS_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_GPU_RUNTIME_WRAPPERS_H_

#include "mlir/ExecutionEngine/RunnerUtils.h"  // from @llvm-project

// Implements a C wrapper around the TensorFlow runtime and CUDA or ROCM that
// allows launching a kernel on the current device and stream from a binary blob
// for the module and function name.

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_tf_launch_kernel(
    void* ctx, void* module_blob, char* kernel_name, intptr_t gridX,
    intptr_t gridY, intptr_t gridZ, intptr_t blockX, intptr_t blockY,
    intptr_t blockZ, void** params);

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TF_GPU_RUNTIME_WRAPPERS_H_

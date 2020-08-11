/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TESTS_TF_FRAMEWORK_C_INTERFACE_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TESTS_TF_FRAMEWORK_C_INTERFACE_H_

#include "mlir/ExecutionEngine/RunnerUtils.h"  // from @llvm-project

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

extern "C" MLIR_RUNNERUTILS_EXPORT void* _mlir_ciface_tf_alloc_raw(
    void* op_kernel_ctx, size_t num_bytes);

extern "C" MLIR_RUNNERUTILS_EXPORT void _mlir_ciface_tf_dealloc_raw(
    void* op_kernel_ctx, void* ptr);

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_TESTS_TF_FRAMEWORK_C_INTERFACE_H_

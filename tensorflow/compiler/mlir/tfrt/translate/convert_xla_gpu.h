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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_CONVERT_XLA_GPU_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_CONVERT_XLA_GPU_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/platform/statusor.h"
#include "tfrt/gpu/system/system.h"  // from @tf_runtime

namespace tensorflow {

// Compile Xla Gpu HLO module to Gpu Program which contain TFRT's Binary
// Executable Format. Gpu Program manages the lifetime of lowered BEF file.
// The program can be execute via gpu::System.
StatusOr<tfrt::gpu::Program> ConvertXlaGpuToGpuProgram(
    std::unique_ptr<xla::HloModule> hlo_module, tfrt::HostContext* host,
    llvm::StringRef platform_name);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_TRANSLATE_CONVERT_XLA_GPU_H_

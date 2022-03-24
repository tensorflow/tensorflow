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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NVPTX_HELPER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NVPTX_HELPER_H_

#include <string>

#include "tensorflow/compiler/xla/service/hlo_module_config.h"

namespace xla {
namespace gpu {

// Logs a warning message that CUDA could not be found in the candidate
// directories.
std::string CantFindCudaMessage(absl::string_view msg,
                                const HloModuleConfig& hlo_module_config);

// Returns the directory containing nvvm libdevice files.
std::string GetLibdeviceDir(const HloModuleConfig& hlo_module_config);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NVPTX_HELPER_H_

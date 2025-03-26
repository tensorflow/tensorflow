/* Copyright 2024 The OpenXLA Authors.
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
#ifndef XLA_SERVICE_GPU_LLVM_GPU_BACKEND_NVPTX_LIBDEVICE_PATH_H_
#define XLA_SERVICE_GPU_LLVM_GPU_BACKEND_NVPTX_LIBDEVICE_PATH_H_
#include <string>

#include "absl/strings/string_view.h"

namespace xla::gpu::nvptx {

// Returns path to libdevice file.
std::string LibDevicePath(absl::string_view xla_gpu_data_dir);

}  // namespace xla::gpu::nvptx

#endif  // XLA_SERVICE_GPU_LLVM_GPU_BACKEND_NVPTX_LIBDEVICE_PATH_H_

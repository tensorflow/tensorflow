/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_EXTENSIONS_ABI_VERSION_GPU_ABI_VERSION_EXTENSION_H_
#define XLA_PJRT_EXTENSIONS_ABI_VERSION_GPU_ABI_VERSION_EXTENSION_H_

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_abi_version_extension.h"

namespace pjrt {

// Creates a PJRT_AbiVersion_Extension that handles GPU specific details of
// parsing serialized PjRtRuntimeAbiVersionProto and
// PjRtExecutableAbiVersionProto.
PJRT_AbiVersion_Extension CreateGpuAbiVersionExtension(
    PJRT_Extension_Base* next = nullptr);

}  // namespace pjrt

#endif  // XLA_PJRT_EXTENSIONS_ABI_VERSION_GPU_ABI_VERSION_EXTENSION_H_

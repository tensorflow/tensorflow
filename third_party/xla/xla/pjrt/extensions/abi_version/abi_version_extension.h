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

#ifndef XLA_PJRT_EXTENSIONS_ABI_VERSION_ABI_VERSION_EXTENSION_H_
#define XLA_PJRT_EXTENSIONS_ABI_VERSION_ABI_VERSION_EXTENSION_H_

#include <memory>

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_abi_version_extension.h"
#include "xla/pjrt/pjrt_abi_version.h"

namespace pjrt {

// Handles parsing the serialized PjRtRuntimeAbiVersionProto and delegates to
// the provided `from_proto` function to create the PjRtRuntimeAbiVersion.
PJRT_Error* CommonRuntimeAbiVersionFromProto(
    absl::FunctionRef<
        absl::StatusOr<std::unique_ptr<xla::PjRtRuntimeAbiVersion>>(
            const xla::PjRtRuntimeAbiVersionProto&)>
        from_proto,
    PJRT_RuntimeAbiVersion_FromProto_Args* args);

// Handles parsing the serialized PjRtExecutableAbiVersionProto and delegates to
// the provided `from_proto` function to create the PjRtExecutableAbiVersion.
PJRT_Error* CommonExecutableAbiVersionFromProto(
    absl::FunctionRef<
        absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>>(
            const xla::PjRtExecutableAbiVersionProto&)>
        from_proto,
    PJRT_ExecutableAbiVersion_FromProto_Args* args);

// Creates a PJRT_AbiVersion_Extension that provides a reasonable default
// implementation for most functions based on PjrtRuntimeAbiVersion and
// PjRtExecutableAbiVersion. The user still needs to provide
// functions for parsing serialized PjRtRuntimeAbiVersionProto and
// PjRtExecutableAbiVersionProto.
PJRT_AbiVersion_Extension CreateAbiVersionExtension(
    PJRT_RuntimeAbiVersion_FromProto* runtime_abi_version_from_proto,
    PJRT_ExecutableAbiVersion_FromProto* executable_abi_version_from_proto,
    PJRT_Extension_Base* next = nullptr);

}  // namespace pjrt

#endif  // XLA_PJRT_EXTENSIONS_ABI_VERSION_ABI_VERSION_EXTENSION_H_

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

#include "xla/pjrt/extensions/abi_version/gpu_abi_version_extension.h"

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_abi_version_extension.h"
#include "xla/pjrt/extensions/abi_version/abi_version_extension.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_runtime_abi_version.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"
#include "xla/pjrt/stream_executor_pjrt_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version_manager.h"

namespace pjrt {

namespace {

PJRT_Error* GpuRuntimeAbiVersionFromProto(
    PJRT_RuntimeAbiVersion_FromProto_Args* args) {
  auto from_proto = [](const xla::PjRtRuntimeAbiVersionProto& proto) {
    return xla::StreamExecutorGpuPjRtRuntimeAbiVersion::FromProto(
        proto, stream_executor::RuntimeAbiVersionManager::GetInstance());
  };
  return CommonRuntimeAbiVersionFromProto(from_proto, args);
}

PJRT_Error* GpuExecutableAbiVersionFromProto(
    PJRT_ExecutableAbiVersion_FromProto_Args* args) {
  return CommonExecutableAbiVersionFromProto(
      xla::StreamExecutorPjRtExecutableAbiVersion::FromProto, args);
}

}  // namespace

PJRT_AbiVersion_Extension CreateGpuAbiVersionExtension(
    PJRT_Extension_Base* next) {
  return CreateAbiVersionExtension(GpuRuntimeAbiVersionFromProto,
                                   GpuExecutableAbiVersionFromProto, next);
}

}  // namespace pjrt

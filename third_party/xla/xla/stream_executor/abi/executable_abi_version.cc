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

#include "xla/stream_executor/abi/executable_abi_version.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/stream_executor/device_description.h"

namespace stream_executor {

static absl::StatusOr<ExecutableAbiVersion> CreateForCuda(
    const DeviceDescription& device_description) {
  stream_executor::ExecutableAbiVersionProto proto;
  proto.set_platform_name("CUDA");
  proto.mutable_cuda_platform_version()->set_cuda_toolkit_version(
      device_description.runtime_version().ToString());
  proto.mutable_cuda_platform_version()->set_cudnn_version(
      device_description.dnn_version().ToString());
  proto.mutable_cuda_platform_version()->set_cub_version(
      device_description.cub_version().ToString());

  return ExecutableAbiVersion::FromProto(std::move(proto));
}

// Returns a minimal ABI version for ROCm with no platform-specific version
// info. Compatibility checks will treat this as always-compatible, preserving
// pre-existing ROCm behavior until proper ABI versioning is designed.
static absl::StatusOr<ExecutableAbiVersion> CreateForRocm(
    const DeviceDescription& /*device_description*/) {
  ExecutableAbiVersionProto proto;
  proto.set_platform_name("ROCm");
  return ExecutableAbiVersion::FromProto(std::move(proto));
}

// Returns a minimal ABI version for oneAPI with no platform-specific version
// info. Compatibility checks will treat this as always-compatible, preserving
// pre-existing oneAPI behavior until proper ABI versioning is designed.
static absl::StatusOr<ExecutableAbiVersion> CreateForOneAPI(
    const DeviceDescription& /*device_description*/) {
  ExecutableAbiVersionProto proto;
  // Platform name is "SYCL" for oneAPI devices.
  proto.set_platform_name("SYCL");
  return ExecutableAbiVersion::FromProto(std::move(proto));
}

absl::StatusOr<ExecutableAbiVersion> ExecutableAbiVersion::FromProto(
    const ExecutableAbiVersionProto& proto) {
  return ExecutableAbiVersion(proto);
}
absl::StatusOr<ExecutableAbiVersion>
ExecutableAbiVersion::FromDeviceDescription(
    const DeviceDescription& device_description) {
  if (device_description.gpu_compute_capability().IsCuda()) {
    return CreateForCuda(device_description);
  }
  if (device_description.gpu_compute_capability().IsRocm()) {
    return CreateForRocm(device_description);
  }
  if (device_description.gpu_compute_capability().IsOneAPI()) {
    return CreateForOneAPI(device_description);
  }

  return absl::UnimplementedError(
      "Deriving the executable ABI version from the device description is not "
      "implemented for the target platform.");
}

absl::string_view ExecutableAbiVersion::platform_name() const {
  return proto_.platform_name();
}

}  // namespace stream_executor

/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TARGET_CONFIG_TARGET_CONFIG_H_
#define XLA_BACKENDS_GPU_TARGET_CONFIG_TARGET_CONFIG_H_

#include <string>

#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

enum class GpuModel {
  A100_PCIE_80,
  A100_SXM_40,
  A100_SXM_80,
  A6000,
  B200,
  B300,
  H100_PCIE,
  H100_SXM,
  MI200,
  P100,
  V100,
};

// Description of a target device for compilation.
struct GpuTargetConfig {
  explicit GpuTargetConfig(stream_executor::StreamExecutor* s);

  static absl::StatusOr<GpuTargetConfig> FromProto(
      const stream_executor::GpuTargetConfigProto& proto);

  stream_executor::GpuTargetConfigProto ToProto() const;

  bool operator==(const GpuTargetConfig& other) const;

  std::string ToString() { return ToProto().DebugString(); }

  stream_executor::DeviceDescription device_description;
  std::string platform_name;
  stream_executor::dnn::VersionInfo dnn_version_info;
  std::string device_description_str;

 private:
  GpuTargetConfig() = default;
};

// Returns the GpuTargetConfigProto for the given GPU model.
absl::StatusOr<stream_executor::GpuTargetConfigProto> GetGpuTargetConfig(
    GpuModel gpu_model);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TARGET_CONFIG_TARGET_CONFIG_H_

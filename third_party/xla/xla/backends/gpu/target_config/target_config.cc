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

#include "xla/backends/gpu/target_config/target_config.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/target_config/embed_gpu_specs.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

absl::StatusOr<absl::string_view> GetEmbeddedGpuTargetConfigData(
    GpuModel gpu_model) {
  switch (gpu_model) {
    case GpuModel::A100_PCIE_80:
      return get_a100_pcie_80();
    case GpuModel::A100_SXM_40:
      return get_a100_sxm_40();
    case GpuModel::A100_SXM_80:
      return get_a100_sxm_80();
    case GpuModel::A6000:
      return get_a6000();
    case GpuModel::B200:
      return get_b200();
    case GpuModel::B300:
      return get_b300();
    case GpuModel::H100_PCIE:
      return get_h100_pcie();
    case GpuModel::H100_SXM:
      return get_h100_sxm();
    case GpuModel::MI200:
      return get_mi200();
    case GpuModel::P100:
      return get_p100();
    case GpuModel::V100:
      return get_v100();
    default:
      return absl::NotFoundError(
          absl::StrCat("Embedded file not found: ", gpu_model, ".txtpb"));
  }
}

}  // namespace

absl::StatusOr<stream_executor::GpuTargetConfigProto> GetGpuTargetConfig(
    GpuModel gpu_model) {
  TF_ASSIGN_OR_RETURN(absl::string_view gpu_spec,
                      GetEmbeddedGpuTargetConfigData(gpu_model));

  stream_executor::GpuTargetConfigProto config;
  if (!google::protobuf::TextFormat::ParseFromString(std::string(gpu_spec), &config)) {
    return absl::InternalError(absl::StrCat(
        "Failed to parse GpuTargetConfigProto from embedded data for: ",
        gpu_model));
  }
  return config;
}

}  // namespace xla::gpu

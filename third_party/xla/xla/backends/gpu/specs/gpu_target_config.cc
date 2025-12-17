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

#include "xla/backends/gpu/specs/gpu_target_config.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/gpu/specs/all_gpu_specs.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

absl::StatusOr<absl::string_view> GetEmbeddedGpuTargetConfigData(
    const std::string& gpu_model) {
  if (gpu_model == "a100_pcie_80") {
    return get_a100_pcie_80();
  }
  if (gpu_model == "a100_sxm_40") {
    return get_a100_sxm_40();
  }
  if (gpu_model == "a100_sxm_80") {
    return get_a100_sxm_80();
  }
  if (gpu_model == "a6000") {
    return get_a6000();
  }
  if (gpu_model == "b200") {
    return get_b200();
  }
  if (gpu_model == "b300") {
    return get_b300();
  }
  if (gpu_model == "h100_pcie") {
    return get_h100_pcie();
  }
  if (gpu_model == "h100_sxm") {
    return get_h100_sxm();
  }
  if (gpu_model == "mi200") {
    return get_mi200();
  }
  if (gpu_model == "p100") {
    return get_p100();
  }
  if (gpu_model == "v100") {
    return get_v100();
  }
  return absl::NotFoundError(
      absl::StrCat("Embedded file not found: ", gpu_model, ".txtpb"));
}

}  // namespace

absl::StatusOr<stream_executor::GpuTargetConfigProto> GetGpuTargetConfig(
    const std::string& gpu_model) {
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

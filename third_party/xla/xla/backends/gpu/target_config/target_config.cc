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
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

namespace se = ::stream_executor;

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
    case GpuModel::BMG_G21:
      return get_bmg_g21();
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

GpuTargetConfig::GpuTargetConfig(se::StreamExecutor* s)
    : device_description(s->GetDeviceDescription()),
      platform_name(s->GetPlatform()->Name()),
      device_description_str(s->GetDeviceDescription().name()) {
  se::dnn::DnnSupport* dnn = s->AsDnn();
  if (dnn != nullptr) {
    absl::StatusOr<se::dnn::VersionInfo> dnn_version = dnn->GetVersion();
    if (dnn_version.ok()) {
      dnn_version_info = *dnn_version;
    }
  }
}

bool GpuTargetConfig::operator==(const GpuTargetConfig& other) const {
  return platform_name == other.platform_name &&
         dnn_version_info == other.dnn_version_info &&
         device_description_str == other.device_description_str &&
         device_description == other.device_description;
}

absl::StatusOr<GpuTargetConfig> GpuTargetConfig::FromProto(
    const se::GpuTargetConfigProto& proto) {
  GpuTargetConfig target_config;
  TF_ASSIGN_OR_RETURN(
      target_config.device_description,
      se::DeviceDescription::FromProto(proto.gpu_device_info()));
  target_config.platform_name = proto.platform_name();
  target_config.dnn_version_info =
      se::dnn::VersionInfo(proto.dnn_version_info());
  target_config.device_description_str = proto.device_description_str();
  if (!target_config.device_description_str.empty()) {
    target_config.device_description.set_name(
        target_config.device_description_str);
  }
  se::SemanticVersion runtime_version(proto.runtime_version().major(),
                                      proto.runtime_version().minor(),
                                      proto.runtime_version().patch());
  target_config.device_description.set_runtime_version(runtime_version);
  se::SemanticVersion dnn_version(
      static_cast<unsigned>(proto.dnn_version_info().major()),
      static_cast<unsigned>(proto.dnn_version_info().minor()),
      static_cast<unsigned>(proto.dnn_version_info().patch()));
  target_config.device_description.set_dnn_version(dnn_version);
  return target_config;
}

se::GpuTargetConfigProto GpuTargetConfig::ToProto() const {
  se::GpuTargetConfigProto proto;
  *proto.mutable_gpu_device_info() = device_description.ToGpuProto();
  proto.set_platform_name(platform_name);
  *proto.mutable_dnn_version_info() = dnn_version_info.ToProto();
  se::RuntimeVersionProto runtime_version_proto;
  runtime_version_proto.set_major(device_description.runtime_version().major());
  runtime_version_proto.set_minor(device_description.runtime_version().minor());
  runtime_version_proto.set_patch(device_description.runtime_version().patch());
  *proto.mutable_runtime_version() = runtime_version_proto;
  proto.set_device_description_str(device_description_str);
  return proto;
}

}  // namespace xla::gpu

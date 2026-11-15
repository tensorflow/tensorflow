/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu_topology.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/env.h"

namespace xla {
namespace {

// Returns the host CPU target machine options associated with `platform_name`,
// or `std::nullopt` when no host CPU target override is configured.
absl::StatusOr<std::optional<cpu::TargetMachineOptions>>
GetHostTargetMachineOptions(absl::string_view platform_name) {
  if (platform_name == "umbriel_b200") {
    return cpu::TargetMachineOptions{
        "x86_64-grtev4-linux-gnu", "emeraldrapids",
        "+64bit,+adx,+aes,+amx-bf16,+amx-int8,+amx-tile,+avx,+avx2,+avx512bf16,"
        "+avx512bitalg,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512fp16,+"
        "avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vl,+avx512vnni,+"
        "avx512vpopcntdq,+avxvnni,+bmi,+bmi2,+cldemote,+clflushopt,+clwb,+cmov,"
        "+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+gfni,+invpcid,+lzcnt,+"
        "mmx,+movbe,+movdir64b,+movdiri,+pclmul,+popcnt,+prefer-no-gather,+"
        "prefer-no-scatter,+prfchw,+rdpid,+rdrnd,+rdseed,+rtm,+sahf,+serialize,"
        "+sha,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+tsxldtrk,+vaes,+"
        "vpclmulqdq,+wbnoinvd,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-avx512,-"
        "amx-complex,-amx-fp16,-amx-fp8,-amx-movrs,-avx10.1,-avx10.2,-"
        "avx10v2aux,-avx512bmm,-avx512vp2intersect,-avxifma,-avxneconvert,-"
        "avxvnniint16,-"
        "avxvnniint8,-ccmp,-cf,-clzero,-cmpccxadd,-egpr,-enqcmd,-fma4,-hreset,-"
        "jmpabs,-kl,-lwp,-movrs,-mwaitx,-ndd,-nf,-pconfig,-pku,-ppx,-prefetchi,"
        "-ptwrite,-push2pop2,-raoint,-rdpru,-sgx,-sha512,-shstk,-sm3,-sm4,-"
        "sse4a,-tbm,-uintr,-usermsr,-waitpkg,-widekl,-xop,-zu"};
  }
  if (platform_name == "oberon_b200" || platform_name == "oberon_b300") {
    return cpu::TargetMachineOptions{
        "aarch64-unknown-linux-gnu", "neoverse-n1",
        "+aes,+crc,+fp-armv8,+lse,+neon,+sha2,+sha3,+sm4,+sve-aes,+sve-sha3,+"
        "sve-sm4,-rand,-sve,-sve2"};
  }
  return std::nullopt;
}

std::optional<absl::StatusOr<GpuTopology>> TryParseInlineGpuTopology(
    absl::string_view spec) {
  absl::string_view platform_version;
  absl::string_view dims_str = spec;
  if (size_t colon_pos = spec.find(':'); colon_pos != absl::string_view::npos) {
    platform_version = spec.substr(0, colon_pos);
    dims_str = spec.substr(colon_pos + 1);
  }
  std::vector<absl::string_view> parts = absl::StrSplit(dims_str, 'x');
  if (parts.size() != 2 && parts.size() != 3) {
    return std::nullopt;
  }
  int32_t dims[3] = {1, 0, 0};
  size_t offset = (parts.size() == 2) ? 1 : 0;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (!absl::SimpleAtoi(parts[i], &dims[offset + i]) ||
        dims[offset + i] <= 0) {
      return std::nullopt;
    }
  }
  if (!platform_version.empty()) {
    return GetGpuTopologyForPlatform(platform_version, dims[0], dims[1],
                                     dims[2]);
  }
  return GpuTopology(/*platform_version=*/"", dims[0], dims[1], dims[2]);
}

}  // namespace

absl::StatusOr<std::unique_ptr<const GpuTopology>> GpuTopology::FromProto(
    const GpuTopologyProto& gpu_topology_proto) {
  std::optional<gpu::GpuTargetConfig> gpu_target_config = std::nullopt;
  if (gpu_topology_proto.has_gpu_target_config()) {
    ABSL_ASSIGN_OR_RETURN(gpu_target_config,
                     gpu::GpuTargetConfig::FromProto(
                         gpu_topology_proto.gpu_target_config()));
  }
  std::optional<cpu::TargetMachineOptions> host_target_machine_options =
      std::nullopt;
  if (gpu_topology_proto.has_host_target_machine_options()) {
    ABSL_ASSIGN_OR_RETURN(host_target_machine_options,
                     cpu::TargetMachineOptions::FromProto(
                         gpu_topology_proto.host_target_machine_options()));
  }
  std::optional<int32_t> num_devices_per_process = std::nullopt;
  if (gpu_topology_proto.has_num_devices_per_process()) {
    num_devices_per_process = gpu_topology_proto.num_devices_per_process();
  }
  return std::make_unique<GpuTopology>(
      gpu_topology_proto.platform_version(),
      gpu_topology_proto.num_partitions(),
      gpu_topology_proto.num_hosts_per_partition(),
      gpu_topology_proto.num_devices_per_host(), std::move(gpu_target_config),
      std::move(host_target_machine_options), num_devices_per_process);
}

GpuTopologyProto GpuTopology::ToProto() const {
  GpuTopologyProto proto;
  proto.set_platform_version(platform_version());
  proto.set_num_partitions(num_partitions());
  proto.set_num_hosts_per_partition(num_hosts_per_partition());
  proto.set_num_devices_per_host(num_devices_per_host());
  if (gpu_target_config_.has_value()) {
    *proto.mutable_gpu_target_config() = gpu_target_config().ToProto();
  }
  if (host_target_machine_options_.has_value()) {
    *proto.mutable_host_target_machine_options() =
        host_target_machine_options()->ToProto();
  }
  if (num_devices_per_process_.has_value()) {
    proto.set_num_devices_per_process(num_devices_per_process());
  }
  return proto;
}

absl::Span<const GpuPlatformConfig> GetSupportedGpuPlatformConfigs() {
  static const absl::NoDestructor<std::vector<GpuPlatformConfig>> kConfigs({
      GpuPlatformConfig{"tesla_a100",
                        "a100",
                        gpu::GpuModel::A100_SXM_40,
                        /*devices_per_host=*/16,
                        /*nvlink_domain_size=*/16,
                        {"NVIDIA A100-SXM4-40GB"}},
      GpuPlatformConfig{"tesla_a100_80gib",
                        "a100_80gib",
                        gpu::GpuModel::A100_SXM_80,
                        /*devices_per_host=*/8,
                        /*nvlink_domain_size=*/8,
                        {"NVIDIA A100-SXM4-80GB"}},
      GpuPlatformConfig{"tesla_v100",
                        "v100",
                        gpu::GpuModel::V100,
                        /*devices_per_host=*/std::nullopt,
                        /*nvlink_domain_size=*/std::nullopt,
                        {"Tesla V100-SXM2-16GB"}},
      GpuPlatformConfig{"tesla_p100",
                        "p100",
                        gpu::GpuModel::P100,
                        /*devices_per_host=*/std::nullopt,
                        /*nvlink_domain_size=*/std::nullopt,
                        {"Tesla P100-SXM2-16GB", "Tesla P100-PCIE-16GB"}},
      GpuPlatformConfig{
          "nvidia_h100",
          "h100",
          gpu::GpuModel::H100_SXM,
          /*devices_per_host=*/8,
          /*nvlink_domain_size=*/8,
          {"NVIDIA H100 80GB HBM3", "NVIDIA H100 80GB HBM3 MIG 1g.10gb"}},
      GpuPlatformConfig{"tesla_h200",
                        "h200",
                        gpu::GpuModel::H200,
                        /*devices_per_host=*/8,
                        /*nvlink_domain_size=*/8,
                        {"NVIDIA H200 141GB HBM3e", "NVIDIA H200"}},
      GpuPlatformConfig{"umbriel_b200",
                        "b200",
                        gpu::GpuModel::B200,
                        /*devices_per_host=*/8,
                        /*nvlink_domain_size=*/8,
                        {"NVIDIA B200"}},
      GpuPlatformConfig{"umbriel_b300",
                        "b300",
                        gpu::GpuModel::B300,
                        /*devices_per_host=*/8,
                        /*nvlink_domain_size=*/8,
                        {"NVIDIA B300", "NVIDIA B300 SXM6 AC"}},
      GpuPlatformConfig{"oberon_b200",
                        "gb200",
                        gpu::GpuModel::GB200,
                        /*devices_per_host=*/4,
                        /*nvlink_domain_size=*/72,
                        {"NVIDIA GB200"}},
      GpuPlatformConfig{"oberon_b300",
                        "gb300",
                        gpu::GpuModel::GB300,
                        /*devices_per_host=*/4,
                        /*nvlink_domain_size=*/72,
                        {"NVIDIA GB300", "NVIDIA GB300 Max-Q"}},
      GpuPlatformConfig{"velorum_r200",
                        "vr200",
                        gpu::GpuModel::VR_NVL72,
                        /*devices_per_host=*/4,
                        /*nvlink_domain_size=*/72,
                        {"NVIDIA VR NVL72", "NVIDIA Graphics Device"}},
  });
  return *kConfigs;
}

absl::StatusOr<const GpuPlatformConfig*> GetGpuPlatformConfig(
    absl::string_view platform_or_device_name) {
  absl::string_view normalized = platform_or_device_name;
  if (normalized == "tesla_a100_v80gib") {
    normalized = "tesla_a100_80gib";
  }
  for (const GpuPlatformConfig& config : GetSupportedGpuPlatformConfigs()) {
    if (config.platform_name == normalized || config.short_name == normalized ||
        absl::c_linear_search(config.device_descriptions, normalized)) {
      return &config;
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported GPU platform type: ", platform_or_device_name));
}

absl::StatusOr<GpuTopology> GetGpuTopologyForPlatform(
    absl::string_view platform_version, int32_t num_partitions,
    int32_t num_hosts_per_partition, int32_t num_devices_per_host) {
  ABSL_ASSIGN_OR_RETURN(const GpuPlatformConfig* platform_config,
                   GetGpuPlatformConfig(platform_version));
  ABSL_ASSIGN_OR_RETURN(
      stream_executor::GpuTargetConfigProto gpu_target_config_proto,
      gpu::GetGpuTargetConfig(platform_config->gpu_model));
  ABSL_ASSIGN_OR_RETURN(gpu::GpuTargetConfig gpu_target_config,
                   gpu::GpuTargetConfig::FromProto(gpu_target_config_proto));
  ABSL_ASSIGN_OR_RETURN(
      std::optional<cpu::TargetMachineOptions> host_target_machine_options,
      GetHostTargetMachineOptions(platform_config->platform_name));
  return GpuTopology(platform_version, num_partitions, num_hosts_per_partition,
                     num_devices_per_host, std::move(gpu_target_config),
                     std::move(host_target_machine_options));
}

absl::StatusOr<GpuTopology> GetGpuTopologyForSliceSpec(
    absl::string_view platform_or_short_name, absl::string_view slice_spec) {
  ABSL_ASSIGN_OR_RETURN(const GpuPlatformConfig* platform_config,
                   GetGpuPlatformConfig(platform_or_short_name));
  std::vector<absl::string_view> parts = absl::StrSplit(slice_spec, 'x');
  if (parts.size() != 1 && parts.size() != 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unable to parse GPU count '", slice_spec,
                     "', invalid topology format."));
  }
  int32_t devices_per_host = 0;
  if (!absl::SimpleAtoi(parts.front(), &devices_per_host) ||
      devices_per_host <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unable to parse GPU count '", slice_spec,
                     "', invalid number of devices per host or hosts."));
  }
  const bool is_multi_host = parts.size() == 2;
  int32_t num_hosts = 1;
  if (is_multi_host) {
    if (!absl::SimpleAtoi(parts[1], &num_hosts) || num_hosts <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unable to parse GPU count '", slice_spec,
                       "', invalid number of hosts."));
    }
    if (!platform_config->devices_per_host.has_value()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "devices_per_host is not set for platform '",
          platform_config->short_name, "' to support multi-host topology."));
    }
    if (devices_per_host != *platform_config->devices_per_host) {
      return absl::InvalidArgumentError(absl::StrCat(
          "'", platform_config->short_name,
          "' cannot support a multi-host topology '", slice_spec, "' with '",
          devices_per_host, "' devices per host (expected '",
          *platform_config->devices_per_host, "')."));
    }
  }

  int32_t num_partitions = num_hosts;
  int32_t num_hosts_per_partition = 1;
  if (platform_config->nvlink_domain_size.has_value() &&
      platform_config->devices_per_host.has_value() &&
      *platform_config->nvlink_domain_size >
          *platform_config->devices_per_host) {
    const int64_t num_devices =
        static_cast<int64_t>(devices_per_host) * num_hosts;
    if (num_devices > *platform_config->nvlink_domain_size) {
      return absl::UnimplementedError(absl::StrCat(
          "For Multi-Node NVLink systems, GPU slice spanning multiple NVLink "
          "domains is not supported. NVLink domain size '",
          *platform_config->nvlink_domain_size, "', slice size '", num_devices,
          "', platform '", platform_config->short_name, "'."));
    }
    num_partitions = 1;
    num_hosts_per_partition = num_hosts;
  }
  return GetGpuTopologyForPlatform(platform_config->platform_name,
                                   num_partitions, num_hosts_per_partition,
                                   devices_per_host);
}

std::string GetGpuDeviceKind(const GpuTopologyProto& gpu_topology_proto) {
  if (!gpu_topology_proto.gpu_target_config()
           .device_description_str()
           .empty()) {
    return gpu_topology_proto.gpu_target_config().device_description_str();
  }
  if (!gpu_topology_proto.gpu_target_config()
           .gpu_device_info()
           .name()
           .empty()) {
    return gpu_topology_proto.gpu_target_config().gpu_device_info().name();
  }
  if (!gpu_topology_proto.platform_version().empty()) {
    return gpu_topology_proto.platform_version();
  }
  return "Unknown GPU version";
}

GpuTopology GetSingleDeviceGpuTopology(
    absl::string_view platform_version,
    const gpu::GpuTargetConfig& gpu_target_config,
    const std::optional<cpu::TargetMachineOptions>&
        host_target_machine_options) {
  return GpuTopology(platform_version, 1, 1, 1, gpu_target_config,
                     host_target_machine_options);
}

absl::StatusOr<GpuTopology> ParseGpuTopology(
    absl::string_view topology_spec_or_filename) {
  TF_RET_CHECK(!topology_spec_or_filename.empty());
  if (std::optional<absl::StatusOr<GpuTopology>> inline_topology =
          TryParseInlineGpuTopology(topology_spec_or_filename);
      inline_topology.has_value()) {
    return *std::move(inline_topology);
  }

  std::string gpu_topology_string;
  ABSL_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(),
                                        std::string(topology_spec_or_filename),
                                        &gpu_topology_string));
  GpuTopologyProto gpu_topology_proto;
  if (!google::protobuf::TextFormat::ParseFromString(gpu_topology_string,
                                           &gpu_topology_proto)) {
    return absl::FailedPreconditionError("Failed to parse GpuTopologyProto");
  }
  if (!gpu_topology_proto.has_gpu_target_config() &&
      !gpu_topology_proto.platform_version().empty()) {
    ABSL_ASSIGN_OR_RETURN(
        const GpuPlatformConfig* platform_config,
        GetGpuPlatformConfig(gpu_topology_proto.platform_version()));
    ABSL_ASSIGN_OR_RETURN(*gpu_topology_proto.mutable_gpu_target_config(),
                     gpu::GetGpuTargetConfig(platform_config->gpu_model));
  }
  if (!gpu_topology_proto.has_host_target_machine_options() &&
      !gpu_topology_proto.platform_version().empty()) {
    absl::StatusOr<const GpuPlatformConfig*> platform_config =
        GetGpuPlatformConfig(gpu_topology_proto.platform_version());
    if (platform_config.ok()) {
      ABSL_ASSIGN_OR_RETURN(
          std::optional<cpu::TargetMachineOptions> host_options,
          GetHostTargetMachineOptions((*platform_config)->platform_name));
      if (host_options.has_value()) {
        *gpu_topology_proto.mutable_host_target_machine_options() =
            host_options->ToProto();
      }
    }
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<const GpuTopology> topology,
                   GpuTopology::FromProto(gpu_topology_proto));
  return *topology;
}

}  // namespace xla

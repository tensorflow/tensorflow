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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/service/gpu_topology.pb.h"

namespace xla {
namespace {

absl::StatusOr<gpu::GpuModel> GetGpuModel(absl::string_view platform_type) {
  if (platform_type == "tesla_a100") {
    return gpu::GpuModel::A100_SXM_40;
  }
  if (platform_type == "nvidia_h100") {
    return gpu::GpuModel::H100_SXM;
  }
  if (platform_type == "umbriel_b200") {
    return gpu::GpuModel::B200;
  }
  if (platform_type == "umbriel_b300") {
    return gpu::GpuModel::B300;
  }
  if (platform_type == "oberon_b200") {
    return gpu::GpuModel::GB200;
  }
  if (platform_type == "oberon_b300") {
    return gpu::GpuModel::GB300;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported GPU platform type: ", platform_type));
}

absl::StatusOr<std::optional<cpu::TargetMachineOptions>>
GetHostTargetMachineOptions(absl::string_view platform_version) {
  if (platform_version == "umbriel_b200") {
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
        "amx-complex,-amx-fp16,-amx-fp8,-amx-movrs,-amx-tf32,-avx10.1,-avx10.2,"
        "-avx512vp2intersect,-avxifma,-avxneconvert,-avxvnniint16,-avxvnniint8,"
        "-ccmp,-cf,-clzero,-cmpccxadd,-egpr,-enqcmd,-fma4,-hreset,-jmpabs,-kl,-"
        "lwp,-movrs,-mwaitx,-ndd,-nf,-pconfig,-pku,-ppx,-prefetchi,-ptwrite,-"
        "push2pop2,-raoint,-rdpru,-sgx,-sha512,-shstk,-sm3,-sm4,-sse4a,-tbm,-"
        "uintr,-usermsr,-waitpkg,-widekl,-xop,-zu"};
  }
  if (platform_version == "oberon_b200") {
    return cpu::TargetMachineOptions{
        "aarch64-linux-gnu", "neoverse-n1",
        "+aes,+crc,+fp-armv8,+lse,+neon,+sha2,+sha3,+sm4,+sve-aes,+sve-sha3,+"
        "sve-sm4,-rand,-sve,-sve2"};
  }
  return std::nullopt;
}

}  // namespace

absl::StatusOr<std::unique_ptr<const GpuTopology>> GpuTopology::FromProto(
    const GpuTopologyProto& gpu_topology_proto) {
  std::optional<gpu::GpuTargetConfig> gpu_target_config = std::nullopt;
  if (gpu_topology_proto.has_gpu_target_config()) {
    ASSIGN_OR_RETURN(gpu_target_config,
                     gpu::GpuTargetConfig::FromProto(
                         gpu_topology_proto.gpu_target_config()));
  }
  std::optional<cpu::TargetMachineOptions> host_target_machine_options =
      std::nullopt;
  if (gpu_topology_proto.has_host_target_machine_options()) {
    ASSIGN_OR_RETURN(host_target_machine_options,
                     cpu::TargetMachineOptions::FromProto(
                         gpu_topology_proto.host_target_machine_options()));
  }
  return std::make_unique<GpuTopology>(
      gpu_topology_proto.platform_version(),
      gpu_topology_proto.num_partitions(),
      gpu_topology_proto.num_hosts_per_partition(),
      gpu_topology_proto.num_devices_per_host(), std::move(gpu_target_config),
      std::move(host_target_machine_options));
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
  return proto;
}

absl::StatusOr<GpuTopology> GetGpuTopologyForPlatform(
    absl::string_view platform_version, int32_t num_partitions,
    int32_t num_hosts_per_partition, int32_t num_devices_per_host) {
  // TODO(b/470487616): Don't use string matching to get the GpuTargetConfig.
  ASSIGN_OR_RETURN(auto spec_name, GetGpuModel(platform_version));
  ASSIGN_OR_RETURN(auto gpu_target_config_proto,
                   gpu::GetGpuTargetConfig(spec_name));
  ASSIGN_OR_RETURN(auto gpu_target_config,
                   gpu::GpuTargetConfig::FromProto(gpu_target_config_proto));
  ASSIGN_OR_RETURN(auto host_target_machine_options,
                   GetHostTargetMachineOptions(platform_version));
  return GpuTopology(platform_version, num_partitions, num_hosts_per_partition,
                     num_devices_per_host, std::move(gpu_target_config),
                     std::move(host_target_machine_options));
}

GpuTopology GetSingleDeviceGpuTopology(
    absl::string_view platform_version,
    const gpu::GpuTargetConfig& gpu_target_config,
    const std::optional<cpu::TargetMachineOptions>&
        host_target_machine_options) {
  return GpuTopology(platform_version, 1, 1, 1, gpu_target_config,
                     host_target_machine_options);
}

}  // namespace xla

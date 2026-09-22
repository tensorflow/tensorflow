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

#include "xla/backends/gpu/autotuner/triton/triton_configs.h"

#include <cstddef>
#include <initializer_list>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/autotuner/triton/embed_default_configs.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace xla::gpu {
namespace {

std::vector<TritonGemmConfig> ParseConfig(absl::string_view config_str) {
  TritonGemmConfigsProto proto;
  CHECK(tsl::protobuf::TextFormat::ParseFromString(config_str, &proto))
      << config_str;
  std::vector<TritonGemmConfig> configs;
  for (const auto& config_proto : proto.config()) {
    absl::StatusOr<TritonGemmConfig> config =
        TritonGemmConfig::FromProto(config_proto);
    CHECK_OK(config);
    configs.push_back(*config);
  }
  return configs;
}

absl::string_view GetDefaultConfigStr(absl::string_view filename) {
  const struct FileToc* toc = configs::embed_default_configs_create();
  for (size_t i = 0; i < configs::embed_default_configs_size(); ++i) {
    if (toc[i].name == filename) {
      return absl::string_view(toc[i].data, toc[i].size);
    }
  }
  LOG(FATAL) << "Embedded file not found: " << filename;
}

}  // namespace

const std::vector<TritonGemmConfig>& GetTritonConfigsForPlatform(
    TritonConfigsPlatform platform) {
  static const absl::NoDestructor<
      absl::flat_hash_map<TritonConfigsPlatform, std::vector<TritonGemmConfig>>>
      kConfigs({{TritonConfigsPlatform::kAmpere,
                 ParseConfig(GetDefaultConfigStr("a100.txtpb"))},
                {TritonConfigsPlatform::kBlackwell,
                 ParseConfig(GetDefaultConfigStr("b200.txtpb"))},
                {TritonConfigsPlatform::kBlackwellConsumer,
                 ParseConfig(GetDefaultConfigStr("sm120.txtpb"))},
                {TritonConfigsPlatform::kDefaultCuda,
                 ParseConfig(GetDefaultConfigStr("cuda.txtpb"))},
                {TritonConfigsPlatform::kDefaultRocm,
                 ParseConfig(GetDefaultConfigStr("rocm.txtpb"))},
                {TritonConfigsPlatform::kHopper,
                 ParseConfig(GetDefaultConfigStr("h100.txtpb"))},
                {TritonConfigsPlatform::kMI300,
                 ParseConfig(GetDefaultConfigStr("mi300.txtpb"))},
                {TritonConfigsPlatform::kMI350,
                 ParseConfig(GetDefaultConfigStr("mi350.txtpb"))}});
  return kConfigs->at(platform);
}

const std::vector<TritonGemmConfig>& GetDefaultTritonConfigs(
    const stream_executor::GpuComputeCapability& compute_capability) {
  if (compute_capability.IsRocm()) {
    const stream_executor::RocmComputeCapability* rocm_cc =
        compute_capability.rocm_compute_capability();
    if (rocm_cc->gfx9_mi300()) {
      return GetTritonConfigsForPlatform(TritonConfigsPlatform::kMI300);
    }
    if (rocm_cc->gfx9_mi350()) {
      return GetTritonConfigsForPlatform(TritonConfigsPlatform::kMI350);
    }
    return GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultRocm);
  }

  CHECK(compute_capability.IsCuda());
  const stream_executor::CudaComputeCapability* cuda_compute_capability =
      compute_capability.cuda_compute_capability();

  if (cuda_compute_capability->IsBlackwell()) {
    // SM 10.0 (datacenter: B200, B100)
    return GetTritonConfigsForPlatform(TritonConfigsPlatform::kBlackwell);
  }
  if (cuda_compute_capability->IsAtLeastBlackwell()) {
    // SM 11.0+ / 12.0+ (consumer: RTX 5090, etc.)
    return GetTritonConfigsForPlatform(
        TritonConfigsPlatform::kBlackwellConsumer);
  }
  if (cuda_compute_capability->IsHopper()) {
    return GetTritonConfigsForPlatform(TritonConfigsPlatform::kHopper);
  }
  if (cuda_compute_capability->IsAmpere()) {
    return GetTritonConfigsForPlatform(TritonConfigsPlatform::kAmpere);
  }
  return GetTritonConfigsForPlatform(TritonConfigsPlatform::kDefaultCuda);
}

}  // namespace xla::gpu

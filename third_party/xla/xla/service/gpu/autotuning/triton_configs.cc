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

#include "xla/service/gpu/autotuning/triton_configs.h"

#include <initializer_list>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/autotuning.pb.h"
#include "xla/service/gpu/autotuning/embed_default_configs.h"
#include "xla/service/gpu/matmul_utils.h"

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
};

}  // namespace

const std::vector<TritonGemmConfig>& GetTritonConfigsForPlatform(
    TritonConfigsPlatform platform) {
  static const absl::NoDestructor<
      absl::flat_hash_map<TritonConfigsPlatform, std::vector<TritonGemmConfig>>>
      kConfigs(
          {{TritonConfigsPlatform::kAmpere, ParseConfig(configs::get_a100())},
           {TritonConfigsPlatform::kBlackwell,
            ParseConfig(configs::get_b200())},
           {TritonConfigsPlatform::kDefaultCuda,
            ParseConfig(configs::get_cuda())},
           {TritonConfigsPlatform::kDefaultRocm,
            ParseConfig(configs::get_rocm())},
           {TritonConfigsPlatform::kHopper, ParseConfig(configs::get_h100())}});
  return kConfigs->at(platform);
}

}  // namespace xla::gpu

/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/cpu/cpu_options.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo_module_config.h"

namespace xla::cpu::options {

bool OptimizeForSizeRequested(const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  return extra_options_map.count(kXlaOptimizeForSizeCpuOption) > 0;
}

bool VectorizedReduceDisabled(const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  return extra_options_map.count(kXlaOptimizeForSizeCpuOption) > 0;
}

bool SlpVectorizerDisabled(const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  return extra_options_map.count(kDisableSlpVectorizer) > 0;
}

bool DisableLoopUnrolling(const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  return extra_options_map.count(kDisableLoopUnrolling) > 0;
}

bool FoldAllConstants(const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  return extra_options_map.count(kFoldAllConstants) > 0;
}

std::optional<int64_t> LlvmIrGemvTilingFactor(const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  auto it = extra_options_map.find(kLlvmIrDotTilingFactor);
  int64_t tiling_factor;
  if (it != extra_options_map.end() &&
      absl::SimpleAtoi(it->second, &tiling_factor)) {
    return tiling_factor;
  }
  return std::nullopt;
}

absl::StatusOr<int64_t> SmallWhileLoopByteThreshold(
    const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();

  auto itr = extra_options_map.find(kSmallWhileLoopByteThreshold);
  if (itr == extra_options_map.end()) {
    return 1024;  // Default value.
  }

  int64_t byte_threshold;
  if (!absl::SimpleAtoi(itr->second, &byte_threshold)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to parse value for: ", kSmallWhileLoopByteThreshold, "."));
  }
  return byte_threshold;
}

bool ForceEnableExperimentalLlvmIrGemm(const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  return extra_options_map.count(kXlaForceEnableExperimentalLlvmIrGemm) > 0;
}

static absl::string_view RemoveSuffix(absl::string_view str,
                                      absl::string_view suffix) {
  CHECK_GE(str.size(), suffix.size());
  CHECK_EQ(str.substr(str.size() - suffix.size()), suffix);
  return str.substr(0, str.size() - suffix.size());
}

std::optional<std::tuple<int64_t, int64_t, int64_t>> LlvmIrGemmTileSize(
    const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  auto it = extra_options_map.find(kLlvmIrGemmTileSize);
  if (it == extra_options_map.end()) {
    return std::nullopt;
  }

  std::vector<std::string> tile_components = absl::StrSplit(it->second, ':');
  CHECK_EQ(tile_components.size(), 3);

  int64_t tile_size_m;
  int64_t tile_size_k;
  int64_t tile_size_n_in_vector_width;

  CHECK(absl::SimpleAtoi(tile_components[0], &tile_size_m));
  CHECK(absl::SimpleAtoi(tile_components[1], &tile_size_k));

  absl::string_view tile_size_n_in_vector_width_str =
      RemoveSuffix(tile_components[2], "*vectwidth");

  CHECK(absl::SimpleAtoi(tile_size_n_in_vector_width_str,
                         &tile_size_n_in_vector_width));

  return std::tuple<int64_t, int64_t, int64_t>(tile_size_m, tile_size_k,
                                               tile_size_n_in_vector_width);
}

}  // namespace xla::cpu::options

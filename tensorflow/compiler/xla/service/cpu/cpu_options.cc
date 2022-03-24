/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/cpu_options.h"

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace {

const char* const kXlaOptimizeForSizeCpuOption = "xla_cpu_optimize_for_size";
const char* const kLlvmIrDotTilingFactor = "xla_llvm_dot_tiling_factor";
const char* const kXlaForceEnableExperimentalLlvmIrGemm =
    "xla_force_enable_experimental_llvm_ir_gemm";
const char* const kLlvmIrGemmTileSize = "xla_llvm_ir_gemm_tile_size";

}  // namespace

namespace xla {
namespace cpu {
namespace options {

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

absl::optional<int64_t> LlvmIrGemvTilingFactor(const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  auto it = extra_options_map.find(kLlvmIrDotTilingFactor);
  int64_t tiling_factor;
  if (it != extra_options_map.end() &&
      absl::SimpleAtoi(it->second, &tiling_factor)) {
    return tiling_factor;
  }
  return absl::nullopt;
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

absl::optional<std::tuple<int64_t, int64_t, int64_t>> LlvmIrGemmTileSize(
    const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  auto it = extra_options_map.find(kLlvmIrGemmTileSize);
  if (it == extra_options_map.end()) {
    return absl::nullopt;
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

}  // namespace options
}  // namespace cpu
}  // namespace xla

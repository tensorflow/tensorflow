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

#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace {

const char* const kXlaOptimizeForSizeCpuOption = "xla_cpu_optimize_for_size";
const char* const kXlaDisableVectorizedReduce = "xla_disable_vectorized_reduce";
const char* const kLlvmIrDotTilingFactor = "xla_llvm_dot_tiling_factor";
const char* const kXlaEnableExperimentalLlvmIrGemm =
    "xla_enable_experimental_llvm_ir_gemm";
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

tensorflow::gtl::optional<int64> LlvmIrGemvTilingFactor(
    const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  auto it = extra_options_map.find(kLlvmIrDotTilingFactor);
  int64 tiling_factor;
  if (it != extra_options_map.end() &&
      tensorflow::strings::safe_strto64(it->second, &tiling_factor)) {
    return tiling_factor;
  }
  return tensorflow::gtl::nullopt;
}

bool EnableExperimentalLlvmIrGemm(const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  return extra_options_map.count(kXlaEnableExperimentalLlvmIrGemm) > 0;
}

static tensorflow::StringPiece RemoveSuffix(tensorflow::StringPiece str,
                                            tensorflow::StringPiece suffix) {
  CHECK_GE(str.size(), suffix.size());
  CHECK_EQ(str.substr(str.size() - suffix.size()), suffix);
  return str.substr(0, str.size() - suffix.size());
}

tensorflow::gtl::optional<std::tuple<int64, int64, int64>> LlvmIrGemmTileSize(
    const HloModuleConfig& config) {
  const auto& extra_options_map =
      config.debug_options().xla_backend_extra_options();
  auto it = extra_options_map.find(kLlvmIrGemmTileSize);
  if (it == extra_options_map.end()) {
    return tensorflow::gtl::nullopt;
  }

  std::vector<string> tile_components =
      tensorflow::str_util::Split(it->second, ':');
  CHECK_EQ(tile_components.size(), 3);

  int64 tile_size_m;
  int64 tile_size_k;
  int64 tile_size_n_in_vector_width;

  CHECK(tensorflow::strings::safe_strto64(tile_components[0], &tile_size_m));
  CHECK(tensorflow::strings::safe_strto64(tile_components[1], &tile_size_k));

  tensorflow::StringPiece tile_size_n_in_vector_width_str =
      RemoveSuffix(tile_components[2], "*vectwidth");

  CHECK(tensorflow::strings::safe_strto64(tile_size_n_in_vector_width_str,
                                          &tile_size_n_in_vector_width));

  return std::tuple<int64, int64, int64>(tile_size_m, tile_size_k,
                                         tile_size_n_in_vector_width);
}

}  // namespace options
}  // namespace cpu
}  // namespace xla

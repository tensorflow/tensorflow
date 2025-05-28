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

#ifndef XLA_SERVICE_CPU_CPU_OPTIONS_H_
#define XLA_SERVICE_CPU_CPU_OPTIONS_H_

#include <cstdint>
#include <optional>
#include <tuple>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo_module_config.h"

// Helper functions for querying options that are specific to the CPU backend.

namespace xla::cpu::options {

inline constexpr absl::string_view kXlaOptimizeForSizeCpuOption =
    "xla_cpu_optimize_for_size";
inline constexpr absl::string_view kLlvmIrDotTilingFactor =
    "xla_llvm_dot_tiling_factor";
inline constexpr absl::string_view kXlaForceEnableExperimentalLlvmIrGemm =
    "xla_force_enable_experimental_llvm_ir_gemm";
inline constexpr absl::string_view kLlvmIrGemmTileSize =
    "xla_llvm_ir_gemm_tile_size";
inline constexpr absl::string_view kDisableSlpVectorizer =
    "xla_cpu_disable_slp_vectorizer";
inline constexpr absl::string_view kDisableLoopUnrolling =
    "xla_cpu_disable_loop_unrolling";
inline constexpr absl::string_view kFoldAllConstants =
    "xla_cpu_fold_all_constants";
inline constexpr absl::string_view kSmallWhileLoopByteThreshold =
    "xla_cpu_small_while_loop_byte_threshold";

bool OptimizeForSizeRequested(const HloModuleConfig& config);
bool VectorizedReduceDisabled(const HloModuleConfig& config);
bool SlpVectorizerDisabled(const HloModuleConfig& config);
bool DisableLoopUnrolling(const HloModuleConfig& config);
bool FoldAllConstants(const HloModuleConfig& config);
bool ForceEnableExperimentalLlvmIrGemm(const HloModuleConfig& config);
std::optional<int64_t> LlvmIrGemvTilingFactor(const HloModuleConfig& config);
std::optional<std::tuple<int64_t, int64_t, int64_t>> LlvmIrGemmTileSize(
    const HloModuleConfig& config);
absl::StatusOr<int64_t> SmallWhileLoopByteThreshold(
    const HloModuleConfig& config);

}  // namespace xla::cpu::options

#endif  // XLA_SERVICE_CPU_CPU_OPTIONS_H_

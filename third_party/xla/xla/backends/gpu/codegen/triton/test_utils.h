/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_TEST_UTILS_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_TEST_UTILS_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

std::vector<xla::PrimitiveType> AllXlaDataTypes();

bool SupportsBF16(const stream_executor::GpuComputeCapability& cc);

absl::Status CreateTritonIrAndFileCheck(const HloModule* hlo_module,
                                        absl::string_view triton_fusion_name,
                                        absl::string_view filecheck_pattern);

absl::Status CreateTritonIrAndFileCheck(
    const HloComputation& computation,
    const BlockLevelParameters& block_level_parameters,
    absl::string_view filecheck_pattern);

absl::Status CreateTritonIrAndFileCheckForDot(
    const HloModule* hlo_module, absl::string_view triton_fusion_name,
    absl::string_view filecheck_pattern);

absl::Status CreateTritonIrAndFileCheckForDot(
    const HloComputation& computation, absl::string_view filecheck_pattern);

inline BlockLevelParameters FromOutputTileSizes(
    std::vector<std::vector<int64_t>> output_tile_sizes) {
  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = std::move(output_tile_sizes);
  return block_level_parameters;
}

absl::StatusOr<bool> ApplyFloatNormalization(
    HloModule* module, const stream_executor::GpuComputeCapability& cc);

}  //  namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_TEST_UTILS_H_

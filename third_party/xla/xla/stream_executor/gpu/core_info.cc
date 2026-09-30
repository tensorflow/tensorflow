/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/core_info.h"

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_core_info_table.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/dtype_core_info.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/rocm/rocm_core_info_table.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace gpu {
namespace {

// Assumed when the architecture has no FP32 vector info and its backend offers
// no better guess. Every untabulated backend/target lands here.
constexpr int kUntabulatedFpusPerCore = 128;

absl::flat_hash_map<int, DTypeCoreInfo> MakeBitwidthToInfoMap(
    absl::Span<const DTypeCoreInfo> infos, bool is_float) {
  absl::flat_hash_map<int, DTypeCoreInfo> bitwidth_to_info;
  for (const auto& info : infos) {
    if (info.dtype.is_float != is_float) {
      continue;
    }
    bitwidth_to_info[info.dtype.bitwidth] = info;
  }
  return bitwidth_to_info;
}

void AddDTypeInfoToDesc(
    xla::PrimitiveType dtype, float base_clock_rate_ghz,
    const absl::flat_hash_map<int, DTypeCoreInfo>& bitwidth_to_info,
    ExecutionUnitDescription& desc) {
  int bitwidth = xla::primitive_util::BitWidth(dtype);
  const auto bitwidth_it = bitwidth_to_info.find(bitwidth);
  if (bitwidth_it == bitwidth_to_info.end()) {
    return;
  }
  const DTypeCoreInfo& perf_info = bitwidth_it->second;
  float clock_rate_ghz = perf_info.clock_scale * base_clock_rate_ghz;
  desc.SetRateInfo(dtype, ExecutionUnitDescription::RateInfo{
                              /*units_per_core=*/perf_info.units_per_core,
                              /*clock_rate_ghz=*/clock_rate_ghz,
                              /*ops_per_clock=*/perf_info.ops_per_clock});
}

ExecutionUnitDescription CreateEuDescription(
    float base_clock_rate_ghz, absl::Span<const DTypeCoreInfo> core_infos) {
  ExecutionUnitDescription desc;
  absl::flat_hash_map<int, DTypeCoreInfo> bitwidth_to_float_info =
      MakeBitwidthToInfoMap(core_infos, /*is_float=*/true);

  xla::primitive_util::FloatingPointTypeForEach([&](auto dtype) {
    AddDTypeInfoToDesc(dtype, base_clock_rate_ghz, bitwidth_to_float_info,
                       desc);
  });

  absl::flat_hash_map<int, DTypeCoreInfo> bitwidth_to_int_info =
      MakeBitwidthToInfoMap(core_infos, /*is_float=*/false);
  xla::primitive_util::IntegralTypeForEach([&](auto dtype) {
    AddDTypeInfoToDesc(dtype, base_clock_rate_ghz, bitwidth_to_int_info, desc);
  });

  return desc;
}

CoreInfo FindCoreInfoForCapability(const GpuComputeCapability& cc) {
  if (const CudaComputeCapability* cuda_cc = cc.cuda_compute_capability()) {
    return FindCudaCoreInfo(*cuda_cc);
  }
  if (const RocmComputeCapability* rocm_cc = cc.rocm_compute_capability()) {
    return FindRocmCoreInfo(*rocm_cc);
  }
  return CoreInfo{};
}

const DTypeCoreInfo* FindFp32Info(absl::Span<const DTypeCoreInfo> infos) {
  for (const DTypeCoreInfo& info : infos) {
    if (info.dtype.is_float && info.dtype.bitwidth == 32) {
      return &info;
    }
  }
  return nullptr;
}

}  // namespace

void FillExecutionUnitDesc(const GpuComputeCapability& cc,
                           float base_clock_rate_ghz, DeviceDescription& desc) {
  // Leaving a field unset is deliberate. Consumers treat it as "unknown" and
  // fall back to their own estimates.
  CoreInfo core_info = FindCoreInfoForCapability(cc);
  if (!core_info.vector_infos.empty()) {
    desc.set_scalar_unit_description(
        CreateEuDescription(base_clock_rate_ghz, core_info.vector_infos));
  }
  if (!core_info.matrix_infos.empty()) {
    desc.set_matrix_unit_description(
        CreateEuDescription(base_clock_rate_ghz, core_info.matrix_infos));
  }
}

int GetFpusPerCore(const GpuComputeCapability& cc) {
  CoreInfo core_info = FindCoreInfoForCapability(cc);
  const DTypeCoreInfo* fp32_info = FindFp32Info(core_info.vector_infos);
  if (fp32_info != nullptr) {
    return fp32_info->units_per_core;
  }
  if (const CudaComputeCapability* cuda_cc = cc.cuda_compute_capability()) {
    return CudaFpusPerCoreFallback(*cuda_cc);
  }
  return kUntabulatedFpusPerCore;
}

}  // namespace gpu
}  // namespace stream_executor

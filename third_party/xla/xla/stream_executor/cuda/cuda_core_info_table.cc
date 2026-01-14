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

#include "xla/stream_executor/cuda/cuda_core_info_table.h"

#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace stream_executor {
namespace gpu {
namespace {

// Instead of using base primitive types we use a simple description that maps
// to several primitive types at once. This way we can keep the types in the
// table below more abstract.
struct DTypeDescr {
  bool is_float;
  int bitwidth;
};

constexpr DTypeDescr kI8 = DTypeDescr{/*is_float=*/false, 8};
constexpr DTypeDescr kI32 = DTypeDescr{/*is_float=*/false, 32};

constexpr DTypeDescr kF8 = DTypeDescr{/*is_float=*/true, 8};
constexpr DTypeDescr kF16 = DTypeDescr{/*is_float=*/true, 16};
constexpr DTypeDescr kF32 = DTypeDescr{/*is_float=*/true, 32};
constexpr DTypeDescr kF64 = DTypeDescr{/*is_float=*/true, 64};

struct DTypeCoreInfo {
  DTypeDescr dtype;
  int units_per_sm;
  int ops_per_clock = 1;    // Note: FMA is considered 1 op.
  float clock_scale = 1.0;  // Ratio of clock rate of this unit vs base device.
};

const std::vector<DTypeCoreInfo>* FindCoreInfoForDType(CudaComputeCapability cc,
                                                       bool is_tensor) {
  struct CoreInfoTableForCC {
    CudaComputeCapability cc;
    std::vector<DTypeCoreInfo> cuda_core_infos;
    std::vector<DTypeCoreInfo> tensor_core_infos;
  };

  // =============== Sources ===============
  // When adding a new source make sure to include the version.
  //
  // [AmpereArch] NVIDIA A100 Tensor Core GPU Architecture, v1.0
  // https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
  //
  // [HopperArch] NVIDIA H100 Tensor Core GPU Architecture, v1.04
  // https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c
  //
  // [ArithmThroughput] CUDA C++ Best Practices Guide, v13.0
  // Table 5 Throughput of Native Arithmetic Instructions.
  // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#throughput-of-native-arithmetic-instructions

  // =============== Constants ===============
  // Make sure to annotate with the sources when adding new data.

  // [HopperArch] Numbers from "Table 3. Comparison of NVIDIA A100 and H100 Data
  // Center GPUs".
  // TC vs non-TC clock rate on H100: 1830 MHz / 1980 MHz
  constexpr float kHopperTcClockScale = 0.924;
  constexpr int kHopperTcPerSm = 4;

  // =============== Lookup table ===============
  // Make sure to annotate with the sources when adding new data.
  static const absl::NoDestructor<std::vector<CoreInfoTableForCC>> kTable(
      std::vector<CoreInfoTableForCC>{
          {CudaComputeCapability::Hopper(),
           // Most numbers are taken from [HopperArch] - "Table 3. Comparison of
           // NVIDIA A100 and H100 Data Center GPUs" unless otherwise specified.
           /*cuda_core_infos=*/
           {
               // DType, Units/SM, Ops/Clk
               {kF16, 128, 2},  // [ArithmThroughput]: 2-way SIMD
               {kF32, 128, 1},
               {kF64, 64, 1},
               {kI32, 64, 1},
           },
           // Ops per clock taken from [AmpereArch] - "A100 SM Architecture" -
           // and multiplied by 2 as mentioned in [HopperArch] - "H100 SM
           // Architecture".
           // Note: numbers in the sources are often per SM, we convert them to
           // per-TC.
           /*tensor_core_infos=*/
           {
               // DType  Units/SM   Ops/Clk  ClockScale
               {kI8, kHopperTcPerSm, 1024, kHopperTcClockScale},
               {kF8, kHopperTcPerSm, 1024, kHopperTcClockScale},
               {kF16, kHopperTcPerSm, 512, kHopperTcClockScale},
               {kF32, kHopperTcPerSm, 256, kHopperTcClockScale},
               // * Ops per clock: [AmpereArch] - "A100 Tensor Cores Accelerate
               // HPC" multiplied by 2 like the numbers above.
               // * FP64 TC runs at base clock rate.
               {kF64, kHopperTcPerSm, 32, 1.0},
           }}});

  for (const auto& config : *kTable) {
    if (config.cc == cc.WithoutAnyFeatureExtension()) {
      return is_tensor ? &config.tensor_core_infos : &config.cuda_core_infos;
    }
  }

  return nullptr;
}

absl::flat_hash_map<int, DTypeCoreInfo> MakeBitwidthToRowMap(
    const std::vector<DTypeCoreInfo>& rows, bool is_float) {
  absl::flat_hash_map<int, DTypeCoreInfo> bitwidth_to_row;
  for (const auto& row : rows) {
    if (row.dtype.is_float != is_float) {
      continue;
    }
    bitwidth_to_row[row.dtype.bitwidth] = row;
  }
  return bitwidth_to_row;
}

void AddDTypeInfoToDesc(
    xla::PrimitiveType dtype, float base_clock_rate_ghz,
    const absl::flat_hash_map<int, DTypeCoreInfo>& bitwidth_to_row,
    ExecutionUnitDescription& desc) {
  int bitwidth = xla::primitive_util::BitWidth(dtype);
  const auto& bitwidth_it = bitwidth_to_row.find(bitwidth);
  if (bitwidth_it == bitwidth_to_row.end()) {
    return;
  }
  const DTypeCoreInfo& perf_info = bitwidth_it->second;
  float clock_rate_ghz = perf_info.clock_scale * base_clock_rate_ghz;
  desc.SetRateInfo(dtype, stream_executor::ExecutionUnitDescription::RateInfo{
                              /*units_per_core=*/perf_info.units_per_sm,
                              /*clock_rate_ghz=*/clock_rate_ghz,
                              /*ops_per_clock=*/perf_info.ops_per_clock});
}

ExecutionUnitDescription CreateEuDescription(
    float base_clock_rate_ghz, const std::vector<DTypeCoreInfo>& perf_rows) {
  ExecutionUnitDescription desc;
  absl::flat_hash_map<int, DTypeCoreInfo> bitwidth_to_float_row =
      MakeBitwidthToRowMap(perf_rows, /*is_float=*/true);

  xla::primitive_util::FloatingPointTypeForEach([&](auto dtype) {
    AddDTypeInfoToDesc(dtype, base_clock_rate_ghz, bitwidth_to_float_row, desc);
  });

  absl::flat_hash_map<int, DTypeCoreInfo> bitwidth_to_int_row =
      MakeBitwidthToRowMap(perf_rows, /*is_float=*/false);
  xla::primitive_util::IntegralTypeForEach([&](auto dtype) {
    AddDTypeInfoToDesc(dtype, base_clock_rate_ghz, bitwidth_to_int_row, desc);
  });

  return desc;
}

}  // namespace

void FillExecutionUnitDesc(CudaComputeCapability cc, float base_clock_rate_ghz,
                           DeviceDescription& desc) {
  const std::vector<DTypeCoreInfo>* cuda_core_rows =
      FindCoreInfoForDType(cc, /*is_tensor=*/false);
  if (cuda_core_rows != nullptr) {
    ExecutionUnitDescription cuda_core_desc =
        CreateEuDescription(base_clock_rate_ghz, *cuda_core_rows);
    desc.set_scalar_unit_description(std::move(cuda_core_desc));
  }

  const std::vector<DTypeCoreInfo>* tc_rows =
      FindCoreInfoForDType(cc, /*is_tensor=*/true);
  if (tc_rows != nullptr) {
    ExecutionUnitDescription tc_desc =
        CreateEuDescription(base_clock_rate_ghz, *tc_rows);
    desc.set_matrix_unit_description(std::move(tc_desc));
  }
}

int GetFpusPerCore(CudaComputeCapability cc) {
  const std::vector<DTypeCoreInfo>* cuda_core_rows =
      FindCoreInfoForDType(cc, /*is_tensor=*/false);

  if (cuda_core_rows != nullptr) {
    for (const auto& perf : *cuda_core_rows) {
      if (perf.dtype.is_float && perf.dtype.bitwidth == 32) {
        return perf.units_per_sm;
      }
    }
  }

  // Fallback to hardcoded values if not found in the table.
  // Source:
  // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#throughput-of-native-arithmetic-instructions
  int n = 128;          // 5.x, 6.1, 6.2, 8.6, 9.0 -> 128.
  if (cc.major == 3) {  // 3.x -> 192.
    n = 192;
  } else if ((cc.major == 6 && cc.minor == 0) || (cc.major == 7) ||
             (cc.major == 8 && cc.minor == 0)) {
    n = 64;  // 6.0, 7.x, 8.0 -> 64.
  }
  return n;
}

}  // namespace gpu
}  // namespace stream_executor

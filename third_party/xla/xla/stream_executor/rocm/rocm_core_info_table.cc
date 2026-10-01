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

#include "xla/stream_executor/rocm/rocm_core_info_table.h"

#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/gpu/dtype_core_info.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"

namespace stream_executor {
namespace gpu {

CoreInfo FindRocmCoreInfo(const RocmComputeCapability& cc) {
  struct CoreInfoTableForArch {
    absl::string_view gfx_version;
    std::vector<DTypeCoreInfo> vector_infos;
    std::vector<DTypeCoreInfo> matrix_infos;
  };

  // =============== Sources ===============
  // [CDNA1] Introducing AMD CDNA Architecture, Table 1, p.7 (MI100).
  //   https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/
  //   white-papers/amd-cdna-white-paper.pdf
  // [CDNA2] AMD CDNA 2 White Paper, Table 1, p.10 (MI250X).
  //   https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/
  //   white-papers/amd-cdna2-white-paper.pdf
  // [CDNA3] AMD CDNA 3 White Paper, Table 1, p.7 (MI300X/MI325X).
  //   https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/
  //   white-papers/amd-cdna-3-white-paper.pdf
  // [CDNA4] Introducing AMD CDNA 4 Architecture, Table 1, p.8 (MI355X).
  //   https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/
  //   white-papers/amd-cdna-4-architecture-whitepaper.pdf
  //
  // Every CDNA CU has 4 SIMD/Matrix engines, hence units_per_core=4 throughout
  // matrix_infos. vector_infos instead counts FP32 FMA equivalent lanes, picked
  // so that units * ops * 2 reproduces the FLOPS/clock/CU in the tables above.

  static const absl::NoDestructor<std::vector<CoreInfoTableForArch>> kTable(
      std::vector<CoreInfoTableForArch>{
          // ===== gfx908 / CDNA1 / MI100 =====
          {"gfx908",
           /*vector_infos=*/
           {
               // DType, Units/CU, Ops/Clk        => FLOPS/clock/CU
               {kF32, 64, 1},  // 128 [CDNA1]
               {kF16, 64, 1},  // 128 (packed; assumed equal)
               {kF64, 32, 1},  //  64 [CDNA1]
           },
           /*matrix_infos=*/
           {
               // DType, Units/CU, Ops/Clk        => FLOPS/clock/CU
               {kF16, 4, 128},  // 1024 [CDNA1]
               // BF16 is 512 [CDNA1], half of FP16, but entries are keyed by
               // bitwidth so both share this one and BF16 comes out 2x high.
               {kF32, 4, 32},  //  256 [CDNA1]
               {kI8, 4, 128},  // 1024 [CDNA1]
           }},
          // ===== gfx90a / CDNA2 / MI210/MI250/MI250X =====
          {"gfx90a",
           /*vector_infos=*/
           {
               {kF32, 64, 1},  // 128 [CDNA2]
               {kF16, 64, 1},  // 128 (packed; assumed equal)
               {kF64, 64, 1},  // 128 [CDNA2]
           },
           /*matrix_infos=*/
           {
               {kF16, 4, 128},  // 1024 [CDNA2]
               {kF32, 4, 32},   //  256 [CDNA2]
               {kF64, 4, 32},   //  256 [CDNA2]
               {kI8, 4, 128},   // 1024 [CDNA2]
           }},
          // ===== gfx942 / CDNA3 / MI300A/MI300X/MI325X =====
          {"gfx942",
           /*vector_infos=*/
           {
               {kF32, 128, 1},  // 256 [CDNA3]
               {kF16, 128, 1},  // 256 (packed; assumed equal)
               {kF64, 64, 1},   // 128 [CDNA3]
           },
           /*matrix_infos=*/
           {
               {kF8, 4, 512},   // 4096 [CDNA3]
               {kF16, 4, 256},  // 2048 [CDNA3]
               {kF32, 4, 32},   //  256 [CDNA3]
               {kF64, 4, 32},   //  256 [CDNA3]
               {kI8, 4, 512},   // 4096 [CDNA3]
           }},
          // ===== gfx950 / CDNA4 / MI350/MI355X =====
          {"gfx950",
           /*vector_infos=*/
           {
               {kF32, 128, 1},  // 256 [CDNA4]
               {kF16, 128, 1},  // 256 [CDNA4] (vector FP16)
               {kF64, 64, 1},   // 128 [CDNA4]
           },
           /*matrix_infos=*/
           {
               {kF4, 4, 2048},  // 16384 [CDNA4] MXFP4
               {kF6, 4, 2048},  // 16384 [CDNA4] MXFP6
               {kF8, 4, 1024},  //  8192 [CDNA4]
               {kF16, 4, 512},  //  4096 [CDNA4]
               {kF32, 4, 32},   //   256 [CDNA4]
               {kF64, 4, 16},   //   128 [CDNA4] (halved)
               {kI8, 4, 1024},  //  8192 [CDNA4]
           }},
      });

  for (const auto& entry : *kTable) {
    if (cc.gfx_version() == entry.gfx_version) {
      return CoreInfo{entry.vector_infos, entry.matrix_infos};
    }
  }
  return CoreInfo{};
}

}  // namespace gpu
}  // namespace stream_executor

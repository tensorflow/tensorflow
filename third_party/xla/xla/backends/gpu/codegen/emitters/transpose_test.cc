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
#include "xla/backends/gpu/codegen/emitters/transpose.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;

class TransposeFusionTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<int64_t> GetPackedTransposeNumThreadsPerBlock(
      absl::string_view hlo_text, const se::DeviceDescription& device_info) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(hlo_text));
    HloFusionAnalysis analysis = HloFusionAnalysis::Create(
        *module->entry_computation()->root_instruction(), device_info);
    std::unique_ptr<MlirKernelEmitter> emitter =
        CreateTransposeFusion(analysis);
    if (dynamic_cast<const PackedTranspose*>(emitter.get()) == nullptr) {
      return absl::InternalError("Expected a PackedTranspose emitter.");
    }
    return emitter->launch_dimensions().num_threads_per_block();
  }

  absl::StatusOr<int64_t> GetStandardTransposeNumThreadsPerBlock(
      absl::string_view hlo_text, const se::DeviceDescription& device_info) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(hlo_text));
    HloFusionAnalysis analysis = HloFusionAnalysis::Create(
        *module->entry_computation()->root_instruction(), device_info);
    std::unique_ptr<MlirKernelEmitter> emitter =
        CreateTransposeFusion(analysis);
    if (dynamic_cast<const TransposeFusion*>(emitter.get()) == nullptr) {
      return absl::InternalError("Expected a TransposeFusion emitter.");
    }
    return emitter->launch_dimensions().num_threads_per_block();
  }
};

constexpr absl::string_view kS4PackedTransposeHlo = R"hlo(
  HloModule s4_packed_transpose
  fusion {
    p0 = s4[2,4096,16384]{2,1,0:E(4)} parameter(0)
    ROOT t = s4[2,16384,4096]{2,1,0:E(4)} transpose(p0), dimensions={0,2,1}
  }
  ENTRY main {
    p0 = s4[2,4096,16384]{2,1,0:E(4)} parameter(0)
    ROOT root = s4[2,16384,4096]{2,1,0:E(4)} fusion(p0), kind=kInput, calls=fusion
  }
)hlo";

constexpr absl::string_view kS8PackedTransposeHlo = R"hlo(
  HloModule s8_packed_transpose
  fusion {
    p0 = s8[2,4096,8192]{2,1,0} parameter(0)
    ROOT t = s8[2,8192,4096]{2,1,0} transpose(p0), dimensions={0,2,1}
  }
  ENTRY main {
    p0 = s8[2,4096,8192]{2,1,0} parameter(0)
    ROOT root = s8[2,8192,4096]{2,1,0} fusion(p0), kind=kInput, calls=fusion
  }
)hlo";

constexpr absl::string_view kBf16PackedTransposeHlo = R"hlo(
  HloModule bf16_packed_transpose
  fusion {
    p0 = bf16[2048,16,32]{2,1,0} parameter(0)
    ROOT t = bf16[32,16,2048]{2,1,0} transpose(p0), dimensions={2,1,0}
  }
  ENTRY main {
    p0 = bf16[2048,16,32]{2,1,0} parameter(0)
    ROOT root = bf16[32,16,2048]{2,1,0} fusion(p0), kind=kInput, calls=fusion
  }
)hlo";

TEST_F(TransposeFusionTest,
       PackedTransposeUses256ThreadsForSubByteTypesOnCuda) {
  EXPECT_THAT(
      GetPackedTransposeNumThreadsPerBlock(
          kS4PackedTransposeHlo, TestGpuDeviceInfo::B200SXMDeviceInfo()),
      IsOkAndHolds(256));
}

TEST_F(TransposeFusionTest, PackedTransposeUses256ThreadsFor8BitTypesOnCuda) {
  EXPECT_THAT(
      GetPackedTransposeNumThreadsPerBlock(
          kS8PackedTransposeHlo, TestGpuDeviceInfo::B200SXMDeviceInfo()),
      IsOkAndHolds(256));
}

TEST_F(TransposeFusionTest, PackedTransposeUses128ThreadsFor16BitTypesOnCuda) {
  EXPECT_THAT(
      GetPackedTransposeNumThreadsPerBlock(
          kBf16PackedTransposeHlo, TestGpuDeviceInfo::B200SXMDeviceInfo()),
      IsOkAndHolds(128));
}

TEST_F(TransposeFusionTest,
       PackedTransposeUses256ThreadsForSubByteTypesOnRocm) {
  EXPECT_THAT(
      GetPackedTransposeNumThreadsPerBlock(
          kS4PackedTransposeHlo, TestGpuDeviceInfo::AMDMI210DeviceInfo()),
      IsOkAndHolds(256));
}

TEST_F(TransposeFusionTest, PackedTransposeUses256ThreadsFor8BitTypesOnRocm) {
  EXPECT_THAT(
      GetPackedTransposeNumThreadsPerBlock(
          kS8PackedTransposeHlo, TestGpuDeviceInfo::AMDMI210DeviceInfo()),
      IsOkAndHolds(256));
}

TEST_F(TransposeFusionTest, PackedTransposeUses256ThreadsFor16BitTypesOnRocm) {
  EXPECT_THAT(
      GetPackedTransposeNumThreadsPerBlock(
          kBf16PackedTransposeHlo, TestGpuDeviceInfo::AMDMI210DeviceInfo()),
      IsOkAndHolds(256));
}

constexpr absl::string_view kF32StandardTransposeHlo = R"hlo(
  HloModule f32_standard_transpose
  fusion {
    p0 = f32[2,4096,4096]{2,1,0} parameter(0)
    ROOT t = f32[2,4096,4096]{2,1,0} transpose(p0), dimensions={0,2,1}
  }
  ENTRY main {
    p0 = f32[2,4096,4096]{2,1,0} parameter(0)
    ROOT root = f32[2,4096,4096]{2,1,0} fusion(p0), kind=kInput, calls=fusion
  }
)hlo";

TEST_F(TransposeFusionTest, StandardTransposeUses128ThreadsOnCuda) {
  EXPECT_THAT(
      GetStandardTransposeNumThreadsPerBlock(
          kF32StandardTransposeHlo, TestGpuDeviceInfo::B200SXMDeviceInfo()),
      IsOkAndHolds(128));
}

TEST_F(TransposeFusionTest, StandardTransposeUses256ThreadsOnRocm) {
  EXPECT_THAT(
      GetStandardTransposeNumThreadsPerBlock(
          kF32StandardTransposeHlo, TestGpuDeviceInfo::AMDMI210DeviceInfo()),
      IsOkAndHolds(256));
}

}  // namespace
}  // namespace xla::gpu

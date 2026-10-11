/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ============================================================================
 */
#include "xla/backends/gpu/codegen/emitters/concatenate.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

class ConcatenateFusionTest : public HloHardwareIndependentTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    auto debug_options = GetDebugOptionsFromFlags();
    debug_options.set_xla_gpu_experimental_max_unroll_factor(32);
    return debug_options;
  }
};

TEST_F(ConcatenateFusionTest, PropagatesUnrollFactorToCompilationPipeline) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    ENTRY main {
      p0 = bf16[1024000] parameter(0)
      p1 = bf16[1024000] parameter(1)
      ROOT result = bf16[2048000] concatenate(p0, p1), dimensions={0}
    })"));

  se::DeviceDescription device_info = TestGpuDeviceInfo::B200SXMDeviceInfo();
  device_info.set_compile_time_toolkit_version(se::SemanticVersion(12, 9, 0));
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(
      *module->entry_computation()->root_instruction(), device_info);
  ConcatenateFusion concatenate_fusion(analysis);
  const MlirKernelEmitter& compilation_pipeline_emitter = concatenate_fusion;

  // Blackwell with CUDA 12.9 vectorizes up to 256 bits, or 16 BF16 elements.
  EXPECT_EQ(compilation_pipeline_emitter.unroll_factor(), 16);
}

}  // namespace
}  // namespace xla::gpu

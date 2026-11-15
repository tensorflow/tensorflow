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

#include "xla/service/gpu/autotuning/gpu_default_backend_policy.h"

#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_module_config.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

class GpuDefaultBackendPolicyTest : public HloHardwareIndependentTestBase {
 protected:
  // Returns the default backend of the root of `hlo_text`, parsed with
  // `debug_options` if given.
  std::optional<autotuner::Backend> GetDefaultBackend(
      absl::string_view hlo_text, const DebugOptions* debug_options = nullptr) {
    HloModuleConfig config = GetModuleConfigForTest();
    if (debug_options != nullptr) {
      config.set_debug_options(*debug_options);
    }
    absl::StatusOr<std::unique_ptr<HloModule>> module =
        ParseAndReturnVerifiedModule(hlo_text, config);
    CHECK_OK(module.status());
    return gpu::GetDefaultBackend(
        *(*module)->entry_computation()->root_instruction());
  }
};

constexpr absl::string_view kSlowLoopTransposeHlo = R"(
transpose {
  p0 = f32[256,128,64]{2,1,0} parameter(0)
  ROOT transpose = f32[64,128,256]{2,1,0} transpose(p0), dimensions={2,1,0}
}

ENTRY main {
  p0 = f32[256,128,64]{2,1,0} parameter(0)
  ROOT fusion = f32[64,128,256]{2,1,0} fusion(p0), kind=kLoop, calls=transpose
})";

TEST_F(GpuDefaultBackendPolicyTest, SlowLoopTransposeDefaultsToBlockLevel) {
  EXPECT_EQ(GetDefaultBackend(kSlowLoopTransposeHlo),
            autotuner::Backend::BLOCK_LEVEL_EMITTER);
}

// The minormost dimension of the input stays the second minormost dimension
// of the output.
TEST_F(GpuDefaultBackendPolicyTest, FastLoopTransposeHasNoDefault) {
  EXPECT_EQ(GetDefaultBackend(R"(
transpose {
  p0 = f32[256,128,64]{2,1,0} parameter(0)
  ROOT transpose = f32[256,64,128]{2,1,0} transpose(p0), dimensions={0,2,1}
}

ENTRY main {
  p0 = f32[256,128,64]{2,1,0} parameter(0)
  ROOT fusion = f32[256,64,128]{2,1,0} fusion(p0), kind=kLoop, calls=transpose
})"),
            std::nullopt);
}

// Triton is only guaranteed to tile a transposed bitcast efficiently if the
// minor dimension of the bitcast is a power of two.
TEST_F(GpuDefaultBackendPolicyTest,
       LoopTransposeOfBitcastNeedsPowerOfTwoMinorDim) {
  // A transpose of a bitcast whose minor dimension is `$1`.
  constexpr absl::string_view kBitcastTransposeHlo = R"(
transpose {
  p0 = f32[1024,$0]{1,0} parameter(0)
  bitcast = f32[1024,1024,$1]{2,1,0} bitcast(p0)
  ROOT transpose = f32[$1,1024,1024]{2,1,0} transpose(bitcast), dimensions={2,1,0}
}

ENTRY main {
  p0 = f32[1024,$0]{1,0} parameter(0)
  ROOT fusion = f32[$1,1024,1024]{2,1,0} fusion(p0), kind=kLoop, calls=transpose
})";
  EXPECT_EQ(GetDefaultBackend(absl::Substitute(kBitcastTransposeHlo, 4096, 4)),
            autotuner::Backend::BLOCK_LEVEL_EMITTER);
  EXPECT_EQ(GetDefaultBackend(absl::Substitute(kBitcastTransposeHlo, 6144, 6)),
            std::nullopt);
}

TEST_F(GpuDefaultBackendPolicyTest, OnlyS32ReductionDefaultsToBlockLevel) {
  // A reduction of type `$0`.
  constexpr absl::string_view kReduceHlo = R"(
add {
  p0 = $0[] parameter(0)
  p1 = $0[] parameter(1)
  ROOT add = $0[] add(p0, p1)
}

fused_reduce {
  p0 = $0[128,256]{1,0} parameter(0)
  c0 = $0[] constant(0)
  ROOT reduce = $0[128]{0} reduce(p0, c0), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = $0[128,256]{1,0} parameter(0)
  ROOT fusion = $0[128]{0} fusion(p0), kind=kInput, calls=fused_reduce
})";
  EXPECT_EQ(GetDefaultBackend(absl::Substitute(kReduceHlo, "s32")),
            autotuner::Backend::BLOCK_LEVEL_EMITTER);
  EXPECT_EQ(GetDefaultBackend(absl::Substitute(kReduceHlo, "f32")),
            std::nullopt);
}

// Without the dot check, this s32 reduction would default to the block-level
// emitter.
TEST_F(GpuDefaultBackendPolicyTest, DotFusionHasNoDefault) {
  EXPECT_EQ(GetDefaultBackend(R"(
add {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  ROOT add = s32[] add(p0, p1)
}

fused_reduce {
  p0 = s32[16,16]{1,0} parameter(0)
  p1 = s32[16,16]{1,0} parameter(1)
  dot = s32[16,16]{1,0} dot(p0, p1), lhs_contracting_dims={1},
    rhs_contracting_dims={0}
  c0 = s32[] constant(0)
  ROOT reduce = s32[16]{0} reduce(dot, c0), dimensions={1}, to_apply=add
}

ENTRY main {
  p0 = s32[16,16]{1,0} parameter(0)
  p1 = s32[16,16]{1,0} parameter(1)
  ROOT fusion = s32[16]{0} fusion(p0, p1), kind=kInput, calls=fused_reduce
})"),
            std::nullopt);
}

TEST_F(GpuDefaultBackendPolicyTest, ScanDefaultsToBlockLevel) {
  // A fusion whose root is the output of a scan at index `$1`, of type `$0`.
  constexpr absl::string_view kScanHlo = R"(
combiner {
  in = f32[] parameter(0)
  carry = f32[] parameter(1)
  add = f32[] add(carry, in)
  ROOT t = (f32[], f32[]) tuple(add, add)
}

fused_scan {
  input = f32[100]{0} parameter(0)
  init = f32[] parameter(1)
  scan = (f32[100]{0}, f32[]) scan(input, init), dimensions={0},
    num_carries=1, to_apply=combiner, is_associative=true
  ROOT gte = $0 get-tuple-element(scan), index=$1
}

ENTRY main {
  input = f32[100]{0} parameter(0)
  init = f32[] constant(0)
  ROOT fusion = $0 fusion(input, init), kind=kLoop, calls=fused_scan
})";
  EXPECT_EQ(GetDefaultBackend(absl::Substitute(kScanHlo, "f32[100]{0}", 0)),
            autotuner::Backend::BLOCK_LEVEL_EMITTER);
  // The carry is the scan output at index 1.
  EXPECT_EQ(GetDefaultBackend(absl::Substitute(kScanHlo, "f32[]", 1)),
            std::nullopt);
}

constexpr absl::string_view kElementwiseHlo = R"(
fusion_computation {
  p0 = f32[128,128] parameter(0)
  ROOT exp = f32[128,128] exponential(p0)
}

ENTRY main {
  p0 = f32[128,128] parameter(0)
  ROOT fusion = f32[128,128] fusion(p0), kind=kLoop, calls=fusion_computation
})";

TEST_F(GpuDefaultBackendPolicyTest, ElementwiseFusionHasNoDefault) {
  EXPECT_EQ(GetDefaultBackend(kElementwiseHlo), std::nullopt);
}

constexpr absl::string_view kSameShapeMultiOutputHlo = R"(
f {
  p0 = f32[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  add = f32[10,10] add(p0, p1)
  sub = f32[10,10] subtract(p0, p1)
  ROOT multi_output = (f32[10,10], f32[10,10]) tuple(add, sub)
}

ENTRY main {
  p0 = f32[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  ROOT fusion = (f32[10,10], f32[10,10]) fusion(p0, p1), kind=kLoop, calls=f
})";

TEST_F(GpuDefaultBackendPolicyTest, MultiOutputFusionNeedsItsFlag) {
  DebugOptions debug_options = GetDebugOptionsForTest();
  EXPECT_EQ(GetDefaultBackend(kSameShapeMultiOutputHlo, &debug_options),
            std::nullopt);
  debug_options.set_xla_gpu_experimental_enable_same_shape_multi_output_fusion(
      true);
  EXPECT_EQ(GetDefaultBackend(kSameShapeMultiOutputHlo, &debug_options),
            autotuner::Backend::BLOCK_LEVEL_EMITTER);
}

}  // namespace
}  // namespace xla::gpu

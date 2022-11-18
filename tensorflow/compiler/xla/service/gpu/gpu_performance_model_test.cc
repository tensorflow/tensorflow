/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_performance_model.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/service/gpu/gpu_device_info_for_tests.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

class GpuPerformanceModelTest : public HloTestBase {
  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const {
    return [&](const Shape& shape) {
      constexpr int64_t kPointerSize = 8;
      return ShapeUtil::ByteSizeOf(shape, kPointerSize);
    };
  }

 public:
  HloCostAnalysis::Options options_{ShapeSizeBytesFunction(),
                                    /*per_second_rates=*/{},
                                    /*count_multiple_input_accesses=*/true};
  GpuHloCostAnalysis analysis_{options_};
  // The reference times in the test cases below are measured
  // on A6000 by profiling the execution of the HLOs.
  GpuDeviceInfo device_info_ = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  GpuPerformanceModelTest() : HloTestBase() {}
};

TEST_F(GpuPerformanceModelTest, LargeWrite) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  c0 = f32[] constant(0)
  ROOT b0 = f32[10000000] broadcast(c0)
}

ENTRY e {
  ROOT r.1 = f32[10000000] fusion(), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  // Dominated by the DRAM bandwidth.
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 57, 10);
}

TEST_F(GpuPerformanceModelTest, SmallReadWrite) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p0 = f32[1000] parameter(0)
  p1 = f32[1000] parameter(1)
  ROOT b0 = f32[1000] add(p0, p1)
}

ENTRY e {
  p0 = f32[1000] parameter(0)
  p1 = f32[1000] parameter(1)
  ROOT r.1 = f32[1000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  // Dominated by the kernel launch overhead.
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 2, 1);
}

TEST_F(GpuPerformanceModelTest, LargeReadWrite) {
  absl::string_view hlo_string = R"(
HloModule m

f {
 p0 = f32[10000000] parameter(0)
 p1 = f32[10000000] parameter(1)
 ROOT a0 = f32[10000000] add(p0, p1)
}

ENTRY e {
 p0 = f32[10000000] parameter(0)
 p1 = f32[10000000] parameter(1)
 ROOT r.1 = f32[10000000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  // Dominated by the DRAM bandwidth.
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 175, 30);
}

TEST_F(GpuPerformanceModelTest, L1CacheEffect) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p0 = f32[10000] parameter(0)
  bc0 = f32[10000,1000] broadcast(p0), dimensions={0}
  b0 = f32[10000000] bitcast(bc0)
  p1 = f32[10000000] parameter(1)
  ROOT a0 = f32[10000000] add(b0, p1)
}

ENTRY e {
  p0 = f32[10000] parameter(0)
  p1 = f32[10000000] parameter(1)
  ROOT r.1 = f32[10000000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  // Parameter 0 read is accelerated by L1 cache even though the total data
  // volume is the same as in the test LargeReadWrite above.
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 118, 12);
}

TEST_F(GpuPerformanceModelTest, L2CacheEffect) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  p0 = f32[1000000] parameter(0)
  bc0 = f32[1000000,10] broadcast(p0), dimensions={0}
  b0 = f32[10000000] bitcast(bc0)
  p1 = f32[10000000] parameter(1)
  ROOT a0 = f32[10000000] add(b0, p1)
}

ENTRY e {
  p0 = f32[1000000] parameter(0)
  p1 = f32[10000000] parameter(1)
  ROOT r.1 = f32[10000000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  // Parameter 0 read is accelerated by L2 cache (does not fit in L1).
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 123, 12);
}

TEST_F(GpuPerformanceModelTest, S32Divide) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  b0 = s32[10000000] parameter(0)
  b1 = s32[10000000] parameter(1)
  d0 = s32[10000000] divide(b0, b1)
  d1 = s32[10000000] divide(d0, b1)
  d2 = s32[10000000] divide(d1, b1)
  d3 = s32[10000000] divide(d2, b1)
  d4 = s32[10000000] divide(d3, b1)
  d5 = s32[10000000] divide(d4, b1)
  d6 = s32[10000000] divide(d5, b1)
  d7 = s32[10000000] divide(d6, b1)
  d8 = s32[10000000] divide(d7, b1)
  d9 = s32[10000000] divide(d8, b1)
  d10 = s32[10000000] divide(d9, b1)
  d11 = s32[10000000] divide(d10, b1)
  d12 = s32[10000000] divide(d11, b1)
  d13 = s32[10000000] divide(d12, b1)
  d14 = s32[10000000] divide(d13, b1)
  d15 = s32[10000000] divide(d14, b1)
  d16 = s32[10000000] divide(d15, b1)
  d17 = s32[10000000] divide(d16, b1)
  d18 = s32[10000000] divide(d17, b1)
  ROOT d19 = s32[10000000] divide(d18, b1)
}

ENTRY e {
  p0 = s32[10000000] parameter(0)
  p1 = s32[10000000] parameter(1)
  ROOT r.1 = s32[10000000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 482, 48);
}

TEST_F(GpuPerformanceModelTest, F32Log) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  b0 = f32[10000000] parameter(0)
  e0 = f32[10000000] log(b0)
  e1 = f32[10000000] log(e0)
  e2 = f32[10000000] log(e1)
  e3 = f32[10000000] log(e2)
  e4 = f32[10000000] log(e3)
  e5 = f32[10000000] log(e4)
  e6 = f32[10000000] log(e5)
  e7 = f32[10000000] log(e6)
  e8 = f32[10000000] log(e7)
  e9 = f32[10000000] log(e8)
  e10 = f32[10000000] log(e9)
  e11 = f32[10000000] log(e10)
  e12 = f32[10000000] log(e11)
  e13 = f32[10000000] log(e12)
  e14 = f32[10000000] log(e13)
  e15 = f32[10000000] log(e14)
  e16 = f32[10000000] log(e15)
  e17 = f32[10000000] log(e16)
  e18 = f32[10000000] log(e17)
  e19 = f32[10000000] log(e18)
  ROOT e20 = f32[10000000] log(e19)
}

ENTRY e {
  p0 = f32[10000000] parameter(0)
  ROOT r.1 = f32[10000000] fusion(p0), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 312, 31);
}

TEST_F(GpuPerformanceModelTest, F64Log) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  b0 = f64[10000000] parameter(0)
  e0 = f64[10000000] log(b0)
  e1 = f64[10000000] log(e0)
  e2 = f64[10000000] log(e1)
  e3 = f64[10000000] log(e2)
  e4 = f64[10000000] log(e3)
  e5 = f64[10000000] log(e4)
  e6 = f64[10000000] log(e5)
  e7 = f64[10000000] log(e6)
  e8 = f64[10000000] log(e7)
  e9 = f64[10000000] log(e8)
  e10 = f64[10000000] log(e9)
  e11 = f64[10000000] log(e10)
  e12 = f64[10000000] log(e11)
  e13 = f64[10000000] log(e12)
  e14 = f64[10000000] log(e13)
  e15 = f64[10000000] log(e14)
  e16 = f64[10000000] log(e15)
  e17 = f64[10000000] log(e16)
  e18 = f64[10000000] log(e17)
  e19 = f64[10000000] log(e18)
  ROOT e20 = f64[10000000] log(e19)
}

ENTRY e {
  p0 = f64[10000000] parameter(0)
  ROOT r.1 = f64[10000000] fusion(p0), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 7100, 700);
}

TEST_F(GpuPerformanceModelTest, F64DivideOnce) {
  absl::string_view hlo_string = R"(
HloModule m

f {
 b0 = f64[10000000] parameter(0)
 b1 = f64[10000000] parameter(1)
 ROOT d0 = f64[10000000] divide(b0, b1)
}

ENTRY e {
 p0 = f64[10000000] parameter(0)
 p1 = f64[10000000] parameter(1)
 ROOT r.1 = f64[10000000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 1100, 110);
}

TEST_F(GpuPerformanceModelTest, F64Exp) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  b0 = f64[10000000] parameter(0)
  e0 = f64[10000000] exponential(b0)
  ROOT r0 = f64[10000000] exponential(e0)
}

ENTRY e {
  p0 = f64[10000000] parameter(0)
  ROOT r.1 = f64[10000000] fusion(p0), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 1400, 140);
}

TEST_F(GpuPerformanceModelTest, F64DivideManyTimes) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  b0 = f64[10000000] parameter(0)
  b1 = f64[10000000] parameter(1)
  d0 = f64[10000000] divide(b0, b1)
  d1 = f64[10000000] divide(d0, b1)
  d2 = f64[10000000] divide(d1, b1)
  d3 = f64[10000000] divide(d2, b1)
  d4 = f64[10000000] divide(d3, b1)
  d5 = f64[10000000] divide(d4, b1)
  d6 = f64[10000000] divide(d5, b1)
  d7 = f64[10000000] divide(d6, b1)
  d8 = f64[10000000] divide(d7, b1)
  d9 = f64[10000000] divide(d8, b1)
  d10 = f64[10000000] divide(d9, b1)
  d11 = f64[10000000] divide(d10, b1)
  d12 = f64[10000000] divide(d11, b1)
  d13 = f64[10000000] divide(d12, b1)
  d14 = f64[10000000] divide(d13, b1)
  d15 = f64[10000000] divide(d14, b1)
  d16 = f64[10000000] divide(d15, b1)
  d17 = f64[10000000] divide(d16, b1)
  d18 = f64[10000000] divide(d17, b1)
  ROOT d19 = f64[10000000] divide(d18, b1)
}

ENTRY e {
  p0 = f64[10000000] parameter(0)
  p1 = f64[10000000] parameter(1)
  ROOT r.1 = f64[10000000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 20000, 2000);
}

TEST_F(GpuPerformanceModelTest, F64Multiply) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  b0 = f64[10000000] parameter(0)
  b1 = f64[10000000] parameter(1)
  d0 = f64[10000000] multiply(b0, b1)
  d1 = f64[10000000] multiply(d0, b1)
  d2 = f64[10000000] multiply(d1, b1)
  d3 = f64[10000000] multiply(d2, b1)
  d4 = f64[10000000] multiply(d3, b1)
  d5 = f64[10000000] multiply(d4, b1)
  d6 = f64[10000000] multiply(d5, b1)
  d7 = f64[10000000] multiply(d6, b1)
  d8 = f64[10000000] multiply(d7, b1)
  d9 = f64[10000000] multiply(d8, b1)
  d10 = f64[10000000] multiply(d9, b1)
  d11 = f64[10000000] multiply(d10, b1)
  d12 = f64[10000000] multiply(d11, b1)
  d13 = f64[10000000] multiply(d12, b1)
  d14 = f64[10000000] multiply(d13, b1)
  d15 = f64[10000000] multiply(d14, b1)
  d16 = f64[10000000] multiply(d15, b1)
  d17 = f64[10000000] multiply(d16, b1)
  d18 = f64[10000000] multiply(d17, b1)
  ROOT d19 = f64[10000000] multiply(d18, b1)
}

ENTRY e {
  p0 = f64[10000000] parameter(0)
  p1 = f64[10000000] parameter(1)
  ROOT r.1 = f64[10000000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 794, 80);
}

TEST_F(GpuPerformanceModelTest, C128Multiply) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  b0 = c128[10000000] parameter(0)
  b1 = c128[10000000] parameter(1)
  d0 = c128[10000000] multiply(b0, b1)
  d1 = c128[10000000] multiply(d0, b1)
  d2 = c128[10000000] multiply(d1, b1)
  d3 = c128[10000000] multiply(d2, b1)
  d4 = c128[10000000] multiply(d3, b1)
  d5 = c128[10000000] multiply(d4, b1)
  d6 = c128[10000000] multiply(d5, b1)
  d7 = c128[10000000] multiply(d6, b1)
  d8 = c128[10000000] multiply(d7, b1)
  d9 = c128[10000000] multiply(d8, b1)
  d10 = c128[10000000] multiply(d9, b1)
  d11 = c128[10000000] multiply(d10, b1)
  d12 = c128[10000000] multiply(d11, b1)
  d13 = c128[10000000] multiply(d12, b1)
  d14 = c128[10000000] multiply(d13, b1)
  d15 = c128[10000000] multiply(d14, b1)
  d16 = c128[10000000] multiply(d15, b1)
  d17 = c128[10000000] multiply(d16, b1)
  d18 = c128[10000000] multiply(d17, b1)
  ROOT d19 = c128[10000000] multiply(d18, b1)
}

ENTRY e {
  p0 = c128[10000000] parameter(0)
  p1 = c128[10000000] parameter(1)
  ROOT r.1 = c128[10000000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 4700, 470);
}

TEST_F(GpuPerformanceModelTest, C128Power) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  b0 = c128[10000000] parameter(0)
  b1 = c128[10000000] parameter(1)
  d0 = c128[10000000] power(b0, b1)
  d1 = c128[10000000] power(d0, b1)
  d2 = c128[10000000] power(d1, b1)
  d3 = c128[10000000] power(d2, b1)
  d4 = c128[10000000] power(d3, b1)
  d5 = c128[10000000] power(d4, b1)
  d6 = c128[10000000] power(d5, b1)
  d7 = c128[10000000] power(d6, b1)
  d8 = c128[10000000] power(d7, b1)
  ROOT d9 = c128[10000000] power(d8, b1)
}

ENTRY e {
  p0 = c128[10000000] parameter(0)
  p1 = c128[10000000] parameter(1)
  ROOT r.1 = c128[10000000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 93000, 9300);
}

TEST_F(GpuPerformanceModelTest, F64Power) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  b0 = f64[10000000] parameter(0)
  b1 = f64[10000000] parameter(1)
  d0 = f64[10000000] power(b0, b1)
  d1 = f64[10000000] power(d0, b1)
  d2 = f64[10000000] power(d1, b1)
  d3 = f64[10000000] power(d2, b1)
  d4 = f64[10000000] power(d3, b1)
  d5 = f64[10000000] power(d4, b1)
  d6 = f64[10000000] power(d5, b1)
  d7 = f64[10000000] power(d6, b1)
  d8 = f64[10000000] power(d7, b1)
  ROOT d9 = f64[10000000] power(d8, b1)
}

ENTRY e {
  p0 = f64[10000000] parameter(0)
  p1 = f64[10000000] parameter(1)
  ROOT r.1 = f64[10000000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 36000, 3600);
}

TEST_F(GpuPerformanceModelTest, F64Tanh) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  b0 = f64[10000000] parameter(0)
  e0 = f64[10000000] tanh(b0)
  e1 = f64[10000000] tanh(e0)
  e2 = f64[10000000] tanh(e1)
  e3 = f64[10000000] tanh(e2)
  e4 = f64[10000000] tanh(e3)
  e5 = f64[10000000] tanh(e4)
  e6 = f64[10000000] tanh(e5)
  e7 = f64[10000000] tanh(e6)
  e8 = f64[10000000] tanh(e7)
  e9 = f64[10000000] tanh(e8)
  e10 = f64[10000000] tanh(e9)
  e11 = f64[10000000] tanh(e10)
  e12 = f64[10000000] tanh(e11)
  e13 = f64[10000000] tanh(e12)
  e14 = f64[10000000] tanh(e13)
  e15 = f64[10000000] tanh(e14)
  e16 = f64[10000000] tanh(e15)
  e17 = f64[10000000] tanh(e16)
  e18 = f64[10000000] tanh(e17)
  e19 = f64[10000000] tanh(e18)
  ROOT e20 = f64[10000000] tanh(e19)
}

ENTRY e {
  p0 = f64[10000000] parameter(0)
  ROOT r.1 = f64[10000000] fusion(p0), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 14000, 1400);
}

TEST_F(GpuPerformanceModelTest, F32Tanh) {
  absl::string_view hlo_string = R"(
HloModule m
 
f {
 b0 = f32[10000000] parameter(0)
 e0 = f32[10000000] tanh(b0)
 e1 = f32[10000000] tanh(e0)
 e2 = f32[10000000] tanh(e1)
 e3 = f32[10000000] tanh(e2)
 e4 = f32[10000000] tanh(e3)
 e5 = f32[10000000] tanh(e4)
 e6 = f32[10000000] tanh(e5)
 e7 = f32[10000000] tanh(e6)
 e8 = f32[10000000] tanh(e7)
 e9 = f32[10000000] tanh(e8)
 e10 = f32[10000000] tanh(e9)
 e11 = f32[10000000] tanh(e10)
 e12 = f32[10000000] tanh(e11)
 e13 = f32[10000000] tanh(e12)
 e14 = f32[10000000] tanh(e13)
 e15 = f32[10000000] tanh(e14)
 e16 = f32[10000000] tanh(e15)
 e17 = f32[10000000] tanh(e16)
 e18 = f32[10000000] tanh(e17)
 e19 = f32[10000000] tanh(e18)
 ROOT e20 = f32[10000000] tanh(e19)
}

ENTRY e {
 p0 = f32[10000000] parameter(0)
 ROOT r.1 = f32[10000000] fusion(p0), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 200, 20);
}

TEST_F(GpuPerformanceModelTest, F64Sqrt) {
  absl::string_view hlo_string = R"(
HloModule m
f {
  b0 = f64[10000000] parameter(0)
  e0 = f64[10000000] sqrt(b0)
  e1 = f64[10000000] sqrt(e0)
  e2 = f64[10000000] sqrt(e1)
  e3 = f64[10000000] sqrt(e2)
  e4 = f64[10000000] sqrt(e3)
  e5 = f64[10000000] sqrt(e4)
  e6 = f64[10000000] sqrt(e5)
  e7 = f64[10000000] sqrt(e6)
  e8 = f64[10000000] sqrt(e7)
  e9 = f64[10000000] sqrt(e8)
  e10 = f64[10000000] sqrt(e9)
  e11 = f64[10000000] sqrt(e10)
  e12 = f64[10000000] sqrt(e11)
  e13 = f64[10000000] sqrt(e12)
  e14 = f64[10000000] sqrt(e13)
  e15 = f64[10000000] sqrt(e14)
  e16 = f64[10000000] sqrt(e15)
  e17 = f64[10000000] sqrt(e16)
  e18 = f64[10000000] sqrt(e17)
  e19 = f64[10000000] sqrt(e18)
  ROOT e20 = f64[10000000] sqrt(e19)
}
ENTRY e {
  p0 = f64[10000000] parameter(0)
  ROOT r.1 = f64[10000000] fusion(p0), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 7800, 780);
}

TEST_F(GpuPerformanceModelTest, C128Sqrt) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  b0 = c128[10000000] parameter(0)
  e0 = c128[10000000] sqrt(b0)
  e1 = c128[10000000] sqrt(e0)
  e2 = c128[10000000] sqrt(e1)
  e3 = c128[10000000] sqrt(e2)
  e4 = c128[10000000] sqrt(e3)
  e5 = c128[10000000] sqrt(e4)
  e6 = c128[10000000] sqrt(e5)
  e7 = c128[10000000] sqrt(e6)
  e8 = c128[10000000] sqrt(e7)
  ROOTe9 = c128[10000000] sqrt(e8)
}

ENTRY e {
  p0 = c128[10000000] parameter(0)
  ROOT r.1 = c128[10000000] fusion(p0), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 83000, 8000);
}

TEST_F(GpuPerformanceModelTest, F64Rsqrt) {
  absl::string_view hlo_string = R"(
HloModule m

f {
 b0 = f64[10000000] parameter(0)
 e0 = f64[10000000] rsqrt(b0)
 e1 = f64[10000000] rsqrt(e0)
 e2 = f64[10000000] rsqrt(e1)
 e3 = f64[10000000] rsqrt(e2)
 e4 = f64[10000000] rsqrt(e3)
 e5 = f64[10000000] rsqrt(e4)
 e6 = f64[10000000] rsqrt(e5)
 e7 = f64[10000000] rsqrt(e6)
 e8 = f64[10000000] rsqrt(e7)
 e9 = f64[10000000] rsqrt(e8)
 e10 = f64[10000000] rsqrt(e9)
 e11 = f64[10000000] rsqrt(e10)
 e12 = f64[10000000] rsqrt(e11)
 e13 = f64[10000000] rsqrt(e12)
 e14 = f64[10000000] rsqrt(e13)
 e15 = f64[10000000] rsqrt(e14)
 e16 = f64[10000000] rsqrt(e15)
 e17 = f64[10000000] rsqrt(e16)
 e18 = f64[10000000] rsqrt(e17)
 e19 = f64[10000000] rsqrt(e18)
 ROOT e20 = f64[10000000] rsqrt(e19)
}

ENTRY e {
 p0 = f64[10000000] parameter(0)
 ROOT r.1 = f64[10000000] fusion(p0), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 6300, 630);
}

TEST_F(GpuPerformanceModelTest, C128Divide) {
  absl::string_view hlo_string = R"(
HloModule m

f {
  b0 = c128[10000000] parameter(0)
  b1 = c128[10000000] parameter(1)
  d0 = c128[10000000] divide(b0, b1)
  d1 = c128[10000000] divide(d0, b1)
  d2 = c128[10000000] divide(d1, b1)
  d3 = c128[10000000] divide(d2, b1)
  d4 = c128[10000000] divide(d3, b1)
  d5 = c128[10000000] divide(d4, b1)
  d6 = c128[10000000] divide(d5, b1)
  d7 = c128[10000000] divide(d6, b1)
  d8 = c128[10000000] divide(d7, b1)
  ROOT d9 = c128[10000000] divide(d8, b1)
}

ENTRY e {
  p0 = c128[10000000] parameter(0)
  p1 = c128[10000000] parameter(1)
  ROOT r.1 = c128[10000000] fusion(p0, p1), kind=kLoop, calls=f
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(root->Accept(&analysis_));

  HloInstruction* instruction = root;
  GpuPerformanceModel::RunTimes t = GpuPerformanceModel::EstimateRunTimes(
      instruction, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 64000, 6400);
}

TEST_F(GpuPerformanceModelTest, UnusedParameter) {
  Shape shape = ShapeUtil::MakeShape(F32, {100000});

  auto module = std::make_unique<HloModule>("m", HloModuleConfig{});
  HloComputation::Builder b("b");
  auto p0 = b.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  auto p1 = b.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));

  HloComputation::Builder sub_builder("subcomp");
  HloInstruction* p0f = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0f"));
  // p1f is not used.
  HloInstruction* p1f = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "p1f"));
  ASSERT_NE(p1f, nullptr);
  sub_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0f));

  HloComputation* subcomp = module->AddEmbeddedComputation(sub_builder.Build());
  auto fusion = HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kLoop, {p0, p1}, subcomp);
  b.AddInstruction(std::move(fusion));
  module->AddEntryComputation(b.Build());

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_IS_OK(module->entry_computation()->Accept(&analysis_));

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimes(root, &analysis_, device_info_);
  EXPECT_NEAR(absl::ToInt64Microseconds(t.time_unfused), 2, 1);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

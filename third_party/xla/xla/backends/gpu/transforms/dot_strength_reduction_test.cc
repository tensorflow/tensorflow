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

#include "xla/backends/gpu/transforms/dot_strength_reduction.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {
namespace {

using DotStrengthReductionTest = HloHardwareIndependentTestBase;

TEST_F(DotStrengthReductionTest, MultipleContractingDimensions) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  p0 = f32[4,256]{1,0} parameter(0)
  p1 = f32[4096,4,256]{2,1,0} parameter(1)
  ROOT dot = f32[4096]{0} dot(p0, p1), lhs_contracting_dims={1,0}, rhs_contracting_dims={2,1}
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Hopper())};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: f32[256,4]{0,1} transpose
// CHECK: f32[4096,256,4]{1,0,2} broadcast
// CHECK: f32[4096,256,4]{1,2,0} transpose
// CHECK: f32[4096,256,4]{1,0,2} multiply
// CHECK: f32[4096]{0} reduce
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                       RunFileCheck(module->ToString(), filecheck_pattern));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(DotStrengthReductionTest, UpcastInReduction) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  p0 = bf16[6144]{0} parameter(0)
  p1 = bf16[6144,256]{1,0} parameter(1)
  ROOT dot = bf16[256]{0} dot(p0, p1), lhs_contracting_dims={0}, rhs_contracting_dims={0}
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Hopper())};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: bf16[256,6144]{1,0} multiply
// CHECK: f32[256,6144]{1,0} convert
// CHECK: f32[] constant(0)
// CHECK: f32[256]{0} reduce
// CHECK: bf16[256]{0} convert
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                       RunFileCheck(module->ToString(), filecheck_pattern));
  EXPECT_TRUE(filecheck_result);
  CHECK_OK(module->Verify());
}

TEST_F(DotStrengthReductionTest, MaintainsMetadata) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  p0 = bf16[6144]{0} parameter(0)
  p1 = bf16[6144,256]{1,0} parameter(1)
  ROOT dot = bf16[256]{0} dot(p0, p1), lhs_contracting_dims={0}, rhs_contracting_dims={0}, metadata={op_name="test"}
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Hopper())};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: bf16[256,6144]{1,0} multiply{{.*}}, metadata={op_name="test"}
// CHECK: f32[256,6144]{1,0} convert{{.*}}, metadata={op_name="test"}
// CHECK: f32[256]{0} reduce{{.*}}, metadata={op_name="test"}
// CHECK: bf16[256]{0} convert{{.*}}, metadata={op_name="test"}
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                       RunFileCheck(module->ToString(), filecheck_pattern));
  EXPECT_TRUE(filecheck_result);
  CHECK_OK(module->Verify());
}

TEST_F(DotStrengthReductionTest, UpcastInReductionF8E4M3FN) {
  const char* hlo_text = R"(
HloModule test

ENTRY test {
  %x = f8e4m3fn[8,16]{1,0} parameter(0)
  %y = f8e4m3fn[8,16,32]{2,1,0} parameter(1)
  ROOT %dot = f8e4m3fn[8,32]{1,0} dot(%x, %y), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Hopper())};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: f8e4m3fn[8,32,16]{2,1,0} multiply
// CHECK: f32[8,32,16]{2,1,0} convert
// CHECK: f32[] constant(0)
// CHECK: f32[8,32]{1,0} reduce
// CHECK: f8e4m3fn[8,32]{1,0} convert
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                       RunFileCheck(module->ToString(), filecheck_pattern));
  EXPECT_TRUE(filecheck_result);
  CHECK_OK(module->Verify());
}

TEST_F(DotStrengthReductionTest, VectorVectorDotShouldBeStrengthReduced) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = f32[32, 500] parameter(0)
  p1 = f32[32, 500] parameter(1)
  ROOT dot = f32[32] dot(p0, p1), lhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: f32[32,500]{1,0} multiply
// CHECK: f32[32]{0} reduce
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                       RunFileCheck(module->ToString(), filecheck_pattern));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(DotStrengthReductionTest, MatrixVectorDotShouldNotBeStrengthReduced) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = f32[32, 5000, 7000] parameter(0)
  p1 = f32[32, 5000] parameter(1)
  ROOT dot = f32[32,7000] dot(p0, p1), lhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1},
    algorithm=dot_bf16_bf16_f32_x6
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotStrengthReductionTest,
       DotWithTypeUnsupportedByGemmFusionShouldBeStrengthReduced) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = c64[32, 5000, 7000] parameter(0)
  p1 = c64[32, 5000] parameter(1)
  ROOT dot = c64[32,7000] dot(p0, p1), lhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: c64[32,7000,5000]{{[^ ]*}} multiply
// CHECK: c64[32,7000]{{[^ ]*}} reduce
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                       RunFileCheck(module->ToString(), filecheck_pattern));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(DotStrengthReductionTest, SmallDotShouldBeStrengthReduced) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = f32[32, 50, 70] parameter(0)
  p1 = f32[32, 50] parameter(1)
  ROOT dot = f32[32,70] dot(p0, p1), lhs_batch_dims={0},
    lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1},
    algorithm=dot_bf16_bf16_f32_x6
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: bf16{{[^ ]*}} multiply
// CHECK: bf16{{[^ ]*}} multiply
// CHECK: bf16{{[^ ]*}} multiply
// CHECK: bf16{{[^ ]*}} multiply
// CHECK: bf16{{[^ ]*}} multiply
// CHECK: bf16{{[^ ]*}} multiply
// CHECK: f32[32,70]{{[^ ]*}} reduce
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                       RunFileCheck(module->ToString(), filecheck_pattern));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(DotStrengthReductionTest, SmallDotShouldBeStrengthReduced2) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = f32[2000, 3000] parameter(0)
  p1 = f32[2000] parameter(1)
  ROOT dot = f32[3000] dot(p0, p1), lhs_contracting_dims={0},
    rhs_contracting_dims={0}, algorithm=dot_bf16_bf16_f32_x6
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: bf16{{[^ ]*}} multiply
// CHECK: bf16{{[^ ]*}} multiply
// CHECK: bf16{{[^ ]*}} multiply
// CHECK: bf16{{[^ ]*}} multiply
// CHECK: bf16{{[^ ]*}} multiply
// CHECK: bf16{{[^ ]*}} multiply
// CHECK: f32[3000]{0} reduce
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                       RunFileCheck(module->ToString(), filecheck_pattern));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(DotStrengthReductionTest,
       DotStrengthReductionWith_F32_F32_F32_Algorithm) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
      a = f32[128,2]{1,0} parameter(0)
      b = f32[2]{0} parameter(1)
      ROOT dot = f32[128]{0} dot(a, b),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0},
        algorithm=dot_f32_f32_f32
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, m.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(m->Verify());

  const char* filecheck_pattern = R"(
// CHECK: f32[128,2]{{[^ ]*}} multiply
// CHECK: f32[128]{0} reduce
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                       RunFileCheck(m->ToString(), filecheck_pattern));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(DotStrengthReductionTest, DotStrengthReductionMixedOperandTypes) {
  constexpr char kModuleStr[] = R"(
    HloModule test
    ENTRY dot {
      a = s32[128,2]{1,0} parameter(0)
      b = s64[2]{0} parameter(1)
      ROOT dot = s32[128]{0} dot(a, b),
        lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, m.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(m->Verify());

  const char* filecheck_pattern = R"(
// CHECK: s32[2]{0} convert
// CHECK: s32[128,2]{{[^ ]*}} multiply
// CHECK: s32[128]{0} reduce
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                       RunFileCheck(m->ToString(), filecheck_pattern));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(DotStrengthReductionTest, S32MatrixMatrixDotShouldBeStrengthReduced) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = s32[32, 50] parameter(0)
  p1 = s32[70, 50] parameter(1)
  ROOT dot = s32[32, 70] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: s32[32,70,50]{{[^ ]*}} multiply
// CHECK: s32[32,70]{{[^ ]*}} reduce
)";

  ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                       RunFileCheck(module->ToString(), filecheck_pattern));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(DotStrengthReductionTest, F32MatrixMatrixDotShouldNotBeStrengthReduced) {
  const std::string& hlo_string = R"(
HloModule m

ENTRY entry {
  p0 = f32[32, 500] parameter(0)
  p1 = f32[700, 500] parameter(1)
  ROOT dot = f32[32, 700] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

struct OneHotDotTestCase {
  std::string test_name;
  absl::string_view hlo;
  bool expected_changed;
  absl::string_view filecheck_pattern = "";
};

class OneHotDotStrengthReductionTest
    : public DotStrengthReductionTest,
      public ::testing::WithParamInterface<OneHotDotTestCase> {};

TEST_P(OneHotDotStrengthReductionTest, PatternMatchAndRewrite) {
  const OneHotDotTestCase& tc = GetParam();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(tc.hlo));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Hopper()),
      /*enable_one_hot_dot_to_gather=*/true};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_EQ(changed, tc.expected_changed);
  CHECK_OK(module->Verify());

  if (!tc.filecheck_pattern.empty()) {
    ASSERT_OK_AND_ASSIGN(
        bool filecheck_result,
        RunFileCheck(module->ToString(), tc.filecheck_pattern));
    EXPECT_TRUE(filecheck_result);
  }
}

INSTANTIATE_TEST_SUITE_P(
    OneHotDotTests, OneHotDotStrengthReductionTest,
    ::testing::Values(
        OneHotDotTestCase{"LhsOneHotToGather",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s32[128] parameter(0)
                bcast_indices = s32[128,256] broadcast(indices), dimensions={0}
                iota = s32[128,256] iota(), iota_dimension=1
                cmp = pred[128,256] compare(bcast_indices, iota), direction=EQ
                lhs = bf16[128,256] convert(cmp)
                rhs = bf16[256,512] parameter(1)
                ROOT dot = bf16[128,512] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
              }
            )",
                          /*expected_changed=*/true,
                          R"(
              // CHECK-DAG: [[GE:%[^ ]+]] = pred[128]{0} compare(%indices, {{.*}}), direction=GE
              // CHECK-DAG: [[LE:%[^ ]+]] = pred[128]{0} compare(%indices, {{.*}}), direction=LE
              // CHECK-DAG: [[IN_BOUNDS:%[^ ]+]] = pred[128]{0} and([[GE]], [[LE]])
              // CHECK-DAG: [[IN_BOUNDS_BCAST:%[^ ]+]] = pred[128,512]{1,0} broadcast([[IN_BOUNDS]]), dimensions={0}
              // CHECK-DAG: [[START_INDICES:%[^ ]+]] = s32[128,1]{1,0} reshape(%indices)
              // CHECK-DAG: [[GATHER:%[^ ]+]] = bf16[128,512]{1,0} gather(%rhs, [[START_INDICES]]), offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,512}
              // CHECK: ROOT {{.*}} = bf16[128,512]{1,0} select([[IN_BOUNDS_BCAST]], [[GATHER]], {{.*}})
            )"},
        OneHotDotTestCase{"SunkBroadcastOnRhs",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s32[8,16] parameter(0)
                bcast_indices = s32[8,16,256] broadcast(indices), dimensions={0,1}
                iota = s32[8,16,256] iota(), iota_dimension=2
                cmp = pred[8,16,256] compare(bcast_indices, iota), direction=EQ
                lhs_reshape = pred[128,256] reshape(cmp)
                lhs = bf16[128,256] convert(lhs_reshape)
                param = bf16[256,16,512] parameter(1)
                rhs_bcast = bf16[1,256,1,16,8,512] broadcast(param), dimensions={1,3,5}
                rhs = bf16[256,65536] reshape(rhs_bcast)
                ROOT dot = bf16[128,65536] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
              }
            )",
                          /*expected_changed=*/true,
                          R"(
              // CHECK-DAG: [[RESHAPED_INDICES:%[^ ]+]] = s32[128]{0} reshape(%indices)
              // CHECK-DAG: [[START_INDICES:%[^ ]+]] = s32[128,1]{1,0} reshape([[RESHAPED_INDICES]])
              // CHECK-DAG: [[GATHER:%[^ ]+]] = bf16[128,16,512]{2,1,0} gather(%param, [[START_INDICES]]), offset_dims={1,2}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,16,512}
              // CHECK: [[SELECT:%[^ ]+]] = bf16[128,16,512]{2,1,0} select({{.*}}, [[GATHER]], {{.*}})
              // CHECK: [[SUNK_BCAST:%[^ ]+]] = bf16[128,1,16,8,512]{4,3,2,1,0} broadcast([[SELECT]]), dimensions={0,2,4}
              // CHECK: ROOT {{.*}} = bf16[128,65536]{1,0} reshape([[SUNK_BCAST]])
            )"},
        OneHotDotTestCase{"RhsOneHotToGather",
                          R"(
              HloModule m
              ENTRY entry {
                lhs = bf16[512,256] parameter(0)
                indices = s32[128] parameter(1)
                bcast_indices = s32[128,256] broadcast(indices), dimensions={0}
                iota = s32[128,256] iota(), iota_dimension=1
                cmp = pred[128,256] compare(bcast_indices, iota), direction=EQ
                rhs = bf16[128,256] convert(cmp)
                ROOT dot = bf16[512,128] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={1}
              }
            )",
                          /*expected_changed=*/true,
                          R"(
              // CHECK-DAG: [[GE:%[^ ]+]] = pred[128]{0} compare(%indices, {{.*}}), direction=GE
              // CHECK-DAG: [[LE:%[^ ]+]] = pred[128]{0} compare(%indices, {{.*}}), direction=LE
              // CHECK-DAG: [[IN_BOUNDS:%[^ ]+]] = pred[128]{0} and([[GE]], [[LE]])
              // CHECK-DAG: [[IN_BOUNDS_BCAST:%[^ ]+]] = pred[512,128]{1,0} broadcast([[IN_BOUNDS]]), dimensions={1}
              // CHECK-DAG: [[START_INDICES:%[^ ]+]] = s32[128,1]{1,0} reshape(%indices)
              // CHECK-DAG: [[GATHER:%[^ ]+]] = bf16[512,128]{1,0} gather(%lhs, [[START_INDICES]]), offset_dims={0}, collapsed_slice_dims={1}, start_index_map={1}, index_vector_dim=1, slice_sizes={512,1}
              // CHECK: ROOT {{.*}} = bf16[512,128]{1,0} select([[IN_BOUNDS_BCAST]], [[GATHER]], {{.*}})
            )"},
        OneHotDotTestCase{"SunkBroadcastOnLhs",
                          R"(
              HloModule m
              ENTRY entry {
                param = bf16[256,16,512] parameter(0)
                lhs_bcast = bf16[1,256,1,16,8,512] broadcast(param), dimensions={1,3,5}
                lhs = bf16[256,65536] reshape(lhs_bcast)
                indices = s32[8,16] parameter(1)
                bcast_indices = s32[8,16,256] broadcast(indices), dimensions={0,1}
                iota = s32[8,16,256] iota(), iota_dimension=2
                cmp = pred[8,16,256] compare(bcast_indices, iota), direction=EQ
                rhs_reshape = pred[128,256] reshape(cmp)
                rhs = bf16[128,256] convert(rhs_reshape)
                ROOT dot = bf16[65536,128] dot(lhs, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={1}
              }
            )",
                          /*expected_changed=*/true,
                          R"(
              // CHECK-DAG: [[RESHAPED_INDICES:%[^ ]+]] = s32[128]{0} reshape(%indices)
              // CHECK-DAG: [[START_INDICES:%[^ ]+]] = s32[128,1]{1,0} reshape([[RESHAPED_INDICES]])
              // CHECK-DAG: [[GATHER:%[^ ]+]] = bf16[16,512,128]{2,1,0} gather(%param, [[START_INDICES]]), offset_dims={0,1}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,16,512}
              // CHECK: [[SELECT:%[^ ]+]] = bf16[16,512,128]{2,1,0} select({{.*}}, [[GATHER]], {{.*}})
              // CHECK: [[SUNK_BCAST:%[^ ]+]] = bf16[1,16,8,512,128]{4,3,2,1,0} broadcast([[SELECT]]), dimensions={1,3,4}
              // CHECK: ROOT {{.*}} = bf16[65536,128]{1,0} reshape([[SUNK_BCAST]])
            )"},
        OneHotDotTestCase{"Rank1WeightsSkipsMaskBroadcast",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s32[128] parameter(0)
                bcast_indices = s32[128,256] broadcast(indices), dimensions={0}
                iota = s32[128,256] iota(), iota_dimension=1
                cmp = pred[128,256] compare(bcast_indices, iota), direction=EQ
                lhs = bf16[128,256] convert(cmp)
                rhs = bf16[256] parameter(1)
                ROOT dot = bf16[128] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
              }
            )",
                          /*expected_changed=*/true,
                          R"(
              // CHECK-DAG: [[GE:%[^ ]+]] = pred[128]{0} compare(%indices, {{.*}}), direction=GE
              // CHECK-DAG: [[LE:%[^ ]+]] = pred[128]{0} compare(%indices, {{.*}}), direction=LE
              // CHECK-DAG: [[IN_BOUNDS:%[^ ]+]] = pred[128]{0} and([[GE]], [[LE]])
              // CHECK-DAG: [[GATHER:%[^ ]+]] = bf16[128]{0} gather(%rhs, {{.*}}), offset_dims={}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1}
              // CHECK: ROOT {{.*}} = bf16[128]{0} select([[IN_BOUNDS]], [[GATHER]], {{.*}})
            )"},
        OneHotDotTestCase{"MixedTypeConvertsAfterGather",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s32[128] parameter(0)
                bcast_indices = s32[128,256] broadcast(indices), dimensions={0}
                iota = s32[128,256] iota(), iota_dimension=1
                cmp = pred[128,256] compare(bcast_indices, iota), direction=EQ
                lhs = bf16[128,256] convert(cmp)
                rhs = bf16[256,512] parameter(1)
                ROOT dot = f32[128,512] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
              }
            )",
                          /*expected_changed=*/true,
                          R"(
              // CHECK-DAG: [[START_INDICES:%[^ ]+]] = s32[128,1]{1,0} reshape(%indices)
              // CHECK-DAG: [[GATHER:%[^ ]+]] = bf16[128,512]{1,0} gather(%rhs, [[START_INDICES]])
              // CHECK-DAG: [[CONVERT:%[^ ]+]] = f32[128,512]{1,0} convert([[GATHER]])
              // CHECK: ROOT {{.*}} = f32[128,512]{1,0} select({{.*}}, [[CONVERT]], {{.*}})
            )"},
        OneHotDotTestCase{"U8With256ClassesDoesNotOverflowUpperBound",
                          R"(
              HloModule m
              ENTRY entry {
                indices = u8[128] parameter(0)
                bcast_indices = u8[128,256] broadcast(indices), dimensions={0}
                iota = u8[128,256] iota(), iota_dimension=1
                cmp = pred[128,256] compare(bcast_indices, iota), direction=EQ
                lhs = bf16[128,256] convert(cmp)
                rhs = bf16[256,512] parameter(1)
                ROOT dot = bf16[128,512] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
              }
            )",
                          /*expected_changed=*/true,
                          R"(
              // CHECK-DAG: [[MAX_CLASS:%[^ ]+]] = u8[] constant(255)
              // CHECK-DAG: [[MAX_BCAST:%[^ ]+]] = u8[128]{0} broadcast([[MAX_CLASS]]), dimensions={}
              // CHECK-DAG: [[IN_BOUNDS:%[^ ]+]] = pred[128]{0} compare(%indices, [[MAX_BCAST]]), direction=LE
              // CHECK-DAG: [[IN_BOUNDS_BCAST:%[^ ]+]] = pred[128,512]{1,0} broadcast([[IN_BOUNDS]]), dimensions={0}
              // CHECK: ROOT {{.*}} = bf16[128,512]{1,0} select([[IN_BOUNDS_BCAST]],
            )"},
        OneHotDotTestCase{"ReversedCompareAndBroadcasted1DIota",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s32[128] parameter(0)
                bcast_indices = s32[128,256] broadcast(indices), dimensions={0}
                iota_1d = s32[256] iota(), iota_dimension=0
                iota = s32[128,256] broadcast(iota_1d), dimensions={1}
                cmp = pred[128,256] compare(iota, bcast_indices), direction=EQ
                lhs = bf16[128,256] convert(cmp)
                rhs = bf16[256,512] parameter(1)
                ROOT dot = bf16[128,512] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
              }
            )",
                          /*expected_changed=*/true,
                          R"(
              // CHECK-DAG: [[START_INDICES:%[^ ]+]] = s32[128,1]{1,0} reshape(%indices)
              // CHECK-DAG: [[GATHER:%[^ ]+]] = bf16[128,512]{1,0} gather(%rhs, [[START_INDICES]])
              // CHECK: ROOT {{.*}} = bf16[128,512]{1,0} select({{.*}}, [[GATHER]], {{.*}})
            )"},
        OneHotDotTestCase{"F32WithHighestPrecision",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s32[128] parameter(0)
                bcast_indices = s32[128,256] broadcast(indices), dimensions={0}
                iota = s32[128,256] iota(), iota_dimension=1
                cmp = pred[128,256] compare(bcast_indices, iota), direction=EQ
                lhs = f32[128,256] convert(cmp)
                rhs = f32[256,512] parameter(1)
                ROOT dot = f32[128,512] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,highest}
              }
            )",
                          /*expected_changed=*/true,
                          R"(
              // CHECK: [[GATHER:%[^ ]+]] = f32[128,512]{1,0} gather(%rhs, {{.*}})
              // CHECK: ROOT {{.*}} = f32[128,512]{1,0} select({{.*}}, [[GATHER]], {{.*}})
            )"},
        OneHotDotTestCase{"F32WithF32Algorithm",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s32[128] parameter(0)
                bcast_indices = s32[128,256] broadcast(indices), dimensions={0}
                iota = s32[128,256] iota(), iota_dimension=1
                cmp = pred[128,256] compare(bcast_indices, iota), direction=EQ
                lhs = f32[128,256] convert(cmp)
                rhs = f32[256,512] parameter(1)
                ROOT dot = f32[128,512] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_f32_f32_f32
              }
            )",
                          /*expected_changed=*/true,
                          R"(
              // CHECK: [[GATHER:%[^ ]+]] = f32[128,512]{1,0} gather(%rhs, {{.*}})
              // CHECK: ROOT {{.*}} = f32[128,512]{1,0} select({{.*}}, [[GATHER]], {{.*}})
            )"},
        OneHotDotTestCase{"NonTrivialLeadingBroadcastDimFallsBackToFullWeights",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s32[128] parameter(0)
                bcast_indices = s32[128,256] broadcast(indices), dimensions={0}
                iota = s32[128,256] iota(), iota_dimension=1
                cmp = pred[128,256] compare(bcast_indices, iota), direction=EQ
                lhs = bf16[128,256] convert(cmp)
                param = bf16[256,64] parameter(1)
                rhs_bcast = bf16[8,256,64] broadcast(param), dimensions={1,2}
                rhs = bf16[256,512] reshape(rhs_bcast)
                ROOT dot = bf16[128,512] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
              }
            )",
                          /*expected_changed=*/true,
                          R"(
              // CHECK: [[GATHER:%[^ ]+]] = bf16[128,512]{1,0} gather(%rhs, {{.*}})
              // CHECK: ROOT {{.*}} = bf16[128,512]{1,0} select({{.*}}, [[GATHER]], {{.*}})
            )"},
        OneHotDotTestCase{"Negative_F32DefaultPrecision",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s32[128] parameter(0)
                bcast_indices = s32[128,256] broadcast(indices), dimensions={0}
                iota = s32[128,256] iota(), iota_dimension=1
                cmp = pred[128,256] compare(bcast_indices, iota), direction=EQ
                lhs = f32[128,256] convert(cmp)
                rhs = f32[256,512] parameter(1)
                ROOT dot = f32[128,512] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
              }
            )",
                          /*expected_changed=*/false},
        OneHotDotTestCase{"Negative_F32AsymmetricPrecision",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s32[128] parameter(0)
                bcast_indices = s32[128,256] broadcast(indices), dimensions={0}
                iota = s32[128,256] iota(), iota_dimension=1
                cmp = pred[128,256] compare(bcast_indices, iota), direction=EQ
                lhs = f32[128,256] convert(cmp)
                rhs = f32[256,512] parameter(1)
                ROOT dot = f32[128,512] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,default}
              }
            )",
                          /*expected_changed=*/false},
        OneHotDotTestCase{"Negative_F32NonF32AlgorithmWithHighestPrecision",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s32[128] parameter(0)
                bcast_indices = s32[128,256] broadcast(indices), dimensions={0}
                iota = s32[128,256] iota(), iota_dimension=1
                cmp = pred[128,256] compare(bcast_indices, iota), direction=EQ
                lhs = f32[128,256] convert(cmp)
                rhs = f32[256,512] parameter(1)
                ROOT dot = f32[128,512] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}, operand_precision={highest,highest}, algorithm=dot_bf16_bf16_f32
              }
            )",
                          /*expected_changed=*/false},
        OneHotDotTestCase{"Negative_S8With256Classes",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s8[128] parameter(0)
                bcast_indices = s8[128,256] broadcast(indices), dimensions={0}
                iota = s8[128,256] iota(), iota_dimension=1
                cmp = pred[128,256] compare(bcast_indices, iota), direction=EQ
                lhs = bf16[128,256] convert(cmp)
                rhs = bf16[256,512] parameter(1)
                ROOT dot = bf16[128,512] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
              }
            )",
                          /*expected_changed=*/false},
        OneHotDotTestCase{"Negative_BatchDimsPresent",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s32[4,128] parameter(0)
                bcast_indices = s32[4,128,256] broadcast(indices), dimensions={0,1}
                iota = s32[4,128,256] iota(), iota_dimension=2
                cmp = pred[4,128,256] compare(bcast_indices, iota), direction=EQ
                lhs = bf16[4,128,256] convert(cmp)
                rhs = bf16[4,256,512] parameter(1)
                ROOT dot = bf16[4,128,512] dot(lhs, rhs), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
              }
            )",
                          /*expected_changed=*/false},
        OneHotDotTestCase{"Negative_NonLastContractingDim",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s32[256] parameter(0)
                bcast_indices = s32[256,128] broadcast(indices), dimensions={0}
                iota = s32[256,128] iota(), iota_dimension=1
                cmp = pred[256,128] compare(bcast_indices, iota), direction=EQ
                lhs = bf16[256,128] convert(cmp)
                rhs = bf16[256,512] parameter(1)
                ROOT dot = bf16[128,512] dot(lhs, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
              }
            )",
                          /*expected_changed=*/false},
        OneHotDotTestCase{"Negative_MergedContractingReshape",
                          R"(
              HloModule m
              ENTRY entry {
                indices = s32[128,2] parameter(0)
                bcast_indices = s32[128,2,128] broadcast(indices), dimensions={0,1}
                iota = s32[128,2,128] iota(), iota_dimension=2
                cmp = pred[128,2,128] compare(bcast_indices, iota), direction=EQ
                lhs_reshape = pred[128,256] reshape(cmp)
                lhs = bf16[128,256] convert(lhs_reshape)
                rhs = bf16[256,512] parameter(1)
                ROOT dot = bf16[128,512] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
              }
            )",
                          /*expected_changed=*/false}),
    [](const ::testing::TestParamInfo<OneHotDotTestCase>& info) {
      return info.param.test_name;
    });

struct OneHotExecutionTestCase {
  std::string test_name;
  absl::string_view hlo;
};

class OneHotDotExecutionTest
    : public DotStrengthReductionTest,
      public ::testing::WithParamInterface<OneHotExecutionTestCase> {};

TEST_P(OneHotDotExecutionTest, EvaluatesIdenticallyWithOutOfBoundsIndices) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(GetParam().hlo));
  Literal indices_lit = LiteralUtil::CreateR2<int32_t>({{-5, 0, 2}, {3, 4, 9}});
  Literal weights_lit = LiteralUtil::CreateR2<float>(
      {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}});

  ASSERT_OK_AND_ASSIGN(
      Literal expected,
      HloEvaluator().Evaluate(*module, {&indices_lit, &weights_lit}));

  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Hopper()),
      /*enable_one_hot_dot_to_gather=*/true};
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  ASSERT_OK_AND_ASSIGN(
      Literal actual,
      HloEvaluator().Evaluate(*module, {&indices_lit, &weights_lit}));
  EXPECT_EQ(expected, actual);
}

INSTANTIATE_TEST_SUITE_P(
    OneHotDotExecutionTests, OneHotDotExecutionTest,
    ::testing::Values(OneHotExecutionTestCase{"LhsOneHot",
                                              R"(
              HloModule m
              ENTRY entry {
                indices = s32[2,3] parameter(0)
                bcast_indices = s32[2,3,4] broadcast(indices), dimensions={0,1}
                iota = s32[2,3,4] iota(), iota_dimension=2
                cmp = pred[2,3,4] compare(bcast_indices, iota), direction=EQ
                lhs = f32[2,3,4] convert(cmp)
                rhs = f32[4,2] parameter(1)
                ROOT dot = f32[2,3,2] dot(lhs, rhs), lhs_contracting_dims={2}, rhs_contracting_dims={0}, algorithm=dot_f32_f32_f32
              }
            )"},
                      OneHotExecutionTestCase{"SunkBroadcastRhs",
                                              R"(
              HloModule m
              ENTRY entry {
                indices = s32[2,3] parameter(0)
                bcast_indices = s32[2,3,4] broadcast(indices), dimensions={0,1}
                iota = s32[2,3,4] iota(), iota_dimension=2
                cmp = pred[2,3,4] compare(bcast_indices, iota), direction=EQ
                lhs_reshape = pred[6,4] reshape(cmp)
                lhs = f32[6,4] convert(lhs_reshape)
                param = f32[4,2] parameter(1)
                rhs_bcast = f32[1,4,3,2] broadcast(param), dimensions={1,3}
                rhs = f32[4,6] reshape(rhs_bcast)
                ROOT dot = f32[6,6] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}, algorithm=dot_f32_f32_f32
              }
            )"},
                      OneHotExecutionTestCase{"SunkBroadcastLhs",
                                              R"(
              HloModule m
              ENTRY entry {
                indices = s32[2,3] parameter(0)
                param = f32[4,2] parameter(1)
                lhs_bcast = f32[1,4,3,2] broadcast(param), dimensions={1,3}
                lhs = f32[4,6] reshape(lhs_bcast)
                bcast_indices = s32[2,3,4] broadcast(indices), dimensions={0,1}
                iota = s32[2,3,4] iota(), iota_dimension=2
                cmp = pred[2,3,4] compare(bcast_indices, iota), direction=EQ
                rhs_reshape = pred[6,4] reshape(cmp)
                rhs = f32[6,4] convert(rhs_reshape)
                ROOT dot = f32[6,6] dot(lhs, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={1}, algorithm=dot_f32_f32_f32
              }
            )"},
                      OneHotExecutionTestCase{"RhsOneHot",
                                              R"(
              HloModule m
              ENTRY entry {
                indices = s32[2,3] parameter(0)
                param = f32[4,2] parameter(1)
                lhs = f32[2,4] transpose(param), dimensions={1,0}
                bcast_indices = s32[2,3,4] broadcast(indices), dimensions={0,1}
                iota = s32[2,3,4] iota(), iota_dimension=2
                cmp = pred[2,3,4] compare(bcast_indices, iota), direction=EQ
                rhs = f32[2,3,4] convert(cmp)
                ROOT dot = f32[2,2,3] dot(lhs, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={2}, algorithm=dot_f32_f32_f32
              }
            )"}),
    [](const ::testing::TestParamInfo<OneHotExecutionTestCase>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace gpu
}  // namespace xla

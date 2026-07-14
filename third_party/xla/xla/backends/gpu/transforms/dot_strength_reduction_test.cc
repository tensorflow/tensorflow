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

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Hopper())};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: f32[256,4]{0,1} transpose
// CHECK: f32[4096,256,4]{1,0,2} broadcast
// CHECK: f32[4096,256,4]{1,2,0} transpose
// CHECK: f32[4096,256,4]{1,0,2} multiply
// CHECK: f32[4096]{0} reduce
)";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Hopper())};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: bf16[256,6144]{1,0} multiply
// CHECK: f32[256,6144]{1,0} convert
// CHECK: f32[] constant(0)
// CHECK: f32[256]{0} reduce
// CHECK: bf16[256]{0} convert
)";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Hopper())};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: bf16[256,6144]{1,0} multiply{{.*}}, metadata={op_name="test"}
// CHECK: f32[256,6144]{1,0} convert{{.*}}, metadata={op_name="test"}
// CHECK: f32[256]{0} reduce{{.*}}, metadata={op_name="test"}
// CHECK: bf16[256]{0} convert{{.*}}, metadata={op_name="test"}
)";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Hopper())};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: f8e4m3fn[8,32,16]{2,1,0} multiply
// CHECK: f32[8,32,16]{2,1,0} convert
// CHECK: f32[] constant(0)
// CHECK: f32[8,32]{1,0} reduce
// CHECK: f8e4m3fn[8,32]{1,0} convert
)";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: f32[32,500]{1,0} multiply
// CHECK: f32[32]{0} reduce
)";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: c64[32,7000,5000]{{[^ ]*}} multiply
// CHECK: c64[32,7000]{{[^ ]*}} reduce
)";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
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

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
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

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
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
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, m.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(m->Verify());

  const char* filecheck_pattern = R"(
// CHECK: f32[128,2]{{[^ ]*}} multiply
// CHECK: f32[128]{0} reduce
)";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
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
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, m.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(m->Verify());

  const char* filecheck_pattern = R"(
// CHECK: s32[2]{0} convert
// CHECK: s32[128,2]{{[^ ]*}} multiply
// CHECK: s32[128]{0} reduce
)";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  CHECK_OK(module->Verify());

  const char* filecheck_pattern = R"(
// CHECK: s32[32,70,50]{{[^ ]*}} multiply
// CHECK: s32[32,70]{{[^ ]*}} reduce
)";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  DotStrengthReduction pass{
      se::GpuComputeCapability(se::CudaComputeCapability::Ampere())};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

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

#include "xla/service/gpu/nvptx_alias_info.h"

#include <memory>
#include <optional>

#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

void ExpectOptionalTrue(std::optional<bool> value) {
  EXPECT_TRUE(value.has_value());
  CHECK(value.has_value());
  EXPECT_TRUE(*value);
}

void ExpectOptionalFalse(std::optional<bool> value) {
  EXPECT_TRUE(value.has_value());
  CHECK(value.has_value());
  EXPECT_FALSE(*value);
}

class NVPTXAliasInfoTest : public HloHardwareIndependentTestBase {
 public:
  std::optional<bool> MayAlias(const HloInstruction* user,
                               const HloInstruction* operand,
                               const ShapeIndex& user_index) {
    return alias_info_.MayAlias(operand, {}, user, user_index);
  }

 private:
  const se::DeviceDescription device_description_{
      xla::gpu::TestGpuDeviceInfo::H100SXMDeviceInfo()};
  NVPTXAliasInfo alias_info_{device_description_};
};

TEST_F(NVPTXAliasInfoTest, BufferCanBeSharedForBiasMatmul) {
  const char* const kModuleString = R"(
HloModule m

ENTRY main {
  lhs = f32[20,20]{1,0} parameter(0)
  rhs = f32[20,30]{1,0} parameter(1)
  bias = f32[20,30]{1,0} parameter(2)
  ROOT cublas-lt-matmul = (f32[20,30]{1,0}, s8[33554432]{0}) custom-call(lhs, rhs, bias), custom_call_target="__cublas$lt$matmul", frontend_attributes={grad_x="false",grad_y="false"}, backend_config={"gemm_backend_config":{"selected_algorithm":"0","alpha_real":1,"beta":1,"dot_dimension_numbers":{"lhs_contracting_dimensions":["0"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["HIGHEST","HIGHEST"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","lhs_stride":"400","rhs_stride":"600","grad_x":false,"grad_y":false,"damax_output":false},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* matmul = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(matmul, matmul->operand(0), {0}));
  ExpectOptionalFalse(MayAlias(matmul, matmul->operand(1), {0}));
  ExpectOptionalTrue(MayAlias(matmul, matmul->operand(2), {0}));
}

TEST_F(NVPTXAliasInfoTest, MatmulWorkspaceBufferCannotBeSharedForBiasMatmul) {
  const char* const kModuleString = R"(
HloModule m

ENTRY main {
  lhs = f32[20,20]{1,0} parameter(0)
  rhs = f32[20,30]{1,0} parameter(1)
  bias = f32[20,30]{1,0} parameter(2)
  ROOT cublas-lt-matmul = (f32[20,30]{1,0}, s8[33554432]{0}) custom-call(lhs,
  rhs, bias), custom_call_target="__cublas$lt$matmul",
  frontend_attributes={grad_x="false",grad_y="false"},
  backend_config={"gemm_backend_config":{"selected_algorithm":"0","alpha_real":1,"beta":1,"dot_dimension_numbers":{"lhs_contracting_dimensions":["0"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["HIGHEST","HIGHEST"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","lhs_stride":"400","rhs_stride":"600","grad_x":false,"grad_y":false,"damax_output":false},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* matmul = module->entry_computation()->root_instruction();
  ExpectOptionalTrue(MayAlias(matmul, matmul->operand(2), {0}));
  // Workspace buffer cannot be shared.
  ExpectOptionalFalse(MayAlias(matmul, matmul->operand(2), {1}));
}

TEST_F(NVPTXAliasInfoTest, DuplicateOperandBufferCannotBeSharedForBiasMatmul) {
  const char* const kModuleString = R"(
HloModule m

ENTRY main {
  lhs = f32[20,20]{1,0} parameter(0)
  rhs = f32[20,30]{1,0} parameter(1)
  ROOT cublas-lt-matmul = (f32[20,30]{1,0}, s8[33554432]{0}) custom-call(lhs, rhs, rhs), custom_call_target="__cublas$lt$matmul", frontend_attributes={grad_x="false",grad_y="false"}, backend_config={"gemm_backend_config":{"selected_algorithm":"0","alpha_real":1,"beta":1,"dot_dimension_numbers":{"lhs_contracting_dimensions":["0"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["HIGHEST","HIGHEST"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","lhs_stride":"400","rhs_stride":"600","grad_x":false,"grad_y":false,"damax_output":false},"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleString));
  HloInstruction* matmul = module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(matmul, matmul->operand(0), {0}));
  ExpectOptionalFalse(MayAlias(matmul, matmul->operand(1), {0}));
  ExpectOptionalFalse(MayAlias(matmul, matmul->operand(2), {0}));
}

TEST_F(NVPTXAliasInfoTest, VariadicAllReduceSharesOnlyUniqueMatchingOperands) {
  const char* const kDistinctModuleString = R"(
HloModule m

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = f32[8]{0} parameter(0)
  p1 = f32[8]{0} parameter(1)
  ROOT ar = (f32[8]{0}, f32[8]{0}) all-reduce(p0, p1), replica_groups={},
      to_apply=add
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> distinct_module,
                       ParseAndReturnVerifiedModule(kDistinctModuleString));
  HloInstruction* ar = distinct_module->entry_computation()->root_instruction();
  ExpectOptionalTrue(MayAlias(ar, ar->operand(0), {0}));
  ExpectOptionalTrue(MayAlias(ar, ar->operand(1), {1}));
  ExpectOptionalFalse(MayAlias(ar, ar->operand(0), {1}));
  ExpectOptionalFalse(MayAlias(ar, ar->operand(1), {0}));

  const char* const kDuplicateModuleString = R"(
HloModule m

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

ENTRY main {
  p0 = f32[8]{0} parameter(0)
  ROOT ar = (f32[8]{0}, f32[8]{0}) all-reduce(p0, p0), replica_groups={},
      to_apply=add
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> duplicate_module,
                       ParseAndReturnVerifiedModule(kDuplicateModuleString));
  HloInstruction* dup_ar =
      duplicate_module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(dup_ar, dup_ar->operand(0), {0}));
  ExpectOptionalFalse(MayAlias(dup_ar, dup_ar->operand(0), {1}));
}

TEST_F(NVPTXAliasInfoTest,
       CollectiveBroadcastCannotShareDuplicateVariadicOperand) {
  const char* const kSingleModuleString = R"(
HloModule m, replica_count=2

ENTRY main {
  p0 = f32[8]{0} parameter(0)
  ROOT cb = f32[8]{0} collective-broadcast(p0), replica_groups={{0,1}}
}
)";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> single_module,
      ParseAndReturnVerifiedModule(
          kSingleModuleString, GetModuleConfigForTest(/*replica_count=*/2)));
  HloInstruction* single_cb =
      single_module->entry_computation()->root_instruction();
  ExpectOptionalTrue(MayAlias(single_cb, single_cb->operand(0), {}));

  const char* const kDuplicateModuleString = R"(
HloModule m, replica_count=2

ENTRY main {
  p0 = f32[8]{0} parameter(0)
  ROOT cb = (f32[8]{0}, f32[8]{0}) collective-broadcast(p0, p0),
      replica_groups={{0,1}}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> dup_module,
                       ParseAndReturnUnverifiedModule(kDuplicateModuleString));
  HloInstruction* dup_cb = dup_module->entry_computation()->root_instruction();
  ExpectOptionalFalse(MayAlias(dup_cb, dup_cb->operand(0), {0}));
  ExpectOptionalFalse(MayAlias(dup_cb, dup_cb->operand(0), {1}));
}

}  // namespace
}  // namespace xla::gpu

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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/tsl/lib/gtl/int_type.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::stream_executor::gpu::AllReduceStrategy;
using ::testing::AllOf;
using ::testing::Field;
using ::testing::HasSubstr;

TSL_LIB_GTL_DEFINE_INT_TYPE(CollectiveKernelEnabled, bool);
TSL_LIB_GTL_DEFINE_INT_TYPE(MultimemEnabled, bool);

class BuildAllReduceInfoTest : public HloHardwareIndependentTestBase {
 protected:
  // Helper to reduce boilerplate while keeping tests independent.
  absl::StatusOr<AllReduceInfo> BuildInfo(
      CollectiveKernelEnabled collective_kernel_enabled,
      MultimemEnabled multimem_enabled, PrimitiveType element_type,
      std::vector<int64_t> shape, HloOpcode hlo_opcode,
      std::vector<int32_t> replica_groups) {
    constexpr absl::string_view kModuleStr = R"(
    HloModule test
     apply_op {
       x = %1$s[] parameter(0)
       y = %1$s[] parameter(1)
       ROOT apply_op = %1$s[] %3$s(x, y)
     }
     ENTRY test_computation {
       param_0 = %1$s[%2$s] parameter(0)
       ROOT all-reduce = %1$s[%2$s] all-reduce(param_0), to_apply=apply_op,
           replica_groups={%4$s}
     }
    )";
    se::DeviceDescription device_info = TestGpuDeviceInfo::H100SXMDeviceInfo();
    std::string replica_groups_str =
        replica_groups.empty()
            ? ""
            : absl::StrFormat("{%s}", absl::StrJoin(replica_groups, ","));
    const std::string module_str = absl::StrFormat(
        kModuleStr, primitive_util::LowercasePrimitiveTypeName(element_type),
        absl::StrJoin(shape, ","), HloOpcodeString(hlo_opcode),
        replica_groups_str);

    SCOPED_TRACE(testing::Message() << "module_str: " << module_str);

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        ParseAndReturnVerifiedModule(
            module_str, replica_groups.empty() ? 1 : replica_groups.size()));
    const HloInstruction* hlo_instr =
        HloHardwareIndependentTestBase::FindInstruction(module.get(),
                                                        HloOpcode::kAllReduce);
    return BuildAllReduceInfo(collective_kernel_enabled.value(),
                              multimem_enabled.value(), device_info,
                              Cast<HloAllReduceInstruction>(hlo_instr));
  }
};

TEST_F(BuildAllReduceInfoTest, ReturnsOneShotStrategyForSmallS32) {
  EXPECT_THAT(BuildInfo(CollectiveKernelEnabled(true), MultimemEnabled(false),
                        S32, {1024, 8}, HloOpcode::kMaximum, {0, 1}),
              IsOkAndHolds(AllOf(
                  Field(&AllReduceInfo::reduction_kind, ReductionKind::MAX),
                  Field(&AllReduceInfo::all_reduce_strategy,
                        AllReduceStrategy::kOneShot))));
}

TEST_F(BuildAllReduceInfoTest, ReturnsTwoShotStrategyForLargerF32) {
  EXPECT_THAT(BuildInfo(CollectiveKernelEnabled(true), MultimemEnabled(false),
                        F32, {128, 1024}, HloOpcode::kAdd, {0, 1}),
              IsOkAndHolds(AllOf(
                  Field(&AllReduceInfo::reduction_kind, ReductionKind::SUM),
                  Field(&AllReduceInfo::all_reduce_strategy,
                        AllReduceStrategy::kTwoShot))));
}

TEST_F(BuildAllReduceInfoTest, ReturnsMultimemStrategy) {
  EXPECT_THAT(BuildInfo(CollectiveKernelEnabled(true), MultimemEnabled(true),
                        F32, {1024}, HloOpcode::kAdd, {0, 1}),
              IsOkAndHolds(AllOf(
                  Field(&AllReduceInfo::reduction_kind, ReductionKind::SUM),
                  Field(&AllReduceInfo::all_reduce_strategy,
                        AllReduceStrategy::kMultimem))));
}

TEST_F(BuildAllReduceInfoTest, FailsIfCollectiveKernelDisabled) {
  EXPECT_THAT(
      BuildInfo(CollectiveKernelEnabled(false), MultimemEnabled(false), F32,
                {1024}, HloOpcode::kAdd, {0, 1}),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("not enabled")));
}

TEST_F(BuildAllReduceInfoTest, FailsForNonPowerOfTwoDevices) {
  EXPECT_THAT(BuildInfo(CollectiveKernelEnabled(true), MultimemEnabled(false),
                        F32, {1024}, HloOpcode::kAdd, {0, 1, 2}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("only supported for power of 2")));
}

TEST_F(BuildAllReduceInfoTest, FailsForTooManyDevices) {
  EXPECT_THAT(BuildInfo(CollectiveKernelEnabled(true), MultimemEnabled(false),
                        F32, {1024}, HloOpcode::kAdd,
                        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("does not support more than 8 ranks")));
}

TEST_F(BuildAllReduceInfoTest, FailsForUnsupportedTypeCombination) {
  EXPECT_THAT(BuildInfo(CollectiveKernelEnabled(true), MultimemEnabled(false),
                        PRED, {1024}, HloOpcode::kAdd, {0, 1}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("combination is not supported")));
}

TEST_F(BuildAllReduceInfoTest, FailsForLargeInputs) {
  EXPECT_THAT(BuildInfo(CollectiveKernelEnabled(true), MultimemEnabled(false),
                        F32, {2, 1024, 1024}, HloOpcode::kAdd, {0, 1}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("only supported for small inputs")));
}

TEST_F(BuildAllReduceInfoTest, FailsIfReplicaGroupsEmpty) {
  EXPECT_THAT(
      BuildInfo(CollectiveKernelEnabled(true), MultimemEnabled(false), F32,
                {1024}, HloOpcode::kAdd, {}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Replica groups must be explicitly provided")));
}

}  // namespace
}  // namespace xla::gpu

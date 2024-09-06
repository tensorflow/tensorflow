/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/host_offload_utils.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace host_offload_utils {
namespace {

class HostOffloadUtilsTest : public HloTestBase {};

TEST_F(HostOffloadUtilsTest, SimpleGetSuccessorsGetPredecessorsTest) {
  const std::string& hlo_string = R"(
HloModule my_module
ENTRY main {
  data_param = f32[1,2048,2048] parameter(0)
  index_param = s32[] parameter(1)
  constant_f32_0 = f32[] constant(0)
  constant_s32_0 = s32[] constant(0)
  broadcast = f32[2,2048,2048] broadcast(constant_f32_0), dimensions={}
  offload_custom_call = f32[1,2048,2048] custom-call(data_param), custom_call_target="MoveToHost"
  dynamic_update_slice = f32[2,2048,2048] dynamic-update-slice(broadcast, offload_custom_call, index_param, constant_s32_0, constant_s32_0)
  dynamic_slice = f32[1,2048,2048] dynamic-slice(dynamic_update_slice, index_param, constant_s32_0, constant_s32_0), dynamic_slice_sizes={1,2048,2048}
  ROOT load_custom_call = f32[1,2048,2048] custom-call(dynamic_slice), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* data_param = FindInstruction(module.get(), "data_param");
  ASSERT_NE(data_param, nullptr);
  HloInstruction* offload_custom_call =
      FindInstruction(module.get(), "offload_custom_call");
  ASSERT_NE(offload_custom_call, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<InstructionAndShapeIndex> succ,
      GetSuccessors(InstructionAndShapeIndex(data_param, {})));
  std::vector<InstructionAndShapeIndex> expected_succ = {
      InstructionAndShapeIndex(offload_custom_call, {})};
  EXPECT_EQ(succ, expected_succ);

  std::vector<InstructionAndShapeIndex> pred =
      GetPredecessors(InstructionAndShapeIndex(offload_custom_call, {}));
  std::vector<InstructionAndShapeIndex> expected_pred = {
      InstructionAndShapeIndex(data_param, {})};
  EXPECT_EQ(pred, expected_pred);
}

TEST_F(HostOffloadUtilsTest, ComputationGetSuccessorsGetPredecessorsTest) {
  const std::string& hlo_string = R"(
HloModule my_module
other_computation {
  param_0 = f32[2048] parameter(0)
  param_1 = f32[2048] parameter(1)
  ROOT tuple = (f32[2048], f32[2048]) tuple(param_0, param_1)
}
ENTRY main {
  data_param = f32[2048] parameter(0)
  other_param = f32[2048] parameter(1)
  offload_custom_call = f32[2048] custom-call(data_param), custom_call_target="MoveToHost"
  call = (f32[2048], f32[2048]) call(offload_custom_call, other_param), to_apply=other_computation
  gte_0 = f32[2048] get-tuple-element(call), index=0
  gte_1 = f32[2048] get-tuple-element(call), index=1
  ROOT load_custom_call = f32[2048] custom-call(gte_0), custom_call_target="MoveToDevice"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* call = FindInstruction(module.get(), "call");
  ASSERT_NE(call, nullptr);
  HloInstruction* gte_0 = FindInstruction(module.get(), "gte_0");
  ASSERT_NE(gte_0, nullptr);
  HloInstruction* tuple = FindInstruction(module.get(), "tuple");
  ASSERT_NE(tuple, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<InstructionAndShapeIndex> succ,
                          GetSuccessors(InstructionAndShapeIndex(call, {0})));
  std::vector<InstructionAndShapeIndex> expected_succ = {
      InstructionAndShapeIndex(gte_0, {})};
  EXPECT_EQ(succ, expected_succ);

  std::vector<InstructionAndShapeIndex> pred =
      GetPredecessors(InstructionAndShapeIndex(call, {0}));
  std::vector<InstructionAndShapeIndex> expected_pred = {
      InstructionAndShapeIndex(tuple, {0})};
  EXPECT_EQ(pred, expected_pred);
}

}  // namespace
}  // namespace host_offload_utils
}  // namespace xla

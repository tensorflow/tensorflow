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

#include "xla/hlo/transforms/simplifiers/call_parameter_cleanup.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

class CallParameterCleanupTest : public HloHardwareIndependentTestBase {
 protected:
  CallParameterCleanupTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/false,
            /*allow_mixed_precision_in_hlo_verifier=*/true) {}
};

namespace {

namespace m = ::xla::match;

TEST_F(CallParameterCleanupTest, DeadParameter) {
  const std::string module_str = R"hlo(
HloModule module

add {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  c = s32[] parameter(2)
  add = s32[] add(a, c)
  ROOT tuple = (s32[]) tuple(add)
}

ENTRY entry {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  ROOT call = (s32[]) call(p0, p1, p2), to_apply=add
}

)hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  CallParameterCleanup cleanup;
  EXPECT_TRUE(cleanup.Run(module.get()).value());

  HloDCE dce;
  TupleSimplifier tuple_simplifier;
  CHECK_OK(dce.Run(module.get()).status());
  CHECK_OK(tuple_simplifier.Run(module.get()).status());

  // We expect the parameter at index 1 to be removed.
  HloInstruction* call;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Call(&call, m::Parameter(0), m::Parameter(2))));
  EXPECT_THAT(call->to_apply()->root_instruction(),
              GmockMatch(m::Tuple(m::Add(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(CallParameterCleanupTest, PassThroughParameter) {
  const std::string module_str = R"hlo(
HloModule module

add {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  c = s32[] parameter(2)
  add = s32[] add(a, c)
  ROOT tuple = (s32[], s32[]) tuple(b, add)
}

ENTRY entry {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  ROOT call = (s32[], s32[]) call(p0, p1, p2), to_apply=add
}

)hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  CallParameterCleanup cleanup;
  EXPECT_TRUE(cleanup.Run(module.get()).value());

  HloDCE dce;
  TupleSimplifier tuple_simplifier;
  CHECK_OK(dce.Run(module.get()).status());
  CHECK_OK(tuple_simplifier.Run(module.get()).status());

  // We expect the parameter at index 1 to be passed through directly from the
  // entry computation parameter, and removed from the call parameters.
  HloInstruction* call;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::Parameter(1),
                  m::GetTupleElement(
                      m::Call(&call, m::Parameter(0), m::Parameter(2)), 0))));
  EXPECT_THAT(call->to_apply()->root_instruction(),
              GmockMatch(m::Tuple(m::Add(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(CallParameterCleanupTest, UsedPassThroughParameter) {
  const std::string module_str = R"hlo(
HloModule module

add {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  c = s32[] parameter(2)
  add = s32[] add(a, c)
  ROOT tuple = (s32[], s32[]) tuple(c, add)
}

ENTRY entry {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  ROOT call = (s32[], s32[]) call(p0, p1, p2), to_apply=add
}

)hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  CallParameterCleanup cleanup;
  EXPECT_TRUE(cleanup.Run(module.get()).value());

  HloDCE dce;
  TupleSimplifier tuple_simplifier;
  CHECK_OK(dce.Run(module.get()).status());
  CHECK_OK(tuple_simplifier.Run(module.get()).status());

  // We expect the parameter at index 2 to be passed through directly from the
  // entry computation parameter, but not removed from the call parameters.
  // Parameter 1 gets removed.
  HloInstruction* call;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::Parameter(2),
                  m::GetTupleElement(
                      m::Call(&call, m::Parameter(0), m::Parameter(2)), 0))));
  EXPECT_THAT(call->to_apply()->root_instruction(),
              GmockMatch(m::Tuple(m::Add(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(CallParameterCleanupTest, DeadPassThroughParameterMultipleUses) {
  const std::string module_str = R"hlo(
HloModule module

add {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  c = s32[] parameter(2)
  add = s32[] add(a, c)
  ROOT tuple = (s32[], s32[], s32[]) tuple(b, add, b)
}

ENTRY entry {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  ROOT call = (s32[], s32[], s32[]) call(p0, p1, p2), to_apply=add
}

)hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  CallParameterCleanup cleanup;
  EXPECT_TRUE(cleanup.Run(module.get()).value());

  HloDCE dce;
  TupleSimplifier tuple_simplifier;
  CHECK_OK(dce.Run(module.get()).status());
  CHECK_OK(tuple_simplifier.Run(module.get()).status());

  HloInstruction* call;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::Parameter(1),
                  m::GetTupleElement(
                      m::Call(&call, m::Parameter(0), m::Parameter(2)), 0),
                  m::Parameter(1))));
  EXPECT_THAT(call->to_apply()->root_instruction(),
              GmockMatch(m::Tuple(m::Add(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(CallParameterCleanupTest, UsedPassThroughParameterNoDeadParams) {
  const std::string module_str = R"hlo(
HloModule module

add {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  add = s32[] add(a, b)
  ROOT tuple = (s32[], s32[]) tuple(b, add)
}

ENTRY entry {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  ROOT call = (s32[], s32[]) call(p0, p1), to_apply=add
}

)hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  CallParameterCleanup cleanup;
  EXPECT_TRUE(cleanup.Run(module.get()).value());

  HloDCE dce;
  TupleSimplifier tuple_simplifier;
  CHECK_OK(dce.Run(module.get()).status());
  CHECK_OK(tuple_simplifier.Run(module.get()).status());

  // We expect the parameter at index 1 to be passed through directly from the
  // entry computation parameter, but not removed from the call parameters.
  HloInstruction* call;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::Parameter(1),
                  m::GetTupleElement(
                      m::Call(&call, m::Parameter(0), m::Parameter(1)), 0))));
  EXPECT_THAT(call->to_apply()->root_instruction(),
              GmockMatch(m::Tuple(m::Add(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(CallParameterCleanupTest, MultipleCallSites) {
  const std::string module_str = R"hlo(
HloModule module

add {
  a = s32[] parameter(0)
  b = s32[] parameter(1)
  c = s32[] parameter(2)
  add = s32[] add(a, c)
  ROOT tuple = (s32[], s32[]) tuple(c, add)
}

wrap {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  ROOT call = (s32[], s32[]) call(p1, p2, p0), to_apply=add
}

ENTRY entry {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  call0 = (s32[], s32[]) call(p0, p1, p2), to_apply=add
  call1 = (s32[], s32[]) call(p0, p1, p2), to_apply=wrap
  gte0 = s32[] get-tuple-element(call0), index=1
  gte1 = s32[] get-tuple-element(call1), index=0
  ROOT mul = s32[] multiply(gte0, gte1)
}

)hlo";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  CallParameterCleanup cleanup;
  EXPECT_TRUE(cleanup.Run(module.get()).value());

  HloDCE dce;
  TupleSimplifier tuple_simplifier;
  CHECK_OK(dce.Run(module.get()).status());
  CHECK_OK(tuple_simplifier.Run(module.get()).status());

  // We expect both call sites to use the same computation, for 3 computations
  // total, rather than 4.
  EXPECT_EQ(module->computation_count(), 3);
}

}  // namespace

}  // namespace xla

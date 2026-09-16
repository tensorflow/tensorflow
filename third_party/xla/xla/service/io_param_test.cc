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

#include "xla/service/io_param.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_value.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::UnorderedElementsAre;

class IOParamTest : public HloHardwareIndependentTestBase {};

TEST_F(IOParamTest, GetNonTrivialUses_AllUsersAreNonTrivial) {
  absl::string_view hlo_string = R"hlo(
HloModule m

ENTRY entry {
  a = f32[10,10]{0,1:T(8,128)} parameter(0)
  b = f32[10,10]{0,1:T(8,128)} add(a, a)
  ROOT c = f32[10,10]{0,1:T(8,128)} add(b, b)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* a = FindInstruction(module.get(), "a");
  HloInstruction* b = FindInstruction(module.get(), "b");

  IOParam param_a(HloPosition{a, {}});
  EXPECT_THAT(param_a.GetNonTrivialUses(),
              IsOkAndHolds(UnorderedElementsAre(IOParam(HloUse(b, 0, {})),
                                                IOParam(HloUse(b, 1, {})))));
}

TEST_F(IOParamTest, GetNonTrivialUses_GetTupleElementUser) {
  absl::string_view hlo_string = R"hlo(
HloModule m

ENTRY entry {
  a = (f32[10,10]{0,1:T(8,128)}, f32[10,10]{0,1:T(8,128)}) parameter(0)
  b = f32[10,10]{0,1:T(8,128)} get-tuple-element(a), index=0
  c = f32[10,10]{0,1:T(8,128)} get-tuple-element(a), index=1
  ROOT d = f32[10,10]{0,1:T(8,128)} add(b, c)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* a = FindInstruction(module.get(), "a");
  HloInstruction* d = FindInstruction(module.get(), "d");

  IOParam param_a_0(HloPosition{a, {0}});
  EXPECT_THAT(param_a_0.GetNonTrivialUses(),
              IsOkAndHolds(UnorderedElementsAre(IOParam(HloUse(d, 0, {})))));

  IOParam param_a_1(HloPosition{a, {1}});
  EXPECT_THAT(param_a_1.GetNonTrivialUses(),
              IsOkAndHolds(UnorderedElementsAre(IOParam(HloUse(d, 1, {})))));
}

TEST_F(IOParamTest, GetNonTrivialUses_TupleUser) {
  absl::string_view hlo_string = R"hlo(
HloModule m

ENTRY entry {
  a = f32[10,10]{0,1:T(8,128)} parameter(0)
  b = f32[10,10]{0,1:T(8,128)} add(a, a)
  c = (f32[10,10]{0,1:T(8,128)}, f32[10,10]{0,1:T(8,128)}) tuple(a, b)
  ROOT d = (f32[10,10]{0,1:T(8,128)}, f32[10,10]{0,1:T(8,128)}) custom-call(c), custom_call_target="cc"
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* a = FindInstruction(module.get(), "a");
  HloInstruction* b = FindInstruction(module.get(), "b");
  HloInstruction* d = FindInstruction(module.get(), "d");

  IOParam param_a(HloPosition{a, {}});
  EXPECT_THAT(param_a.GetNonTrivialUses(),
              IsOkAndHolds(UnorderedElementsAre(IOParam(HloUse(b, 0, {})),
                                                IOParam(HloUse(b, 1, {})),
                                                IOParam(HloUse(d, 0, {0})))));
}

TEST_F(IOParamTest, GetNonTrivialUses_BitcastUser) {
  absl::string_view hlo_string = R"hlo(
HloModule m

ENTRY entry {
  a = f32[10,10]{0,1:T(8,128)} parameter(0)
  b = f32[10,10]{0,1:T(8,128)} bitcast(a)
  ROOT c = f32[10,10]{0,1:T(8,128)} add(b, b)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* a = FindInstruction(module.get(), "a");
  HloInstruction* c = FindInstruction(module.get(), "c");

  IOParam param_a(HloPosition{a, {}});
  EXPECT_THAT(param_a.GetNonTrivialUses(),
              IsOkAndHolds(UnorderedElementsAre(IOParam(HloUse(c, 0, {})),
                                                IOParam(HloUse(c, 1, {})))));
}

TEST_F(IOParamTest, GetNonTrivialUses_ComplexUsers) {
  absl::string_view hlo_string = R"hlo(
HloModule m

ENTRY entry {
  a = f32[10,10]{0,1:T(8,128)} parameter(0)
  b = f32[10,10]{0,1:T(8,128)} bitcast(a)
  c = f32[10,10]{0,1:T(8,128)} add(b, b)
  d = (f32[10,10]{0,1:T(8,128)}, f32[10,10]{0,1:T(8,128)}) tuple(b, c)
  e = f32[10,10]{0,1:T(8,128)} get-tuple-element(d), index=0
  ROOT f = f32[10,10]{0,1:T(8,128)} add(e, e)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* a = FindInstruction(module.get(), "a");
  HloInstruction* c = FindInstruction(module.get(), "c");
  HloInstruction* f = FindInstruction(module.get(), "f");

  IOParam param_a(HloPosition{a, {}});
  EXPECT_THAT(param_a.GetNonTrivialUses(),
              IsOkAndHolds(UnorderedElementsAre(
                  IOParam(HloUse(c, 0, {})), IOParam(HloUse(c, 1, {})),
                  IOParam(HloUse(f, 0, {})), IOParam(HloUse(f, 1, {})))));
}

TEST_F(IOParamTest, GetNonTrivialSourcePosition_NonTrivialSource) {
  absl::string_view hlo_string = R"hlo(
HloModule m

ENTRY entry {
  a = f32[10,10]{0,1:T(8,128)} parameter(0)
  b = f32[10,10]{0,1:T(8,128)} add(a, a)
  ROOT c = f32[10,10]{0,1:T(8,128)} add(b, b)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* b = FindInstruction(module.get(), "b");
  HloInstruction* c = FindInstruction(module.get(), "c");

  IOParam use_c_0(HloUse(c, 0, {}));
  EXPECT_THAT(use_c_0.GetNonTrivialSourcePosition(),
              IsOkAndHolds(IOParam(HloPosition{b, {}})));

  IOParam use_c_1(HloUse(c, 1, {}));
  EXPECT_THAT(use_c_1.GetNonTrivialSourcePosition(),
              IsOkAndHolds(IOParam(HloPosition{b, {}})));
}

TEST_F(IOParamTest, GetNonTrivialSourcePosition_GetTupleElementSource) {
  absl::string_view hlo_string = R"hlo(
HloModule m

ENTRY entry {
  a = (f32[10,10]{0,1:T(8,128)}, f32[10,10]{0,1:T(8,128)}) parameter(0)
  b = f32[10,10]{0,1:T(8,128)} get-tuple-element(a), index=0
  c = f32[10,10]{0,1:T(8,128)} get-tuple-element(a), index=1
  ROOT d = f32[10,10]{0,1:T(8,128)} add(b, c)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* a = FindInstruction(module.get(), "a");
  HloInstruction* d = FindInstruction(module.get(), "d");

  IOParam use_d_0(HloUse(d, 0, {}));
  EXPECT_THAT(use_d_0.GetNonTrivialSourcePosition(),
              IsOkAndHolds(IOParam(HloPosition{a, {0}})));

  IOParam use_d_1(HloUse(d, 1, {}));
  EXPECT_THAT(use_d_1.GetNonTrivialSourcePosition(),
              IsOkAndHolds(IOParam(HloPosition{a, {1}})));
}

TEST_F(IOParamTest, GetNonTrivialSourcePosition_TupleSource) {
  absl::string_view hlo_string = R"hlo(
HloModule m

ENTRY entry {
  a = f32[10,10]{0,1:T(8,128)} parameter(0)
  b = f32[10,10]{0,1:T(8,128)} add(a, a)
  c = (f32[10,10]{0,1:T(8,128)}, f32[10,10]{0,1:T(8,128)}) tuple(a, b)
  ROOT d = (f32[10,10]{0,1:T(8,128)}, f32[10,10]{0,1:T(8,128)}) custom-call(c), custom_call_target="cc"
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* a = FindInstruction(module.get(), "a");
  HloInstruction* b = FindInstruction(module.get(), "b");
  HloInstruction* d = FindInstruction(module.get(), "d");

  IOParam use_d_0(HloUse(d, 0, {0}));
  EXPECT_THAT(use_d_0.GetNonTrivialSourcePosition(),
              IsOkAndHolds(IOParam(HloPosition{a, {}})));

  IOParam use_d_1(HloUse(d, 0, {1}));
  EXPECT_THAT(use_d_1.GetNonTrivialSourcePosition(),
              IsOkAndHolds(IOParam(HloPosition{b, {}})));
}

TEST_F(IOParamTest, GetNonTrivialSourcePosition_BitcastSource) {
  absl::string_view hlo_string = R"hlo(
HloModule m

ENTRY entry {
  a = f32[10,10]{0,1:T(8,128)} parameter(0)
  b = f32[10,10]{0,1:T(8,128)} bitcast(a)
  ROOT c = f32[10,10]{0,1:T(8,128)} add(b, b)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* a = FindInstruction(module.get(), "a");
  HloInstruction* c = FindInstruction(module.get(), "c");

  IOParam use_c_0(HloUse(c, 0, {}));
  EXPECT_THAT(use_c_0.GetNonTrivialSourcePosition(),
              IsOkAndHolds(IOParam(HloPosition{a, {}})));

  IOParam use_c_1(HloUse(c, 1, {}));
  EXPECT_THAT(use_c_1.GetNonTrivialSourcePosition(),
              IsOkAndHolds(IOParam(HloPosition{a, {}})));
}

TEST_F(IOParamTest, GetNonTrivialSourcePosition_ComplexSource) {
  absl::string_view hlo_string = R"hlo(
HloModule m

ENTRY entry {
  a = f32[10,10]{0,1:T(8,128)} parameter(0)
  b = f32[10,10]{0,1:T(8,128)} bitcast(a)
  c = f32[10,10]{0,1:T(8,128)} add(b, b)
  d = (f32[10,10]{0,1:T(8,128)}, f32[10,10]{0,1:T(8,128)}) tuple(b, c)
  e = f32[10,10]{0,1:T(8,128)} get-tuple-element(d), index=0
  ROOT f = f32[10,10]{0,1:T(8,128)} add(e, e)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* a = FindInstruction(module.get(), "a");
  HloInstruction* c = FindInstruction(module.get(), "c");
  HloInstruction* f = FindInstruction(module.get(), "f");

  IOParam use_c_0(HloUse(c, 0, {}));
  EXPECT_THAT(use_c_0.GetNonTrivialSourcePosition(),
              IsOkAndHolds(IOParam(HloPosition{a, {}})));

  IOParam use_f_0(HloUse(f, 0, {}));
  EXPECT_THAT(use_f_0.GetNonTrivialSourcePosition(),
              IsOkAndHolds(IOParam(HloPosition{a, {}})));
}
}  // namespace
}  // namespace xla

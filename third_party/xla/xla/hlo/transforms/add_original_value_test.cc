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

#include "xla/hlo/transforms/add_original_value.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using AddOriginalValueTest = HloHardwareIndependentTestBase;

using ::absl::string_view;

TEST_F(AddOriginalValueTest, Basic) {
  constexpr absl::string_view hlo_string = R"(
HloModule test, entry_computation_layout={(s32[]{:T(256)})->u32[2]{0:T(256)}}

ENTRY test {
  Arg_0.1 = s32[] parameter(0)
  constant.2 = s32[] constant(32)
  shift-right-logical.3 = s32[] shift-right-logical(Arg_0.1, constant.2)
  convert.4 = u32[] convert(shift-right-logical.3)
  reshape.5 = u32[1]{0} reshape(convert.4)
  convert.6 = u32[] convert(Arg_0.1)
  reshape.7 = u32[1]{0} reshape(convert.6)
  ROOT concatenate.8 = u32[2]{0} concatenate(reshape.5, reshape.7), dimensions={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AddOriginalValue pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(AddOriginalValueTest, Tuple) {
  constexpr absl::string_view hlo_string = R"(
HloModule test, entry_computation_layout={(f32[], f32[3]{0}, f32[2,3]{1,0})->((f32[], f32[3]{0}), f32[2,3]{1,0})}

ENTRY test (v1: f32[], v2: f32[3], v3: f32[2,3]) -> ((f32[], f32[3]{0}), f32[2,3]{1,0}) {
  v1 = f32[] parameter(0)
  v2 = f32[3]{0} parameter(1)
  v3 = f32[2,3]{1,0} parameter(2)
  t1 = (f32[], f32[3]{0}) tuple(f32[] v1, f32[3]{0} v2)
  ROOT t2 = ((f32[], f32[3]{0}), f32[2,3]{1,0}) tuple((f32[], f32[3]{0}) t1, f32[2,3]{1,0} v3)
}

)";

  RunAndFilecheckHloRewrite(hlo_string, AddOriginalValue(), R"(
CHECK:  %[[V1:.*]] = f32[] parameter(0), origin={{[{]}}{"[[V1]]"}
CHECK:  %[[V2:.*]] = f32[3]{0} parameter(1), origin={{[{]}}{"[[V2]]"}
CHECK:  %[[TUPLE:.*]] = (f32[], f32[3]{0}) tuple(%[[V1]], %[[V2]]), origin={({"[[V1]]"}, {"[[V2]]"})}
CHECK:  %[[V3:.*]] = f32[2,3]{1,0} parameter(2), origin={{[{]}}{"[[V3]]"}
CHECK:  ((f32[], f32[3]{0}), f32[2,3]{1,0}) tuple(%[[TUPLE]], %[[V3]]), origin={(({"v1"}, {"v2"}), {"v3"})}
  )");
}

TEST_F(AddOriginalValueTest, GetTupleElement) {
  constexpr absl::string_view hlo_string = R"(
HloModule test, entry_computation_layout={()->s32[2,3]{1,0}}

ENTRY test {
  constant = f32[3]{0} constant({1, 2, 3})
  constant.1 = s32[2,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 } })
  tuple = (f32[3]{0}, s32[2,3]{1,0}) tuple(f32[3]{0} constant, s32[2,3]{1,0} constant.1)
  ROOT get-tuple-element = s32[2,3]{1,0} get-tuple-element((f32[3]{0}, s32[2,3]{1,0}) tuple), index=1
}

)";

  RunAndFilecheckHloRewrite(hlo_string, AddOriginalValue(), R"(
CHECK:  %[[CONSTANT1:.*]] = f32[3]{0} constant({1, 2, 3}), origin={{[{]}}{"[[CONSTANT1]]"}
CHECK:  %[[CONSTANT2:.*]] = s32[2,3]{1,0} constant({ { 1, 2, 3 }, { 4, 5, 6 } }), origin={{[{]}}{"[[CONSTANT2]]"}
CHECK:  %[[TUPLE:.*]] = (f32[3]{0}, s32[2,3]{1,0}) tuple(%[[CONSTANT1]], %[[CONSTANT2]]), origin={({"[[CONSTANT1]]"}, {"[[CONSTANT2]]"})}
CHECK:  s32[2,3]{1,0} get-tuple-element(%[[TUPLE]]), index=1, origin={{[{]}}{"[[CONSTANT2]]"}
  )");
}

TEST_F(AddOriginalValueTest, GetTupleElementNonSymbolic) {
  constexpr absl::string_view hlo_string = R"(
HloModule test, entry_computation_layout={((f32[], s32[]))->s32[]}

ENTRY test {
  p = (f32[], s32[]) parameter(0)
  ROOT get-tuple-element = s32[] get-tuple-element(p), index=1
}

)";

  RunAndFilecheckHloRewrite(hlo_string, AddOriginalValue(), R"(
CHECK:  %[[PARAM:.*]] = (f32[], s32[]) parameter(0), origin={({"p" {0}{{[}]}}, {"p" {1}})}
CHECK:  s32[] get-tuple-element(%[[PARAM]]), index=1, origin={{[{]}}{"[[PARAM]]" {1}
  )");
}

}  // namespace
}  // namespace xla

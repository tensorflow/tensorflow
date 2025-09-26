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

#include "xla/hlo/utils/hlo_stack_trace.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

// Defines the data structure for each parameterized test case.
struct StackTraceTestCase {
  std::string test_name;
  std::string hlo_string;
  std::string expected_output;
};

class HloStackTraceTest : public HloHardwareIndependentTestBase {};

class HloStackTraceParameterizedTest
    : public HloStackTraceTest,
      public ::testing::WithParamInterface<StackTraceTestCase> {
 protected:
  void RunStackTraceTest(const std::string& hlo_string,
                         const std::string& expected_output) {
    auto module_or_status = ParseAndReturnVerifiedModule(hlo_string);
    ASSERT_TRUE(module_or_status.ok());
    auto module = std::move(module_or_status.value());

    auto dataflow_or_status = HloDataflowAnalysis::Run(*module);
    ASSERT_TRUE(dataflow_or_status.ok());
    auto& dataflow = dataflow_or_status.value();

    std::vector<std::pair<int64_t, const HloValue*>> buffers;
    for (const HloValue* value : dataflow->values()) {
      if (value->shape().IsTuple()) {
        continue;
      }
      buffers.emplace_back(ShapeUtil::ByteSizeOf(value->shape()), value);
    }

    std::string actual_output =
        FormatStackTraceBreakdown(buffers, module.get());
    EXPECT_EQ(actual_output, expected_output);
  }
};

TEST_F(HloStackTraceTest, HandlesEmptyBufferList) {
  std::vector<std::pair<int64_t, const HloValue*>> sized_buffers;
  std::string expected = "  Stack trace breakdown for peak usage: 0 bytes\n";
  auto module = CreateNewVerifiedModule();
  std::string actual = FormatStackTraceBreakdown(sized_buffers, module.get());
  EXPECT_EQ(actual, expected);
}

TEST_P(HloStackTraceParameterizedTest, Run) {
  const auto& param = GetParam();
  RunStackTraceTest(param.hlo_string, param.expected_output);
}

const char* const kRegexMatchHlo = R"(
HloModule RegexMatchModule
ENTRY main {
  p0 = f32[1] parameter(0), metadata={op_name="params['transformer/layer_1/mlp/linear'].weight"}
  p1 = f32[1] parameter(1), metadata={op_name="var['a']['b']['c']"}
  p2 = f32[1] parameter(2), metadata={op_name="invalid-string"}
  p3 = f32[1] parameter(3), metadata={op_name="no_key.attr1.attr2"}
  p4 = f32[1] parameter(4), metadata={op_name="only_var"}
  p5 = f32[1] parameter(5), metadata={op_name="p['a']"}
  ROOT tuple = (f32[1], f32[1], f32[1], f32[1], f32[1], f32[1]) tuple(p0, p1, p2, p3, p4, p5)
}
)";

const char* const kRegexMatchExpected =
    R"(  Stack trace breakdown for peak usage: 24 bytes
    main (100.0%, total: 24 bytes, current: 0 bytes, remaining: 24 bytes)
      ├── invalid-string (16.7%, total: 4 bytes, current: 4 bytes, remaining: 20 bytes)
      ├── no_key.attr1.attr2 (16.7%, total: 4 bytes, current: 4 bytes, remaining: 16 bytes)
      ├── only_var (16.7%, total: 4 bytes, current: 4 bytes, remaining: 12 bytes)
      ├── p (16.7%, total: 4 bytes, current: 0 bytes, remaining: 12 bytes)
      │   └── a (16.7%, total: 4 bytes, current: 4 bytes, remaining: 8 bytes)
      ├── params (16.7%, total: 4 bytes, current: 0 bytes, remaining: 8 bytes)
      │   └── transformer (16.7%, total: 4 bytes, current: 0 bytes, remaining: 8 bytes)
      │       └── layer_1 (16.7%, total: 4 bytes, current: 0 bytes, remaining: 8 bytes)
      │           └── mlp (16.7%, total: 4 bytes, current: 0 bytes, remaining: 8 bytes)
      │               └── linear (16.7%, total: 4 bytes, current: 0 bytes, remaining: 8 bytes)
      │                   └── weight (16.7%, total: 4 bytes, current: 4 bytes, remaining: 4 bytes)
      └── var (16.7%, total: 4 bytes, current: 0 bytes, remaining: 4 bytes)
          └── a (16.7%, total: 4 bytes, current: 0 bytes, remaining: 4 bytes)
              └── b (16.7%, total: 4 bytes, current: 0 bytes, remaining: 4 bytes)
                  └── c (16.7%, total: 4 bytes, current: 4 bytes, remaining: 0 bytes)
)";

const char* const kComplexHlo = R"(
HloModule MemoryTraceModule
InnerComp {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  ROOT multiply = f32[4] multiply(p0, p1), metadata={op_name="multiply"}
}
MiddleComp {
  p0 = f32[4] parameter(0)
  p1 = f32[4] parameter(1)
  p2 = f32[4] parameter(2)
  call_inner = f32[4] call(p0, p1), to_apply=InnerComp, metadata={op_name="call"}
  ROOT add = f32[4] add(call_inner, p2)
}
ENTRY main {
  arg1 = f32[4] parameter(0), metadata={op_name="arg1"}
  arg2 = f32[4] parameter(1), metadata={op_name="arg2"}
  cst = f32[4] constant({1, 1, 1, 1})
  call_middle = f32[4] call(arg1, arg2, cst), to_apply=MiddleComp
  ROOT final_add = f32[4] add(call_middle, arg1)
}
)";
const char* const kComplexExpected =
    R"(  Stack trace breakdown for peak usage: 96 bytes
    main (100.0%, total: 96 bytes, current: 0 bytes, remaining: 96 bytes)
      ├── call_middle (33.3%, total: 32 bytes, current: 0 bytes, remaining: 96 bytes)
      │   └── MiddleComp (33.3%, total: 32 bytes, current: 0 bytes, remaining: 96 bytes)
      │       ├── add (16.7%, total: 16 bytes, current: 16 bytes, remaining: 80 bytes)
      │       └── call (16.7%, total: 16 bytes, current: 0 bytes, remaining: 80 bytes)
      │           └── InnerComp (16.7%, total: 16 bytes, current: 0 bytes, remaining: 80 bytes)
      │               └── multiply (16.7%, total: 16 bytes, current: 16 bytes, remaining: 64 bytes)
      ├── arg1 (16.7%, total: 16 bytes, current: 16 bytes, remaining: 48 bytes)
      ├── arg2 (16.7%, total: 16 bytes, current: 16 bytes, remaining: 32 bytes)
      ├── cst (16.7%, total: 16 bytes, current: 16 bytes, remaining: 16 bytes)
      └── final_add (16.7%, total: 16 bytes, current: 16 bytes, remaining: 0 bytes)
)";

const char* const kQuotedParamsHlo = R"(
HloModule TransformerLayerModule
ENTRY main {
  p0 = f32[128,256] parameter(0), metadata={op_name="params['transformer/layer_1/mlp/linear/AqtEinsum_0']"}
  p1 = f32[128,256] parameter(1), metadata={op_name="params['transformer/layer_1/mlp/linear/AqtEinsum_0/AqtDotGeneral_0']"}
  p2 = s8[256,512] parameter(2), metadata={op_name="params['transformer/layer_1/mlp/linear/AqtEinsum_0/AqtDotGeneral_0/qrhs']['frozen'].qvalue"}
  p2_f32 = f32[256,512] convert(p2)
  ROOT dot_product = f32[128,512] dot(p1, p2_f32), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
const char* const kQuotedParamsExpected =
    R"(  Stack trace breakdown for peak usage: 1 179 648 bytes
    main (100.0%, total: 1 179 648 bytes, current: 0 bytes, remaining: 1 179 648 bytes)
      ├── p2_f32 (44.4%, total: 524 288 bytes, current: 524 288 bytes, remaining: 655 360 bytes)
      ├── params (33.3%, total: 393 216 bytes, current: 0 bytes, remaining: 655 360 bytes)
      │   └── transformer (33.3%, total: 393 216 bytes, current: 0 bytes, remaining: 655 360 bytes)
      │       └── layer_1 (33.3%, total: 393 216 bytes, current: 0 bytes, remaining: 655 360 bytes)
      │           └── mlp (33.3%, total: 393 216 bytes, current: 0 bytes, remaining: 655 360 bytes)
      │               └── linear (33.3%, total: 393 216 bytes, current: 0 bytes, remaining: 655 360 bytes)
      │                   └── AqtEinsum_0 (33.3%, total: 393 216 bytes, current: 131 072 bytes, remaining: 524 288 bytes)
      │                       └── AqtDotGeneral_0 (22.2%, total: 262 144 bytes, current: 131 072 bytes, remaining: 393 216 bytes)
      │                           └── qrhs (11.1%, total: 131 072 bytes, current: 0 bytes, remaining: 393 216 bytes)
      │                               └── frozen (11.1%, total: 131 072 bytes, current: 0 bytes, remaining: 393 216 bytes)
      │                                   └── qvalue (11.1%, total: 131 072 bytes, current: 131 072 bytes, remaining: 262 144 bytes)
      └── dot_product (22.2%, total: 262 144 bytes, current: 262 144 bytes, remaining: 0 bytes)
)";

const char* const kSingleInstructionHlo = R"(
HloModule main
ENTRY main {
  ROOT dup = f32[2] parameter(0)
}
)";
const char* const kSingleInstructionExpected =
    R"(  Stack trace breakdown for peak usage: 8 bytes
    main (100.0%, total: 8 bytes, current: 0 bytes, remaining: 8 bytes)
      └── dup (100.0%, total: 8 bytes, current: 8 bytes, remaining: 0 bytes)
)";

const char* const kAliasedMetadataHlo = R"(
HloModule main
ENTRY main {
  input1 = f32[256] parameter(0)
  input2 = f32[256] parameter(1)
  add1 = f32[256] add(input1, input2), metadata={op_name="params['layer1/fc']"}
  add2 = f32[256] add(input1, input2), metadata={op_name="params['layer1']['fc']"}
  ROOT final_add = f32[256] add(add1, add2)
}
)";
const char* const kAliasedMetadataExpected =
    R"(  Stack trace breakdown for peak usage: 5 120 bytes
    main (100.0%, total: 5 120 bytes, current: 0 bytes, remaining: 5 120 bytes)
      ├── params (40.0%, total: 2 048 bytes, current: 0 bytes, remaining: 5 120 bytes)
      │   └── layer1 (40.0%, total: 2 048 bytes, current: 0 bytes, remaining: 5 120 bytes)
      │       └── fc (40.0%, total: 2 048 bytes, current: 2 048 bytes, remaining: 3 072 bytes)
      ├── final_add (20.0%, total: 1 024 bytes, current: 1 024 bytes, remaining: 2 048 bytes)
      ├── input1 (20.0%, total: 1 024 bytes, current: 1 024 bytes, remaining: 1 024 bytes)
      └── input2 (20.0%, total: 1 024 bytes, current: 1 024 bytes, remaining: 0 bytes)
)";

const char* const kMultiCallsiteAliasedHlo = R"(
HloModule AliasedCalls
InnerComputation {
  p = f32[256] parameter(0)
  ROOT add_inside = f32[256] add(p, p)
}
ENTRY main {
  input = f32[256] parameter(0)
  call1 = f32[256] call(input), to_apply=InnerComputation, metadata={op_name="calls['layer1/fc']"}
  call2 = f32[256] call(input), to_apply=InnerComputation, metadata={op_name="calls['layer1']['fc']"}
  ROOT final_add = f32[256] add(call1, call2)
}
)";
const char* const kMultiCallsiteAliasedExpected =
    R"(  Stack trace breakdown for peak usage: 3 072 bytes
    main (100.0%, total: 3 072 bytes, current: 0 bytes, remaining: 3 072 bytes)
      ├── calls (33.3%, total: 1 024 bytes, current: 0 bytes, remaining: 3 072 bytes)
      │   └── layer1 (33.3%, total: 1 024 bytes, current: 0 bytes, remaining: 3 072 bytes)
      │       └── fc (33.3%, total: 1 024 bytes, current: 0 bytes, remaining: 3 072 bytes)
      │           └── InnerComputation (33.3%, total: 1 024 bytes, current: 0 bytes, remaining: 3 072 bytes)
      │               └── add_inside (33.3%, total: 1 024 bytes, current: 1 024 bytes, remaining: 2 048 bytes)
      ├── final_add (33.3%, total: 1 024 bytes, current: 1 024 bytes, remaining: 1 024 bytes)
      └── input (33.3%, total: 1 024 bytes, current: 1 024 bytes, remaining: 0 bytes)
)";

const char* const kTupleRootHlo = R"(
HloModule TupleRootModule
ENTRY main {
  p0 = f32[2] parameter(0)
  p1 = f32[2] parameter(1)
  add_op = f32[2] add(p0, p1)
  mul_op = f32[2] multiply(p0, p1)
  ROOT root_tuple = (f32[2], f32[2]) tuple(add_op, mul_op)
}
)";
const char* const kTupleRootExpected =
    R"(  Stack trace breakdown for peak usage: 32 bytes
    main (100.0%, total: 32 bytes, current: 0 bytes, remaining: 32 bytes)
      ├── add_op (25.0%, total: 8 bytes, current: 8 bytes, remaining: 24 bytes)
      ├── mul_op (25.0%, total: 8 bytes, current: 8 bytes, remaining: 16 bytes)
      ├── p0 (25.0%, total: 8 bytes, current: 8 bytes, remaining: 8 bytes)
      └── p1 (25.0%, total: 8 bytes, current: 8 bytes, remaining: 0 bytes)
)";

const char* const kRemainderHlo = R"(
HloModule RemainderModule
InnerComp {
  p = pred[] parameter(0)
  ROOT bc = pred[1] broadcast(p)
}
ENTRY main {
  p_in = pred[] parameter(0)
  call1 = pred[1] call(p_in), to_apply=InnerComp, metadata={op_name="call1"}
  call2 = pred[1] call(p_in), to_apply=InnerComp, metadata={op_name="call2"}
  call3 = pred[1] call(p_in), to_apply=InnerComp, metadata={op_name="call3"}
  and1 = pred[1] and(call1, call2)
  ROOT final_and = pred[1] and(and1, call3)
}
)";
const char* const kRemainderExpected =
    R"(  Stack trace breakdown for peak usage: 4 bytes
    main (100.0%, total: 4 bytes, current: 0 bytes, remaining: 4 bytes)
      ├── and1 (25.0%, total: 1 bytes, current: 1 bytes, remaining: 3 bytes)
      ├── call1 (25.0%, total: 1 bytes, current: 0 bytes, remaining: 3 bytes)
      │   └── InnerComp (25.0%, total: 1 bytes, current: 0 bytes, remaining: 3 bytes)
      │       └── bc (25.0%, total: 1 bytes, current: 1 bytes, remaining: 2 bytes)
      ├── final_and (25.0%, total: 1 bytes, current: 1 bytes, remaining: 1 bytes)
      └── p_in (25.0%, total: 1 bytes, current: 1 bytes, remaining: 0 bytes)
)";

INSTANTIATE_TEST_SUITE_P(
    StackTraceTests, HloStackTraceParameterizedTest,
    ::testing::ValuesIn(std::vector<StackTraceTestCase>{
        {"RegexMatch", kRegexMatchHlo, kRegexMatchExpected},
        {"FormatsComplexStackTraceCorrectly", kComplexHlo, kComplexExpected},
        {"ParsesAndFormatsQuotedParameterNames", kQuotedParamsHlo,
         kQuotedParamsExpected},
        {"HandlesSingleInstructionBuffer", kSingleInstructionHlo,
         kSingleInstructionExpected},
        {"HandlesAliasedMetadataPaths", kAliasedMetadataHlo,
         kAliasedMetadataExpected},
        {"HandlesMultipleCallsitesWithAliasedMetadata",
         kMultiCallsiteAliasedHlo, kMultiCallsiteAliasedExpected},
        {"HandlesTupleReturningRoot", kTupleRootHlo, kTupleRootExpected},
        {"DistributesSizeWithRemainder", kRemainderHlo, kRemainderExpected}}),
    [](const ::testing::TestParamInfo<
        HloStackTraceParameterizedTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace xla

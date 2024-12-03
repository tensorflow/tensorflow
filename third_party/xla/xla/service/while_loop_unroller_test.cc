/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/while_loop_unroller.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class WhileLoopUnrollerTest : public HloTestBase {
 protected:
  [[nodiscard]] std::unique_ptr<VerifiedHloModule> MakeModuleWithSimpleLoop(
      int num_iters);
  [[nodiscard]] std::unique_ptr<VerifiedHloModule>
  MakeModuleWithLoopBodyIndirectInc(int num_iters);
  [[nodiscard]] std::unique_ptr<VerifiedHloModule>
  MakeModuleWithNestedLoopBodyIndirectInc(int num_iters);
  [[nodiscard]] std::unique_ptr<VerifiedHloModule>
  MakeModuleWithWhileFeedingAnotherWhile(int num_iters);
  [[nodiscard]] std::unique_ptr<VerifiedHloModule>
  MakeModuleWithSimpleLoopAllReduce(int num_iters);

 public:
  void UnrollAndCompare(std::unique_ptr<HloModule> module,
                        absl::Span<Literal* const> arguments,
                        int64_t unroll_factor = -1, bool wrap_in_loop = false) {
    Literal before_unroll = ExecuteAndTransfer(module->Clone(), arguments);
    VLOG(2) << "before unroll value: " << before_unroll.ToString();

    EXPECT_TRUE(WhileLoopUnroller(unroll_factor, wrap_in_loop)
                    .Run(module.get())
                    .value());

    Literal after_unroll = ExecuteAndTransfer(std::move(module), arguments);
    VLOG(2) << "after unroll value: " << after_unroll.ToString();

    ASSERT_TRUE(LiteralTestUtil::NearOrEqual(/*expected=*/before_unroll,
                                             /*actual=*/after_unroll,
                                             std::nullopt));
  }
};

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithSimpleLoop(int num_iters) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    output = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[]{:T(128)}, s32[3]{0}) tuple(idx, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[]{:T(128)}, s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithLoopBodyIndirectInc(int num_iters) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[] get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[3]{0} get-tuple-element(loop_var.1), index=2
    output = s32[3]{0} add(get-tuple-element.3, get-tuple-element.3)
    inc = s32[] add(get-tuple-element.1, get-tuple-element.2)
    ROOT tuple = (s32[], s32[], s32[3]{0}) tuple(inc, get-tuple-element.2, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.1 = s32[] constant(1)
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[], s32[3]{0}) tuple(constant.3, constant.1, constant.4)
    ROOT while = (s32[], s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithNestedLoopBodyIndirectInc(int num_iters) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[] get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[3]{0} get-tuple-element(loop_var.1), index=2
    output = s32[3]{0} add(get-tuple-element.3, get-tuple-element.3)
    inc = s32[] add(get-tuple-element.1, get-tuple-element.2)
    ROOT tuple = (s32[], s32[], s32[3]{0}) tuple(inc, get-tuple-element.2, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  SimpleLoop {
    constant.1 = s32[] constant(1)
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[], s32[3]{0}) tuple(constant.3, constant.1, constant.4)
    ROOT while = (s32[], s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  OuterLoop.body {
    loop_var.1 = (s32[], s32[], s32[3]{0}, s32[10]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[] get-tuple-element(loop_var.1), index=1
    get-tuple-element.22 = s32[3]{0} get-tuple-element(loop_var.1), index=2
    get-tuple-element.3 = s32[10]{0} get-tuple-element(loop_var.1), index=3
    output = s32[10]{0} add(get-tuple-element.3, get-tuple-element.3)
    /* inner loop call*/
    constant.1 = s32[] constant(1)
    constant.3 = s32[] constant(0)
    tuple.1 = (s32[], s32[], s32[3]{0}) tuple(constant.3, constant.1, get-tuple-element.22)
    inner-while = (s32[], s32[], s32[3]{0}) while(tuple.1), condition=
        SimpleLoop.condition, body=SimpleLoop.body
    get-tuple-element.6 = s32[3]{0} get-tuple-element(inner-while), index=2
    inc = s32[] add(get-tuple-element.1, get-tuple-element.2)
    ROOT tuple = (s32[], s32[], s32[3]{0}, s32[10]{0}) tuple(inc, get-tuple-element.2, get-tuple-element.6, output)
  }
  OuterLoop.condition {
    loop_var.2 = (s32[], s32[], s32[3]{0}, s32[10]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY OuterLoop {
    constant.1 = s32[] constant(1)
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    constant.5 = s32[10]{0} constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
    tuple.1 = (s32[], s32[], s32[3]{0}, s32[10]{0}) tuple(constant.3, constant.1, constant.4, constant.5)
    ROOT while = (s32[], s32[], s32[3]{0}, s32[10]{0}) while(tuple.1), condition=
        OuterLoop.condition, body=OuterLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithWhileFeedingAnotherWhile(int num_iters) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    const1 = s32[] constant(1)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.3 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    output = s32[3]{0} add(get-tuple-element.3, get-tuple-element.3)
    inc = s32[] add(get-tuple-element.1, const1)
    ROOT tuple = (s32[], s32[3]{0}) tuple(inc, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  SimpleLoop {
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  OuterLoop.body {
    loop_var.1 = (s32[], s32[3]{0}, s32[10]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.22 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s32[10]{0} get-tuple-element(loop_var.1), index=2
    output1 = s32[3]{0} add(get-tuple-element.22, get-tuple-element.22)
    output2 = s32[10]{0} add(get-tuple-element.3, get-tuple-element.3)
    one = s32[] constant(1)
    inc = s32[] add(get-tuple-element.1, one)
    ROOT tuple = (s32[], s32[3]{0}, s32[10]{0}) tuple(inc, output1, output2)
  }
  OuterLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}, s32[10]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY entry.comp {
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    constant.5 = s32[10]{0} constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
    /* inner loop call*/
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    inner-while = (s32[], s32[3]{0}) while(tuple.1), condition=
        SimpleLoop.condition, body=SimpleLoop.body
    get-tuple-element.6 = s32[3]{0} get-tuple-element(inner-while), index=1
    tuple.2 = (s32[], s32[3]{0}, s32[10]{0}) tuple(constant.3, get-tuple-element.6, constant.5)
    ROOT while = (s32[], s32[3]{0}, s32[10]{0}) while(tuple.2), condition=
        OuterLoop.condition, body=OuterLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

std::unique_ptr<VerifiedHloModule>
WhileLoopUnrollerTest::MakeModuleWithSimpleLoopAllReduce(int num_iters) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop

  %reduction {
    %x = f32[] parameter(0)
    %y = f32[] parameter(1)
    ROOT %add = f32[] add(f32[] %x, f32[] %y)
  }

  SimpleLoop.body {
    loop_var.1 = (s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = f32[1024, 1024] get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = f32[1024, 1024] get-tuple-element(loop_var.1), index=2

    %all-reduce = f32[1024, 1024] all-reduce(f32[1024, 1024] get-tuple-element.2), channel_id=1, replica_groups={{0}}, to_apply=%reduction
    %accumulation = f32[1024, 1024] add(f32[1024, 1024] %all-reduce, f32[1024, 1024] get-tuple-element.3)

    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[], f32[1024, 1024], f32[1024, 1024]) tuple(add, get-tuple-element.2, %accumulation)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], f32[1024, 1024], f32[1024, 1024]) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    %param.1 = f32[1024, 1024] parameter(0)
    constant.3 = s32[] constant(0)

    %accumulation_buffer_init = f32[] constant(0)
    %accumulation_buffer = f32[1024, 1024] broadcast(f32[] %accumulation_buffer_init), dimensions={}

    tuple.1 =    (s32[], f32[1024, 1024], f32[1024, 1024]) tuple(constant.3, %param.1, %accumulation_buffer)
    ROOT while = (s32[], f32[1024, 1024], f32[1024, 1024]) while(tuple.1), condition=SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(num_iters)}});
  return ParseAndReturnVerifiedModule(hlo_string).value();
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopUnroll) {
  UnrollAndCompare(MakeModuleWithSimpleLoop(/*num_iters=*/5), {}, -1, false);
  UnrollAndCompare(MakeModuleWithSimpleLoop(/*num_iters=*/5), {}, -1, true);
}

// This test passes because we run WhileLoopConstantSinking before unrolling.
TEST_F(WhileLoopUnrollerTest, SimpleLoopUnrollNeedPrepare) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s64[], s32[3]{0}, s64[]) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.1), index=2
    add = s64[] add(get-tuple-element.1, get-tuple-element.3)
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}, s64[]) tuple(add, multiply, get-tuple-element.3)
  }
  SimpleLoop.condition {
    loop_var.2 = (s64[], s32[3]{0}, s64[]) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s64[] constant(0)
    one = s64[] constant(1)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s64[], s32[3]{0}, s64[]) tuple(constant.3, constant.4, one)
    while = (s64[], s32[3]{0}, s64[]) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT result = s32[3]{0} get-tuple-element(while), index=1
  }
  )";
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   false);
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   true);
}

// This test passes because we run TupleSimplifier before unrolling.
TEST_F(WhileLoopUnrollerTest, SimpleLoopUnrollNeedPrepare2) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s64[], s32[3]{0}, s64[]) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.1), index=2
    add = s64[] add(get-tuple-element.1, get-tuple-element.3)
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}, s64[]) tuple(add, multiply, get-tuple-element.3)
  }
  SimpleLoop.condition {
    loop_var.2 = (s64[], s32[3]{0}, s64[]) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s64[] constant(0)
    one = s64[] constant(1)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s64[], s32[3]{0}, s64[]) tuple(constant.3, constant.4, one)
    gte1 = s64[] get-tuple-element(tuple.1), index=0
    gte2 = s32[3]{0} get-tuple-element(tuple.1), index=1
    gte3 = s64[] get-tuple-element(tuple.1), index=2
    tuple = (s64[], s32[3]{0}, s64[]) tuple(gte1, gte2, gte3)
    while = (s64[], s32[3]{0}, s64[]) while(tuple), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT result = s32[3]{0} get-tuple-element(while), index=1
  }
  )";
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   false);
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   true);
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopNotRoot) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s64[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s64[], s32[3]{0}) tuple(constant.3, constant.4)
    while = (s64[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT result = s32[3]{0} get-tuple-element(while), index=1
  }
  )";
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   false);
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   true);
}

TEST_F(WhileLoopUnrollerTest, GetUnrollableLoops) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  SimpleLoop.body.2 {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition.2 {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  SimpleLoop.body.3 {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] multiply(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition.3 {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s64[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s64[], s32[3]{0}) tuple(constant.3, constant.4)
    while1 = (s64[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    while3 = (s64[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition.3, body=SimpleLoop.body.3
    while2 = (s64[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition.2, body=SimpleLoop.body.2
    o1 = s32[3]{0} get-tuple-element(while1), index=1
    o2 = s32[3]{0} get-tuple-element(while2), index=1
    ROOT result = (s32[3]{0}, s32[3]{0}) tuple(o1,o2)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  auto unrollable_loops = WhileLoopUnroller::GetUnrollableLoops(
      module.get(), {}, /*unroll_config=*/std::nullopt);
  // Only while1 and while2 are unrollable
  EXPECT_EQ(unrollable_loops.size(), 2);
}

TEST_F(WhileLoopUnrollerTest, UnrollMutipleLoops) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  SimpleLoop.body.2 {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition.2 {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s64[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s64[], s32[3]{0}) tuple(constant.3, constant.4)
    while1 = (s64[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    input = s32[3]{0} get-tuple-element(while1), index=1
    tuple.2 = (s64[], s32[3]{0}) tuple(constant.3, input)
    while2 = (s64[], s32[3]{0}) while(tuple.2), condition=
      SimpleLoop.condition.2, body=SimpleLoop.body.2
    o1 = s32[3]{0} get-tuple-element(while1), index=1
    o2 = s32[3]{0} get-tuple-element(while2), index=1
    ROOT result = (s32[3]{0}, s32[3]{0}) tuple(o1,o2)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Unroll the first loop
  TF_ASSERT_OK_AND_ASSIGN(
      UnrollResult unrolled_result,
      WhileLoopUnroller::UnrollAndReturnReplacement(
          module->entry_computation()->GetInstructionWithName("while1")));
  bool unrolled1 = unrolled_result.unrolled;
  EXPECT_TRUE(unrolled1);

  // There should be no call instructions after unrolling either loops since we
  // inline all the calls after unrolling.
  std::vector<HloInstruction*> call_instrs_1;
  for (auto* comp : module->MakeComputationPostOrder()) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(call_instrs_1),
                    HloPredicateIsOp<HloOpcode::kCall>);
  }
  EXPECT_EQ(call_instrs_1.size(), 0);

  // Unroll the second loop
  TF_ASSERT_OK_AND_ASSIGN(
      UnrollResult unrolled_result2,
      WhileLoopUnroller::UnrollAndReturnReplacement(
          module->entry_computation()->GetInstructionWithName("while2")));
  bool unrolled2 = unrolled_result2.unrolled;
  EXPECT_TRUE(unrolled2);
  std::vector<HloInstruction*> call_instrs_2;
  for (auto* comp : module->MakeComputationPostOrder()) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(call_instrs_2),
                    HloPredicateIsOp<HloOpcode::kCall>);
  }
  EXPECT_EQ(call_instrs_2.size(), 0);
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopNonZeroInit) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s64[] get-tuple-element(loop_var.1), index=0
    constant.1 = s64[] constant(1)
    add = s64[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s64[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s64[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s64[] get-tuple-element(loop_var.2), index=0
    constant.2 = s64[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s64[] constant(4)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s64[], s32[3]{0}) tuple(constant.3, constant.4)
    while = (s64[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT result = s32[3]{0} get-tuple-element(while), index=1
  }
  )";
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   false);
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   true);
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopS16IndVar) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s16[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s16[] get-tuple-element(loop_var.1), index=0
    constant.1 = s16[] constant(1)
    add = s16[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s16[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s16[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s16[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s16[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s16[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s16[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s16[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   false);
  UnrollAndCompare(ParseAndReturnVerifiedModule(hlo_string).value(), {}, -1,
                   true);
}

TEST_F(WhileLoopUnrollerTest, LoopWithControlDep) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s16[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s16[] get-tuple-element(loop_var.1), index=0
    constant.1 = s16[] constant(1)
    add = s16[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s16[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s16[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s16[] get-tuple-element(loop_var.2), index=0
    /* number of iterations is 10 */
    constant.2 = s16[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s16[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s16[], s32[3]{0}) tuple(constant.3, constant.4)
    while1 = (s16[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    copy1 = copy(constant.3), control-predecessors={while1}
    ROOT add = add(copy1, constant.3)
  }
  )";
  EXPECT_FALSE(WhileLoopUnroller()
                   .Run(ParseAndReturnVerifiedModule(hlo_string).value().get())
                   .value());
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopPartialUnroll) {
  auto m = MakeModuleWithSimpleLoop(/*num_iters=*/5);
  EXPECT_FALSE(WhileLoopUnroller(/*unroll_factor=*/3).Run(m.get()).value());
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopNoUnrollDueToTripCountThreshold) {
  auto m = MakeModuleWithSimpleLoop(/*num_iters=*/5);
  UnrollConfig config;
  config.trip_count_threshold = 0;  // Set the trip count threshold to 0.
  EXPECT_FALSE(WhileLoopUnroller(/*unroll_factor=*/-1,
                                 /*wrap_in_trivial_loop=*/false, config)
                   .Run(m.get())
                   .value());
}

TEST_F(WhileLoopUnrollerTest, IndirectBodyInc) {
  std::unique_ptr<HloModule> module =
      MakeModuleWithLoopBodyIndirectInc(/*num_iters=*/5);
  UnrollAndCompare(MakeModuleWithLoopBodyIndirectInc(/*num_iters=*/5), {}, -1,
                   false);
  UnrollAndCompare(MakeModuleWithLoopBodyIndirectInc(/*num_iters=*/5), {}, -1,
                   true);
}

TEST_F(WhileLoopUnrollerTest, NestedIndirectBodyInc) {
  std::unique_ptr<HloModule> module =
      MakeModuleWithNestedLoopBodyIndirectInc(/*num_iters=*/5);
  UnrollAndCompare(MakeModuleWithNestedLoopBodyIndirectInc(/*num_iters=*/5), {},
                   -1, false);
  UnrollAndCompare(MakeModuleWithNestedLoopBodyIndirectInc(/*num_iters=*/5), {},
                   -1, true);
}

TEST_F(WhileLoopUnrollerTest, WhileFeedingWhile) {
  UnrollAndCompare(MakeModuleWithWhileFeedingAnotherWhile(/*num_iters=*/5), {},
                   -1, false);
  UnrollAndCompare(MakeModuleWithWhileFeedingAnotherWhile(/*num_iters=*/5), {},
                   -1, true);
}

TEST_F(WhileLoopUnrollerTest, LoopWithCollective) {
  int64_t num_iters = 5;
  auto module = MakeModuleWithSimpleLoopAllReduce(num_iters);

  EXPECT_TRUE(
      WhileLoopUnroller(/*unroll_factor=*/-1).Run(module.get()).value());

  EXPECT_EQ(absl::c_count_if(module->entry_computation()->instructions(),
                             [](const HloInstruction* instruction) {
                               return instruction->opcode() ==
                                      HloOpcode::kAllReduce;
                             }),
            num_iters);
}

TEST_F(WhileLoopUnrollerTest, LoopWithCollective2) {
  std::string hlo_string = R"(
  HloModule module, entry_computation_layout={(s8[1,32,2048]{2,1,0:T(8,128)(4,1)S(1)}, s8[1,2048,4096]{2,1,0:T(8,128)(4,1)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)})->(s8[1,32,2048]{2,1,0:T(8,128)(4,1)S(1)}, s8[1,2048,4096]{2,1,0:T(8,128)(4,1)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, u32[]{:T(128)}, /*index=5*/u32[]{:T(128)}, u32[256]{0:T(256)}, u32[]{:T(128)}, u32[]{:T(128)}, s32[]{:T(128)}, /*index=10*/u32[]{:T(128)}, u32[]{:T(128)}, u32[]{:T(128)})}

  fused_computation.70.clone.clone.clone {
    param_0.10545 = s8[1,32,2048]{2,1,0:T(8,128)(4,1)S(1)} parameter(0)
    ROOT bitcast.7213 = s8[32,2048,1]{1,0,2:T(8,128)(4,1)} bitcast(param_0.10545)
  }

  fused_computation.68.clone.clone.clone {
    param_1.12561 = s8[1,2048,1,4096]{3,1,2,0:T(8,128)(4,1)S(1)} parameter(1)
    constant.26622 = s8[]{:T(512)} constant(0)
    pad.3783 = s8[1,2048,2,4096]{3,1,2,0:T(8,128)(4,1)} pad(param_1.12561, constant.26622), padding=0_0x0_0x0_1x0_0
    constant.26621 = s32[]{:T(128)} constant(0)
    param_2.10214 = s32[]{:T(128)S(6)} parameter(2)
    dynamic-slice.5474 = s8[1,2048,2,256]{3,1,2,0:T(8,128)(4,1)} dynamic-slice(pad.3783, constant.26621, constant.26621, constant.26621, param_2.10214), dynamic_slice_sizes={1,2048,2,256}
    pad.3782 = s8[1,2048,2,4096]{3,1,2,0:T(8,128)(4,1)} pad(param_1.12561, constant.26622), padding=0_0x0_0x1_0x0_0
    param_0.10544 = s32[]{:T(128)S(6)} parameter(0)
    dynamic-slice.5473 = s8[1,2048,2,256]{3,1,2,0:T(8,128)(4,1)} dynamic-slice(pad.3782, constant.26621, constant.26621, constant.26621, param_0.10544), dynamic_slice_sizes={1,2048,2,256}
    add.10207 = s8[1,2048,2,256]{3,1,2,0:T(8,128)(4,1)} add(dynamic-slice.5474, dynamic-slice.5473)
    ROOT bitcast.7212 = s8[2048,2,256]{2,0,1:T(8,128)(4,1)} bitcast(add.10207)
  }

  fused_computation.71.clone {
    param_3.7588 = s8[1,32,2048]{2,1,0:T(8,128)(4,1)S(1)} parameter(3)
    fusion.4288 = s8[32,2048,1]{1,0,2:T(8,128)(4,1)} fusion(param_3.7588), kind=kLoop, calls=fused_computation.70.clone.clone.clone
    param_0.10546 = s32[]{:T(128)S(6)} parameter(0)
    param_1.12562 = s8[1,2048,1,4096]{3,1,2,0:T(8,128)(4,1)S(1)} parameter(1)
    param_2.10215 = s32[]{:T(128)S(6)} parameter(2)
    fusion.4287 = s8[2048,2,256]{2,0,1:T(8,128)(4,1)} fusion(param_0.10546, param_1.12562, param_2.10215), kind=kLoop, calls=fused_computation.68.clone.clone.clone
    convolution.802 = s32[32,2,256]{2,0,1:T(8,128)} convolution(fusion.4288, fusion.4287), window={size=2 pad=1_1 rhs_reversal=1}, dim_labels=bf0_i0o->b0f
    ROOT bitcast.7214 = s32[1,32,2,256]{3,1,2,0:T(8,128)S(1)} bitcast(convolution.802)
  }

  fused_computation.76.clone {
    param_0.10547 = s32[1,32,256]{2,1,0:T(8,128)S(1)} parameter(0)
    param_1.12563 = s32[1,32,2,256]{3,1,2,0:T(8,128)S(1)} parameter(1)
    slice.12606 = s32[1,32,1,256]{3,1,2,0:T(8,128)} slice(param_1.12563), slice={[0:1], [0:32], [1:2], [0:256]}
    bitcast.7215 = s32[1,32,256]{2,1,0:T(8,128)} bitcast(slice.12606)
    add.10208 = s32[1,32,256]{2,1,0:T(8,128)S(1)} add(param_0.10547, bitcast.7215)
    param_2.10216 = s32[1,32,256]{2,1,0:T(8,128)S(1)} parameter(2)
    slice.12000.clone.2 = s32[1,32,1,256]{3,1,2,0:T(8,128)} slice(param_1.12563), slice={[0:1], [0:32], [0:1], [0:256]}
    bitcast.1776.clone.2 = s32[1,32,256]{2,1,0:T(8,128)} bitcast(slice.12000.clone.2)
    add.6006.clone.2 = s32[1,32,256]{2,1,0:T(8,128)S(1)} add(param_2.10216, bitcast.1776.clone.2)
    ROOT tuple.2892 = (s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}) tuple(add.10208, add.6006.clone.2)
  }

  fused_computation.69.clone.clone.clone {
    param_0.10549 = s8[1,32,2048]{2,1,0:T(8,128)(4,1)S(1)} parameter(0)
    ROOT bitcast.7217 = s8[32,2048,1]{1,0,2:T(8,128)(4,1)} bitcast(param_0.10549)
  }

  fused_computation.66.clone.clone.clone {
    param_1.12564 = s8[1,2048,1,4096]{3,1,2,0:T(8,128)(4,1)S(1)} parameter(1)
    constant.26625 = s8[]{:T(512)} constant(0)
    pad.3785 = s8[1,2048,2,4096]{3,1,2,0:T(8,128)(4,1)} pad(param_1.12564, constant.26625), padding=0_0x0_0x0_1x0_0
    constant.26624 = s32[]{:T(128)} constant(0)
    param_2.10217 = s32[]{:T(128)S(6)} parameter(2)
    dynamic-slice.5476 = s8[1,2048,2,256]{3,1,2,0:T(8,128)(4,1)} dynamic-slice(pad.3785, constant.26624, constant.26624, constant.26624, param_2.10217), dynamic_slice_sizes={1,2048,2,256}
    pad.3784 = s8[1,2048,2,4096]{3,1,2,0:T(8,128)(4,1)} pad(param_1.12564, constant.26625), padding=0_0x0_0x1_0x0_0
    param_0.10548 = s32[]{:T(128)S(6)} parameter(0)
    dynamic-slice.5475 = s8[1,2048,2,256]{3,1,2,0:T(8,128)(4,1)} dynamic-slice(pad.3784, constant.26624, constant.26624, constant.26624, param_0.10548), dynamic_slice_sizes={1,2048,2,256}
    add.10212 = s8[1,2048,2,256]{3,1,2,0:T(8,128)(4,1)} add(dynamic-slice.5476, dynamic-slice.5475)
    ROOT bitcast.7216 = s8[2048,2,256]{2,0,1:T(8,128)(4,1)} bitcast(add.10212)
  }

  fused_computation.72.clone {
    param_3.7589 = s8[1,32,2048]{2,1,0:T(8,128)(4,1)S(1)} parameter(3)
    fusion.4292 = s8[32,2048,1]{1,0,2:T(8,128)(4,1)} fusion(param_3.7589), kind=kLoop, calls=fused_computation.69.clone.clone.clone
    param_0.10550 = s32[]{:T(128)S(6)} parameter(0)
    param_1.12565 = s8[1,2048,1,4096]{3,1,2,0:T(8,128)(4,1)S(1)} parameter(1)
    param_2.10218 = s32[]{:T(128)S(6)} parameter(2)
    fusion.4291 = s8[2048,2,256]{2,0,1:T(8,128)(4,1)} fusion(param_0.10550, param_1.12565, param_2.10218), kind=kLoop, calls=fused_computation.66.clone.clone.clone
    convolution.803 = s32[32,2,256]{2,0,1:T(8,128)} convolution(fusion.4292, fusion.4291), window={size=2 pad=1_1 rhs_reversal=1}, dim_labels=bf0_i0o->b0f
    ROOT bitcast.7218 = s32[1,32,2,256]{3,1,2,0:T(8,128)S(1)} bitcast(convolution.803)
  }

  fused_computation.74.clone {
    param_0.10551 = s32[1,32,256]{2,1,0:T(8,128)S(1)} parameter(0)
    param_1.12566 = s32[1,32,2,256]{3,1,2,0:T(8,128)S(1)} parameter(1)
    slice.12607 = s32[1,32,1,256]{3,1,2,0:T(8,128)} slice(param_1.12566), slice={[0:1], [0:32], [1:2], [0:256]}
    bitcast.7219 = s32[1,32,256]{2,1,0:T(8,128)} bitcast(slice.12607)
    add.10213 = s32[1,32,256]{2,1,0:T(8,128)S(1)} add(param_0.10551, bitcast.7219)
    param_2.10219 = s32[1,32,256]{2,1,0:T(8,128)S(1)} parameter(2)
    slice.11997.clone.2 = s32[1,32,1,256]{3,1,2,0:T(8,128)} slice(param_1.12566), slice={[0:1], [0:32], [0:1], [0:256]}
    bitcast.1773.clone.2 = s32[1,32,256]{2,1,0:T(8,128)} bitcast(slice.11997.clone.2)
    add.6005.clone.2 = s32[1,32,256]{2,1,0:T(8,128)S(1)} add(param_2.10219, bitcast.1773.clone.2)
    ROOT tuple.2893 = (s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}) tuple(add.10213, add.6005.clone.2)
  }

  wide.windowed_dot_general_body {
    wide_param.41 = (s8[1,32,2048]{2,1,0:T(8,128)(4,1)S(1)}, s8[1,2048,4096]{2,1,0:T(8,128)(4,1)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, u32[]{:T(128)}, /*index=5*/u32[]{:T(128)}, u32[256]{0:T(256)}, u32[]{:T(128)}, u32[]{:T(128)}, s32[]{:T(128)}, /*index=10*/u32[]{:T(128)}, u32[]{:T(128)}, u32[]{:T(128)}) parameter(0)
    get-tuple-element.29000 = s8[1,32,2048]{2,1,0:T(8,128)(4,1)S(1)} get-tuple-element(wide_param.41), index=0
    get-tuple-element.29001 = s8[1,2048,4096]{2,1,0:T(8,128)(4,1)S(1)} get-tuple-element(wide_param.41), index=1
    get-tuple-element.28990 = s32[1,32,256]{2,1,0:T(8,128)S(1)} get-tuple-element(wide_param.41), index=3
    collective-permute-start = (s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, u32[]{:S(2)}, u32[]{:S(2)}) collective-permute-start(get-tuple-element.28990), channel_id=18, source_target_pairs={{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,8},{8,9},{9,10},{10,11},{11,12},{12,13},{13,14},{14,15},{15,0},{16,17},{17,18},{18,19},{19,20},{20,21},{21,22},{22,23},{23,24},{24,25},{25,26},{26,27},{27,28},{28,29},{29,30},{30,31},{31,16},{32,33},{33,34},{34,35},{35,36},{36,37},{37,38},{38,39},{39,40},{40,41},{41,42},{42,43},{43,44},{44,45},{45,46},{46,47},{47,32},{48,49},{49,50},{50,51},{51,52},{52,53},{53,54},{54,55},{55,56},{56,57},{57,58},{58,59},{59,60},{60,61},{61,62},{62,63},{63,48},{64,65},{65,66},{66,67},{67,68},{68,69},{69,70},{70,71},{71,72},{72,73},{73,74},{74,75},{75,76},{76,77},{77,78},{78,79},{79,64},{80,81},{81,82},{82,83},{83,84},{84,85},{85,86},{86,87},{87,88},{88,89},{89,90},{90,91},{91,92},{92,93},{93,94},{94,95},{95,80},{96,97},{97,98},{98,99},{99,100},{100,101},{101,102},{102,103},{103,104},{104,105},{105,106},{106,107},{107,108},{108,109},{109,110},{110,111},{111,96},{112,113},{113,114},{114,115},{115,116},{116,117},{117,118},{118,119},{119,120},{120,121},{121,122},{122,123},{123,124},{124,125},{125,126},{126,127},{127,112},{128,129},{129,130},{130,131},{131,132},{132,133},{133,134},{134,135},{135,136},{136,137},{137,138},{138,139},{139,140},{140,141},{141,142},{142,143},{143,128},{144,145},{145,146},{146,147},{147,148},{148,149},{149,150},{150,151},{151,152},{152,153},{153,154},{154,155},{155,156},{156,157},{157,158},{158,159},{159,144},{160,161},{161,162},{162,163},{163,164},{164,165},{165,166},{166,167},{167,168},{168,169},{169,170},{170,171},{171,172},{172,173},{173,174},{174,175},{175,160},{176,177},{177,178},{178,179},{179,180},{180,181},{181,182},{182,183},{183,184},{184,185},{185,186},{186,187},{187,188},{188,189},{189,190},{190,191},{191,176},{192,193},{193,194},{194,195},{195,196},{196,197},{197,198},{198,199},{199,200},{200,201},{201,202},{202,203},{203,204},{204,205},{205,206},{206,207},{207,192},{208,209},{209,210},{210,211},{211,212},{212,213},{213,214},{214,215},{215,216},{216,217},{217,218},{218,219},{219,220},{220,221},{221,222},{222,223},{223,208},{224,225},{225,226},{226,227},{227,228},{228,229},{229,230},{230,231},{231,232},{232,233},{233,234},{234,235},{235,236},{236,237},{237,238},{238,239},{239,224},{240,241},{241,242},{242,243},{243,244},{244,245},{245,246},{246,247},{247,248},{248,249},{249,250},{250,251},{251,252},{252,253},{253,254},{254,255},{255,240}}
    collective-permute-done = s32[1,32,256]{2,1,0:T(8,128)S(1)} collective-permute-done(collective-permute-start)
    get-tuple-element.29005 = u32[]{:T(128)} get-tuple-element(wide_param.41), index=5
    get-tuple-element.29006 = u32[256]{0:T(256)} get-tuple-element(wide_param.41), index=6
    partition-id.101 = u32[] partition-id()
    dynamic-slice.5472 = u32[1]{0:T(128)} dynamic-slice(get-tuple-element.29006, partition-id.101), dynamic_slice_sizes={1}
    bitcast.7210 = u32[]{:T(128)} bitcast(dynamic-slice.5472)
    get-tuple-element.29007 = u32[]{:T(128)} get-tuple-element(wide_param.41), index=7
    add.10204 = u32[]{:T(128)S(6)} add(bitcast.7210, get-tuple-element.29007)
    get-tuple-element.28991 = u32[]{:T(128)} get-tuple-element(wide_param.41), index=4
    subtract.2863 = u32[]{:T(128)S(6)} subtract(add.10204, get-tuple-element.28991)
    get-tuple-element.29008 = u32[]{:T(128)} get-tuple-element(wide_param.41), index=8
    and.400 = u32[]{:T(128)S(6)} and(subtract.2863, get-tuple-element.29008)
    clamp.1712 = u32[]{:T(128)S(6)} clamp(get-tuple-element.29005, and.400, get-tuple-element.29008)
    convert.8615 = s32[]{:T(128)S(6)} convert(clamp.1712)
    get-tuple-element.29009 = s32[]{:T(128)} get-tuple-element(wide_param.41), index=9
    multiply.14830 = s32[]{:T(128)S(6)} multiply(convert.8615, get-tuple-element.29009)
    bitcast.8823 = s8[1,2048,1,4096]{3,1,2,0:T(8,128)(4,1)S(1)} bitcast(get-tuple-element.29001)
    add.10205 = u32[]{:T(128)S(6)} add(get-tuple-element.28991, bitcast.7210)
    get-tuple-element.29010 = u32[]{:T(128)} get-tuple-element(wide_param.41), index=10
    add.10206 = u32[]{:T(128)S(6)} add(add.10205, get-tuple-element.29010)
    and.401 = u32[]{:T(128)S(6)} and(add.10206, get-tuple-element.29008)
    clamp.1713 = u32[]{:T(128)S(6)} clamp(get-tuple-element.29005, and.401, get-tuple-element.29008)
    convert.8616 = s32[]{:T(128)S(6)} convert(clamp.1713)
    multiply.14831 = s32[]{:T(128)S(6)} multiply(convert.8616, get-tuple-element.29009)
    fusion.4289 = s32[1,32,2,256]{3,1,2,0:T(8,128)S(1)} fusion(multiply.14830, bitcast.8823, multiply.14831, get-tuple-element.29000), kind=kOutput, calls=fused_computation.71.clone
    get-tuple-element.28989 = s32[1,32,256]{2,1,0:T(8,128)S(1)} get-tuple-element(wide_param.41), index=2
    collective-permute-start.1 = (s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, u32[]{:S(2)}, u32[]{:S(2)}) collective-permute-start(get-tuple-element.28989), channel_id=17, source_target_pairs={{0,15},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6},{8,7},{9,8},{10,9},{11,10},{12,11},{13,12},{14,13},{15,14},{16,31},{17,16},{18,17},{19,18},{20,19},{21,20},{22,21},{23,22},{24,23},{25,24},{26,25},{27,26},{28,27},{29,28},{30,29},{31,30},{32,47},{33,32},{34,33},{35,34},{36,35},{37,36},{38,37},{39,38},{40,39},{41,40},{42,41},{43,42},{44,43},{45,44},{46,45},{47,46},{48,63},{49,48},{50,49},{51,50},{52,51},{53,52},{54,53},{55,54},{56,55},{57,56},{58,57},{59,58},{60,59},{61,60},{62,61},{63,62},{64,79},{65,64},{66,65},{67,66},{68,67},{69,68},{70,69},{71,70},{72,71},{73,72},{74,73},{75,74},{76,75},{77,76},{78,77},{79,78},{80,95},{81,80},{82,81},{83,82},{84,83},{85,84},{86,85},{87,86},{88,87},{89,88},{90,89},{91,90},{92,91},{93,92},{94,93},{95,94},{96,111},{97,96},{98,97},{99,98},{100,99},{101,100},{102,101},{103,102},{104,103},{105,104},{106,105},{107,106},{108,107},{109,108},{110,109},{111,110},{112,127},{113,112},{114,113},{115,114},{116,115},{117,116},{118,117},{119,118},{120,119},{121,120},{122,121},{123,122},{124,123},{125,124},{126,125},{127,126},{128,143},{129,128},{130,129},{131,130},{132,131},{133,132},{134,133},{135,134},{136,135},{137,136},{138,137},{139,138},{140,139},{141,140},{142,141},{143,142},{144,159},{145,144},{146,145},{147,146},{148,147},{149,148},{150,149},{151,150},{152,151},{153,152},{154,153},{155,154},{156,155},{157,156},{158,157},{159,158},{160,175},{161,160},{162,161},{163,162},{164,163},{165,164},{166,165},{167,166},{168,167},{169,168},{170,169},{171,170},{172,171},{173,172},{174,173},{175,174},{176,191},{177,176},{178,177},{179,178},{180,179},{181,180},{182,181},{183,182},{184,183},{185,184},{186,185},{187,186},{188,187},{189,188},{190,189},{191,190},{192,207},{193,192},{194,193},{195,194},{196,195},{197,196},{198,197},{199,198},{200,199},{201,200},{202,201},{203,202},{204,203},{205,204},{206,205},{207,206},{208,223},{209,208},{210,209},{211,210},{212,211},{213,212},{214,213},{215,214},{216,215},{217,216},{218,217},{219,218},{220,219},{221,220},{222,221},{223,222},{224,239},{225,224},{226,225},{227,226},{228,227},{229,228},{230,229},{231,230},{232,231},{233,232},{234,233},{235,234},{236,235},{237,236},{238,237},{239,238},{240,255},{241,240},{242,241},{243,242},{244,243},{245,244},{246,245},{247,246},{248,247},{249,248},{250,249},{251,250},{252,251},{253,252},{254,253},{255,254}}
    collective-permute-done.1 = s32[1,32,256]{2,1,0:T(8,128)S(1)} collective-permute-done(collective-permute-start.1)
    fusion.4290 = (s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}) fusion(collective-permute-done, fusion.4289, collective-permute-done.1), kind=kLoop, calls=fused_computation.76.clone
    get-tuple-element.22079 = s32[1,32,256]{2,1,0:T(8,128)S(1)} get-tuple-element(fusion.4290), index=0
    collective-permute-start.2 = (s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, u32[]{:S(2)}, u32[]{:S(2)}) collective-permute-start(get-tuple-element.22079), channel_id=20, source_target_pairs={{0,1},{1,2},{2,3},{3,4},{4,5},{5,6},{6,7},{7,8},{8,9},{9,10},{10,11},{11,12},{12,13},{13,14},{14,15},{15,0},{16,17},{17,18},{18,19},{19,20},{20,21},{21,22},{22,23},{23,24},{24,25},{25,26},{26,27},{27,28},{28,29},{29,30},{30,31},{31,16},{32,33},{33,34},{34,35},{35,36},{36,37},{37,38},{38,39},{39,40},{40,41},{41,42},{42,43},{43,44},{44,45},{45,46},{46,47},{47,32},{48,49},{49,50},{50,51},{51,52},{52,53},{53,54},{54,55},{55,56},{56,57},{57,58},{58,59},{59,60},{60,61},{61,62},{62,63},{63,48},{64,65},{65,66},{66,67},{67,68},{68,69},{69,70},{70,71},{71,72},{72,73},{73,74},{74,75},{75,76},{76,77},{77,78},{78,79},{79,64},{80,81},{81,82},{82,83},{83,84},{84,85},{85,86},{86,87},{87,88},{88,89},{89,90},{90,91},{91,92},{92,93},{93,94},{94,95},{95,80},{96,97},{97,98},{98,99},{99,100},{100,101},{101,102},{102,103},{103,104},{104,105},{105,106},{106,107},{107,108},{108,109},{109,110},{110,111},{111,96},{112,113},{113,114},{114,115},{115,116},{116,117},{117,118},{118,119},{119,120},{120,121},{121,122},{122,123},{123,124},{124,125},{125,126},{126,127},{127,112},{128,129},{129,130},{130,131},{131,132},{132,133},{133,134},{134,135},{135,136},{136,137},{137,138},{138,139},{139,140},{140,141},{141,142},{142,143},{143,128},{144,145},{145,146},{146,147},{147,148},{148,149},{149,150},{150,151},{151,152},{152,153},{153,154},{154,155},{155,156},{156,157},{157,158},{158,159},{159,144},{160,161},{161,162},{162,163},{163,164},{164,165},{165,166},{166,167},{167,168},{168,169},{169,170},{170,171},{171,172},{172,173},{173,174},{174,175},{175,160},{176,177},{177,178},{178,179},{179,180},{180,181},{181,182},{182,183},{183,184},{184,185},{185,186},{186,187},{187,188},{188,189},{189,190},{190,191},{191,176},{192,193},{193,194},{194,195},{195,196},{196,197},{197,198},{198,199},{199,200},{200,201},{201,202},{202,203},{203,204},{204,205},{205,206},{206,207},{207,192},{208,209},{209,210},{210,211},{211,212},{212,213},{213,214},{214,215},{215,216},{216,217},{217,218},{218,219},{219,220},{220,221},{221,222},{222,223},{223,208},{224,225},{225,226},{226,227},{227,228},{228,229},{229,230},{230,231},{231,232},{232,233},{233,234},{234,235},{235,236},{236,237},{237,238},{238,239},{239,224},{240,241},{241,242},{242,243},{243,244},{244,245},{245,246},{246,247},{247,248},{248,249},{249,250},{250,251},{251,252},{252,253},{253,254},{254,255},{255,240}}
    collective-permute-done.2 = s32[1,32,256]{2,1,0:T(8,128)S(1)} collective-permute-done(collective-permute-start.2)
    get-tuple-element.29011 = u32[]{:T(128)} get-tuple-element(wide_param.41), index=11
    add.10209 = u32[]{:T(128)S(6)} add(get-tuple-element.28991, get-tuple-element.29011)
    subtract.2864 = u32[]{:T(128)S(6)} subtract(add.10204, add.10209)
    and.402 = u32[]{:T(128)S(6)} and(subtract.2864, get-tuple-element.29008)
    clamp.1714 = u32[]{:T(128)S(6)} clamp(get-tuple-element.29005, and.402, get-tuple-element.29008)
    convert.8617 = s32[]{:T(128)S(6)} convert(clamp.1714)
    multiply.14832 = s32[]{:T(128)S(6)} multiply(convert.8617, get-tuple-element.29009)
    bitcast.8824 = s8[1,2048,1,4096]{3,1,2,0:T(8,128)(4,1)S(1)} bitcast(get-tuple-element.29001)
    add.10210 = u32[]{:T(128)S(6)} add(add.10209, bitcast.7210)
    add.10211 = u32[]{:T(128)S(6)} add(add.10210, get-tuple-element.29010)
    and.403 = u32[]{:T(128)S(6)} and(add.10211, get-tuple-element.29008)
    clamp.1715 = u32[]{:T(128)S(6)} clamp(get-tuple-element.29005, and.403, get-tuple-element.29008)
    convert.8618 = s32[]{:T(128)S(6)} convert(clamp.1715)
    multiply.14833 = s32[]{:T(128)S(6)} multiply(convert.8618, get-tuple-element.29009)
    fusion.4293 = s32[1,32,2,256]{3,1,2,0:T(8,128)S(1)} fusion(multiply.14832, bitcast.8824, multiply.14833, get-tuple-element.29000), kind=kOutput, calls=fused_computation.72.clone
    get-tuple-element.22080 = s32[1,32,256]{2,1,0:T(8,128)S(1)} get-tuple-element(fusion.4290), index=1
    collective-permute-start.3 = (s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, u32[]{:S(2)}, u32[]{:S(2)}) collective-permute-start(get-tuple-element.22080), channel_id=19, source_target_pairs={{0,15},{1,0},{2,1},{3,2},{4,3},{5,4},{6,5},{7,6},{8,7},{9,8},{10,9},{11,10},{12,11},{13,12},{14,13},{15,14},{16,31},{17,16},{18,17},{19,18},{20,19},{21,20},{22,21},{23,22},{24,23},{25,24},{26,25},{27,26},{28,27},{29,28},{30,29},{31,30},{32,47},{33,32},{34,33},{35,34},{36,35},{37,36},{38,37},{39,38},{40,39},{41,40},{42,41},{43,42},{44,43},{45,44},{46,45},{47,46},{48,63},{49,48},{50,49},{51,50},{52,51},{53,52},{54,53},{55,54},{56,55},{57,56},{58,57},{59,58},{60,59},{61,60},{62,61},{63,62},{64,79},{65,64},{66,65},{67,66},{68,67},{69,68},{70,69},{71,70},{72,71},{73,72},{74,73},{75,74},{76,75},{77,76},{78,77},{79,78},{80,95},{81,80},{82,81},{83,82},{84,83},{85,84},{86,85},{87,86},{88,87},{89,88},{90,89},{91,90},{92,91},{93,92},{94,93},{95,94},{96,111},{97,96},{98,97},{99,98},{100,99},{101,100},{102,101},{103,102},{104,103},{105,104},{106,105},{107,106},{108,107},{109,108},{110,109},{111,110},{112,127},{113,112},{114,113},{115,114},{116,115},{117,116},{118,117},{119,118},{120,119},{121,120},{122,121},{123,122},{124,123},{125,124},{126,125},{127,126},{128,143},{129,128},{130,129},{131,130},{132,131},{133,132},{134,133},{135,134},{136,135},{137,136},{138,137},{139,138},{140,139},{141,140},{142,141},{143,142},{144,159},{145,144},{146,145},{147,146},{148,147},{149,148},{150,149},{151,150},{152,151},{153,152},{154,153},{155,154},{156,155},{157,156},{158,157},{159,158},{160,175},{161,160},{162,161},{163,162},{164,163},{165,164},{166,165},{167,166},{168,167},{169,168},{170,169},{171,170},{172,171},{173,172},{174,173},{175,174},{176,191},{177,176},{178,177},{179,178},{180,179},{181,180},{182,181},{183,182},{184,183},{185,184},{186,185},{187,186},{188,187},{189,188},{190,189},{191,190},{192,207},{193,192},{194,193},{195,194},{196,195},{197,196},{198,197},{199,198},{200,199},{201,200},{202,201},{203,202},{204,203},{205,204},{206,205},{207,206},{208,223},{209,208},{210,209},{211,210},{212,211},{213,212},{214,213},{215,214},{216,215},{217,216},{218,217},{219,218},{220,219},{221,220},{222,221},{223,222},{224,239},{225,224},{226,225},{227,226},{228,227},{229,228},{230,229},{231,230},{232,231},{233,232},{234,233},{235,234},{236,235},{237,236},{238,237},{239,238},{240,255},{241,240},{242,241},{243,242},{244,243},{245,244},{246,245},{247,246},{248,247},{249,248},{250,249},{251,250},{252,251},{253,252},{254,253},{255,254}}
    collective-permute-done.3 = s32[1,32,256]{2,1,0:T(8,128)S(1)} collective-permute-done(collective-permute-start.3)
    fusion.4294 = (s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}) fusion(collective-permute-done.2, fusion.4293, collective-permute-done.3), kind=kLoop, calls=fused_computation.74.clone
    get-tuple-element.29002 = s32[1,32,256]{2,1,0:T(8,128)S(1)} get-tuple-element(fusion.4294), index=1
    get-tuple-element.29003 = s32[1,32,256]{2,1,0:T(8,128)S(1)} get-tuple-element(fusion.4294), index=0
    get-tuple-element.29012 = u32[]{:T(128)} get-tuple-element(wide_param.41), index=12
    constant.28871 = u32[]{:T(128)} constant(2)
    add.10214 = u32[]{:T(128)} add(get-tuple-element.28991, constant.28871)
    ROOT tuple.3341 = (s8[1,32,2048]{2,1,0:T(8,128)(4,1)S(1)}, s8[1,2048,4096]{2,1,0:T(8,128)(4,1)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, u32[]{:T(128)}, /*index=5*/u32[]{:T(128)}, u32[256]{0:T(256)}, u32[]{:T(128)}, u32[]{:T(128)}, s32[]{:T(128)}, /*index=10*/u32[]{:T(128)}, u32[]{:T(128)}, u32[]{:T(128)}) tuple(get-tuple-element.29000, get-tuple-element.29001, get-tuple-element.29002, get-tuple-element.29003, add.10214, get-tuple-element.29005, get-tuple-element.29006, get-tuple-element.29007, get-tuple-element.29008, get-tuple-element.29009, get-tuple-element.29010, get-tuple-element.29011, get-tuple-element.29012)
  }

  wide.windowed_dot_general_cond {
    wide_param.40 = (s8[1,32,2048]{2,1,0:T(8,128)(4,1)S(1)}, s8[1,2048,4096]{2,1,0:T(8,128)(4,1)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, u32[]{:T(128)}, /*index=5*/u32[]{:T(128)}, u32[256]{0:T(256)}, u32[]{:T(128)}, u32[]{:T(128)}, s32[]{:T(128)}, /*index=10*/u32[]{:T(128)}, u32[]{:T(128)}, u32[]{:T(128)}) parameter(0)
    get-tuple-element.22055 = u32[]{:T(128)} get-tuple-element(wide_param.40), index=4
    constant.26614 = u32[]{:T(128)} constant(8)
    ROOT compare.2683 = pred[]{:T(512)} compare(get-tuple-element.22055, constant.26614), direction=LT
  }

  ENTRY test {
    fusion.4456 = s8[1,32,2048]{2,1,0:T(8,128)(4,1)S(1)} parameter(0)
    fusion.4457 = s8[1,2048,4096]{2,1,0:T(8,128)(4,1)S(1)} parameter(1)
    broadcast.26239 = s32[1,32,256]{2,1,0:T(8,128)S(1)} parameter(2)
    broadcast.26239.clone = s32[1,32,256]{2,1,0:T(8,128)S(1)} parameter(3)
    constant.28863 = u32[]{:T(128)} constant(0)
    constant.28864 = u32[]{:T(128)} constant(0)
    constant.28865 = u32[256]{0:T(256)} constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255})
    constant.28866 = u32[]{:T(128)} constant(8)
    constant.28867 = u32[]{:T(128)} constant(15)
    constant.28868 = s32[]{:T(128)} constant(256)
    constant.28869 = u32[]{:T(128)} constant(9)
    constant.28870 = u32[]{:T(128)} constant(1)
    constant.28871 = u32[]{:T(128)} constant(2)
    tuple.3339 = (s8[1,32,2048]{2,1,0:T(8,128)(4,1)S(1)}, s8[1,2048,4096]{2,1,0:T(8,128)(4,1)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, u32[]{:T(128)}, /*index=5*/u32[]{:T(128)}, u32[256]{0:T(256)}, u32[]{:T(128)}, u32[]{:T(128)}, s32[]{:T(128)}, /*index=10*/u32[]{:T(128)}, u32[]{:T(128)}, u32[]{:T(128)}) tuple(fusion.4456, fusion.4457, broadcast.26239, broadcast.26239.clone, constant.28863, constant.28864, constant.28865, constant.28866, constant.28867, constant.28868, constant.28869, constant.28870, constant.28871)
    ROOT while.636 = (s8[1,32,2048]{2,1,0:T(8,128)(4,1)S(1)}, s8[1,2048,4096]{2,1,0:T(8,128)(4,1)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, s32[1,32,256]{2,1,0:T(8,128)S(1)}, u32[]{:T(128)}, /*index=5*/u32[]{:T(128)}, u32[256]{0:T(256)}, u32[]{:T(128)}, u32[]{:T(128)}, s32[]{:T(128)}, /*index=10*/u32[]{:T(128)}, u32[]{:T(128)}, u32[]{:T(128)}) while(tuple.3339), condition=wide.windowed_dot_general_cond, body=wide.windowed_dot_general_body
  })";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  int64_t fusion_instr_count = absl::c_count_if(
      module->GetComputationWithName("wide.windowed_dot_general_body")
          ->instructions(),
      [](const HloInstruction* instr) {
        return (instr->IsLoopFusion() || instr->IsOutputFusion());
      });

  // Fully unroll the specific loop (trip count is 4)
  EXPECT_TRUE(
      WhileLoopUnroller(/*unroll_factor=*/-1).Run(module.get()).value());

  int64_t fusion_instr_count_after_unroll = absl::c_count_if(
      module->entry_computation()->instructions(),
      [](const HloInstruction* instr) {
        return (instr->IsLoopFusion() || instr->IsOutputFusion());
      });

  // The total number of fusions in the unrolled version in the entry must be
  // equal to loop_trip_count * fusion_instr_count
  EXPECT_EQ(fusion_instr_count * 4, fusion_instr_count_after_unroll);
}

TEST_F(WhileLoopUnrollerTest, MatchShapeCoveringDS) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[3,10]{1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3,10]{1,0} get-tuple-element(loop_var.1), index=1
    zero = s32[] constant(0)
    slice = s32[1,10] dynamic-slice(get-tuple-element.2, get-tuple-element.1, zero), dynamic_slice_sizes={1,10}
    output = s32[3,10]{1,0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[]{:T(128)}, s32[3,10]{1,0}) tuple(idx, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[3,10]{1,0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant(0)
    constant.4 = s32[3,10]{1,0} constant({...})
    tuple.1 = (s32[]{:T(128)}, s32[3,10]{1,0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[3,10]{1,0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";

  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(3)}});
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  HloInstruction* loop = module->entry_computation()->root_instruction();
  auto config = WhileLoopUnroller::IsLoopUnrollable(loop);
  EXPECT_TRUE(config.has_value());
  HloComputation* body = module->GetComputationWithName("SimpleLoop.body");
  HloInstruction* input = body->GetInstructionWithName("get-tuple-element.2");
  HloInstruction* instr = body->GetInstructionWithName("slice");
  EXPECT_TRUE(MatchShapeCoveringDynamicIndexInstruction(
                  instr, input, HloOpcode::kDynamicSlice, config.value())
                  .has_value());
}

TEST_F(WhileLoopUnrollerTest, MatchShapeCoveringDSShapeMismatch) {
  const std::string hlo_string = R"(
  HloModule SimpleLoop
  body {
    param = (s32[]{:T(128)}, s32[3,10]{1,0}, s32[3,11]{1,0}) parameter(0)
    idx = s32[]{:T(128)} get-tuple-element(param), index=0
    constant1 = s32[]{:T(128)} constant(1)
    new-idx = s32[]{:T(128)} add(idx, constant1)
    update = s32[3,10]{1,0} get-tuple-element(param), index=1
    input = s32[3,11]{1,0} get-tuple-element(param), index=2
    zero = s32[] constant(0)
    slice = s32[1,10] dynamic-slice(input, idx, zero), dynamic_slice_sizes={1,10}
    new-update = s32[3,10]{1,0} dynamic-update-slice(update, slice, idx, zero)
    ROOT tuple = (s32[]{:T(128)}, s32[3,10]{1,0}, s32[3,11]{1,0}) tuple(new-idx, new-update, input)
  }
  condition {
    param = (s32[]{:T(128)}, s32[3,10]{1,0}, s32[3,11]{1,0}) parameter(0)
    idx = s32[] get-tuple-element(param), index=0
    constant3 = s32[]{:T(128)} constant(3)
    ROOT less-than = pred[] compare(idx, constant3), direction=LT
  }
  ENTRY main {
    constant0 = s32[]{:T(128)} constant(0)
    init-update = s32[3,10]{1,0} constant({...})
    init-input = s32[3,11]{1,0} constant({...})
    init-while = (s32[]{:T(128)}, s32[3,10]{1,0}, s32[3,11]{1,0}) tuple(constant0, init-update, init-input)
    ROOT while = (s32[]{:T(128)}, s32[3,10]{1,0}, s32[3,11]{1,0}) while(init-while), condition=
      condition, body=body
  }
  )";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  HloInstruction* loop = module->entry_computation()->root_instruction();
  auto config = WhileLoopUnroller::IsLoopUnrollable(loop);
  EXPECT_TRUE(config.has_value());
  HloComputation* body = module->GetComputationWithName("body");
  HloInstruction* input = body->GetInstructionWithName("input");
  HloInstruction* instr = body->GetInstructionWithName("slice");
  EXPECT_FALSE(MatchShapeCoveringDynamicIndexInstruction(
                   instr, input, HloOpcode::kDynamicSlice, config.value())
                   .has_value());
}

TEST_F(WhileLoopUnrollerTest, MatchShapeCoveringDSNested) {
  std::string hlo_string_template = R"(
  HloModule SimpleLoop
  %fused_computation.slice (param_0.51117: s32[3,10], p1: s32[]) -> s32[10] {
    %param_0.51117 = s32[3,10] parameter(0)
    p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    slice = s32[1,10] dynamic-slice(s32[3,10] %param_0.51117, p1, s32[] %constant.85694), dynamic_slice_sizes={1,10}
    ROOT %bitcast.31250 = s32[10] bitcast(s32[1,10] slice)
  }

  %fused_computation.outer (param_1.30691: s32[3,10], p2: s32[]) -> s32[10] {
    %param_1.30691 = s32[3,10] parameter(0)
    p2 = s32[] parameter(1)
    inner.fusion = s32[10] fusion(s32[3,10] %param_1.30691, p2), kind=kLoop, calls=%fused_computation.slice
    ROOT out = s32[10] add(inner.fusion, inner.fusion)
  }
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[3,10]{1,0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3,10]{1,0} get-tuple-element(loop_var.1), index=1
    zero = s32[] constant(0)
    outer.fusion = s32[10] fusion(get-tuple-element.2, get-tuple-element.1), kind=kOutput, calls=%fused_computation.outer
    output = s32[3,10]{1,0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[]{:T(128)}, s32[3,10]{1,0}) tuple(idx, output)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[3,10]{1,0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant({{LOOP_BOUND}})
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant(0)
    constant.4 = s32[3,10]{1,0} constant({...})
    tuple.1 = (s32[]{:T(128)}, s32[3,10]{1,0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[3,10]{1,0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";

  std::string hlo_string = absl::StrReplaceAll(
      hlo_string_template, {{"{{LOOP_BOUND}}", absl::StrCat(3)}});
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  HloInstruction* loop = module->entry_computation()->root_instruction();
  auto config = WhileLoopUnroller::IsLoopUnrollable(loop);
  EXPECT_TRUE(config.has_value());
  HloComputation* inner_fusion_comp =
      module->GetComputationWithName("fused_computation.slice");
  HloInstruction* instr = inner_fusion_comp->GetInstructionWithName("slice");
  EXPECT_TRUE(MatchShapeCoveringDynamicIndexInstruction(
                  instr, inner_fusion_comp->parameter_instruction(0),
                  HloOpcode::kDynamicSlice, config.value())
                  .has_value());
}

// Unroller pass must remove all the DynamicGte custom-calls.
TEST_F(WhileLoopUnrollerTest, UnrollLoopWithDynamicGte) {
  std::string hlo_string = R"(
  HloModule SimpleLoop, entry_computation_layout={(s8[6,128,128]{2,1,0}, bf16[8,128]{1,0})->bf16[8,128]{1,0}}
    %fused_computation (param_0: s8[1,128,128]) -> s8[128,128] {
      %param_0 = s8[1,128,128]{2,1,0} parameter(0)
      ROOT %bitcast.1 = s8[128,128]{1,0} bitcast(s8[1,128,128]{2,1,0} %param_0)
    }

    %fused_computation.inner (param_0.34523: bf16[8,128], sliced: s8[1,128,128]) -> bf16[8,128] {
      %sliced = s8[1,128,128]{2,1,0} parameter(1)
      %param_0.34523 = bf16[8,128]{1,0} parameter(0)
      %fusion = s8[128,128]{1,0} fusion(s8[1,128,128]{2,1,0} %sliced), kind=kLoop, calls=%fused_computation
      ROOT %convolution.3447 = bf16[8,128]{1,0} convolution(bf16[8,128]{1,0} %param_0.34523, s8[128,128]{1,0} %fusion), dim_labels=bf_io->bf
    }

    %while.body (unstacked: (s32[], bf16[8,128], (s8[1,128,128], s8[1,128,128], s8[1,128,128], s8[1,128,128], s8[1,128,128], /*index=5*/s8[1,128,128]))) -> (s32[], bf16[8,128], (s8[1,128,128], s8[1,128,128], s8[1,128,128], s8[1,128,128], s8[1,128,128], /*index=5*/s8[1,128,128])) {
      %unstacked = (s32[], bf16[8,128]{1,0}, (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0})) parameter(0)
      %i = s32[] get-tuple-element((s32[], bf16[8,128]{1,0}, (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0})) %unstacked), index=0
      %one = s32[] constant(1)
      %inc = s32[] add(s32[] %i, s32[] %one)
      %p0 = bf16[8,128]{1,0} get-tuple-element((s32[], bf16[8,128]{1,0}, (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0})) %unstacked), index=1
      %p1.1 = (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0}) get-tuple-element((s32[], bf16[8,128]{1,0}, (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0})) %unstacked), index=2
      %two = s32[] constant(2)
      %mult = s32[] multiply(s32[] %i, s32[] %two)
      %custom-call = s8[1,128,128]{2,1,0} custom-call((s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0}) %p1.1, s32[] %mult), custom_call_target="DynamicGte"
      %fusion.conv = bf16[8,128]{1,0} fusion(bf16[8,128]{1,0} %p0, s8[1,128,128]{2,1,0} %custom-call), kind=kOutput, calls=%fused_computation.inner
      ROOT %out = (s32[], bf16[8,128]{1,0}, (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0})) tuple(s32[] %inc, bf16[8,128]{1,0} %fusion.conv, (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0}) %p1.1)
    }

    %while.cond (unstacked.1: (s32[], bf16[8,128], (s8[1,128,128], s8[1,128,128], s8[1,128,128], s8[1,128,128], s8[1,128,128], /*index=5*/s8[1,128,128]))) -> pred[] {
      %unstacked.1 = (s32[], bf16[8,128]{1,0}, (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0})) parameter(0)
      %i.1 = s32[] get-tuple-element((s32[], bf16[8,128]{1,0}, (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0})) %unstacked.1), index=0
      %constant.12857 = s32[] constant(3)
      ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] %i.1, s32[] %constant.12857), direction=LT
    }

    ENTRY %main (p0.1: s8[6,128,128], p1.2: bf16[8,128]) -> bf16[8,128] {
      %init = s32[] constant(0)
      %p1.2 = bf16[8,128]{1,0} parameter(1)
      %p0.1 = s8[6,128,128]{2,1,0} parameter(0)
      %while.input = (s32[], bf16[8,128]{1,0}, s8[6,128,128]{2,1,0}) tuple(s32[] %init, bf16[8,128]{1,0} %p1.2, s8[6,128,128]{2,1,0} %p0.1)
      %slice = s8[1,128,128]{2,1,0} slice(s8[6,128,128]{2,1,0} %p0.1), slice={[0:1], [0:128], [0:128]}
      %slice.1 = s8[1,128,128]{2,1,0} slice(s8[6,128,128]{2,1,0} %p0.1), slice={[1:2], [0:128], [0:128]}
      %slice.2 = s8[1,128,128]{2,1,0} slice(s8[6,128,128]{2,1,0} %p0.1), slice={[2:3], [0:128], [0:128]}
      %slice.3 = s8[1,128,128]{2,1,0} slice(s8[6,128,128]{2,1,0} %p0.1), slice={[3:4], [0:128], [0:128]}
      %slice.4 = s8[1,128,128]{2,1,0} slice(s8[6,128,128]{2,1,0} %p0.1), slice={[4:5], [0:128], [0:128]}
      %slice.5 = s8[1,128,128]{2,1,0} slice(s8[6,128,128]{2,1,0} %p0.1), slice={[5:6], [0:128], [0:128]}
      %tuple = (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0}) tuple(s8[1,128,128]{2,1,0} %slice, s8[1,128,128]{2,1,0} %slice.1, s8[1,128,128]{2,1,0} %slice.2, s8[1,128,128]{2,1,0} %slice.3, s8[1,128,128]{2,1,0} %slice.4, /*index=5*/s8[1,128,128]{2,1,0} %slice.5)
      %tuple.1 = (s32[], bf16[8,128]{1,0}, (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0})) tuple(s32[] %init, bf16[8,128]{1,0} %p1.2, (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0}) %tuple)
      %while.out = (s32[], bf16[8,128]{1,0}, (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0})) while((s32[], bf16[8,128]{1,0}, (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0})) %tuple.1), condition=%while.cond, body=%while.body
      %while_use = (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0}) get-tuple-element((s32[], bf16[8,128]{1,0}, (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0})) %while.out), index=2
      ROOT %out.1 = bf16[8,128]{1,0} get-tuple-element((s32[], bf16[8,128]{1,0}, (s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, s8[1,128,128]{2,1,0}, /*index=5*/s8[1,128,128]{2,1,0})) %while.out), index=1
  })";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  HloInstruction* loop =
      module->entry_computation()->root_instruction()->mutable_operand(0);
  TF_ASSERT_OK_AND_ASSIGN(UnrollResult unrolled_result,
                          WhileLoopUnroller::UnrollAndReturnReplacement(
                              loop, -1, false, true, true));
  bool unrolled = unrolled_result.unrolled;
  EXPECT_TRUE(unrolled);
  // Below method is successful only if all the DynamicGte and DynamicTuple
  // custom-calls are removed.
  for (HloInstruction* instr :
       module->entry_computation()->MakeInstructionPostOrder()) {
    EXPECT_FALSE(instr->IsCustomCall("DynamicGte"));
    EXPECT_FALSE(instr->IsCustomCall("DynamicTuple"));
  }
}

TEST_F(WhileLoopUnrollerTest, IsEffectivelyStaticDynamicSlice) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  %fused_computation.slice (param_0.51117: s8[6,128,128], p1: s32[]) -> s8[128,128] {
    %param_0.51117 = s8[6,128,128] parameter(0)
    static.p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    %dynamic-slice.static = s8[1,128,128] dynamic-slice(s8[6,128,128] %param_0.51117, static.p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
    ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.static)
  }

  %fused_computation.slice.2 (param_0.51117: s8[6,128,128], p1: s32[]) -> s8[128,128] {
    %param_0.51117 = s8[6,128,128] parameter(0)
    dynamic.p1 = s32[] parameter(1)
    %constant.85694 = s32[] constant(0)
    %dynamic-slice.dynamic = s8[1,128,128] dynamic-slice(s8[6,128,128] %param_0.51117, dynamic.p1, s32[] %constant.85694, s32[] %constant.85694), dynamic_slice_sizes={1,128,128}
    ROOT %bitcast.31250 = s8[128,128] bitcast(s8[1,128,128] %dynamic-slice.dynamic)
  }

  %fused_computation.inner (param_0.34523: bf16[8,128], param_1.30691: s8[6,128,128], p2: s32[], p3: s32[]) -> bf16[8,128] {
    %param_0.34523 = bf16[8,128] parameter(0)
    %param_1.30691 = s8[6,128,128] parameter(1)
    static.p2 = s32[] parameter(2)
    %fusion.1 = s8[128,128] fusion(s8[6,128,128] %param_1.30691, static.p2), kind=kLoop, calls=%fused_computation.slice
    dynamic.p3 = s32[] parameter(3)
    %fusion.2 = s8[128,128] fusion(s8[6,128,128] %param_1.30691, dynamic.p3), kind=kLoop, calls=%fused_computation.slice.2
    out = s8[128,128] add(%fusion.1, %fusion.2)
    ROOT %convolution.3447 = bf16[8,128] convolution(bf16[8,128] %param_0.34523, s8[128,128] out), dim_labels=bf_io->bf
  }

  %while.body (wide_param: (s32[], bf16[8,128], s8[6,128,128], s32[])) -> (s32[], bf16[8,128], s8[6,128,128], s32[]) {
    wide_p = (s32[], bf16[8,128], s8[6,128,128], s32[]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    p0 = bf16[8,128] get-tuple-element(wide_p), index=1
    p1 = s8[6,128,128] get-tuple-element(wide_p), index=2
    dynamic.p2 = s32[] get-tuple-element(wide_p), index=3
    one = s32[] constant(1)
    inc = s32[] add(i, one)
    two = s32[] constant(2)
    mult = s32[] multiply(i, two)
    fusion.conv = bf16[8,128] fusion(p0, p1, mult, dynamic.p2), kind=kOutput, calls=%fused_computation.inner
    ROOT out = (s32[], bf16[8,128], s8[6,128,128], s32[]) tuple(inc, fusion.conv, p1, dynamic.p2)
  }

  %while.cond (wide_param: (s32[], bf16[8,128], s8[6,128,128], s32[])) -> pred[] {
    wide_p = (s32[], bf16[8,128], s8[6,128,128], s32[]) parameter(0)
    i = s32[] get-tuple-element(wide_p), index=0
    %constant.12857 = s32[] constant(3)
    ROOT %compare.1921 = pred[]{:T(512)} compare(s32[] i, s32[] %constant.12857), direction=LT
  }

  ENTRY main {
    p0 = s8[6,128,128] parameter(0)
    p1 = bf16[8,128] parameter(1)
    p2 = s32[] parameter(2)
    init = s32[] constant(0)
    while.input = (s32[], bf16[8,128], s8[6,128,128], s32[]) tuple(init, p1, p0, p2)
    while.out = (s32[], bf16[8,128], s8[6,128,128], s32[]) while(while.input), condition=%while.cond , body=%while.body
    while_use = s8[6,128,128] get-tuple-element(while.out), index=2
    ROOT out = bf16[8,128] get-tuple-element(while.out), index=1
  }
  )";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  HloInstruction* loop =
      module->entry_computation()->root_instruction()->mutable_operand(0);
  std::optional<WhileLoopConfig> config =
      WhileLoopUnroller::IsLoopUnrollable(loop);
  EXPECT_TRUE(config.has_value());
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    HloInstruction* static_slice =
        comp->GetInstructionWithName("dynamic-slice.static");
    if (static_slice != nullptr) {
      auto index = MatchEffectivelyStaticDynamicSliceInsideLoop(
          static_slice, static_slice->operand(0), *config);
      EXPECT_TRUE(index.has_value());
    }
    HloInstruction* dynamic_slice =
        comp->GetInstructionWithName("dynamic-slice.dynamic");
    if (dynamic_slice != nullptr) {
      auto index = MatchEffectivelyStaticDynamicSliceInsideLoop(
          dynamic_slice, dynamic_slice->operand(0), *config);
      EXPECT_FALSE(index.has_value());
    }
  }
}
// We do not support case where there is no tuple for input.
TEST_F(WhileLoopUnrollerTest, SimpleLoopWithCustomCallNoTuple) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    custom-call.1 = (s32[]{:T(128)}, s32[3]{0}) custom-call(get-tuple-element.1, get-tuple-element.2), custom_call_target="CustomCallStart"
    get-tuple-element.3 = s32[]{:T(128)} get-tuple-element(custom-call.1), index=0
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.3, constant.1)
    get-tuple-element.4 = s32[3]{0} get-tuple-element(custom-call.1), index=1
    output = s32[3]{0} add(get-tuple-element.4, get-tuple-element.4)
    tuple = (s32[]{:T(128)}, s32[3]{0}) tuple(idx, output)
    ROOT custom-call.2 = (s32[]{:T(128)}, s32[3]{0}) custom-call(idx, output), custom_call_target="CustomCallEnd"
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    get-tuple-element.5 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(5)
    ROOT less-than = pred[] compare(get-tuple-element.5, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[]{:T(128)}, s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  auto m = ParseAndReturnVerifiedModule(hlo_string).value();
  UnrollConfig config;
  EXPECT_FALSE(WhileLoopUnroller(/*unroll_factor=*/-1,
                                 /*wrap_in_trivial_loop=*/false, config)
                   .Run(m.get())
                   .value());
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopWithCustomCallNonTupleForRoot) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    custom-call.1 = (s32[]{:T(128)}, s32[3]{0}) custom-call(loop_var.1), custom_call_target="CustomCallStart"
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(custom-call.1), index=0
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(custom-call.1), index=1
    output = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT custom-call.2 = (s32[]{:T(128)}, s32[3]{0}) custom-call(idx, output), custom_call_target="CustomCallEnd"
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    get-tuple-element.5 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(5)
    ROOT less-than = pred[] compare(get-tuple-element.5, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[]{:T(128)}, s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  auto m = ParseAndReturnVerifiedModule(hlo_string).value();
  UnrollConfig config;
  EXPECT_FALSE(WhileLoopUnroller(/*unroll_factor=*/-1,
                                 /*wrap_in_trivial_loop=*/false, config)
                   .Run(m.get())
                   .value());
}

TEST_F(WhileLoopUnrollerTest, SimpleLoopWithCustomCall) {
  std::string hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    custom-call.1 = (s32[]{:T(128)}, s32[3]{0}) custom-call(loop_var.1), custom_call_target="CustomCallStart"
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(custom-call.1), index=0
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(custom-call.1), index=1
    output = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    tuple = (s32[]{:T(128)}, s32[3]{0}) tuple(idx, output)
    ROOT custom-call.2 = (s32[]{:T(128)}, s32[3]{0}) custom-call(tuple), custom_call_target="CustomCallEnd"
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(5)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[]{:T(128)}, s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  auto m = ParseAndReturnVerifiedModule(hlo_string).value();
  UnrollConfig config;
  EXPECT_FALSE(WhileLoopUnroller(/*unroll_factor=*/-1,
                                 /*wrap_in_trivial_loop=*/false, config)
                   .Run(m.get())
                   .value());
}

}  // namespace
}  // namespace xla

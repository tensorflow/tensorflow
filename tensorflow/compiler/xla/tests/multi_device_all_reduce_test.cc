/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

class MultiDeviceAllReduceTest : public HloTestBase {};

XLA_TEST_F(MultiDeviceAllReduceTest, TwoReplicasOneOperand) {
  const char* module_str = R"(
  HloModule test

  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    add = f32[] add(x, y)
  }

  ENTRY test_computation {
    p = f32[3] parameter(0)
    ROOT crs = f32[3] all-reduce(p), to_apply=add
  })";
  auto config = GetModuleConfigForTest();
  config.set_replica_count(2);
  auto module = ParseHloString(module_str, config).ValueOrDie();
  auto literal = LiteralUtil::CreateR1<float>({1, 2, 3});
  auto expected = LiteralUtil::CreateR1<float>({2, 4, 6});
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(std::move(module), {&literal}, 2,
                                            /*use_threads=*/true));
  EXPECT_EQ(expected, results[0]);
  EXPECT_EQ(expected, results[1]);
}

}  // namespace
}  // namespace xla

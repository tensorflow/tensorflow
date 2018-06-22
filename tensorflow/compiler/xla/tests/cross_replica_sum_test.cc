/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

class TrivialCrossReplicaSumTest : public HloTestBase {};

// Currently the CPU and GPU backends only support CrossReplicaSum with one
// replica.  But we can at least check this.

XLA_TEST_F(TrivialCrossReplicaSumTest, OneOperand) {
  const char* module_str = R"(
  HloModule test

  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    add = f32[] add(x, y)
  }

  ENTRY test_computation {
    p = f32[3] parameter(0)
    ROOT crs = f32[3] cross-replica-sum(p), to_apply=add
  })";
  auto module =
      ParseHloString(module_str, GetModuleConfigForTest()).ValueOrDie();
  auto literal = Literal::CreateR1<float>({1, 2, 3});
  EXPECT_EQ(*literal, *ExecuteAndTransfer(std::move(module), {literal.get()}));
}

XLA_TEST_F(TrivialCrossReplicaSumTest, MultipleOperands) {
  const char* module_str = R"(
  HloModule test

  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    add = f32[] add(x, y)
  }

  ENTRY test_computation {
    p0 = f32[3] parameter(0)
    p1 = f32[2] parameter(1)
    ROOT crs = (f32[3], f32[2]) cross-replica-sum(p0, p1), to_apply=add
  })";
  auto module =
      ParseHloString(module_str, GetModuleConfigForTest()).ValueOrDie();
  auto literal0 = Literal::CreateR1<float>({1, 2, 3});
  auto literal1 = Literal::CreateR1<float>({10, 20});
  EXPECT_EQ(
      *Literal::MakeTuple({literal0.get(), literal1.get()}),
      *ExecuteAndTransfer(std::move(module), {literal0.get(), literal1.get()}));
}

// On the GPU backend, constants get special handling.  Someone might pass a
// constant to CRS to e.g. count the number of replicas -- we need to make sure
// it works.
XLA_TEST_F(TrivialCrossReplicaSumTest, ConstantOperand) {
  const char* module_str = R"(
  HloModule test

  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    add = f32[] add(x, y)
  }

  ENTRY test_computation {
    p0 = f32[3] parameter(0)
    p1 = f32[2] constant({10, 20})
    ROOT crs = (f32[3], f32[2]) cross-replica-sum(p0, p1), to_apply=add
  })";
  auto module =
      ParseHloString(module_str, GetModuleConfigForTest()).ValueOrDie();
  auto literal0 = Literal::CreateR1<float>({1, 2, 3});
  auto literal1 = Literal::CreateR1<float>({10, 20});
  EXPECT_EQ(*Literal::MakeTuple({literal0.get(), literal1.get()}),
            *ExecuteAndTransfer(std::move(module), {literal0.get()}));
}

}  // namespace
}  // namespace xla

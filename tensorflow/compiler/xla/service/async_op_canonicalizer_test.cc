/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/async_op_canonicalizer.h"

#include <string>

#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

using AsyncOpCanonicalizerTest = HloTestBase;

TEST_F(AsyncOpCanonicalizerTest, AsyncCallsSingleComputation) {
  std::string hlo_string = R"(
HloModule AsyncCall

%called_computation (param_0: f32[4096], param_1: f32[4096]) -> f32[4096] {
  %param_0 = f32[4096]{0} parameter(0)
  %param_1 = f32[4096]{0} parameter(1)
  %negate_0 = f32[4096]{0} negate(f32[4096]{0} %param_0)
  %negate_1 = f32[4096]{0} negate(f32[4096]{0} %param_1)
  ROOT %result.1 = f32[4096]{0} add(f32[4096]{0} %negate_0, f32[4096]{0} %negate_1)
}

%async_wrapped (async_param: f32[4096], async_param.1: f32[4096]) -> f32[4096] {
  %async_param = f32[4096]{0} parameter(0)
  %async_param.1 = f32[4096]{0} parameter(1)
  ROOT %call = f32[4096]{0} call(f32[4096]{0} %async_param, f32[4096]{0} %async_param.1), to_apply=%called_computation
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %b = f32[4096]{0} parameter(1)
  %async-start = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) async-start(f32[4096]{0} %a, f32[4096]{0} %b), calls=%async_wrapped
  %negate_2 = f32[4096]{0} negate(f32[4096]{0} %a)
  %async-update = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) async-update(((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) %async-start), calls=%async_wrapped
  %negate_3 = f32[4096]{0} negate(f32[4096]{0} %b)
  %add_0 = f32[4096]{0} add(f32[4096]{0} %negate_2, f32[4096]{0} %negate_3)
  %async-done = f32[4096]{0} async-done(((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) %async-update), calls=%async_wrapped
  ROOT %add_1 = f32[4096]{0} add(f32[4096]{0} %add_0, f32[4096]{0} %async-done)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AsyncOpCanonicalizer canonicalizer;
  EXPECT_TRUE(canonicalizer.Run(module.get()).ValueOrDie());

  HloInstruction* async_start = FindInstruction(module.get(), "async-start");
  HloInstruction* async_update = FindInstruction(module.get(), "async-update");
  HloInstruction* async_done = FindInstruction(module.get(), "async-done");

  EXPECT_EQ(async_start->async_group_id(), 0);
  EXPECT_EQ(async_update->async_group_id(), 0);
  EXPECT_EQ(async_done->async_group_id(), 0);
  EXPECT_EQ(async_start->async_wrapped_computation(),
            async_update->async_wrapped_computation());
  EXPECT_EQ(async_start->async_wrapped_computation(),
            async_done->async_wrapped_computation());
}

TEST_F(AsyncOpCanonicalizerTest, AsyncCallsMultipleComputations) {
  std::string hlo_string = R"(
HloModule AsyncCall

%called_computation (param_0: f32[4096], param_1: f32[4096]) -> f32[4096] {
  %param_0 = f32[4096]{0} parameter(0)
  %param_1 = f32[4096]{0} parameter(1)
  %negate_0 = f32[4096]{0} negate(f32[4096]{0} %param_0)
  %negate_1 = f32[4096]{0} negate(f32[4096]{0} %param_1)
  ROOT %result.1 = f32[4096]{0} add(f32[4096]{0} %negate_0, f32[4096]{0} %negate_1)
}

%async_wrapped.1 (async_param: f32[4096], async_param.1: f32[4096]) -> f32[4096] {
  %async_param = f32[4096]{0} parameter(0)
  %async_param.1 = f32[4096]{0} parameter(1)
  ROOT %call = f32[4096]{0} call(f32[4096]{0} %async_param, f32[4096]{0} %async_param.1), to_apply=%called_computation
}

%async_wrapped.2 (async_param: f32[4096], async_param.1: f32[4096]) -> f32[4096] {
  %async_param = f32[4096]{0} parameter(0)
  %async_param.1 = f32[4096]{0} parameter(1)
  ROOT %call = f32[4096]{0} call(f32[4096]{0} %async_param, f32[4096]{0} %async_param.1), to_apply=%called_computation
}

%async_wrapped.3 (async_param: f32[4096], async_param.1: f32[4096]) -> f32[4096] {
  %async_param = f32[4096]{0} parameter(0)
  %async_param.1 = f32[4096]{0} parameter(1)
  ROOT %call = f32[4096]{0} call(f32[4096]{0} %async_param, f32[4096]{0} %async_param.1), to_apply=%called_computation
}

ENTRY %main (a: f32[4096], b: f32[4096]) -> f32[4096] {
  %a = f32[4096]{0} parameter(0)
  %b = f32[4096]{0} parameter(1)
  %async-start = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) async-start(f32[4096]{0} %a, f32[4096]{0} %b), calls=%async_wrapped.1
  %negate_2 = f32[4096]{0} negate(f32[4096]{0} %a)
  %async-update = ((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) async-update(((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) %async-start), calls=%async_wrapped.2
  %negate_3 = f32[4096]{0} negate(f32[4096]{0} %b)
  %add_0 = f32[4096]{0} add(f32[4096]{0} %negate_2, f32[4096]{0} %negate_3)
  %async-done = f32[4096]{0} async-done(((f32[4096]{0}, f32[4096]{0}), f32[4096]{0}, u32[]) %async-update), calls=%async_wrapped.3
  ROOT %add_1 = f32[4096]{0} add(f32[4096]{0} %add_0, f32[4096]{0} %async-done)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  AsyncOpCanonicalizer canonicalizer;
  EXPECT_TRUE(canonicalizer.Run(module.get()).ValueOrDie());
  HloDCE dce;
  dce.Run(module.get()).ValueOrDie();

  HloInstruction* async_start = FindInstruction(module.get(), "async-start");
  HloInstruction* async_update = FindInstruction(module.get(), "async-update");
  HloInstruction* async_done = FindInstruction(module.get(), "async-done");

  EXPECT_EQ(async_start->async_group_id(), 0);
  EXPECT_EQ(async_update->async_group_id(), 0);
  EXPECT_EQ(async_done->async_group_id(), 0);
  EXPECT_EQ(async_start->async_wrapped_computation(),
            async_update->async_wrapped_computation());
  EXPECT_EQ(async_start->async_wrapped_computation(),
            async_done->async_wrapped_computation());
}

}  // namespace
}  // namespace xla

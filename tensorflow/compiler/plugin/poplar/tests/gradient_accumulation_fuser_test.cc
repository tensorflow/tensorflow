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
#include "tensorflow/compiler/plugin/poplar/driver/passes/gradient_accumulation_fuser.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_factor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using GradientAccumulationFuserTest = HloTestBase;

TEST_F(GradientAccumulationFuserTest, TestGradAccumAndAllReduce) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

entry  {
  %arg0 = f16[4] parameter(0)
  %ga = f16[4] custom-call(arg0), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  ROOT %a1 = f16[4] all-reduce(ga), to_apply=add
}
  )";

  auto config = GetModuleConfigForTest();
  auto module = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  GradientAccumulationFuser fuser(annotations);
  EXPECT_TRUE(fuser.Run(module0).ValueOrDie());
  auto root = module0->entry_computation()->root_instruction();
  auto cast = DynCast<HloStatefulGradientAccumulateAndAllReduce>(root);
  ASSERT_TRUE(cast);
  EXPECT_EQ(cast->MiniBatchesToAccumulate(), 4);
}

TEST_F(GradientAccumulationFuserTest, TestAllReduceAndGradAccum) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

entry  {
  %arg0 = f16[4] parameter(0)
  %a1 = f16[4] all-reduce(arg0), to_apply=add
  ROOT %ga = f16[4] custom-call(a1), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":10}\n"
}
  )";

  auto config = GetModuleConfigForTest();
  auto module = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  GradientAccumulationFuser fuser(annotations);
  EXPECT_TRUE(fuser.Run(module0).ValueOrDie());
  auto root = module0->entry_computation()->root_instruction();
  auto ga_and_ar = DynCast<HloStatefulGradientAccumulateAndAllReduce>(root);
  ASSERT_TRUE(ga_and_ar);
  EXPECT_EQ(ga_and_ar->MiniBatchesToAccumulate(), 10);
}

TEST_F(GradientAccumulationFuserTest, TestAllReduceAndNormalizeAndGradAccum) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

entry  {
  %arg0 = f16[4] parameter(0)
  %a1 = f16[4] all-reduce(arg0), to_apply=add
  %norm = f16[4] custom-call(a1), custom_call_target="Poputil::ReplicationNormalise", backend_config="{}\n"
  ROOT %ga = f16[4] custom-call(norm), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":10}\n"
}
  )";

  auto config = GetModuleConfigForTest();
  auto module = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  GradientAccumulationFuser fuser(annotations);
  EXPECT_TRUE(fuser.Run(module0).ValueOrDie());
  auto root = module0->entry_computation()->root_instruction();
  auto normalise = DynCast<HloReplicationNormaliseInstruction>(root);
  ASSERT_TRUE(normalise);
  auto ga_and_ar =
      DynCast<HloStatefulGradientAccumulateAndAllReduce>(normalise->operand(0));
  ASSERT_TRUE(ga_and_ar);
  EXPECT_EQ(ga_and_ar->MiniBatchesToAccumulate(), 10);
}

TEST_F(GradientAccumulationFuserTest, TestUnsupportedAllReduce) {
  std::string hlo_string = R"(
HloModule top

divide {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  divide = f32[] divide(x, y)
}

entry  {
  %arg0 = f16[4] parameter(0)
  %ga = f16[4] custom-call(arg0), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":4}\n"
  ROOT %a1 = f16[4] all-reduce(ga), to_apply=divide
}
  )";

  auto config = GetModuleConfigForTest();
  auto module = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  GradientAccumulationFuser fuser(annotations);
  EXPECT_FALSE(fuser.Run(module0).ValueOrDie());
}

TEST_F(GradientAccumulationFuserTest, TestMoreThanOneUser) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

entry  {
  %arg0 = f16[4] parameter(0)
  %a1 = f16[4] all-reduce(arg0), to_apply=add
  %norm = f16[4] custom-call(a1), custom_call_target="Poputil::ReplicationNormalise", backend_config="{}\n"
  %ga = f16[4] custom-call(norm), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":10}\n"
  ROOT %tuple = (f16[4], f16[4]) tuple(ga, norm)
}
  )";

  auto config = GetModuleConfigForTest();
  auto module = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  GradientAccumulationFuser fuser(annotations);
  EXPECT_FALSE(fuser.Run(module0).ValueOrDie());
}

TEST_F(GradientAccumulationFuserTest, TestMoreThanOneUser2) {
  std::string hlo_string = R"(
HloModule top

add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  add = f32[] add(x, y)
}

entry  {
  %arg0 = f16[4] parameter(0)
  %a1 = f16[4] all-reduce(arg0), to_apply=add
  %norm = f16[4] custom-call(a1), custom_call_target="Poputil::ReplicationNormalise", backend_config="{}\n"
  %ga = f16[4] custom-call(norm), custom_call_target="Poputil::StatefulGradientAccumulate", backend_config="{\"num_mini_batches\":10}\n"
  ROOT %tuple = (f16[4], f16[4]) tuple(ga, a1)
}
  )";

  auto config = GetModuleConfigForTest();
  auto module = ParseHloString(hlo_string, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);
  CustomOpReplacer custom_op_replacer;
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  GradientAccumulationFuser fuser(annotations);
  EXPECT_FALSE(fuser.Run(module0).ValueOrDie());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla

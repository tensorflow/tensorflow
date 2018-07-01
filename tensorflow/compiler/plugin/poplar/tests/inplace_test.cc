/* Copyright 2018 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/update_op_dependencies.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using HloInplaceDependencyTest = HloTestBase;


TEST_F(HloInplaceDependencyTest, ResourceUpdate) {
std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = s32[20] parameter(0)
  p1 = s32[20] parameter(1)

  s = s32[20] subtract(p0, p1), metadata={op_type="ResourceApplyGradientDescent" op_name="name"}

  ROOT t = (s32[20]) tuple(s)
}

)";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations;

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  EXPECT_THAT(annotations.inplace_instructions.size(), 1);

  const auto* inst = *(annotations.inplace_instructions.begin());
  EXPECT_THAT(inst->name(), "s");
}


TEST_F(HloInplaceDependencyTest, DynamicSliceUpdateInWhile) {
std::string hlo = R"(
HloModule top

body {
  p_b = (s32[], s32[20], s32[20]) parameter(0)
  p0_b = s32[] get-tuple-element(p_b), index=0
  p1_b = s32[20] get-tuple-element(p_b), index=1
  p2_b = s32[20] get-tuple-element(p_b), index=2
  i_b = s32[1] reshape(p0_b)
  a_b = s32[1] dynamic-slice(p1_b, i_b), dynamic_slice_sizes={1}
  t1_b = s32[1] dynamic-slice(p2_b, a_b), dynamic_slice_sizes={1}
  t2_b = s32[1] dynamic-slice(p2_b, i_b), dynamic_slice_sizes={1}
  u0_b = s32[20] dynamic-update-slice(p2_b, t2_b, a_b)
  u1_b = s32[20] dynamic-update-slice(u0_b, t1_b, i_b)
  ROOT root_b = (s32[], s32[20], s32[20]) tuple(p0_b, p1_b, u1_b)
}

cond {
  p_c = (s32[], s32[20], s32[20]) parameter(0)
  p0_c = s32[] get-tuple-element(p_c), index=0
  z_c = s32[] constant(0)
  ROOT eq_c = pred[] equal-to(p0_c, z_c)
}

ENTRY c1 {
  p0 = s32[] parameter(0)
  p1 = s32[20] parameter(1)
  p2 = s32[20] parameter(2)
  t = (s32[], s32[20], s32[20]) tuple(p0, p1, p2)
  ROOT while = (s32[], s32[20], s32[20]) while(t), condition=cond, body=body
}

)";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations;

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  std::set<std::string> in_place_ops = {"u0_b", "u1_b"};
  EXPECT_THAT(annotations.inplace_instructions.size(), 2);
  for (const auto* inst : annotations.inplace_instructions) {
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }
}


TEST_F(HloInplaceDependencyTest, DynamicUpdateSlice) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = s32[] parameter(0)
  p1 = s32[20] parameter(1)
  p2 = s32[20] parameter(2)
  i = s32[1] reshape(s32[] p0)
  a = s32[1] dynamic-slice(p1, i), dynamic_slice_sizes={1}
  t1 = s32[1] dynamic-slice(p2, a), dynamic_slice_sizes={1}
  t2 = s32[1] dynamic-slice(p2, i), dynamic_slice_sizes={1}
  u0 = s32[20] dynamic-update-slice(p2, t2, a)
  u1 = s32[20] dynamic-update-slice(u0, t1, i)
  ROOT root = (s32[20]) tuple(u1)
 }
)";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* entry = module0->entry_computation();


  CompilerAnnotations annotations;

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  std::set<std::string> in_place_ops = {"u0", "u1"};
  EXPECT_THAT(annotations.inplace_instructions.size(), 2);
  for (const auto* inst : annotations.inplace_instructions) {
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }

  UpdateOpDependenctOrdering updateOpDependenctOrdering(annotations);
  EXPECT_TRUE(updateOpDependenctOrdering.Run(module0).ValueOrDie());

  std::vector<const HloInstruction*> instruction_order =
      Scheduler::schedule(entry).ValueOrDie();

  EXPECT_THAT(instruction_order.size(), 10);

  std::map<std::string, unsigned int> order;
  for (unsigned int i=0; i<instruction_order.size(); i++) {
    order[instruction_order[i]->name()] = i;
  }

  // Normal ordering
  EXPECT_TRUE(order.at("p0") < order.at("i"));
  EXPECT_TRUE(order.at("i") < order.at("a"));
  EXPECT_TRUE(order.at("i") < order.at("t2"));
  EXPECT_TRUE(order.at("i") < order.at("u1"));
  EXPECT_TRUE(order.at("a") < order.at("t1"));
  EXPECT_TRUE(order.at("a") < order.at("u0"));
  EXPECT_TRUE(order.at("t1") < order.at("u1"));
  EXPECT_TRUE(order.at("t2") < order.at("u0"));
  EXPECT_TRUE(order.at("u0") < order.at("u1"));

  // All updates need to occur after all reads
  EXPECT_TRUE(order.at("t1") < order.at("u0"));
  EXPECT_TRUE(order.at("t2") < order.at("u1"));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla

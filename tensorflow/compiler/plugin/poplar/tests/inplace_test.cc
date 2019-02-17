// /* Copyright 2018 Graphcore Ltd

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================*/

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/fuse_ops_late.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"
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

  s = s32[20] subtract(p0, p1),
  metadata={op_type="ResourceApplyGradientDescent" op_name="name"}

  ROOT t = (s32[20]) tuple(s)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = annotations.inplace_instructions;

  EXPECT_THAT(inplace_instructions.size(), 2);
  std::set<std::string> in_place_ops = {"s", "t"};
  for (const auto* inst : inplace_instructions) {
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }
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

  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(3);
  config.set_input_mapping({0, 1, 2});
  config.set_resource_update_to_input_index({0, 1, 2});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  std::set<std::string> in_place_ops = {
      "p0_b", "p1_b", "p2_b", "u0_b", "u1_b", "root_b", "p0_c", "t", "while"};
  auto inplace_instructions = annotations.inplace_instructions;

  EXPECT_THAT(inplace_instructions.size(), 9);
  for (const auto* inst : inplace_instructions) {
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

  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(1);
  config.set_input_mapping({0, 1, 2});
  config.set_resource_update_to_input_index({2});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* entry = module0->entry_computation();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  std::set<std::string> in_place_ops = {"u0", "u1", "root", "i"};
  auto inplace_instructions = annotations.inplace_instructions;

  EXPECT_THAT(inplace_instructions.size(), 4);
  for (const auto* inst : inplace_instructions) {
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }

  Scheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module0).ValueOrDie());

  auto instruction_order = module0->schedule().sequence(entry).instructions();

  EXPECT_THAT(instruction_order.size(), 10);

  std::map<std::string, unsigned int> order;
  for (unsigned int i = 0; i < instruction_order.size(); i++) {
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

TEST_F(HloInplaceDependencyTest, MultipleUpdateInPlacePeers) {
  std::string hlo = R"(
  HloModule top

  ENTRY c1 {
    p0 = s32[20] parameter(0)
    p1 = s32[20] parameter(1)
    u0 = s32[20] add(p0, p1),
    metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
    u1 = s32[20] subtract(p0, p1),
    metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
    ROOT root = (s32[20], s32[20]) tuple(u0, u1)
   }
  )";

  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1});
  config.set_resource_update_to_input_index({0, 1});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* entry = module0->entry_computation();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  std::set<std::string> either_in_place_ops = {"u0", "u1"};
  std::set<std::string> in_place_ops = {"root"};
  auto& inplace_instructions = annotations.inplace_instructions;
  // Only one of the binary ops can be update in place
  EXPECT_THAT(inplace_instructions.size(), 2);
  for (const auto* inst : inplace_instructions) {
    EXPECT_TRUE(either_in_place_ops.count(inst->name()) == 1 ||
                in_place_ops.count(inst->name()) == 1);
  }

  Scheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module0).ValueOrDie());

  auto instruction_order = module0->schedule().sequence(entry).instructions();
  EXPECT_THAT(instruction_order.size(), 5);

  std::map<std::string, unsigned int> order;
  for (unsigned int i = 0; i < instruction_order.size(); i++) {
    order[instruction_order[i]->name()] = i;
  }

  // Normal ordering
  EXPECT_TRUE(order.at("p0") < order.at("u0"));
  EXPECT_TRUE(order.at("p1") < order.at("u0"));
  EXPECT_TRUE(order.at("p0") < order.at("u1"));
  EXPECT_TRUE(order.at("p1") < order.at("u1"));
  EXPECT_TRUE(order.at("u0") < order.at("root"));
  EXPECT_TRUE(order.at("u1") < order.at("root"));
}

TEST_F(HloInplaceDependencyTest, MultipleInplaceWithInterdependency) {
  std::string hlo = R"(
    HloModule top

    ENTRY c1 {
      p0 = s32[20] parameter(0)
      p1 = s32[20] parameter(1)
      u0 = s32[20] add(p0, p1),
      metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
      u1 = s32[20] subtract(u0, p0),
      metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
      ROOT root = (s32[20]) tuple(u1)
     }
    )";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* entry = module0->entry_computation();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto& inplace_instructions = annotations.inplace_instructions;

  // Only one of the binary ops can be update in place
  std::set<std::string> in_place_ops = {"u1", "root"};
  EXPECT_THAT(inplace_instructions.size(), 2);
  for (const auto* inst : inplace_instructions) {
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }

  Scheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module0).ValueOrDie());

  auto instruction_order = module0->schedule().sequence(entry).instructions();

  EXPECT_THAT(instruction_order.size(), 5);

  std::map<std::string, unsigned int> order;
  for (unsigned int i = 0; i < instruction_order.size(); i++) {
    order[instruction_order[i]->name()] = i;
  }

  // Normal ordering
  EXPECT_TRUE(order.at("p0") < order.at("u0"));
  EXPECT_TRUE(order.at("p1") < order.at("u0"));
  EXPECT_TRUE(order.at("u0") < order.at("u1"));
  EXPECT_TRUE(order.at("u1") < order.at("root"));
}

TEST_F(HloInplaceDependencyTest, MultipleInplaceWithRightOrder) {
  std::string hlo = R"(
    HloModule top

    ENTRY c1 {
      p0 = s32[20] parameter(0)
      p1 = s32[20] parameter(1)
      p2 = s32[20] parameter(2)
      u0 = s32[20] add(p0, p1),
      metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
      u1 = s32[20] add(p1, p2),
      metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
      ROOT root = (s32[20], s32[20]) tuple(u0, u1)
     }
    )";

  auto config = GetModuleConfigForTest();
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1, 2});
  config.set_resource_update_to_input_index({1, 2});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* entry = module0->entry_computation();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto& inplace_instructions = annotations.inplace_instructions;

  std::set<std::string> in_place_ops = {"u0", "u1", "root"};
  EXPECT_THAT(inplace_instructions.size(), 3);
  for (const auto* inst : inplace_instructions) {
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }

  Scheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module0).ValueOrDie());

  auto instruction_order = module0->schedule().sequence(entry).instructions();

  EXPECT_THAT(instruction_order.size(), 6);

  std::map<std::string, unsigned int> order;
  for (unsigned int i = 0; i < instruction_order.size(); i++) {
    order[instruction_order[i]->name()] = i;
  }

  // Normal ordering
  EXPECT_TRUE(order.at("p0") < order.at("u0"));
  EXPECT_TRUE(order.at("p1") < order.at("u0"));
  EXPECT_TRUE(order.at("p2") < order.at("u1"));
  EXPECT_TRUE(order.at("u0") < order.at("u1"));
  EXPECT_TRUE(order.at("u1") < order.at("root"));
}

TEST_F(HloInplaceDependencyTest, InplaceCorrectDependencies) {
  std::string hlo = R"(
    HloModule top

    ENTRY c1 {
      p0 = s32[20] parameter(0)
      p1 = s32[20] parameter(1)
      u0 = s32[20] add(p0, p1),
      metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
      u1 = s32[20] add(p0, u0),
      metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
      ROOT root = (s32[20], s32[20]) tuple(u0, u1)
     }
    )";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();
  auto* entry = module0->entry_computation();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());
  auto& inplace_instructions = annotations.inplace_instructions;

  std::set<std::string> in_place_ops = {"u1", "root"};
  EXPECT_THAT(inplace_instructions.size(), 2);
  for (const auto* inst : inplace_instructions) {
    EXPECT_THAT(in_place_ops.count(inst->name()), 1);
  }

  Scheduler scheduler;
  EXPECT_TRUE(scheduler.Run(module0).ValueOrDie());

  auto instruction_order = module0->schedule().sequence(entry).instructions();

  EXPECT_THAT(instruction_order.size(), 5);

  std::map<std::string, unsigned int> order;
  for (unsigned int i = 0; i < instruction_order.size(); i++) {
    order[instruction_order[i]->name()] = i;
  }

  // Normal ordering
  EXPECT_TRUE(order.at("p0") < order.at("u0"));
  EXPECT_TRUE(order.at("p1") < order.at("u0"));
  EXPECT_TRUE(order.at("u0") < order.at("u1"));
  EXPECT_TRUE(order.at("u1") < order.at("root"));
}

TEST_F(HloInplaceDependencyTest, InplaceInputOuputStreamedAndResourceVariable) {
  std::string hlo = R"(
    HloModule top

    ENTRY c1 {
      p0 = s32[20] parameter(0)
      p1 = s32[20] parameter(1)
      u0 = s32[20] add(p1, p0)
      u1 = s32[20] add(p0, u0),
      metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
      ROOT root = (s32[20], s32[20]) tuple(u0, u1)
     }
    )";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(1);
  config.set_resource_input_count(1);
  config.set_input_mapping({0, 1});
  config.set_resource_update_to_input_index({1});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto& inplace_instructions = annotations.inplace_instructions;

  EXPECT_THAT(inplace_instructions.size(), 3);
  std::set<std::string> in_place_ops = {"u0", "u1", "root"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
}

TEST_F(HloInplaceDependencyTest, InplaceElementwiseBinary) {
  std::string hlo = R"(
    HloModule top

    ENTRY c1 {
      p0 = s32[20] parameter(0)
      p1 = s32[20] parameter(1)
      ROOT u0 = s32[20] add(p1, p0)
     }
    )";

  auto config = GetModuleConfigForTest();
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto& inplace_instructions = annotations.inplace_instructions;
  EXPECT_THAT(inplace_instructions.size(), 1);

  auto* inst = *(inplace_instructions.begin());
  EXPECT_THAT(inst->name(), "u0");
}

TEST_F(HloInplaceDependencyTest, ScaledInplaceHighPriority) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  a = f32[20] parameter(0)
  b = f32[20] parameter(1)
  c = f32[] constant(2)
  c_bcast = f32[20] broadcast(f32[] %c), dimensions={}
  bc = f32[20] multiply(%b, %c_bcast)
  ROOT res = f32[20] add(%a, %bc),
  metadata={op_type="ResourceApplyGradientDescent" op_name="name"}
}

)";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  FuseOpsLate fuseOpsLate(annotations);
  EXPECT_TRUE(fuseOpsLate.Run(module0).ValueOrDie());
  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  // Make sure that the only inplace instruction is a call to scaled add to.
  auto& inplace_instructions = annotations.inplace_instructions;
  EXPECT_THAT(inplace_instructions.size(), 1);

  EXPECT_TRUE(module0->entry_computation()->root_instruction() ==
              *inplace_instructions.begin());
  EXPECT_TRUE(IsPopOpsFusion(module0->entry_computation()->root_instruction(),
                             "scaled_inplace"));
}

TEST_F(HloInplaceDependencyTest, InplaceInsideWhile) {
  const char* const hlo = R"(
HloModule ModuleWithWhile

body {
  p_body = (s32[],s32[]) parameter(0)
  p_body.0 = s32[] get-tuple-element((s32[],s32[]) p_body), index=0
  const = s32[] constant(1)
  add = s32[] add(p_body.0, const)
  p_body.1 = s32[] get-tuple-element((s32[],s32[]) p_body), index=1
  ROOT root = (s32[],s32[]) tuple(add, p_body.1)
}

condition {
  p_cond = (s32[],s32[]) parameter(0)
  p_cond.0 = s32[] get-tuple-element((s32[],s32[]) p_cond), index=0
  const = s32[] constant(10)
  ROOT result = pred[] less-than(p_cond.0, const)
}

ENTRY entry {
  const_0 = s32[] constant(0)
  const_1 = s32[] constant(10)
  while_init = (s32[],s32[]) tuple(const_0, const_1)
  ROOT while = (s32[],s32[]) while(while_init), condition=condition, body=body
}
)";

  auto module =
      HloRunner::CreateModuleFromString(hlo, GetDebugOptionsForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto& inplace_instructions = annotations.inplace_instructions;
  EXPECT_THAT(inplace_instructions.size(), 7);
  std::set<std::string> in_place_ops = {
      "p_body.2", "root", "p_body.1", "add", "p_cond.1", "while_init", "while"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
}

TEST_F(HloInplaceDependencyTest, CustomPoplibsOpInplace) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = s32[20] parameter(0)
  p1 = s32[20] parameter(1)

  c = s32[20] custom-call(p0, p1), custom_call_target="Popops::Sqrt", opaque="{\"allocating_indexes\":[],\"layout_dependencies\":{\"keys\":[],\"values\":[]},\"num_inplace_operands\":1}\n"

  ROOT t = (s32[20]) tuple(c)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(2);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = annotations.inplace_instructions;
  EXPECT_THAT(inplace_instructions.size(), 2);
  std::set<std::string> in_place_ops = {"c", "t"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
}

TEST_F(HloInplaceDependencyTest, CustomPoplibsOpNotInplace) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = s32[20] parameter(0)
  p1 = s32[20] parameter(1)

  c = s32[20] custom-call(p0, p1), custom_call_target="Popnn::LstmLayerFwd", opaque="{\"allocating_indexes\":[],\"layout_dependencies\":{\"keys\":[],\"values\":[]},\"num_inplace_operands\":0}\n"

  ROOT t = (s32[20]) tuple(c)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(2);
  config.set_resource_input_count(2);
  config.set_input_mapping({0, 1});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = annotations.inplace_instructions;

  EXPECT_THAT(inplace_instructions.size(), 1);
  auto* inst = *(inplace_instructions.begin());
  EXPECT_THAT(inst->name(), "t");
}

TEST_F(HloInplaceDependencyTest, TestAllGTEsInplace) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = (s32[20], s32[20], s32[20], s32[20]) parameter(0)
  p0_0 = s32[20] get-tuple-element(p0), index=0
  p0_1 = s32[20] get-tuple-element(p0), index=1
  p0_2 = s32[20] get-tuple-element(p0), index=2
  a = s32[20] add(p0_0, p0_1)
  ROOT a2 = s32[20] add(a, p0_2)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(4);
  config.set_resource_input_count(0);
  config.set_input_mapping({0});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = annotations.inplace_instructions;
  EXPECT_THAT(inplace_instructions.size(), 5);
  std::set<std::string> in_place_ops = {"p0_0", "p0_1", "p0_2", "a", "a2"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
}

TEST_F(HloInplaceDependencyTest, TestGTENotInplace) {
  std::string hlo = R"(
HloModule top

ENTRY c1 {
  p0 = (s32[20], s32[20], s32[20], s32[20]) parameter(0)
  p0_0 = s32[20] get-tuple-element(p0), index=0
  c = s32[20] custom-call(p0), custom_call_target="Popnn::LstmLayerFwd", opaque="{\"allocating_indexes\":[],\"layout_dependencies\":{\"keys\":[],\"values\":[]},\"num_inplace_operands\":0}\n"

  ROOT a = s32[20] add(p0_0, c)
}

)";

  auto config = GetModuleConfigForTest();
  config.set_argument_count(4);
  config.set_resource_input_count(2);
  config.set_input_mapping({0});
  config.set_resource_update_to_input_index({0});
  auto module = ParseHloString(hlo, config);
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  CompilerAnnotations annotations(module0);

  InplaceFinder inplaceFinder(annotations);
  EXPECT_TRUE(inplaceFinder.Run(module0).ValueOrDie());

  auto inplace_instructions = annotations.inplace_instructions;
  EXPECT_THAT(inplace_instructions.size(), 1);
  std::set<std::string> in_place_ops = {"a"};
  for (auto i : inplace_instructions) {
    EXPECT_TRUE(in_place_ops.count(i->name()));
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla

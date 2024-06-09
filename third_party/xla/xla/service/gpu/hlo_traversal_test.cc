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
#include "xla/service/gpu/hlo_traversal.h"

#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

using ::testing::ElementsAre;
using ::testing::IsEmpty;

MATCHER_P(InstructionAdaptorName, name, "") { return arg.name() == name; }

class HloTraversalTest : public HloTestBase {};

const char kTestModule[] = R"(
    HloModule test

    scalar_add_computation {
      scalar_lhs.0 = f32[] parameter(0)
      scalar_rhs.0 = f32[] parameter(1)
      ROOT add.0 = f32[] add(scalar_lhs.0, scalar_rhs.0)
    }

    fused_computation {
      p0.1 = f32[] parameter(0)
      p1.1 = f32[128] parameter(1)
      mul = f32[128] multiply(p1.1, p1.1)
      ROOT reduce.1 = f32[] reduce(mul, p0.1), dimensions={0}, to_apply=scalar_add_computation
    }

    fused_computation_1 {
      p0.2 = f32[] parameter(0)
      zero = f32[] constant(0.0)
      is_positive = pred[] compare(p0.2, zero), direction=GE
      not = pred[] not(is_positive)
      ROOT tuple = (pred[], pred[]) tuple(is_positive, not)
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[128] parameter(1)
      sum = f32[128] add(p1, p1)
      log = f32[128] log(sum)
      negate = f32[128] negate(log)
      fusion = f32[] fusion(p0, negate), kind=kLoop, calls=fused_computation
      fusion2 = (pred[], pred[]) fusion(fusion), kind=kLoop, calls=fused_computation_1
      gte = pred[] get-tuple-element(fusion2), index=0
      ROOT select = f32[] select(gte, fusion, p0)
    })";

TEST_F(HloTraversalTest, AdaptorOperands) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();

  auto fusion_adaptor = HloFusionAdaptor::ForProducerConsumer(
      module->entry_computation()->GetInstructionWithName("fusion2"),
      module->entry_computation()->GetInstructionWithName("select"));

  HloInstructionAdaptor instr = fusion_adaptor->GetRoots()[0];

  EXPECT_THAT(instr.GetOperands(),
              ElementsAre(InstructionAdaptorName("is_positive"),
                          InstructionAdaptorName("fusion"),
                          InstructionAdaptorName("p0")));
}

TEST_F(HloTraversalTest, AdaptorUsers) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    fused_computation {
      p0 = f32[] parameter(0)
      neg = f32[] negate(p0)
      add = f32[] add(p0, neg)
      ROOT t = (f32[], f32[]) tuple(neg, add)
    }

    fused_computation_1 {
      p0.0 = f32[] parameter(0)
      mul = f32[] multiply(p0.0, p0.0)
      ROOT neg.1 = f32[] negate(mul)
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      fusion = (f32[], f32[]) fusion(p0), kind=kLoop, calls=fused_computation
      gte = f32[] get-tuple-element(fusion), index=0
      add.1 = f32[] add(p0, gte)
      fusion2 = f32[] fusion(gte), kind=kLoop, calls=fused_computation_1
      exp.1 = f32[] exponential(fusion2)
      ROOT res = (f32[], (f32[], f32[]), f32[], f32[]) tuple(add.1, fusion, fusion2, exp.1)
    }
  )")
                    .value();

  auto fusion_adaptor1 = HloFusionAdaptor::ForProducerConsumer(
      module->entry_computation()->GetInstructionWithName("fusion"),
      module->entry_computation()->GetInstructionWithName("fusion2"));

  HloInstructionAdaptor add{*module->GetComputationWithName("fused_computation")
                                 ->GetInstructionWithName("add"),
                            fusion_adaptor1.get()};
  EXPECT_THAT(add.GetUsers(), ElementsAre(InstructionAdaptorName("add.1"),
                                          InstructionAdaptorName("mul"),
                                          InstructionAdaptorName("res")));

  auto fusion_adaptor2 = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion2"));

  HloInstructionAdaptor mul{
      *module->GetComputationWithName("fused_computation_1")
           ->GetInstructionWithName("mul"),
      fusion_adaptor2.get()};
  EXPECT_THAT(mul.GetUsers(), ElementsAre(InstructionAdaptorName("neg.1")));

  HloInstructionAdaptor neg{
      *module->GetComputationWithName("fused_computation_1")
           ->GetInstructionWithName("neg.1"),
      fusion_adaptor2.get()};
  EXPECT_THAT(neg.GetUsers(), ElementsAre(InstructionAdaptorName("exp.1")));
}

TEST_F(HloTraversalTest, TraverseFusionConsumerFirst) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> visited_nodes;
  std::vector<std::string> visited_args;
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  HloBfsConsumersFirstTraversal(
      fusion->GetRoots(), *fusion,
      [&](HloInstructionAdaptor node) {
        visited_nodes.emplace_back(node.name());
        return TraversalResult::kAdvance;
      },
      [&](HloInstructionAdaptor arg) {
        visited_args.emplace_back(arg.name());
      });

  EXPECT_THAT(visited_nodes, ElementsAre("reduce.1", "mul"));
  EXPECT_THAT(visited_args, ElementsAre("p0", "negate"));
}

TEST_F(HloTraversalTest,
       TraverseFusionConsumerFirstFromFusionRootAndInnerNode) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> visited_nodes;
  std::vector<std::string> visited_args;
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  auto root = fusion->GetRoots()[0];
  HloBfsConsumersFirstTraversal(
      {root, root.GetOperand(0)}, *fusion,
      [&](HloInstructionAdaptor node) {
        visited_nodes.emplace_back(node.name());
        return TraversalResult::kAdvance;
      },
      [&](HloInstructionAdaptor arg) {
        visited_args.emplace_back(arg.name());
      });

  EXPECT_THAT(visited_nodes, ElementsAre("reduce.1", "mul"));
  EXPECT_THAT(visited_args, ElementsAre("p0", "negate"));
}

TEST_F(HloTraversalTest, TraverseFusionProducerFirst) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  std::vector<std::string> visited_nodes;
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  auto root = fusion->GetRoots()[0];
  HloBfsProducersFirstTraversal({root.GetOperand(0)}, *fusion,
                                [&](HloInstructionAdaptor node) {
                                  visited_nodes.emplace_back(node.name());
                                  return TraversalResult::kAdvance;
                                });

  EXPECT_THAT(visited_nodes, ElementsAre("mul", "reduce.1"));
}

TEST_F(HloTraversalTest, AbortTraversal) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  std::vector<std::string> visited_nodes;
  HloBfsConsumersFirstTraversal(fusion->GetRoots(), *fusion,
                                [&](HloInstructionAdaptor node) {
                                  visited_nodes.emplace_back(node.name());
                                  return node.opcode() == HloOpcode::kReduce
                                             ? TraversalResult::kAdvance
                                             : TraversalResult::kInterrupt;
                                });

  EXPECT_THAT(visited_nodes, ElementsAre("reduce.1", "mul"));
}

TEST_F(HloTraversalTest, FindArguments) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  std::vector<std::string> producers;
  absl::c_for_each(fusion->GetParameters(),
                   [&](const HloInstruction* producer) {
                     producers.emplace_back(producer->name());
                   });

  EXPECT_THAT(producers, ElementsAre("p0", "negate"));
}

TEST_F(HloTraversalTest, FindArgumentsAfterFusion) {
  // Verifies that we correctly find the arguments after fusing the negation.
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  auto fusion = HloFusionAdaptor::ForProducerConsumer(
      module->entry_computation()->GetInstructionWithName("negate"),
      module->entry_computation()->GetInstructionWithName("fusion"));
  std::vector<std::string> producers;
  absl::c_for_each(fusion->GetParameters(),
                   [&](const HloInstruction* producer) {
                     producers.emplace_back(producer->name());
                   });
  EXPECT_THAT(producers, ElementsAre("p0", "log"));
}

TEST_F(HloTraversalTest, FindIf) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  auto result =
      HloFindIf(fusion->GetRoots(), *fusion, [&](HloInstructionAdaptor node) {
        return node.opcode() == HloOpcode::kMultiply;
      });
  ASSERT_NE(result, std::nullopt);
  ASSERT_EQ(result->name(), "mul");
}

TEST_F(HloTraversalTest, NotFound) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  auto result = HloFindIf(fusion->GetRoots(), *fusion,
                          [&](HloInstructionAdaptor node) { return false; });
  ASSERT_EQ(result, std::nullopt);
}

const char kTwoFusions[] = R"(
    HloModule test

    scalar_add_computation {
      scalar_lhs.0 = f32[] parameter(0)
      scalar_rhs.0 = f32[] parameter(1)
      ROOT add.0 = f32[] add(scalar_lhs.0, scalar_rhs.0)
    }

    fused_computation_1 {
      p0.1 = f32[] parameter(0)
      p1.1 = f32[128] parameter(1)
      mul = f32[128] multiply(p1.1, p1.1)
      ROOT reduce.1 = f32[] reduce(mul, p0.1), dimensions={0}, to_apply=scalar_add_computation
    }

    fused_computation_2 {
      p0.2 = f32[] parameter(0)
      p1.2 = f32[128] parameter(1)
      ROOT reduce.2 = f32[] reduce(p1.2, p0.2), dimensions={0}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[128] parameter(1)
      sum = f32[128] add(p1, p1)
      negate = f32[128] negate(sum)
      fusion.1 = f32[] fusion(p0, negate), kind=kLoop, calls=fused_computation_1
      fusion.2 = f32[] fusion(fusion.1, negate), kind=kLoop, calls=fused_computation_2
      ROOT difference = f32[] subtract(fusion.2, p0)
    })";

TEST_F(HloTraversalTest, FuseFusionConsumer) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();

  auto producer = module->entry_computation()->GetInstructionWithName("negate");
  auto consumer =
      module->entry_computation()->GetInstructionWithName("fusion.1");
  auto fusion = HloFusionAdaptor::ForProducerConsumer(producer, consumer);

  HloInstructionAdaptor reduce_1(
      *module->GetComputationWithName("fused_computation_1")
           ->GetInstructionWithName("reduce.1"),
      fusion.get());

  EXPECT_THAT(reduce_1.GetUsers(),
              ElementsAre(InstructionAdaptorName("fusion.2")));

  std::vector<std::string> nodes;
  std::vector<std::string> params;
  HloBfsConsumersFirstTraversal(
      fusion->GetRoots(), *fusion,
      [&](HloInstructionAdaptor node) {
        nodes.emplace_back(node.name());
        return TraversalResult::kAdvance;
      },
      [&](HloInstructionAdaptor param) { params.emplace_back(param.name()); });

  EXPECT_THAT(nodes, ElementsAre("reduce.1", "mul", "negate"));
  EXPECT_THAT(params, ElementsAre("p0", "sum"));
}

TEST_F(HloTraversalTest, FuseFusionProducer) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();

  auto producer =
      module->entry_computation()->GetInstructionWithName("fusion.2");
  auto consumer =
      module->entry_computation()->GetInstructionWithName("difference");
  auto fusion = HloFusionAdaptor::ForProducerConsumer(producer, consumer);

  HloInstructionAdaptor reduce_2(
      *module->GetComputationWithName("fused_computation_2")
           ->GetInstructionWithName("reduce.2"),
      fusion.get());

  EXPECT_THAT(reduce_2.GetOperands(),
              ElementsAre(InstructionAdaptorName("negate"),
                          InstructionAdaptorName("fusion.1")));

  std::vector<std::string> nodes;
  std::vector<std::string> params;
  HloBfsConsumersFirstTraversal(
      fusion->GetRoots(), *fusion,
      [&](HloInstructionAdaptor node) {
        nodes.emplace_back(node.name());
        return TraversalResult::kAdvance;
      },
      [&](HloInstructionAdaptor arg) { params.emplace_back(arg.name()); });

  EXPECT_THAT(nodes, ElementsAre("difference", "reduce.2"));
  EXPECT_THAT(params, ElementsAre("p0", "negate", "fusion.1"));
}

TEST_F(HloTraversalTest, FuseFusionConsumerAndProducer) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();
  auto producer =
      module->entry_computation()->GetInstructionWithName("fusion.1");
  auto consumer =
      module->entry_computation()->GetInstructionWithName("fusion.2");
  auto fusion = HloFusionAdaptor::ForProducerConsumer(producer, consumer);

  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal(fusion->GetRoots(), *fusion,
                                [&](HloInstructionAdaptor node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kAdvance;
                                });
  std::vector<std::string> params;
  absl::c_for_each(fusion->GetParameters(), [&](const HloInstruction* param) {
    params.emplace_back(param->name());
  });

  EXPECT_THAT(nodes, ElementsAre("reduce.2", "reduce.1", "mul"));
  EXPECT_THAT(params, ElementsAre("negate", "p0"));
}

TEST_F(HloTraversalTest, FuseNonFusionConsumerAndProducer) {
  auto module = ParseAndReturnVerifiedModule(kTestModule).value();

  auto producer = module->entry_computation()->GetInstructionWithName("log");
  auto consumer = module->entry_computation()->GetInstructionWithName("negate");
  auto fusion = HloFusionAdaptor::ForProducerConsumer(producer, consumer);

  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal(fusion->GetRoots(), *fusion,
                                [&](HloInstructionAdaptor node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kAdvance;
                                });

  EXPECT_THAT(nodes, ElementsAre("negate", "log"));
}

TEST_F(HloTraversalTest, SingleInstructionFusionOfFusion) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion.1"));

  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal(fusion->GetRoots(), *fusion,
                                [&](HloInstructionAdaptor node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kAdvance;
                                });

  EXPECT_THAT(nodes, ElementsAre("reduce.1", "mul"));
}

TEST_F(HloTraversalTest, SingleInstructionFusionOfInstruction) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("negate"));

  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal(fusion->GetRoots(), *fusion,
                                [&](HloInstructionAdaptor node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kAdvance;
                                });

  EXPECT_THAT(nodes, ElementsAre("negate"));
}

TEST_F(HloTraversalTest, MultiOutputFusionDuplicateRoot) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    fused_computation {
      p0.1 = f32[128] parameter(0)
      p1.1 = f32[128] parameter(1)
      mul = f32[128] multiply(p0.1, p1.1)
      ROOT res = (f32[128], f32[128]) tuple(mul, mul)
    }

    ENTRY entry {
      p0 = f32[128] parameter(0)
      p1 = f32[128] parameter(1)
      ROOT fusion = (f32[128], f32[128]) fusion(p0, p1), kind=kLoop, calls=fused_computation
    })")
                    .value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  EXPECT_THAT(fusion->GetRoots(), ElementsAre(InstructionAdaptorName("mul"),
                                              InstructionAdaptorName("mul")));
}

TEST_F(HloTraversalTest, MakeInstructionsPostOrder_SingleInstruction) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("negate"));

  auto nodes = fusion->MakeInstructionPostOrder();
  EXPECT_THAT(nodes, ElementsAre(InstructionAdaptorName("negate")));
}

TEST_F(HloTraversalTest, MakeInstructionsPostOrder_TwoFusions) {
  auto module = ParseAndReturnVerifiedModule(kTwoFusions).value();
  auto fusion = HloFusionAdaptor::ForProducerConsumer(
      module->entry_computation()->GetInstructionWithName("fusion.1"),
      module->entry_computation()->GetInstructionWithName("fusion.2"));

  auto nodes = fusion->MakeInstructionPostOrder();
  EXPECT_THAT(nodes, ElementsAre(InstructionAdaptorName("mul"),
                                 InstructionAdaptorName("reduce.1"),
                                 InstructionAdaptorName("reduce.2")));
}

TEST_F(HloTraversalTest, MakeInstructionsPostOrder_TwoMultiOutputFusions) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test

    scalar_add_computation {
      scalar_lhs.0 = f32[] parameter(0)
      scalar_rhs.0 = f32[] parameter(1)
      ROOT add.0 = f32[] add(scalar_lhs.0, scalar_rhs.0)
    }

    fused_computation_1 {
      p0.1 = f32[] parameter(0)
      p1.1 = f32[128] parameter(1)
      mul = f32[128] multiply(p1.1, p1.1)
      reduce.1 = f32[] reduce(mul, p0.1), dimensions={0}, to_apply=scalar_add_computation
      ROOT t = (f32[128], f32[]) tuple(mul, reduce.1)
    }

    fused_computation_2 {
      p0.2 = f32[] parameter(0)
      p1.2 = f32[128] parameter(1)
      neg = f32[128] negate(p1.2)
      reduce.2 = f32[] reduce(neg, p0.2), dimensions={0}, to_apply=scalar_add_computation
      ROOT t2 = (f32[], f32[128]) tuple(reduce.2, neg)
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[128] parameter(1)
      sum = f32[128] add(p1, p1)
      negate = f32[128] negate(sum)
      fusion.1 = (f32[128], f32[]) fusion(p0, negate), kind=kLoop, calls=fused_computation_1
      gte1 = f32[128] get-tuple-element(fusion.1), index=0
      gte2 = f32[] get-tuple-element(fusion.1), index=1
      fusion.2 = (f32[], f32[128]) fusion(p0, gte1), kind=kLoop, calls=fused_computation_2
      gte3 = f32[] get-tuple-element(fusion.2), index=0
      gte4 = f32[128] get-tuple-element(fusion.2), index=1
      difference = f32[] subtract(gte3, p0)
      ROOT res = (f32[], f32[128]) tuple(difference, gte4)
    })")
                    .value();
  auto fusion = HloFusionAdaptor::ForProducerConsumer(
      module->entry_computation()->GetInstructionWithName("fusion.1"),
      module->entry_computation()->GetInstructionWithName("fusion.2"));

  auto nodes = fusion->MakeInstructionPostOrder();
  EXPECT_THAT(nodes, ElementsAre(InstructionAdaptorName("mul"),
                                 InstructionAdaptorName("reduce.1"),
                                 InstructionAdaptorName("neg"),
                                 InstructionAdaptorName("reduce.2")));
}

const char kTwoMultiOutputFusions[] = R"(
    HloModule mof
    mof_producer {
      param0 = f32[10]{0} parameter(0)
      param1 = f32[10]{0} parameter(1)
      param2 = f32[10]{0} parameter(2)
      add = f32[10]{0} add(param0, param1)
      sub = f32[10]{0} subtract(param0, param1)
      ROOT res = (f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}) tuple(param1, add, sub, param0, param2)
    }

    mof_consumer {
      param0.0 = f32[10]{0} parameter(0)
      param1.0 = f32[10]{0} parameter(1)
      param2.0 = f32[10]{0} parameter(2)
      mul = f32[10]{0} multiply(param0.0, param1.0)
      div = f32[10]{0} divide(param0.0, param1.0)
      ROOT res = (f32[10]{0}, f32[10]{0}, f32[10]{0}) tuple(mul, div, param2.0)
    }

    ENTRY main {
      p0 = f32[10]{0} parameter(0)
      p1 = f32[10]{0} parameter(1)
      p2 = f32[10]{0} parameter(2)
      producer = (f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}) fusion(p0, p1, p2), kind=kLoop, calls=mof_producer
      gte0 = f32[10]{0} get-tuple-element(producer), index=0
      gte1 = f32[10]{0} get-tuple-element(producer), index=1
      gte2 = f32[10]{0} get-tuple-element(producer), index=2
      gte3 = f32[10]{0} get-tuple-element(producer), index=3
      gte4 = f32[10]{0} get-tuple-element(producer), index=4
      consumer = (f32[10]{0}, f32[10]{0}, f32[10]{0}) fusion(gte1, gte2, gte3), kind=kLoop, calls=mof_consumer
      gte5 = f32[10]{0} get-tuple-element(consumer), index=0
      gte6 = f32[10]{0} get-tuple-element(consumer), index=1
      gte7 = f32[10]{0} get-tuple-element(consumer), index=2
      ROOT res = tuple(gte0, gte1, gte3, gte4, gte5, gte6, gte7)
    })";

TEST_F(HloTraversalTest, GetParametersMultiOutputFusion) {
  auto module = ParseAndReturnVerifiedModule(kTwoMultiOutputFusions).value();
  auto producer =
      module->entry_computation()->GetInstructionWithName("producer");
  auto consumer =
      module->entry_computation()->GetInstructionWithName("consumer");
  auto fusion_adaptor =
      HloFusionAdaptor::ForProducerConsumer(producer, consumer);
  auto p0 = module->entry_computation()->GetInstructionWithName("p0");
  auto p1 = module->entry_computation()->GetInstructionWithName("p1");
  EXPECT_THAT(fusion_adaptor->GetParameters(), ElementsAre(p0, p1));
  // Double-check that after performing the actual fusion, we get the same
  // parameters.
  consumer->MergeFusionInstructionIntoMultiOutput(producer);
  EXPECT_THAT(consumer->operands(), ElementsAre(p0, p1));
}

TEST_F(HloTraversalTest, GetRootsMultiOutputFusion) {
  auto module = ParseAndReturnVerifiedModule(kTwoMultiOutputFusions).value();
  auto consumer_fusion_instr =
      module->entry_computation()->GetInstructionWithName("consumer");
  auto producer_fusion_instr =
      module->entry_computation()->GetInstructionWithName("producer");
  auto fusion_adaptor = HloFusionAdaptor::ForProducerConsumer(
      producer_fusion_instr, consumer_fusion_instr);
  auto producer_computation = module->GetComputationWithName("mof_producer");
  auto producer = HloFusionAdaptor::ForComputation(producer_computation);
  auto consumer_computation = module->GetComputationWithName("mof_consumer");
  auto consumer = HloFusionAdaptor::ForComputation(consumer_computation);
  EXPECT_THAT(fusion_adaptor->GetRoots(),
              ElementsAre(
                  HloInstructionAdaptor{
                      *consumer_computation->GetInstructionWithName("mul"),
                      consumer.get()},
                  HloInstructionAdaptor{
                      *consumer_computation->GetInstructionWithName("div"),
                      consumer.get()},
                  HloInstructionAdaptor{
                      *producer_computation->GetInstructionWithName("param0"),
                      producer.get()},
                  HloInstructionAdaptor{
                      *producer_computation->GetInstructionWithName("add"),
                      producer.get()}));
  // Double-check that after performing the actual fusion, we get the same
  // roots.
  consumer_fusion_instr->MergeFusionInstructionIntoMultiOutput(
      producer_fusion_instr);
  EXPECT_THAT(consumer_fusion_instr->fused_expression_root(),
              GmockMatch(m::Tuple(
                  m::Multiply(m::Add(m::Parameter(0), m::Parameter(1)),
                              m::Subtract(m::Parameter(0), m::Parameter(1))),
                  m::Divide(m::Add(m::Parameter(0), m::Parameter(1)),
                            m::Subtract(m::Parameter(0), m::Parameter(1))),
                  m::Parameter(0), m::Add(m::Parameter(0), m::Parameter(1)))));
}

TEST_F(HloTraversalTest, HloFindUseChain) {
  auto module = ParseAndReturnVerifiedModule(R"(
    fusion {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      negate = f32[] negate(p0)
      log = f32[] log(p0)
      sum = f32[] add(p0, log)
      exp = f32[] exponential(p1)
      ROOT call = f32[] custom-call(negate, exp, sum), custom_call_target="it"
    }

    ENTRY main {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT fusion = f32[] fusion(p0, p1), kind=kLoop, calls=fusion
    }
    )")
                    .value();

  auto* fusion_computation = module->GetComputationWithName("fusion");
  auto fusion = HloFusionAdaptor::ForComputation(fusion_computation);
  auto get = [&](absl::string_view name) {
    return HloInstructionAdaptor{
        *fusion_computation->GetInstructionWithName(name), fusion.get()};
  };
  auto p0 = get("p0");
  auto p1 = get("p1");
  auto log = get("log");
  auto sum = get("sum");
  auto negate = get("negate");
  auto exp = get("exp");
  auto call = get("call");

  EXPECT_THAT(HloFindUseChain(p0, p0), ElementsAre(p0));
  EXPECT_THAT(HloFindUseChain(p0, p1), IsEmpty());
  EXPECT_THAT(HloFindUseChain(p0, call), ElementsAre(p0, negate, call));
  EXPECT_THAT(HloFindUseChain(p0, sum), ElementsAre(p0, log, sum));
  EXPECT_THAT(HloFindUseChain(p1, exp), ElementsAre(p1, exp));
  EXPECT_THAT(HloFindUseChain(negate, exp), IsEmpty());
  EXPECT_THAT(HloFindUseChain(call, p0), IsEmpty());
}

}  // namespace
}  // namespace gpu
}  // namespace xla

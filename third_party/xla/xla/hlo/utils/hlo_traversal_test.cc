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
#include "xla/hlo/utils/hlo_traversal.h"

#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

// Pretty-print HloInstructionAdaptor.
template <typename Sink>
void AbslStringify(Sink& sink, const HloInstructionAdaptor& adaptor) {
  absl::Format(&sink, "%s", adaptor.ToString());
}

namespace {

namespace m = ::xla::match;

using ::testing::ElementsAre;
using ::testing::IsEmpty;

MATCHER_P(InstructionAdaptorName, name, "") { return arg.name() == name; }

class HloTraversalTest : public HloTestBase {};

const char kTestModule[] = R"(
    accumulate {
      p0.0 = f32[] parameter(0)
      p1.0 = f32[] parameter(1)
      ROOT add = f32[] add(p0.0, p1.0)
    }

    computation1 {
      p0.1 = f32[] parameter(0)
      p1.1 = f32[128] parameter(1)
      mul = f32[128] multiply(p1.1, p1.1)
      ROOT reduce = f32[] reduce(mul, p0.1), dimensions={0}, to_apply=accumulate
    }

    computation2 {
      p0.2 = f32[] parameter(0)
      zero = f32[] constant(0.0)
      is_positive = pred[] compare(p0.2, zero), direction=GE
      not = pred[] not(is_positive)
      ROOT tuple = (pred[], pred[]) tuple(is_positive, not)
    }

    ENTRY entry {
      p0.3 = f32[] parameter(0)
      p1.3 = f32[128] parameter(1)
      sum = f32[128] add(p1.3, p1.3)
      log = f32[128] log(sum)
      negate = f32[128] negate(log)
      fusion1 = f32[] fusion(p0.3, negate), kind=kLoop, calls=computation1
      fusion2 = (pred[], pred[]) fusion(fusion1), kind=kLoop, calls=computation2
      gte = pred[] get-tuple-element(fusion2), index=0
      ROOT select = f32[] select(gte, fusion1, p0.3)
    })";

TEST_F(HloTraversalTest, AdaptorOperands) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTestModule));

  auto fusion_adaptor = HloFusionAdaptor::ForProducerConsumer(
      module->entry_computation()->GetInstructionWithName("fusion2"),
      module->entry_computation()->GetInstructionWithName("select"));

  HloInstructionAdaptor select = fusion_adaptor->GetRoots()[0];
  EXPECT_THAT(select.GetOperands(),
              ElementsAre(InstructionAdaptorName("is_positive"),
                          InstructionAdaptorName("fusion1"),
                          InstructionAdaptorName("p0.3")));
}

TEST_F(HloTraversalTest, AdaptorUsers) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    computation1 {
      p0.1 = f32[] parameter(0)
      neg.1 = f32[] negate(p0.1)
      add.1 = f32[] add(p0.1, neg.1)
      ROOT tuple.1 = (f32[], f32[]) tuple(neg.1, add.1)
    }

    computation2 {
      p0.2 = f32[] parameter(0)
      mul = f32[] multiply(p0.2, p0.2)
      ROOT neg.2 = f32[] negate(mul)
    }

    ENTRY entry {
      p0.3 = f32[] parameter(0)
      fusion1 = (f32[], f32[]) fusion(p0.3), kind=kLoop, calls=computation1
      gte = f32[] get-tuple-element(fusion1), index=0
      add.3 = f32[] add(p0.3, gte)
      fusion2 = f32[] fusion(gte), kind=kLoop, calls=computation2
      exp = f32[] exponential(fusion2)
      ROOT tuple.3 = (f32[], (f32[], f32[]), f32[], f32[]) tuple(add.3, fusion1, fusion2, exp)
    }
  )"));

  auto fusion_adaptor1 = HloFusionAdaptor::ForProducerConsumer(
      module->entry_computation()->GetInstructionWithName("fusion1"),
      module->entry_computation()->GetInstructionWithName("fusion2"));

  HloInstructionAdaptor add{*module->GetComputationWithName("computation1")
                                 ->GetInstructionWithName("add.1"),
                            fusion_adaptor1.get()};
  EXPECT_THAT(add.GetUsers(), ElementsAre(InstructionAdaptorName("mul")));

  auto fusion_adaptor2 = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion2"));

  HloInstructionAdaptor mul{*module->GetComputationWithName("computation2")
                                 ->GetInstructionWithName("mul"),
                            fusion_adaptor2.get()};
  EXPECT_THAT(mul.GetUsers(), ElementsAre(InstructionAdaptorName("neg.2")));

  HloInstructionAdaptor neg{*module->GetComputationWithName("computation2")
                                 ->GetInstructionWithName("neg.2"),
                            fusion_adaptor2.get()};
  EXPECT_TRUE(neg.GetUsers().empty());
}

TEST_F(HloTraversalTest, TraverseFusionConsumerFirst) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTestModule));
  std::vector<std::string> visited_nodes;
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion1"));
  HloBfsConsumersFirstTraversal(fusion->GetRoots(), *fusion,
                                [&](HloInstructionAdaptor node) {
                                  visited_nodes.emplace_back(node.name());
                                  return TraversalResult::kAdvance;
                                });

  EXPECT_THAT(visited_nodes, ElementsAre("reduce", "mul"));
}

TEST_F(HloTraversalTest,
       TraverseFusionConsumerFirstFromFusionRootAndInnerNode) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTestModule));
  std::vector<std::string> visited_nodes;
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion1"));
  auto root = fusion->GetRoots()[0];
  HloBfsConsumersFirstTraversal({root, root.GetOperand(0)}, *fusion,
                                [&](HloInstructionAdaptor node) {
                                  visited_nodes.emplace_back(node.name());
                                  return TraversalResult::kAdvance;
                                });

  EXPECT_THAT(visited_nodes, ElementsAre("reduce", "mul"));
}

TEST_F(HloTraversalTest, TraverseFusionProducerFirst) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTestModule));
  std::vector<std::string> visited_nodes;
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion1"));
  auto root = fusion->GetRoots()[0];
  HloBfsProducersFirstTraversal({root.GetOperand(0)}, *fusion,
                                [&](HloInstructionAdaptor node) {
                                  visited_nodes.emplace_back(node.name());
                                  return TraversalResult::kAdvance;
                                });

  EXPECT_THAT(visited_nodes, ElementsAre("mul", "reduce"));
}

TEST_F(HloTraversalTest, AbortTraversal) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTestModule));
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion1"));
  std::vector<std::string> visited_nodes;
  HloBfsConsumersFirstTraversal(fusion->GetRoots(), *fusion,
                                [&](HloInstructionAdaptor node) {
                                  visited_nodes.emplace_back(node.name());
                                  return node.opcode() == HloOpcode::kReduce
                                             ? TraversalResult::kAdvance
                                             : TraversalResult::kInterrupt;
                                });

  EXPECT_THAT(visited_nodes, ElementsAre("reduce", "mul"));
}

TEST_F(HloTraversalTest, FindArguments) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTestModule));
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion1"));
  std::vector<std::string> producers;
  absl::c_for_each(fusion->GetParameters(),
                   [&](const HloInstruction* producer) {
                     producers.emplace_back(producer->name());
                   });

  EXPECT_THAT(producers, ElementsAre("p0.3", "negate"));
}

TEST_F(HloTraversalTest, FindArgumentsAfterFusion) {
  // Verifies that we correctly find the arguments after fusing the negation.
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTestModule));
  auto fusion = HloFusionAdaptor::ForProducerConsumer(
      module->entry_computation()->GetInstructionWithName("negate"),
      module->entry_computation()->GetInstructionWithName("fusion1"));
  std::vector<std::string> producers;
  absl::c_for_each(fusion->GetParameters(),
                   [&](const HloInstruction* producer) {
                     producers.emplace_back(producer->name());
                   });
  EXPECT_THAT(producers, ElementsAre("p0.3", "log"));
}

TEST_F(HloTraversalTest, HloBfsFindIf_Found) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTestModule));
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion1"));
  auto result = HloBfsFindIf(fusion->GetRoots(), *fusion,
                             [&](HloInstructionAdaptor node) {
                               return node.opcode() == HloOpcode::kMultiply;
                             });
  ASSERT_NE(result, std::nullopt);
  ASSERT_EQ(result->name(), "mul");
}

TEST_F(HloTraversalTest, HloBfsFindIf_NotFound) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTestModule));
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion1"));
  auto result = HloBfsFindIf(fusion->GetRoots(), *fusion,
                             [&](HloInstructionAdaptor node) { return false; });
  ASSERT_EQ(result, std::nullopt);
}

TEST_F(HloTraversalTest, HloAnyOf_Found) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTestModule));
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion1"));
  EXPECT_TRUE(HloAnyOf(*fusion, [&](HloInstructionAdaptor node) {
    return node.opcode() == HloOpcode::kMultiply;
  }));
}

TEST_F(HloTraversalTest, HloAnyOf_NotFound) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTestModule));
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion1"));
  EXPECT_FALSE(
      HloAnyOf(*fusion, [&](HloInstructionAdaptor node) { return false; }));
}

TEST_F(HloTraversalTest, FindAllMultiple) {
  const char kConverts[] = R"(
    ENTRY entry {
      p0 = s8[128] parameter(0)
      p1 = pred[128] parameter(1)
      p0.f16 = f16[128] convert(p0)
      p1.s8 = s8[128] convert(p1)
      p1.f16 = f16[128] convert(p1.s8)
      ROOT diff = f16[128] subtract(p0.f16, p1.f16)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kConverts));
  auto root = module->entry_computation()->GetInstructionWithName("diff");
  std::vector<const HloInstruction*> converts =
      HloBfsFindAll({root}, [&](const HloInstruction* node) {
        return node->opcode() == HloOpcode::kConvert;
      });

  auto get = [&](absl::string_view name) {
    return module->entry_computation()->GetInstructionWithName(name);
  };

  EXPECT_THAT(converts,
              ElementsAre(get("p0.f16"), get("p1.f16"), get("p1.s8")));
}

TEST_F(HloTraversalTest, FindAllNotFound) {
  const char kConverts[] = R"(
    ENTRY entry {
      p0 = s8[128] parameter(0)
      p1 = f16[128] parameter(1)
      p0f16 = f16[128] convert(p0)
      ROOT diff = f16[128] subtract(p0f16, p1)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kConverts));
  auto root = module->entry_computation()->GetInstructionWithName("diff");
  std::vector<const HloInstruction*> converts =
      HloBfsFindAll({root}, [&](const HloInstruction* node) {
        return node->opcode() == HloOpcode::kAdd;
      });
  EXPECT_THAT(converts, IsEmpty());
}

const char kTwoFusions[] = R"(
    accumulate {
      p0.0 = f32[] parameter(0)
      p1.0 = f32[] parameter(1)
      ROOT add = f32[] add(p0.0, p1.0)
    }

    computation1 {
      p0.1 = f32[] parameter(0)
      p1.1 = f32[128] parameter(1)
      mul = f32[128] multiply(p1.1, p1.1)
      ROOT reduce.1 = f32[] reduce(mul, p0.1), dimensions={0}, to_apply=accumulate
    }

    computation2 {
      p0.2 = f32[] parameter(0)
      p1.2 = f32[128] parameter(1)
      ROOT reduce.2 = f32[] reduce(p1.2, p0.2), dimensions={0}, to_apply=accumulate
    }

    ENTRY entry {
      p0.3 = f32[] parameter(0)
      p1.3 = f32[128] parameter(1)
      sum = f32[128] add(p1.3, p1.3)
      negate = f32[128] negate(sum)
      fusion1 = f32[] fusion(p0.3, negate), kind=kLoop, calls=computation1
      fusion2 = f32[] fusion(fusion1, negate), kind=kLoop, calls=computation2
      ROOT difference = f32[] subtract(fusion2, p0.3)
    })";

TEST_F(HloTraversalTest, FuseFusionConsumer) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTwoFusions));

  auto producer = module->entry_computation()->GetInstructionWithName("negate");
  auto consumer =
      module->entry_computation()->GetInstructionWithName("fusion1");
  auto fusion = HloFusionAdaptor::ForProducerConsumer(producer, consumer);

  HloInstructionAdaptor reduce_1(*module->GetComputationWithName("computation1")
                                      ->GetInstructionWithName("reduce.1"),
                                 fusion.get());

  EXPECT_TRUE(reduce_1.GetUsers().empty());

  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal(fusion->GetRoots(), *fusion,
                                [&](HloInstructionAdaptor node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kAdvance;
                                });

  EXPECT_THAT(nodes, ElementsAre("reduce.1", "mul", "negate"));
}

TEST_F(HloTraversalTest, FuseFusionProducer) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTwoFusions));

  auto producer =
      module->entry_computation()->GetInstructionWithName("fusion2");
  auto consumer =
      module->entry_computation()->GetInstructionWithName("difference");
  auto fusion = HloFusionAdaptor::ForProducerConsumer(producer, consumer);

  HloInstructionAdaptor reduce_2(*module->GetComputationWithName("computation2")
                                      ->GetInstructionWithName("reduce.2"),
                                 fusion.get());

  EXPECT_THAT(reduce_2.GetOperands(),
              ElementsAre(InstructionAdaptorName("negate"),
                          InstructionAdaptorName("fusion1")));

  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal(fusion->GetRoots(), *fusion,
                                [&](HloInstructionAdaptor node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kAdvance;
                                });

  EXPECT_THAT(nodes, ElementsAre("difference", "reduce.2"));
}

TEST_F(HloTraversalTest, FuseFusionConsumerAndProducer) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTwoFusions));
  auto producer =
      module->entry_computation()->GetInstructionWithName("fusion1");
  auto consumer =
      module->entry_computation()->GetInstructionWithName("fusion2");
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
  EXPECT_THAT(params, ElementsAre("negate", "p0.3"));
}

TEST_F(HloTraversalTest, FuseNonFusionConsumerAndProducer) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTestModule));

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
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTwoFusions));
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion1"));

  std::vector<std::string> nodes;
  HloBfsConsumersFirstTraversal(fusion->GetRoots(), *fusion,
                                [&](HloInstructionAdaptor node) {
                                  nodes.emplace_back(node.name());
                                  return TraversalResult::kAdvance;
                                });

  EXPECT_THAT(nodes, ElementsAre("reduce.1", "mul"));
}

TEST_F(HloTraversalTest, SingleInstructionFusionOfInstruction) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTwoFusions));
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    computation {
      p0.1 = f32[128] parameter(0)
      p1.1 = f32[128] parameter(1)
      mul = f32[128] multiply(p0.1, p1.1)
      ROOT tuple = (f32[128], f32[128]) tuple(mul, mul)
    }

    ENTRY entry {
      p0.2 = f32[128] parameter(0)
      p1.2 = f32[128] parameter(1)
      ROOT fusion = (f32[128], f32[128]) fusion(p0.2, p1.2), kind=kLoop, calls=computation
    })"));
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("fusion"));
  EXPECT_THAT(fusion->GetRoots(), ElementsAre(InstructionAdaptorName("mul"),
                                              InstructionAdaptorName("mul")));
}

TEST_F(HloTraversalTest, MakeInstructionsPostOrder_SingleInstruction) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTwoFusions));
  auto fusion = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->GetInstructionWithName("negate"));

  auto nodes = fusion->MakeInstructionPostOrder();
  EXPECT_THAT(nodes, ElementsAre(InstructionAdaptorName("negate")));
}

TEST_F(HloTraversalTest, MakeInstructionsPostOrder_TwoFusions) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTwoFusions));
  auto fusion = HloFusionAdaptor::ForProducerConsumer(
      module->entry_computation()->GetInstructionWithName("fusion1"),
      module->entry_computation()->GetInstructionWithName("fusion2"));

  auto nodes = fusion->MakeInstructionPostOrder();
  EXPECT_THAT(nodes, ElementsAre(InstructionAdaptorName("mul"),
                                 InstructionAdaptorName("reduce.1"),
                                 InstructionAdaptorName("reduce.2")));
}

TEST_F(HloTraversalTest, MakeInstructionsPostOrder_TwoMultiOutputFusions) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    accumulate {
      p0.0 = f32[] parameter(0)
      p1.0 = f32[] parameter(1)
      ROOT add = f32[] add(p0.0, p1.0)
    }

    computation1 {
      p0.1 = f32[] parameter(0)
      p1.1 = f32[128] parameter(1)
      mul = f32[128] multiply(p1.1, p1.1)
      reduce.1 = f32[] reduce(mul, p0.1), dimensions={0}, to_apply=accumulate
      ROOT tuple.1 = (f32[128], f32[]) tuple(mul, reduce.1)
    }

    computation2 {
      p0.2 = f32[] parameter(0)
      p1.2 = f32[128] parameter(1)
      neg = f32[128] negate(p1.2)
      reduce.2 = f32[] reduce(neg, p0.2), dimensions={0}, to_apply=accumulate
      ROOT tuple.2 = (f32[], f32[128]) tuple(reduce.2, neg)
    }

    ENTRY entry {
      p0.3 = f32[] parameter(0)
      p1.3 = f32[128] parameter(1)
      sum = f32[128] add(p1.3, p1.3)
      negate = f32[128] negate(sum)
      fusion1 = (f32[128], f32[]) fusion(p0.3, negate), kind=kLoop, calls=computation1
      gte1 = f32[128] get-tuple-element(fusion1), index=0
      gte2 = f32[] get-tuple-element(fusion1), index=1
      fusion2 = (f32[], f32[128]) fusion(p0.3, gte1), kind=kLoop, calls=computation2
      gte3 = f32[] get-tuple-element(fusion2), index=0
      gte4 = f32[128] get-tuple-element(fusion2), index=1
      difference = f32[] subtract(gte3, p0.3)
      ROOT tuple.3 = (f32[], f32[128]) tuple(difference, gte4)
    })"));
  auto fusion = HloFusionAdaptor::ForProducerConsumer(
      module->entry_computation()->GetInstructionWithName("fusion1"),
      module->entry_computation()->GetInstructionWithName("fusion2"));

  auto nodes = fusion->MakeInstructionPostOrder();
  EXPECT_THAT(nodes, ElementsAre(InstructionAdaptorName("mul"),
                                 InstructionAdaptorName("reduce.1"),
                                 InstructionAdaptorName("neg"),
                                 InstructionAdaptorName("reduce.2")));
}

TEST_F(HloTraversalTest, GetRootsForProducerConsumerFusion) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
      computation1 {
        p0.1 = f32[10]{0} parameter(0)
        ROOT neg = f32[10]{0} negate(p0.1)
      }

      computation2 {
        p0.2 = f32[10]{0} parameter(0)
        ROOT add = f32[10]{0} add(p0.2, p0.2)
      }

      ENTRY entry {
        p0.3 = f32[10]{0} parameter(0)
        fusion1 = f32[10]{0} fusion(p0.3), kind=kLoop, calls=computation1
        fusion2 = f32[10]{0} fusion(fusion1), kind=kLoop, calls=computation2
        ROOT tuple = (f32[10]{0}, f32[10]{0}) tuple(fusion1, fusion2)
      }
  )"));
  auto producer_instr =
      module->entry_computation()->GetInstructionWithName("fusion1");
  auto consumer_instr =
      module->entry_computation()->GetInstructionWithName("fusion2");
  auto fusion_adaptor =
      HloFusionAdaptor::ForProducerConsumer(producer_instr, consumer_instr);
  auto producer_computation = module->GetComputationWithName("computation1");
  auto producer = HloFusionAdaptor::ForComputation(producer_computation);
  auto consumer_computation = module->GetComputationWithName("computation2");
  auto consumer = HloFusionAdaptor::ForComputation(consumer_computation);
  auto add = HloInstructionAdaptor{
      *consumer_computation->GetInstructionWithName("add"), consumer.get()};
  EXPECT_THAT(fusion_adaptor->GetRoots(), ElementsAre(add));
  fusion_adaptor = HloFusionAdaptor::ForProducerConsumer(
      producer_instr, consumer_instr, /*with_extra_outputs=*/true);
  auto neg = HloInstructionAdaptor{
      *producer_computation->GetInstructionWithName("neg"), producer.get()};
  EXPECT_THAT(fusion_adaptor->GetRoots(), ElementsAre(add, neg));
}

const char kTwoMultiOutputFusions[] = R"(
    computation1 {
      p0.1 = f32[10]{0} parameter(0)
      p1.1 = f32[10]{0} parameter(1)
      computation1.p2 = f32[10]{0} parameter(2)
      add = f32[10]{0} add(p0.1, p1.1)
      sub = f32[10]{0} subtract(p0.1, p1.1)
      ROOT tuple.1 = (f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}) tuple(p1.1, add, sub, p0.1, computation1.p2)
    }

    computation2 {
      p0.2 = f32[10]{0} parameter(0)
      p1.2 = f32[10]{0} parameter(1)
      p2.2 = f32[10]{0} parameter(2)
      mul = f32[10]{0} multiply(p0.2, p1.2)
      div = f32[10]{0} divide(p0.2, p1.2)
      ROOT tuple.2 = (f32[10]{0}, f32[10]{0}, f32[10]{0}) tuple(mul, div, p2.2)
    }

    ENTRY entry {
      p0.3 = f32[10]{0} parameter(0)
      p1.3 = f32[10]{0} parameter(1)
      p2.3 = f32[10]{0} parameter(2)
      fusion1 = (f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}, f32[10]{0}) fusion(p0.3, p1.3, p2.3), kind=kLoop, calls=computation1
      gte0 = f32[10]{0} get-tuple-element(fusion1), index=0
      gte1 = f32[10]{0} get-tuple-element(fusion1), index=1
      gte2 = f32[10]{0} get-tuple-element(fusion1), index=2
      gte3 = f32[10]{0} get-tuple-element(fusion1), index=3
      gte4 = f32[10]{0} get-tuple-element(fusion1), index=4
      fusion2 = (f32[10]{0}, f32[10]{0}, f32[10]{0}) fusion(gte1, gte2, gte3), kind=kLoop, calls=computation2
      gte5 = f32[10]{0} get-tuple-element(fusion2), index=0
      gte6 = f32[10]{0} get-tuple-element(fusion2), index=1
      gte7 = f32[10]{0} get-tuple-element(fusion2), index=2
      ROOT tuple.3 = tuple(gte0, gte1, gte3, gte4, gte5, gte6, gte7)
    })";

TEST_F(HloTraversalTest, GetParametersMultiOutputFusion) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTwoMultiOutputFusions));
  auto producer =
      module->entry_computation()->GetInstructionWithName("fusion1");
  auto consumer =
      module->entry_computation()->GetInstructionWithName("fusion2");
  auto fusion_adaptor =
      HloFusionAdaptor::ForProducerConsumer(producer, consumer);
  auto p0 = module->entry_computation()->GetInstructionWithName("p0.3");
  auto p1 = module->entry_computation()->GetInstructionWithName("p1.3");
  EXPECT_THAT(fusion_adaptor->GetParameters(), ElementsAre(p0, p1));
  // Double-check that after performing the actual fusion, we get the same
  // parameters.
  consumer->MergeFusionInstructionIntoMultiOutput(producer);
  EXPECT_THAT(consumer->operands(), ElementsAre(p0, p1));
}

TEST_F(HloTraversalTest, GetRootsMultiOutputFusion) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kTwoMultiOutputFusions));
  auto consumer_fusion_instr =
      module->entry_computation()->GetInstructionWithName("fusion2");
  auto producer_fusion_instr =
      module->entry_computation()->GetInstructionWithName("fusion1");
  auto fusion_adaptor = HloFusionAdaptor::ForProducerConsumer(
      producer_fusion_instr, consumer_fusion_instr,
      /*with_extra_outputs=*/true);
  auto producer_computation = module->GetComputationWithName("computation1");
  auto producer = HloFusionAdaptor::ForComputation(producer_computation);
  auto consumer_computation = module->GetComputationWithName("computation2");
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
                      *producer_computation->GetInstructionWithName("p0.1"),
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    computation {
      p0.1 = f32[] parameter(0)
      p1.1 = f32[] parameter(1)
      negate = f32[] negate(p0.1)
      log = f32[] log(p0.1)
      sum = f32[] add(p0.1, log)
      exp = f32[] exponential(p1.1)
      ROOT call = f32[] custom-call(negate, exp, sum), custom_call_target="it"
    }

    ENTRY entry {
      p0.2 = f32[] parameter(0)
      p1.2 = f32[] parameter(1)
      ROOT fusion = f32[] fusion(p0.2, p1.2), kind=kLoop, calls=computation
    }
    )"));

  auto* fusion_computation = module->GetComputationWithName("computation");
  auto fusion = HloFusionAdaptor::ForComputation(fusion_computation);
  auto get = [&](absl::string_view name) {
    return HloInstructionAdaptor{
        *fusion_computation->GetInstructionWithName(name), fusion.get()};
  };
  auto p0 = get("p0.1");
  auto p1 = get("p1.1");
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

TEST_F(HloTraversalTest, DoNotResolveIntoNestedFusions) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    computation1 {
      p0.1 = f32[] parameter(0)
      ROOT mul = f32[] multiply(p0.1, p0.1)
    }

    computation2 {
      p0.2 = f32[] parameter(0)
      fusion.2 = f32[] fusion(p0.2), kind=kLoop, calls=computation1
      ROOT neg = f32[] negate(fusion.2)
    }

    ENTRY entry {
      p0.3 = f32[] parameter(0)
      ROOT fusion.3 = f32[] fusion(p0.3), kind=kLoop, calls=computation2
    }
  )"));

  auto* computation2 = module->GetComputationWithName("computation2");
  auto fusion_adaptor = HloFusionAdaptor::ForComputation(computation2);

  HloInstructionAdaptor param0(*computation2->GetInstructionWithName("p0.2"),
                               fusion_adaptor.get());
  EXPECT_THAT(param0.GetUsers(),
              ElementsAre(InstructionAdaptorName("fusion.2")));

  HloInstructionAdaptor fusion2(
      *computation2->GetInstructionWithName("fusion.2"), fusion_adaptor.get());
  EXPECT_THAT(fusion2.GetOperands(),
              ElementsAre(InstructionAdaptorName("p0.3")));
  EXPECT_THAT(fusion2.GetUsers(), ElementsAre(InstructionAdaptorName("neg")));

  HloInstructionAdaptor negate(*computation2->GetInstructionWithName("neg"),
                               fusion_adaptor.get());
  EXPECT_THAT(negate.GetOperands(),
              ElementsAre(InstructionAdaptorName("fusion.2")));
}

}  // namespace
}  // namespace xla

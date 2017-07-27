/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/hlo_matcher.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace poplarplugin {
namespace {

using HloMatcherTest = HloTestBase;

class TestMatcher : public HloMatcher {
public:
  TestMatcher(const std::vector<HloMatcherPattern>& patterns,
              bool root_only,
              bool drop_last_instruction)
          : HloMatcher(patterns, root_only)
          , drop_last_instruction_(drop_last_instruction) {}

private:
  ReplacedInstructions ReplaceNodes(unsigned int pattern,
                                    const HloMatcherMatched& match) override {
    replace_count++;
    match_pattern.push_back(pattern);
    match_count.push_back(match.instructions.size());

    ReplacedInstructions replaced = match.instructions;
    if (drop_last_instruction_) {
      replaced.pop_back();
    }

    return replaced;
  }

  bool drop_last_instruction_;

public:
  int replace_count=0;
  std::vector<unsigned int> match_pattern;
  std::vector<unsigned int> match_count;
};




TEST_F(HloMatcherTest, MatchTestSimpleReplacementTwice) {
  Shape shape = ShapeUtil::MakeShape(F32, {10, 10});

  auto builder = HloComputation::Builder(TestName());
  auto i1 = builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape, "in1"));
  auto i2 = builder.AddInstruction(
          HloInstruction::CreateParameter(1, shape, "in2"));
  auto i3 = builder.AddInstruction(
          HloInstruction::CreateParameter(2, shape, "in3"));
  auto add1 = builder.AddInstruction(
          HloInstruction::CreateBinary(shape, HloOpcode::kAdd, i1, i2));
  auto add2 = builder.AddInstruction(
          HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add1, i3));

  builder.AddInstruction(
          HloInstruction::CreateTuple({add2}));

  auto computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));


  std::vector<HloMatcherPattern> patterns = {
    {{HloOpcode::kAdd, true, nullptr, {-1, -1}}}
  };
  TestMatcher matcher(patterns, false, false);

  EXPECT_TRUE(matcher.Run(hlo_module.get()).ValueOrDie());
  EXPECT_EQ(2, matcher.replace_count);
}





TEST_F(HloMatcherTest, MatchTestTwoPatterns) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {10, 10});
  Shape shape2 = ShapeUtil::MakeShape(F32, {10});

  auto builder = HloComputation::Builder(TestName());
  auto i1 = builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape1, "in1"));
  auto i2 = builder.AddInstruction(
          HloInstruction::CreateParameter(1, shape1, "in2"));
  auto i3 = builder.AddInstruction(
          HloInstruction::CreateParameter(2, shape2, "in3"));
  auto b1 = builder.AddInstruction(
          HloInstruction::CreateBroadcast(shape1, i3, {1}));
  auto add1 = builder.AddInstruction(
          HloInstruction::CreateBinary(shape1, HloOpcode::kAdd, i1, i2));
  auto add2 = builder.AddInstruction(
          HloInstruction::CreateBinary(shape1, HloOpcode::kAdd, add1, b1));

  builder.AddInstruction(
          HloInstruction::CreateTuple({add2}));

  auto computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));


  std::vector<HloMatcherPattern> patterns = {
    {{HloOpcode::kAdd, true, nullptr, {-1, 1}},
     {HloOpcode::kBroadcast, true, nullptr, {-1}}},

    {{HloOpcode::kAdd, true, nullptr, {-1, -1}}}
  };
  TestMatcher matcher(patterns, false, false);

  EXPECT_TRUE(matcher.Run(hlo_module.get()).ValueOrDie());
  EXPECT_EQ(2, matcher.replace_count);
}




TEST_F(HloMatcherTest, MatchTestGraphWithPathsJoining) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {10, 10});
  Shape shape2 = ShapeUtil::MakeShape(F32, {10});

  auto builder = HloComputation::Builder(TestName());
  auto i1 = builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape1, "in1"));
  auto i2 = builder.AddInstruction(
          HloInstruction::CreateParameter(1, shape1, "in2"));
  auto i3 = builder.AddInstruction(
          HloInstruction::CreateParameter(2, shape2, "in3"));
  auto b1 = builder.AddInstruction(
          HloInstruction::CreateBroadcast(shape1, i3, {1}));
  auto sub1 = builder.AddInstruction(
          HloInstruction::CreateBinary(shape1, HloOpcode::kSubtract, i1, b1));
  auto add1 = builder.AddInstruction(
          HloInstruction::CreateBinary(shape1, HloOpcode::kAdd, i2, b1));

  auto sub2 = builder.AddInstruction(
          HloInstruction::CreateBinary(shape1, HloOpcode::kSubtract, add1, sub1));

  builder.AddInstruction(
          HloInstruction::CreateTuple({sub2}));

  auto computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));


  std::vector<HloMatcherPattern> patterns = {
    {{HloOpcode::kAdd, true, nullptr, {-1, 1}},
     {HloOpcode::kBroadcast, true, nullptr, {-1}}}
  };
  TestMatcher matcher(patterns, false, false);

  EXPECT_TRUE(matcher.Run(hlo_module.get()).ValueOrDie());
  EXPECT_EQ(1, matcher.replace_count);
}





TEST_F(HloMatcherTest, MatchTestGraphWithPathsJoiningOnMultipleMatchNode) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {10, 10});
  Shape shape2 = ShapeUtil::MakeShape(F32, {10});

  auto builder = HloComputation::Builder(TestName());
  auto i1 = builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape1, "in1"));
  auto i2 = builder.AddInstruction(
          HloInstruction::CreateParameter(1, shape1, "in2"));
  auto i3 = builder.AddInstruction(
          HloInstruction::CreateParameter(2, shape2, "in3"));
  auto b1 = builder.AddInstruction(
          HloInstruction::CreateBroadcast(shape1, i3, {1}));
  auto add1 = builder.AddInstruction(
          HloInstruction::CreateBinary(shape1, HloOpcode::kAdd, i1, b1));
  auto add2 = builder.AddInstruction(
          HloInstruction::CreateBinary(shape1, HloOpcode::kAdd, i2, b1));

  auto sub1 = builder.AddInstruction(
          HloInstruction::CreateBinary(shape1, HloOpcode::kSubtract, add1, add2));

  builder.AddInstruction(
          HloInstruction::CreateTuple({sub1}));

  auto computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));


  std::vector<HloMatcherPattern> patterns = {
    {{HloOpcode::kAdd, true, nullptr, {-1, 1}},
     {HloOpcode::kBroadcast, true, nullptr, {-1}}}
  };
  TestMatcher matcher(patterns, false, true);

  EXPECT_TRUE(matcher.Run(hlo_module.get()).ValueOrDie());
  EXPECT_EQ(2, matcher.replace_count);
}





TEST_F(HloMatcherTest, MatchTestGraphWithMatchedByNonRemovedNodes) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {10, 10});
  Shape shape2 = ShapeUtil::MakeShape(F32, {10});

  auto builder = HloComputation::Builder(TestName());
  auto i1 = builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape1, "in1"));
  auto i2 = builder.AddInstruction(
          HloInstruction::CreateParameter(1, shape1, "in2"));
  auto i3 = builder.AddInstruction(
          HloInstruction::CreateParameter(2, shape2, "in3"));
  auto b1 = builder.AddInstruction(
          HloInstruction::CreateBroadcast(shape1, i3, {1}));
  auto sub1 = builder.AddInstruction(
          HloInstruction::CreateBinary(shape1, HloOpcode::kSubtract, i1, b1));
  auto add1 = builder.AddInstruction(
          HloInstruction::CreateBinary(shape1, HloOpcode::kAdd, i2, b1));

  auto sub2 = builder.AddInstruction(
          HloInstruction::CreateBinary(shape1, HloOpcode::kSubtract, add1, sub1));

  builder.AddInstruction(
          HloInstruction::CreateTuple({sub2}));

  auto computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));


  std::vector<HloMatcherPattern> patterns = {
    {{HloOpcode::kSubtract, true, nullptr, {1, -1}},
     {HloOpcode::kAdd, true, nullptr, {-1, 2}},
     {HloOpcode::kBroadcast, false, nullptr, {-1}}}
  };
  TestMatcher matcher(patterns, false, false);

  EXPECT_TRUE(matcher.Run(hlo_module.get()).ValueOrDie());
  EXPECT_EQ(1, matcher.replace_count);
  EXPECT_EQ(2, matcher.match_count[0]);
}


}
}
}

/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/slice_sinker.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace m = match;
using ::testing::ElementsAre;

class SliceSinkerTest : public HloHardwareIndependentTestBase {};

TEST_F(SliceSinkerTest, TernaryOperation) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = pred[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      p2 = f32[8,9] parameter(2)
      s00 = pred[2,9] slice(pred[8,9] p0), slice={[0:2], [0:9]}
      s01 = pred[6,9] slice(pred[8,9] p0), slice={[2:8], [0:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[6,9] slice(f32[8,9] p1), slice={[2:8], [0:9]}
      s20 = f32[2,9] slice(f32[8,9] p2), slice={[0:2], [0:9]}
      s21 = f32[6,9] slice(f32[8,9] p2), slice={[2:8], [0:9]}
      sel0 = f32[2,9] select(pred[2,9] s00, f32[2,9] s10, f32[2,9] s20)
      sel1 = f32[6,9] select(pred[6,9] s01, f32[6,9] s11, f32[6,9] s21)
      ROOT tuple = (f32[2,9], f32[6,9]) tuple(sel0, sel1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  HloInstruction* inst = module->entry_computation()->root_instruction();
  const HloInstruction* slice0;
  const HloInstruction* slice1;
  EXPECT_THAT(inst,
              GmockMatch(m::Tuple(
                  m::Slice(&slice0, m::Select(m::Parameter(0), m::Parameter(1),
                                              m::Parameter(2))),
                  m::Slice(&slice1, m::Select(m::Parameter(0), m::Parameter(1),
                                              m::Parameter(2))))));
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(2, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1));
}

TEST_F(SliceSinkerTest, OverlappingPartialSlicesBeneficial) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[5,9] slice(f32[8,9] p0), slice={[3:8], [0:9]}
      s02 = f32[8,4] slice(f32[8,9] p0), slice={[0:8], [0:4]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[5,9] slice(f32[8,9] p1), slice={[3:8], [0:9]}
      s12 = f32[8,4] slice(f32[8,9] p1), slice={[0:8], [0:4]}
      add0 = f32[2,9] add(f32[2,9] s00, f32[2,9] s10)
      add1 = f32[5,9] add(f32[5,9] s01, f32[5,9] s11)
      add2 = f32[8,4] add(f32[8,4] s02, f32[8,4] s12)
      ROOT tuple = (f32[2,9], f32[5,9], f32[8,4]) tuple(add0, add1, add2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  HloInstruction* inst = module->entry_computation()->root_instruction();
  const HloInstruction* slice0;
  const HloInstruction* slice1;
  const HloInstruction* slice2;
  EXPECT_THAT(
      inst, GmockMatch(m::Tuple(
                m::Slice(&slice0, m::Add(m::Parameter(0), m::Parameter(1))),
                m::Slice(&slice1, m::Add(m::Parameter(0), m::Parameter(1))),
                m::Slice(&slice2, m::Add(m::Parameter(0), m::Parameter(1))))));
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(3, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice2->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice2->slice_limits(), ElementsAre(8, 4));
  EXPECT_THAT(slice2->slice_strides(), ElementsAre(1, 1));
}

TEST_F(SliceSinkerTest, SameSliceSourcesTwoPeerGroups) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[6,9] slice(f32[8,9] p0), slice={[2:8], [0:9]}
      s02 = f32[8,2] slice(f32[8,9] p0), slice={[0:8], [0:2]}
      s03 = f32[8,7] slice(f32[8,9] p0), slice={[0:8], [2:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[6,9] slice(f32[8,9] p1), slice={[2:8], [0:9]}
      s12 = f32[8,2] slice(f32[8,9] p1), slice={[0:8], [0:2]}
      s13 = f32[8,7] slice(f32[8,9] p1), slice={[0:8], [2:9]}
      add0 = f32[2,9] add(f32[2,9] s00, f32[2,9] s10)
      add1 = f32[6,9] add(f32[6,9] s01, f32[6,9] s11)
      mul0 = f32[8,2] multiply(f32[8,2] s02, f32[8,2] s12)
      mul1 = f32[8,7] multiply(f32[8,7] s03, f32[8,7] s13)
      ROOT tuple = (f32[2,9], f32[6,9], f32[8,2], f32[8,7]) tuple(add0, add1, mul0, mul1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  HloInstruction* inst = module->entry_computation()->root_instruction();
  const HloInstruction* slice0;
  const HloInstruction* slice1;
  const HloInstruction* slice2;
  const HloInstruction* slice3;
  EXPECT_THAT(
      inst,
      GmockMatch(m::Tuple(
          m::Slice(&slice0, m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(&slice1, m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(&slice2, m::Multiply(m::Parameter(0), m::Parameter(1))),
          m::Slice(&slice3, m::Multiply(m::Parameter(0), m::Parameter(1))))));
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(2, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice2->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice2->slice_limits(), ElementsAre(8, 2));
  EXPECT_THAT(slice2->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice3->slice_starts(), ElementsAre(0, 2));
  EXPECT_THAT(slice3->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice3->slice_strides(), ElementsAre(1, 1));
}

TEST_F(SliceSinkerTest, OverlappingMultipleSlices) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[5,9] slice(f32[8,9] p0), slice={[3:8], [0:9]}
      s02 = f32[3,9] slice(f32[8,9] p0), slice={[2:5], [0:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[5,9] slice(f32[8,9] p1), slice={[3:8], [0:9]}
      s12 = f32[3,9] slice(f32[8,9] p1), slice={[2:5], [0:9]}
      add0 = f32[2,9] add(f32[2,9] s00, f32[2,9] s10)
      add1 = f32[5,9] add(f32[5,9] s01, f32[5,9] s11)
      add2 = f32[3,9] add(f32[3,9] s02, f32[3,9] s12)
      ROOT tuple = (f32[2,9], f32[5,9], f32[3,9]) tuple(add0, add1, add2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  HloInstruction* inst = module->entry_computation()->root_instruction();
  const HloInstruction* slice0;
  const HloInstruction* slice1;
  const HloInstruction* slice2;
  EXPECT_THAT(
      inst, GmockMatch(m::Tuple(
                m::Slice(&slice0, m::Add(m::Parameter(0), m::Parameter(1))),
                m::Slice(&slice1, m::Add(m::Parameter(0), m::Parameter(1))),
                m::Slice(&slice2, m::Add(m::Parameter(0), m::Parameter(1))))));
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(3, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice2->slice_starts(), ElementsAre(2, 0));
  EXPECT_THAT(slice2->slice_limits(), ElementsAre(5, 9));
  EXPECT_THAT(slice2->slice_strides(), ElementsAre(1, 1));
}

TEST_F(SliceSinkerTest, DisjointedPartialSlices) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[5,9] slice(f32[8,9] p0), slice={[2:7], [0:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[5,9] slice(f32[8,9] p1), slice={[2:7], [0:9]}
      add0 = f32[2,9] add(f32[2,9] s00, f32[2,9] s10)
      add1 = f32[5,9] add(f32[5,9] s01, f32[5,9] s11)
      ROOT tuple = (f32[2,9], f32[5,9]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, OverlappingPartialSlicesNotBeneficial) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,7] slice(f32[8,9] p0), slice={[0:2], [0:7]}
      s01 = f32[6,7] slice(f32[8,9] p0), slice={[2:8], [0:7]}
      s10 = f32[2,7] slice(f32[8,9] p1), slice={[0:2], [0:7]}
      s11 = f32[6,7] slice(f32[8,9] p1), slice={[2:8], [0:7]}
      add0 = f32[2,7] add(f32[2,7] s00, f32[2,7] s10)
      add1 = f32[6,7] add(f32[6,7] s01, f32[6,7] s11)
      ROOT tuple = (f32[2,7], f32[6,7]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, DifferentOrderingOfSliceSources) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,7] parameter(0)
      p1 = f32[8,7] parameter(1)
      s00 = f32[2,7] slice(f32[8,7] p0), slice={[0:2], [0:7]}
      s01 = f32[6,7] slice(f32[8,7] p0), slice={[2:8], [0:7]}
      s10 = f32[2,7] slice(f32[8,7] p1), slice={[0:2], [0:7]}
      s11 = f32[6,7] slice(f32[8,7] p1), slice={[2:8], [0:7]}
      add0 = f32[2,7] add(f32[2,7] s00, f32[2,7] s10)
      add1 = f32[6,7] add(f32[6,7] s11, f32[6,7] s01)
      ROOT tuple = (f32[2,7], f32[6,7]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, SlicesFromDifferentIndices) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[4,9] slice(f32[8,9] p0), slice={[0:4], [0:9]}
      s01 = f32[4,9] slice(f32[8,9] p0), slice={[4:8], [0:9]}
      s10 = f32[4,9] slice(f32[8,9] p1), slice={[0:4], [0:9]}
      s11 = f32[4,9] slice(f32[8,9] p1), slice={[4:8], [0:9]}
      add0 = f32[4,9] add(f32[4,9] s01, f32[4,9] s10)
      add1 = f32[4,9] add(f32[4,9] s00, f32[4,9] s11)
      ROOT tuple = (f32[4,9], f32[4,9]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, DifferentOperator) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[6,9] slice(f32[8,9] p0), slice={[2:8], [0:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[6,9] slice(f32[8,9] p1), slice={[2:8], [0:9]}
      mul = f32[2,9] multiply(f32[2,9] s00, f32[2,9] s10)
      add = f32[6,9] add(f32[6,9] s01, f32[6,9] s11)
      ROOT tuple = (f32[2,9], f32[6,9]) tuple(mul, add)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, SameOperatorDifferentAttributes) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[6,9] slice(f32[8,9] p0), slice={[2:8], [0:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[6,9] slice(f32[8,9] p1), slice={[2:8], [0:9]}
      cmp1 = pred[2,9] compare(f32[2,9] s00, f32[2,9] s10), direction=GT
      cmp2 = pred[6,9] compare(f32[6,9] s01, f32[6,9] s11), direction=LT
      ROOT tuple = (pred[2,9], pred[6,9]) tuple(cmp1, cmp2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, SlicesWithMultiUsers) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[6,9] slice(f32[8,9] p0), slice={[2:8], [0:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[6,9] slice(f32[8,9] p1), slice={[2:8], [0:9]}
      add0 = f32[2,9] add(f32[2,9] s00, f32[2,9] s10)
      add1 = f32[6,9] add(f32[6,9] s01, f32[6,9] s11)
      mul0 = f32[2,9] multiply(f32[2,9] s00, f32[2,9] s10)
      mul1 = f32[6,9] multiply(f32[6,9] s01, f32[6,9] s11)
      ROOT tuple = (f32[2,9], f32[6,9], f32[2,9], f32[6,9]) tuple(add0, add1, mul0, mul1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  HloInstruction* inst = module->entry_computation()->root_instruction();
  const HloInstruction* slice0;
  const HloInstruction* slice1;
  const HloInstruction* slice2;
  const HloInstruction* slice3;
  EXPECT_THAT(
      inst,
      GmockMatch(m::Tuple(
          m::Slice(&slice0, m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(&slice1, m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(&slice2, m::Multiply(m::Parameter(0), m::Parameter(1))),
          m::Slice(&slice3, m::Multiply(m::Parameter(0), m::Parameter(1))))));
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(2, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice2->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice2->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(slice2->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice3->slice_starts(), ElementsAre(2, 0));
  EXPECT_THAT(slice3->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice3->slice_strides(), ElementsAre(1, 1));
}

TEST_F(SliceSinkerTest, NonElementWise) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8] parameter(0)
      s00 = f32[2] slice(f32[8] p0), slice={[0:2]}
      s01 = f32[6] slice(f32[8] p0), slice={[2:8]}
      bc0 = f32[2,9] broadcast(f32[2] s00), dimensions={0}
      bc1 = f32[6,9] broadcast(f32[6] s01), dimensions={0}
      ROOT tuple = (f32[2,9], f32[6,9]) tuple(bc0, bc1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, SlicesWithNontrivialStrides) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[4,9] slice(f32[8,9] p0), slice={[0:7:2], [0:9]}
      s01 = f32[4,9] slice(f32[8,9] p0), slice={[1:8:2], [0:9]}
      s10 = f32[4,9] slice(f32[8,9] p1), slice={[0:7:2], [0:9]}
      s11 = f32[4,9] slice(f32[8,9] p1), slice={[1:8:2], [0:9]}
      add0 = f32[4,9] add(f32[4,9] s00, f32[4,9] s10)
      add1 = f32[4,9] add(f32[4,9] s01, f32[4,9] s11)
      ROOT tuple = (f32[4,9], f32[4,9]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  HloInstruction* inst = module->entry_computation()->root_instruction();
  const HloInstruction* slice0;
  const HloInstruction* slice1;
  EXPECT_THAT(
      inst, GmockMatch(m::Tuple(
                m::Slice(&slice0, m::Add(m::Parameter(0), m::Parameter(1))),
                m::Slice(&slice1, m::Add(m::Parameter(0), m::Parameter(1))))));
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(7, 9));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(2, 1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(1, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(2, 1));
}

TEST_F(SliceSinkerTest, NotAllSliceOperand) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[2,9] parameter(1)
      p2 = f32[6,9] parameter(2)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[6,9] slice(f32[8,9] p0), slice={[2:8], [0:9]}
      abs0 = f32[2,9] abs(f32[2,9] p1)
      abs1 = f32[6,9] abs(f32[6,9] p2)
      add0 = f32[2,9] add(f32[2,9] s00, f32[2,9] abs0)
      add1 = f32[6,9] add(f32[6,9] s01, f32[6,9] abs1)
      ROOT tuple = (f32[2,9], f32[6,9]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceSinkerTest, Cascade) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      p1 = f32[8,9] parameter(1)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[6,9] slice(f32[8,9] p0), slice={[2:8], [0:9]}
      s10 = f32[2,9] slice(f32[8,9] p1), slice={[0:2], [0:9]}
      s11 = f32[6,9] slice(f32[8,9] p1), slice={[2:8], [0:9]}
      abs0 = f32[2,9] abs(f32[2,9] s10)
      abs1 = f32[6,9] abs(f32[6,9] s11)
      add0 = f32[2,9] add(f32[2,9] s00, f32[2,9] abs0)
      add1 = f32[6,9] add(f32[6,9] s01, f32[6,9] abs1)
      ROOT tuple = (f32[2,9], f32[6,9]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_TRUE(result);
  HloInstruction* inst = module->entry_computation()->root_instruction();
  const HloInstruction* slice0;
  const HloInstruction* slice1;
  EXPECT_THAT(
      inst,
      GmockMatch(m::Tuple(
          m::Slice(&slice0, m::Add(m::Parameter(0), m::Abs(m::Parameter(1)))),
          m::Slice(&slice1,
                   m::Add(m::Parameter(0), m::Abs(m::Parameter(1)))))));
  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(2, 9));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(2, 0));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(8, 9));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1));
}

TEST_F(SliceSinkerTest, SameOpcodeDifferentResultElementTypes) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,9] parameter(0)
      s00 = f32[2,9] slice(f32[8,9] p0), slice={[0:2], [0:9]}
      s01 = f32[6,9] slice(f32[8,9] p0), slice={[2:8], [0:9]}
      convert0 = s32[2,9] convert(f32[2,9] s00)
      convert1 = s64[6,9] convert(f32[6,9] s01)
      ROOT tuple = (s32[2,9], s64[6,9]) tuple(convert0, convert1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceSinker slice_sinker;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&slice_sinker, module.get()));
  EXPECT_FALSE(result);
}

}  // namespace
}  // namespace xla

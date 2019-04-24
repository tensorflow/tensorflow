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

#include <memory>
#include <vector>
#include "tensorflow/compiler/xla/service/slice_delaying.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
namespace m = match;

class SliceDelayingTest : public HloTestBase {
};

TEST_F(SliceDelayingTest, Basic) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      s00 = f32[2,8] slice(f32[8,8] p0), slice={[0:2], [0:8]}
      s01 = f32[6,8] slice(f32[8,8] p0), slice={[2:8], [0:8]}
      s10 = f32[2,8] slice(f32[8,8] p1), slice={[0:2], [0:8]}
      s11 = f32[6,8] slice(f32[8,8] p1), slice={[2:8], [0:8]}
      add0 = f32[2,8] add(f32[2,8] s00, f32[2,8] s10)
      add1 = f32[6,8] add(f32[6,8] s01, f32[6,8] s11)
      ROOT tuple = (f32[2,8], f32[6,8]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_TRUE(result);
  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(result,
                          RunHloPass(&dce, module.get()));
  EXPECT_TRUE(result);
  EXPECT_EQ(6, module->entry_computation()->instruction_count());
  HloInstruction* inst = module->entry_computation()->root_instruction();
  EXPECT_THAT(inst,
      GmockMatch(m::Tuple(m::Slice(m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Add(m::Parameter(0), m::Parameter(1))))));
  HloInstruction* slice0 = inst->mutable_operand(0);
  EXPECT_EQ(slice0->slice_starts(), std::vector<int64>({0, 0}));
  EXPECT_EQ(slice0->slice_limits(), std::vector<int64>({2, 8}));
  EXPECT_EQ(slice0->slice_strides(), std::vector<int64>({1, 1}));
  HloInstruction* slice1 = inst->mutable_operand(1);
  EXPECT_EQ(slice1->slice_starts(), std::vector<int64>({2, 0}));
  EXPECT_EQ(slice1->slice_limits(), std::vector<int64>({8, 8}));
  EXPECT_EQ(slice1->slice_strides(), std::vector<int64>({1, 1}));
}

TEST_F(SliceDelayingTest, SliceDualDimension) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      s00 = f32[2,8] slice(f32[8,8] p0), slice={[0:2], [0:8]}
      s01 = f32[6,8] slice(f32[8,8] p0), slice={[2:8], [0:8]}
      s02 = f32[8,4] slice(f32[8,8] p0), slice={[0:8], [0:4]}
      s10 = f32[2,8] slice(f32[8,8] p1), slice={[0:2], [0:8]}
      s11 = f32[6,8] slice(f32[8,8] p1), slice={[2:8], [0:8]}
      s12 = f32[8,4] slice(f32[8,8] p1), slice={[0:8], [0:4]}
      add0 = f32[2,8] add(f32[2,8] s00, f32[2,8] s10)
      add1 = f32[6,8] add(f32[6,8] s01, f32[6,8] s11)
      add2 = f32[8,4] add(f32[8,4] s02, f32[8,4] s12)
      ROOT tuple = (f32[2,8], f32[6,8], f32[8,4]) tuple(add0, add1, add2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_TRUE(result);
  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(result,
                          RunHloPass(&dce, module.get()));
  EXPECT_TRUE(result);
  EXPECT_EQ(7, module->entry_computation()->instruction_count());
  HloInstruction* inst = module->entry_computation()->root_instruction();
  EXPECT_THAT(inst,
      GmockMatch(m::Tuple(m::Slice(m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Add(m::Parameter(0), m::Parameter(1))))));
  HloInstruction* slice0 = inst->mutable_operand(0);
  EXPECT_EQ(slice0->slice_starts(), std::vector<int64>({0, 0}));
  EXPECT_EQ(slice0->slice_limits(), std::vector<int64>({2, 8}));
  EXPECT_EQ(slice0->slice_strides(), std::vector<int64>({1, 1}));
  HloInstruction* slice1 = inst->mutable_operand(1);
  EXPECT_EQ(slice1->slice_starts(), std::vector<int64>({2, 0}));
  EXPECT_EQ(slice1->slice_limits(), std::vector<int64>({8, 8}));
  EXPECT_EQ(slice1->slice_strides(), std::vector<int64>({1, 1}));
  HloInstruction* slice2 = inst->mutable_operand(2);
  EXPECT_EQ(slice2->slice_starts(), std::vector<int64>({0, 0}));
  EXPECT_EQ(slice2->slice_limits(), std::vector<int64>({8, 4}));
  EXPECT_EQ(slice2->slice_strides(), std::vector<int64>({1, 1}));
}

TEST_F(SliceDelayingTest, SplitDualDimension) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      s00 = f32[2,8] slice(f32[8,8] p0), slice={[0:2], [0:8]}
      s01 = f32[6,8] slice(f32[8,8] p0), slice={[2:8], [0:8]}
      s02 = f32[8,2] slice(f32[8,8] p0), slice={[0:8], [0:2]}
      s03 = f32[8,6] slice(f32[8,8] p0), slice={[0:8], [2:8]}
      s10 = f32[2,8] slice(f32[8,8] p1), slice={[0:2], [0:8]}
      s11 = f32[6,8] slice(f32[8,8] p1), slice={[2:8], [0:8]}
      s12 = f32[8,2] slice(f32[8,8] p1), slice={[0:8], [0:2]}
      s13 = f32[8,6] slice(f32[8,8] p1), slice={[0:8], [2:8]}
      add0 = f32[2,8] add(f32[2,8] s00, f32[2,8] s10)
      add1 = f32[6,8] add(f32[6,8] s01, f32[6,8] s11)
      mul0 = f32[8,2] multiply(f32[8,2] s02, f32[8,2] s12)
      mul1 = f32[8,6] multiply(f32[8,6] s03, f32[8,6] s13)
      ROOT tuple = (f32[2,8], f32[6,8], f32[8,2], f32[8,6]) tuple(add0, add1, mul0, mul1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_TRUE(result);
  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(result,
                          RunHloPass(&dce, module.get()));
  EXPECT_TRUE(result);
  EXPECT_EQ(9, module->entry_computation()->instruction_count());
  HloInstruction* inst = module->entry_computation()->root_instruction();
  EXPECT_THAT(inst,
      GmockMatch(m::Tuple(m::Slice(m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Multiply(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Multiply(m::Parameter(0), m::Parameter(1))))));
  HloInstruction* slice0 = inst->mutable_operand(0);
  EXPECT_EQ(slice0->slice_starts(), std::vector<int64>({0, 0}));
  EXPECT_EQ(slice0->slice_limits(), std::vector<int64>({2, 8}));
  EXPECT_EQ(slice0->slice_strides(), std::vector<int64>({1, 1}));
  HloInstruction* slice1 = inst->mutable_operand(1);
  EXPECT_EQ(slice1->slice_starts(), std::vector<int64>({2, 0}));
  EXPECT_EQ(slice1->slice_limits(), std::vector<int64>({8, 8}));
  EXPECT_EQ(slice1->slice_strides(), std::vector<int64>({1, 1}));
  HloInstruction* slice2 = inst->mutable_operand(2);
  EXPECT_EQ(slice2->slice_starts(), std::vector<int64>({0, 0}));
  EXPECT_EQ(slice2->slice_limits(), std::vector<int64>({8, 2}));
  EXPECT_EQ(slice2->slice_strides(), std::vector<int64>({1, 1}));
  HloInstruction* slice3 = inst->mutable_operand(3);
  EXPECT_EQ(slice3->slice_starts(), std::vector<int64>({0, 2}));
  EXPECT_EQ(slice3->slice_limits(), std::vector<int64>({8, 8}));
  EXPECT_EQ(slice3->slice_strides(), std::vector<int64>({1, 1}));
}

TEST_F(SliceDelayingTest, OverlapSlice) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      s00 = f32[2,8] slice(f32[8,8] p0), slice={[0:2], [0:8]}
      s01 = f32[5,8] slice(f32[8,8] p0), slice={[3:8], [0:8]}
      s02 = f32[3,8] slice(f32[8,8] p0), slice={[2:5], [0:8]}
      s10 = f32[2,8] slice(f32[8,8] p1), slice={[0:2], [0:8]}
      s11 = f32[5,8] slice(f32[8,8] p1), slice={[3:8], [0:8]}
      s12 = f32[3,8] slice(f32[8,8] p1), slice={[2:5], [0:8]}
      add0 = f32[2,8] add(f32[2,8] s00, f32[2,8] s10)
      add1 = f32[5,8] add(f32[5,8] s01, f32[5,8] s11)
      add2 = f32[3,8] add(f32[3,8] s02, f32[3,8] s12)
      ROOT tuple = (f32[2,8], f32[5,8], f32[3,8]) tuple(add0, add1, add2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_TRUE(result);
  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(result,
                          RunHloPass(&dce, module.get()));
  EXPECT_TRUE(result);
  EXPECT_EQ(7, module->entry_computation()->instruction_count());
  HloInstruction* inst = module->entry_computation()->root_instruction();
  EXPECT_THAT(inst,
      GmockMatch(m::Tuple(m::Slice(m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Add(m::Parameter(0), m::Parameter(1))))));
  HloInstruction* slice0 = inst->mutable_operand(0);
  EXPECT_EQ(slice0->slice_starts(), std::vector<int64>({0, 0}));
  EXPECT_EQ(slice0->slice_limits(), std::vector<int64>({2, 8}));
  EXPECT_EQ(slice0->slice_strides(), std::vector<int64>({1, 1}));
  HloInstruction* slice1 = inst->mutable_operand(1);
  EXPECT_EQ(slice1->slice_starts(), std::vector<int64>({3, 0}));
  EXPECT_EQ(slice1->slice_limits(), std::vector<int64>({8, 8}));
  EXPECT_EQ(slice1->slice_strides(), std::vector<int64>({1, 1}));
  HloInstruction* slice2 = inst->mutable_operand(2);
  EXPECT_EQ(slice2->slice_starts(), std::vector<int64>({2, 0}));
  EXPECT_EQ(slice2->slice_limits(), std::vector<int64>({5, 8}));
  EXPECT_EQ(slice2->slice_strides(), std::vector<int64>({1, 1}));
}

TEST_F(SliceDelayingTest, PartialSplit) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      s00 = f32[2,8] slice(f32[8,8] p0), slice={[0:2], [0:8]}
      s01 = f32[5,8] slice(f32[8,8] p0), slice={[2:7], [0:8]}
      s10 = f32[2,8] slice(f32[8,8] p1), slice={[0:2], [0:8]}
      s11 = f32[5,8] slice(f32[8,8] p1), slice={[2:7], [0:8]}
      add0 = f32[2,8] add(f32[2,8] s00, f32[2,8] s10)
      add1 = f32[5,8] add(f32[5,8] s01, f32[5,8] s11)
      ROOT tuple = (f32[2,8], f32[5,8]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceDelayingTest, PartialSlice) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      s00 = f32[2,7] slice(f32[8,8] p0), slice={[0:2], [0:7]}
      s01 = f32[6,7] slice(f32[8,8] p0), slice={[2:8], [0:7]}
      s10 = f32[2,7] slice(f32[8,8] p1), slice={[0:2], [0:7]}
      s11 = f32[6,7] slice(f32[8,8] p1), slice={[2:8], [0:7]}
      add0 = f32[2,7] add(f32[2,7] s00, f32[2,7] s10)
      add1 = f32[6,7] add(f32[6,7] s01, f32[6,7] s11)
      ROOT tuple = (f32[2,7], f32[6,7]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceDelayingTest, OperantDisorder) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      s00 = f32[2,7] slice(f32[8,8] p0), slice={[0:2], [0:7]}
      s01 = f32[6,7] slice(f32[8,8] p0), slice={[2:8], [0:7]}
      s10 = f32[2,7] slice(f32[8,8] p1), slice={[0:2], [0:7]}
      s11 = f32[6,7] slice(f32[8,8] p1), slice={[2:8], [0:7]}
      add0 = f32[2,7] add(f32[2,7] s00, f32[2,7] s10)
      add1 = f32[6,7] add(f32[6,7] s11, f32[6,7] s01)
      ROOT tuple = (f32[2,7], f32[6,7]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceDelayingTest, SliceDisorder) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      s00 = f32[4,8] slice(f32[8,8] p0), slice={[0:4], [0:8]}
      s01 = f32[4,8] slice(f32[8,8] p0), slice={[4:8], [0:8]}
      s10 = f32[4,8] slice(f32[8,8] p1), slice={[0:4], [0:8]}
      s11 = f32[4,8] slice(f32[8,8] p1), slice={[4:8], [0:8]}
      add0 = f32[4,8] add(f32[4,8] s01, f32[4,8] s10)
      add1 = f32[4,8] add(f32[4,8] s00, f32[4,8] s11)
      ROOT tuple = (f32[4,8], f32[4,8]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceDelayingTest, DifferentOperator) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      s00 = f32[2,8] slice(f32[8,8] p0), slice={[0:2], [0:8]}
      s01 = f32[6,8] slice(f32[8,8] p0), slice={[2:8], [0:8]}
      s10 = f32[2,8] slice(f32[8,8] p1), slice={[0:2], [0:8]}
      s11 = f32[6,8] slice(f32[8,8] p1), slice={[2:8], [0:8]}
      mul = f32[2,8] multiply(f32[2,8] s00, f32[2,8] s10)
      add = f32[6,8] add(f32[6,8] s01, f32[6,8] s11)
      ROOT tuple = (f32[2,8], f32[6,8]) tuple(mul, add)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceDelayingTest, MultiUsers) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      s00 = f32[2,8] slice(f32[8,8] p0), slice={[0:2], [0:8]}
      s01 = f32[6,8] slice(f32[8,8] p0), slice={[2:8], [0:8]}
      s10 = f32[2,8] slice(f32[8,8] p1), slice={[0:2], [0:8]}
      s11 = f32[6,8] slice(f32[8,8] p1), slice={[2:8], [0:8]}
      add0 = f32[2,8] add(f32[2,8] s00, f32[2,8] s10)
      add1 = f32[6,8] add(f32[6,8] s01, f32[6,8] s11)
      mul0 = f32[2,8] multiply(f32[2,8] s00, f32[2,8] s10)
      mul1 = f32[6,8] multiply(f32[6,8] s01, f32[6,8] s11)
      ROOT tuple = (f32[2,8], f32[6,8]) tuple(add0, add1, mul0, mul1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_TRUE(result);
  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(result,
                          RunHloPass(&dce, module.get()));
  EXPECT_TRUE(result);
  EXPECT_EQ(9, module->entry_computation()->instruction_count());
  HloInstruction* inst = module->entry_computation()->root_instruction();
  EXPECT_THAT(inst,
      GmockMatch(m::Tuple(m::Slice(m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Multiply(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Multiply(m::Parameter(0), m::Parameter(1))))));
  HloInstruction* slice0 = inst->mutable_operand(0);
  EXPECT_EQ(slice0->slice_starts(), std::vector<int64>({0, 0}));
  EXPECT_EQ(slice0->slice_limits(), std::vector<int64>({2, 8}));
  EXPECT_EQ(slice0->slice_strides(), std::vector<int64>({1, 1}));
  HloInstruction* slice1 = inst->mutable_operand(1);
  EXPECT_EQ(slice1->slice_starts(), std::vector<int64>({2, 0}));
  EXPECT_EQ(slice1->slice_limits(), std::vector<int64>({8, 8}));
  EXPECT_EQ(slice1->slice_strides(), std::vector<int64>({1, 1}));
  HloInstruction* slice2 = inst->mutable_operand(2);
  EXPECT_EQ(slice2->slice_starts(), std::vector<int64>({0, 0}));
  EXPECT_EQ(slice2->slice_limits(), std::vector<int64>({2, 8}));
  EXPECT_EQ(slice2->slice_strides(), std::vector<int64>({1, 1}));
  HloInstruction* slice3 = inst->mutable_operand(3);
  EXPECT_EQ(slice3->slice_starts(), std::vector<int64>({2, 0}));
  EXPECT_EQ(slice3->slice_limits(), std::vector<int64>({8, 8}));
  EXPECT_EQ(slice3->slice_strides(), std::vector<int64>({1, 1}));
}

TEST_F(SliceDelayingTest, NonElementWise) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8] parameter(0)
      s00 = f32[2] slice(f32[8] p0), slice={[0:2]}
      s01 = f32[6] slice(f32[8] p0), slice={[2:8]}
      bc0 = f32[2,8] broadcast(f32[2] s00), dimensions={0}
      bc1 = f32[6,8] broadcast(f32[6] s01), dimensions={0}
      ROOT tuple = (f32[2,8], f32[6,8]) tuple(bc0, bc1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceDelayingTest, Stride) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      s00 = f32[4,8] slice(f32[8,8] p0), slice={[0:7:2], [0:8]}
      s01 = f32[4,8] slice(f32[8,8] p0), slice={[1:8:2], [0:8]}
      s10 = f32[4,8] slice(f32[8,8] p1), slice={[0:7:2], [0:8]}
      s11 = f32[4,8] slice(f32[8,8] p1), slice={[1:8:2], [0:8]}
      add0 = f32[4,8] add(f32[4,8] s00, f32[4,8] s10)
      add1 = f32[4,8] add(f32[4,8] s01, f32[4,8] s11)
      ROOT tuple = (f32[4,8], f32[4,8]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_TRUE(result);
  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(result,
                          RunHloPass(&dce, module.get()));
  EXPECT_TRUE(result);
  EXPECT_EQ(6, module->entry_computation()->instruction_count());
  HloInstruction* inst = module->entry_computation()->root_instruction();
  EXPECT_THAT(inst,
      GmockMatch(m::Tuple(m::Slice(m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Add(m::Parameter(0), m::Parameter(1))))));
  HloInstruction* slice0 = inst->mutable_operand(0);
  EXPECT_EQ(slice0->slice_starts(), std::vector<int64>({0, 0}));
  EXPECT_EQ(slice0->slice_limits(), std::vector<int64>({7, 8}));
  EXPECT_EQ(slice0->slice_strides(), std::vector<int64>({2, 1}));
  HloInstruction* slice1 = inst->mutable_operand(1);
  EXPECT_EQ(slice1->slice_starts(), std::vector<int64>({1, 0}));
  EXPECT_EQ(slice1->slice_limits(), std::vector<int64>({8, 8}));
  EXPECT_EQ(slice1->slice_strides(), std::vector<int64>({2, 1}));
}

TEST_F(SliceDelayingTest, NotAllSliceOperand) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,8] parameter(0)
      p1 = f32[2,8] parameter(1)
      p2 = f32[6,8] parameter(2)
      s00 = f32[2,8] slice(f32[8,8] p0), slice={[0:2], [0:8]}
      s01 = f32[6,8] slice(f32[8,8] p0), slice={[2:8], [0:8]}
      abs0 = f32[2,8] abs(f32[2,8] p1)
      abs1 = f32[6,8] abs(f32[6,8] p2)
      add0 = f32[2,8] add(f32[2,8] s00, f32[2,8] abs0)
      add1 = f32[6,8] add(f32[6,8] s01, f32[6,8] abs1)
      ROOT tuple = (f32[2,8], f32[6,8]) tuple(add0, add1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_FALSE(result);
}

TEST_F(SliceDelayingTest, Cascade) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      p2 = f32[8,8] parameter(2)
      s00 = f32[2,8] slice(f32[8,8] p0), slice={[0:2], [0:8]}
      s01 = f32[6,8] slice(f32[8,8] p0), slice={[2:8], [0:8]}
      s10 = f32[2,8] slice(f32[8,8] p1), slice={[0:2], [0:8]}
      s11 = f32[6,8] slice(f32[8,8] p1), slice={[2:8], [0:8]}
      add0 = f32[2,8] add(f32[2,8] s00, f32[2,8] s10)
      s21 = f32[6,8] slice(f32[8,8] p2), slice={[2:8], [0:8]}
      abs1 = f32[6,8] abs(f32[6,8] s21)
      add3 = f32[6,8] add(f32[6,8] s01, f32[6,8] abs1)
      add1 = f32[6,8] add(f32[6,8] s01, f32[6,8] s11)
      s20 = f32[2,8] slice(f32[8,8] p2), slice={[0:2], [0:8]}
      abs0 = f32[2,8] abs(f32[2,8] s20)
      add2 = f32[2,8] add(f32[2,8] s00, f32[2,8] abs0)
      ROOT tuple = (f32[2,8], f32[6,8]) tuple(add0, add1, add2, add3)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  SliceDelaying slice_delaying;
  TF_ASSERT_OK_AND_ASSIGN(bool result,
                          RunHloPass(&slice_delaying, module.get()));
  EXPECT_TRUE(result);
  HloDCE dce;
  TF_ASSERT_OK_AND_ASSIGN(result,
                          RunHloPass(&dce, module.get()));
  EXPECT_TRUE(result);
  EXPECT_EQ(11, module->entry_computation()->instruction_count());
  HloInstruction* inst = module->entry_computation()->root_instruction();
  EXPECT_THAT(inst,
      GmockMatch(m::Tuple(m::Slice(m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Add(m::Parameter(0), m::Parameter(1))),
          m::Slice(m::Add(m::Parameter(0), m::Abs(m::Parameter(2)))),
          m::Slice(m::Add(m::Parameter(0), m::Abs(m::Parameter(2)))))));
  HloInstruction* slice0 = inst->mutable_operand(0);
  EXPECT_EQ(slice0->slice_starts(), std::vector<int64>({0, 0}));
  EXPECT_EQ(slice0->slice_limits(), std::vector<int64>({2, 8}));
  EXPECT_EQ(slice0->slice_strides(), std::vector<int64>({1, 1}));
  HloInstruction* slice1 = inst->mutable_operand(1);
  EXPECT_EQ(slice1->slice_starts(), std::vector<int64>({2, 0}));
  EXPECT_EQ(slice1->slice_limits(), std::vector<int64>({8, 8}));
  EXPECT_EQ(slice1->slice_strides(), std::vector<int64>({1, 1}));
  HloInstruction* slice2 = inst->mutable_operand(2);
  EXPECT_EQ(slice2->slice_starts(), std::vector<int64>({0, 0}));
  EXPECT_EQ(slice2->slice_limits(), std::vector<int64>({2, 8}));
  EXPECT_EQ(slice2->slice_strides(), std::vector<int64>({1, 1}));
  HloInstruction* slice3 = inst->mutable_operand(3);
  EXPECT_EQ(slice3->slice_starts(), std::vector<int64>({2, 0}));
  EXPECT_EQ(slice3->slice_limits(), std::vector<int64>({8, 8}));
  EXPECT_EQ(slice3->slice_strides(), std::vector<int64>({1, 1}));
}

}  // namespace
}  // namespace xla

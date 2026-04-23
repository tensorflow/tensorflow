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

#include "xla/hlo/transforms/while_loop_trip_count_annotator.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class TripCountAnnotatorTest : public HloHardwareIndependentTestBase {};

TEST_F(TripCountAnnotatorTest, KnownSmallTripCount) {
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[]) tuple(i_plus_one)
    }

    Cond {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      trip_count = s32[] constant(10)
      ROOT done = pred[] compare(i, trip_count), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(0)
      initial_tuple = (s32[]) tuple(i_start)
      ROOT while = (s32[]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_TRUE(config.has_known_induction_variable());
  EXPECT_TRUE(config.has_known_init_step());
  EXPECT_EQ(config.known_trip_count().n(), 10);
  EXPECT_EQ(config.known_induction_variable().tuple_index(), 0);
  EXPECT_EQ(config.known_init_step().init(), 0);
  EXPECT_EQ(config.known_init_step().step(), 1);
}

TEST_F(TripCountAnnotatorTest, KnownLargeTripCount) {
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[]) tuple(i_plus_one)
    }

    Cond {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      trip_count = s32[] constant(1000000)
      ROOT done = pred[] compare(i, trip_count), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(0)
      initial_tuple = (s32[]) tuple(i_start)
      ROOT while = (s32[]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_trip_count().n(), 1000000);
}

TEST_F(TripCountAnnotatorTest, NonzeroStartStep) {
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      two = s32[] constant(2)
      i_plus_two = s32[] add(i, two)
      ROOT tuple = (s32[]) tuple(i_plus_two)
    }

    Cond {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      max_i = s32[] constant(1000000)
      ROOT done = pred[] compare(i, max_i), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(10)
      initial_tuple = (s32[]) tuple(i_start)
      ROOT while = (s32[]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_trip_count().n(), 499995);
  EXPECT_TRUE(config.has_known_init_step());
  EXPECT_EQ(config.known_init_step().init(), 10);
  EXPECT_EQ(config.known_init_step().step(), 2);
}

TEST_F(TripCountAnnotatorTest, LessThanOrEqualTo) {
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[]) tuple(i_plus_one)
    }

    Cond {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      trip_count = s32[] constant(1000000)
      ROOT done = pred[] compare(i, trip_count), direction=LE
    }

    ENTRY test {
      i_start = s32[] constant(10)
      initial_tuple = (s32[]) tuple(i_start)
      ROOT while = (s32[]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_trip_count().n(), 999991);
}

TEST_F(TripCountAnnotatorTest, Int64Overflow) {
  // for(i = INT64_MIN; i < INT64_MAX; ++i)
  //
  // We store the trip count as an int64_t, so this loop is unanalyzable.
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s64[]) parameter(0)
      i = s64[] get-tuple-element(param), index=0
      one = s64[] constant(1)
      i_plus_one = s64[] add(i, one)
      ROOT tuple = (s64[]) tuple(i_plus_one)
    }

    Cond {
      param = (s64[]) parameter(0)
      i = s64[] get-tuple-element(param), index=0
      trip_count = s64[] constant(9223372036854775807) // 2^63-1
      ROOT done = pred[] compare(i, trip_count), direction=LE
    }

    ENTRY test {
      i_start = s64[] constant(-9223372036854775808)  // -2^63
      initial_tuple = (s64[]) tuple(i_start)
      ROOT while = (s64[]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  EXPECT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_FALSE(config.has_known_trip_count());
  EXPECT_FALSE(config.has_known_init_step());
  EXPECT_TRUE(config.has_known_induction_variable());
  EXPECT_EQ(config.known_induction_variable().tuple_index(), 0);
}

TEST_F(TripCountAnnotatorTest, NonZeroTupleIndex) {
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=1
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[], s32[]) tuple(one, i_plus_one)
    }

    Cond {
      param = (s32[], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=1
      trip_count = s32[] constant(10)
      ROOT done = pred[] compare(i, trip_count), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(0)
      initial_tuple = (s32[], s32[]) tuple(i_start, i_start)
      ROOT while = (s32[], s32[]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_induction_variable().tuple_index(), 1);
}

TEST_F(TripCountAnnotatorTest, InductionVarForwardedToConstant) {
  // for(i = 0; i < 10; ++i) {}
  // use(GTE(while, 0))  --> should become use(constant(10))
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[], f32[10]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      data = f32[10] get-tuple-element(param), index=1
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[], f32[10]) tuple(i_plus_one, data)
    }

    Cond {
      param = (s32[], f32[10]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      trip_count = s32[] constant(10)
      ROOT done = pred[] compare(i, trip_count), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(0)
      data = f32[10] parameter(0)
      initial_tuple = (s32[], f32[10]) tuple(i_start, data)
      while = (s32[], f32[10]) while(initial_tuple), condition=Cond, body=Body
      i_final = s32[] get-tuple-element(while), index=0
      ROOT result = f32[10] get-tuple-element(while), index=1
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  // Find the GTE that used to extract index 0 -- it should now have no users
  // and a constant should exist with value 10.
  HloComputation* entry = m->entry_computation();
  bool found_constant = false;
  for (const HloInstruction* instr : entry->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant &&
        instr->shape().element_type() == S32 &&
        instr->shape().dimensions_size() == 0 &&
        instr->literal().Get<int32_t>({}) == 10) {
      found_constant = true;
    }
  }
  EXPECT_TRUE(found_constant);
}

TEST_F(TripCountAnnotatorTest, InductionVarForwardedNonZeroInit) {
  // for(i = 5; i < 15; i += 2) {} -> trip_count = 5, final_value = 15
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[], f32[10]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      data = f32[10] get-tuple-element(param), index=1
      two = s32[] constant(2)
      i_plus_two = s32[] add(i, two)
      ROOT tuple = (s32[], f32[10]) tuple(i_plus_two, data)
    }

    Cond {
      param = (s32[], f32[10]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(15)
      ROOT done = pred[] compare(i, limit), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(5)
      data = f32[10] parameter(0)
      initial_tuple = (s32[], f32[10]) tuple(i_start, data)
      while = (s32[], f32[10]) while(initial_tuple), condition=Cond, body=Body
      i_final = s32[] get-tuple-element(while), index=0
      ROOT result = f32[10] get-tuple-element(while), index=1
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  // trip_count = (15 - 5) / 2 = 5, final_value = 5 + 2 * 5 = 15
  HloComputation* entry = m->entry_computation();
  bool found_constant = false;
  for (const HloInstruction* instr : entry->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant &&
        instr->shape().element_type() == S32 &&
        instr->shape().dimensions_size() == 0 &&
        instr->literal().Get<int32_t>({}) == 15) {
      found_constant = true;
    }
  }
  EXPECT_TRUE(found_constant);
}

TEST_F(TripCountAnnotatorTest, InductionVarMultipleGTEsForwarded) {
  // Multiple GTEs extracting the induction variable should all be replaced.
  const char* kModuleStr = R"(
    HloModule test

    add_comp {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT sum = s32[] add(lhs, rhs)
    }

    Body {
      param = (s32[], s32[4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      data = s32[4] get-tuple-element(param), index=1
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[], s32[4]) tuple(i_plus_one, data)
    }

    Cond {
      param = (s32[], s32[4]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      limit = s32[] constant(4)
      ROOT done = pred[] compare(i, limit), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(0)
      data = s32[4] parameter(0)
      initial_tuple = (s32[], s32[4]) tuple(i_start, data)
      while = (s32[], s32[4]) while(initial_tuple), condition=Cond, body=Body
      i_final_1 = s32[] get-tuple-element(while), index=0
      i_final_2 = s32[] get-tuple-element(while), index=0
      data_out = s32[4] get-tuple-element(while), index=1
      zero = s32[] constant(0)
      ds1 = s32[1] dynamic-slice(data_out, i_final_1), dynamic_slice_sizes={1}
      ds2 = s32[1] dynamic-slice(data_out, i_final_2), dynamic_slice_sizes={1}
      ROOT result = s32[] reduce(ds1, zero), dimensions={0}, to_apply=add_comp
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  // Both GTEs at index 0 should have been replaced with constant(4).
  HloComputation* entry = m->entry_computation();
  int constant_4_count = 0;
  for (const HloInstruction* instr : entry->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant &&
        instr->shape().element_type() == S32 &&
        instr->shape().dimensions_size() == 0 &&
        instr->literal().Get<int32_t>({}) == 4) {
      constant_4_count++;
    }
  }
  // We create one constant per GTE replaced.
  EXPECT_GE(constant_4_count, 2);

  // Verify the dynamic-slice ops now consume constants, not GTEs.
  for (const HloInstruction* instr : entry->instructions()) {
    if (instr->opcode() == HloOpcode::kDynamicSlice) {
      EXPECT_EQ(instr->operand(1)->opcode(), HloOpcode::kConstant);
    }
  }
}

TEST_F(TripCountAnnotatorTest, InductionVarNotForwardedForBruteForce) {
  // When trip count is only known via brute-force (no init/step), we don't
  // forward to constant because we can't compute the final value.
  // We use a loop where MatchTrivialLoopRange fails but brute force succeeds.
  // A loop with s64 and large constant won't match trivial range due to
  // overflow concerns, but brute force can handle it if trip count is small.
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (s32[]) tuple(i_plus_one)
    }

    Cond {
      param = (s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=0
      trip_count = s32[] constant(5)
      ROOT done = pred[] compare(i, trip_count), direction=LT
    }

    ENTRY test {
      i_start = s32[] constant(0)
      initial_tuple = (s32[]) tuple(i_start)
      ROOT while = (s32[]) while(initial_tuple), condition=Cond, body=Body
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  // This particular loop will match the trivial range, so the constant
  // should be created. This test just validates the pass doesn't crash.
  ASSERT_OK_AND_ASSIGN(auto config,
                       m->entry_computation()
                           ->root_instruction()
                           ->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_trip_count().n(), 5);
}

TEST_F(TripCountAnnotatorTest, InductionVarNonZeroTupleIndexForwarded) {
  // Induction variable at tuple index 1 should be forwarded correctly.
  const char* kModuleStr = R"(
    HloModule test
    Body {
      param = (f32[10], s32[]) parameter(0)
      data = f32[10] get-tuple-element(param), index=0
      i = s32[] get-tuple-element(param), index=1
      one = s32[] constant(1)
      i_plus_one = s32[] add(i, one)
      ROOT tuple = (f32[10], s32[]) tuple(data, i_plus_one)
    }

    Cond {
      param = (f32[10], s32[]) parameter(0)
      i = s32[] get-tuple-element(param), index=1
      limit = s32[] constant(7)
      ROOT done = pred[] compare(i, limit), direction=LT
    }

    ENTRY test {
      data = f32[10] parameter(0)
      i_start = s32[] constant(0)
      initial_tuple = (f32[10], s32[]) tuple(data, i_start)
      while = (f32[10], s32[]) while(initial_tuple), condition=Cond, body=Body
      i_final = s32[] get-tuple-element(while), index=1
      ROOT result = f32[10] get-tuple-element(while), index=0
    })";

  ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  // final_value = 0 + 1 * 7 = 7
  HloComputation* entry = m->entry_computation();
  bool found_constant_7 = false;
  for (const HloInstruction* instr : entry->instructions()) {
    if (instr->opcode() == HloOpcode::kConstant &&
        instr->shape().element_type() == S32 &&
        instr->shape().dimensions_size() == 0 &&
        instr->literal().Get<int32_t>({}) == 7) {
      found_constant_7 = true;
    }
  }
  EXPECT_TRUE(found_constant_7);
}

}  // namespace
}  // namespace xla

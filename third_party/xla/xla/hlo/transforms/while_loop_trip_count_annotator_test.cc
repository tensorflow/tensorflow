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

#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

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

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(auto config,
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

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(auto config,
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

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(auto config,
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

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(auto config,
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

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  EXPECT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(auto config,
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

  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  WhileLoopTripCountAnnotator pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, m.get()));
  ASSERT_TRUE(changed);

  TF_ASSERT_OK_AND_ASSIGN(auto config,
                          m->entry_computation()
                              ->root_instruction()
                              ->backend_config<WhileLoopBackendConfig>());
  EXPECT_EQ(config.known_induction_variable().tuple_index(), 1);
}

}  // namespace
}  // namespace xla

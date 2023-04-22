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

#include "tensorflow/compiler/xla/service/while_loop_trip_count_annotator.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class TripCountAnnotatorTest : public HloTestBase {};

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
  EXPECT_EQ(10, config.known_trip_count().n());
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
  EXPECT_EQ(1000000, config.known_trip_count().n());
}

TEST_F(TripCountAnnotatorTest, NonzeroStart) {
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
  EXPECT_EQ(999990, config.known_trip_count().n());
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
  EXPECT_EQ(999991, config.known_trip_count().n());
}

TEST_F(TripCountAnnotatorTest, Int64Overflow) {
  // for(i = INT64_MIN; i < INT64_MAX; ++i)
  //
  // We store the trip count as an int64, so this loop is unanalyzable.
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
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla

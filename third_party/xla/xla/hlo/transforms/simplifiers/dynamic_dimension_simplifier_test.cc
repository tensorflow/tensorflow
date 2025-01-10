/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/dynamic_dimension_simplifier.h"

#include <memory>

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/shape_inference.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/types.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace m = match;

class DynamicDimensionSimplifierTest : public HloHardwareIndependentTestBase {};

TEST_F(DynamicDimensionSimplifierTest, ForwardConcat) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[1] parameter(0)
      p1 = s32[1] parameter(1)
      p2 = s32[1] parameter(2)
      concat1 = s32[2] concatenate(p0, p1), dimensions={0}
      ROOT concat2 = s32[3] concatenate(concat1, p2), dimensions={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  DynamicDimensionSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Concatenate(m::Parameter(0), m::Parameter(1),
                                        m::Parameter(2))));
}

TEST_F(DynamicDimensionSimplifierTest, DoNotForwardConcatMultipleDims) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[1, 1] parameter(0)
      p1 = s32[1, 1] parameter(1)
      p2 = s32[2, 1] parameter(2)
      concat1 = s32[2, 1] concatenate(p0, p1), dimensions={0}
      ROOT concat2 = s32[2, 2] concatenate(concat1, p2), dimensions={1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  DynamicDimensionSimplifier simplifier;
  ASSERT_FALSE(simplifier.Run(m.get()).value());
}

TEST_F(DynamicDimensionSimplifierTest, ForwardConcatSlice) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[1] parameter(0)
      p1 = s32[1] parameter(1)
      p2 = s32[1] parameter(2)
      concat = s32[3] concatenate(p0, p1, p2), dimensions={0}
      ROOT slice = s32[1] slice(concat), slice={[1:2]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  DynamicDimensionSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(1)));
}

TEST_F(DynamicDimensionSimplifierTest, DoNotForwardConcatSliceSizeMismatch) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[1] parameter(0)
      p1 = s32[1] parameter(1)
      p2 = s32[1] parameter(2)
      concat = s32[3] concatenate(p0, p1, p2), dimensions={0}
      ROOT slice = s32[2] slice(concat), slice={[1:3]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  DynamicDimensionSimplifier simplifier;
  ASSERT_FALSE(simplifier.Run(m.get()).value());
}

TEST_F(DynamicDimensionSimplifierTest, DoNotForwardConcatSliceStrided) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[1] parameter(0)
      p1 = s32[1] parameter(1)
      p2 = s32[1] parameter(2)
      concat = s32[3] concatenate(p0, p1, p2), dimensions={0}
      ROOT slice = s32[1] slice(concat), slice={[1:2:2]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  DynamicDimensionSimplifier simplifier;
  ASSERT_FALSE(simplifier.Run(m.get()).value());
}

TEST_F(DynamicDimensionSimplifierTest, BroadcastReshapeForwarding) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[] parameter(0)
      broadcast = s32[1] broadcast(p0), dimensions={}
      ROOT reshape = s32[] reshape(broadcast)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  DynamicDimensionSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(DynamicDimensionSimplifierTest, ReshapeReshapeForwarding) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[] parameter(0)
      reshape = s32[1] reshape(p0)
      ROOT reshape2 = s32[] reshape(reshape)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  DynamicDimensionSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST_F(DynamicDimensionSimplifierTest,
       DoNotReshapeReshapeForwardingShapeMismatch) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[1, 1] parameter(0)
      reshape = s32[1] reshape(p0)
      ROOT reshape2 = s32[] reshape(reshape)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  DynamicDimensionSimplifier simplifier;
  ASSERT_FALSE(simplifier.Run(m.get()).value());
}

TEST_F(DynamicDimensionSimplifierTest, IdConvertRemoving) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[1] parameter(0)
      ROOT reshape2 = s32[1] convert(p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kModuleStr));
  DynamicDimensionSimplifier simplifier;
  ASSERT_TRUE(simplifier.Run(m.get()).value());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

}  // namespace
}  // namespace xla

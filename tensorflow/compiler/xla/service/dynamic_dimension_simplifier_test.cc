/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/dynamic_dimension_simplifier.h"

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace m = match;

class DynamicDimensionSimplifierTest : public HloTestBase {};

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
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
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
  ASSERT_FALSE(simplifier.Run(m.get()).ValueOrDie());
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
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
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
  ASSERT_FALSE(simplifier.Run(m.get()).ValueOrDie());
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
  ASSERT_FALSE(simplifier.Run(m.get()).ValueOrDie());
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
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
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
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
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
  ASSERT_FALSE(simplifier.Run(m.get()).ValueOrDie());
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
  ASSERT_TRUE(simplifier.Run(m.get()).ValueOrDie());
  EXPECT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

}  // namespace
}  // namespace xla

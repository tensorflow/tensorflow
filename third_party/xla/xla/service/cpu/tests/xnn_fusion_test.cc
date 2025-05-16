/* Copyright 2024 The OpenXLA Authors.

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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/test.h"

namespace xla::cpu {
namespace {

using ::testing::HasSubstr;
using XnnFusionTest = HloTestBase;

TEST_F(XnnFusionTest, AddAndMultiply) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule add_and_multiply

    xnn_fusion {
      %lhs = f32[4] parameter(0)
      %rhs = f32[4] parameter(1)
      %add = f32[4] add(%lhs, %rhs)
      ROOT %mul = f32[4] multiply(%add, %add)
    }

    ENTRY entry {
      %p0 = f32[4] parameter(0)
      %p1 = f32[4] parameter(1)
      ROOT %fusion = f32[4] fusion(%p0, %p1), kind=kCustom, calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    })";

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{0.0}));
}

TEST_F(XnnFusionTest, BatchedDot) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule dot_add_multiply

    xnn_fusion {
      %lhs = f32[2,3,4,5] parameter(0)
      %rhs = f32[2,3,5,6] parameter(1)
      ROOT %dot = f32[2,3,4,6] dot(%lhs, %rhs),
        lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
        lhs_contracting_dims={3}, rhs_contracting_dims={2}
    }

    ENTRY entry {
      %lhs = f32[2,3,4,5] parameter(0)
      %rhs = f32[2,3,5,6] parameter(1)
      ROOT %fusion = f32[2,3,4,6] fusion(%lhs, %rhs),
        kind=kCustom, calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    })";

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{1e-7}));
}

TEST_F(XnnFusionTest, DotAddMultiply) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule dot_add_multiply

    xnn_fusion {
      %lhs = f32[4,5] parameter(0)
      %rhs = f32[5,6] parameter(1)
      %addend = f32[4,6] parameter(2)
      %multiplier = f32[4,6] parameter(3)
      %dot = f32[4,6] dot(%lhs, %rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      %add = f32[4,6] add(%dot, %addend)
      ROOT %mul = f32[4,6] multiply(%add, %multiplier)
    }

    ENTRY entry {
      %lhs = f32[4,5] parameter(0)
      %rhs = f32[5,6] parameter(1)
      %addend = f32[4, 6] parameter(2)
      %multiplier = f32[4, 6] parameter(3)
      ROOT %fusion = f32[4,6] fusion(%lhs, %rhs, %addend, %multiplier),
        kind=kCustom, calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    })";

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{1e-7}));
}

TEST_F(XnnFusionTest, DotRhsTransposedAndMultiply) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule dot_rhs_transposed_and_multiply

    xnn_fusion {
      %lhs = f32[4,5] parameter(0)
      %rhs = f32[6,5] parameter(1)
      %multiplier = f32[4,6] parameter(2)
      %dot = f32[4,6] dot(%lhs, %rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={1}
      ROOT %mul = f32[4,6] multiply(%dot, %multiplier)
    }

    ENTRY entry {
      %lhs = f32[4,5] parameter(0)
      %rhs = f32[6,5] parameter(1)
      %multiplier = f32[4, 6] parameter(2)
      ROOT %fusion = f32[4,6] fusion(%lhs, %rhs, %multiplier),
        kind=kCustom, calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    })";

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{1e-7}));
}

TEST_F(XnnFusionTest, UnsupportedDot) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule unsupported_dot

    xnn_fusion {
      %lhs = f32[5,4] parameter(0)
      %rhs = f32[5,6] parameter(1)
      ROOT %dot = f32[4,6] dot(%lhs, %rhs),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    }

    ENTRY entry {
      %lhs = f32[5,4] parameter(0)
      %rhs = f32[5,6] parameter(1)
      ROOT %fusion = f32[4,6] fusion(%lhs, %rhs),
        kind=kCustom, calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    })";

  auto status = RunAndCompare(kModuleStr, ErrorSpec{0.0});
  EXPECT_FALSE(status);
  EXPECT_THAT(status.message(),
              HasSubstr("Unsupported XNNPACK Dot op variation"));
}

TEST_F(XnnFusionTest, UnsupportedOp) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule unsupported_sqrt

    xnn_fusion {
      %x = f32[10] parameter(0)
      ROOT %sqrt = f32[10] sqrt(%x)
    }

    ENTRY entry {
      %x = f32[10] parameter(0)
      ROOT %sqrt = f32[10] fusion(%x), kind=kCustom, calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    })";

  auto status = RunAndCompare(kModuleStr, ErrorSpec{0.0});
  EXPECT_FALSE(status);
  EXPECT_THAT(status.message(),
              HasSubstr("Unsupported XNNPACK fusion instruction"));
}

}  // namespace
}  // namespace xla::cpu

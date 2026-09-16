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

#include "xla/hlo/transforms/simplifiers/gemv_rewriter.h"

#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class GemvRewriterTest : public HloHardwareIndependentTestBase,
                         public ::testing::WithParamInterface<bool> {
 protected:
  GemvRewriterTest() {
    is_layout_sensitive_ = GetParam();
    reshape_or_bitcast_ = is_layout_sensitive_ ? "bitcast" : "reshape";
  }

  bool is_layout_sensitive_;
  const char* reshape_or_bitcast_;
};

TEST_P(GemvRewriterTest, RewriteMatrixVectorMultiplicationToGemm) {
  const char* hlo = R"(
  HloModule m

  ENTRY e {
    p0 = f32[32,7] parameter(0)
    p1 = f32[7] parameter(1)
    ROOT d = f32[32] dot(p0, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  // clang-format off
  std::string expected = absl::StrCat(R"(
// CHECK:  %[[P0:.*]] = f32[32,7]{1,0} parameter(0)
// CHECK:  %[[P1:.*]] = f32[7]{0} parameter(1)
// CHECK:  %[[RESHAPE_OR_BITCAST:.*]] = f32[7,1]{1,0}
// CHECK-SAME: )", reshape_or_bitcast_, R"((%[[P1]])
// CHECK:  %[[DOT:.*]] = f32[32,1]{1,0} dot(%[[P0]], %[[RESHAPE_OR_BITCAST]]),
// CHECK-SAME: lhs_contracting_dims={1}, rhs_contracting_dims={0}
// CHECK:  ROOT %[[ROOT:.*]] = f32[32]{0}
// CHECK-SAME: )", reshape_or_bitcast_, R"((%[[DOT]])
)");
  // clang-format on

  RunAndFilecheckHloRewrite(hlo, GemvRewriter(is_layout_sensitive_), expected);
}

TEST_P(GemvRewriterTest, RewriteVectorMatrixMultiplicationToGemm) {
  const char* hlo = R"(
  HloModule m

  ENTRY e {
    p0 = f32[7] parameter(0)
    p1 = f32[7,32] parameter(1)
    ROOT d = f32[32] dot(p0, p1),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}
  })";

  // clang-format off
  std::string expected = absl::StrCat(R"(
// CHECK:  %[[P0:.*]] = f32[7]{0} parameter(0)
// CHECK:  %[[RESHAPE_OR_BITCAST:.*]] = f32[7,1]{1,0}
// CHECK-SAME: )", reshape_or_bitcast_, R"((%[[P0]])
// CHECK:  %[[P1:.*]] = f32[7,32]{1,0} parameter(1)
// CHECK:  %[[DOT:.*]] = f32[1,32]{1,0} dot(%[[RESHAPE_OR_BITCAST]], %[[P1]]),
// CHECK-SAME: lhs_contracting_dims={0}, rhs_contracting_dims={0}
// CHECK:  ROOT %[[ROOT:.*]].1 = f32[32]{0}
// CHECK-SAME: )", reshape_or_bitcast_, R"((%[[DOT]])
)");
  // clang-format on

  RunAndFilecheckHloRewrite(hlo, GemvRewriter(is_layout_sensitive_), expected);
}

TEST_P(GemvRewriterTest, RewriteMatrixVectorMultiplicationWithBatch) {
  const char* hlo = R"(
  HloModule m

  ENTRY e {
    p0 = f32[2,5,32,7] parameter(0)
    p1 = f32[2,5,7] parameter(1)
    ROOT d = f32[2,5,32] dot(p0, p1),
      lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
      lhs_contracting_dims={3}, rhs_contracting_dims={2}
  })";

  // clang-format off
  std::string expected = absl::StrCat(R"(
// CHECK:  %[[P0:.*]] = f32[2,5,32,7]{3,2,1,0} parameter(0)
// CHECK:  %[[P1:.*]] = f32[2,5,7]{2,1,0} parameter(1)
// CHECK:  %[[RESHAPE_OR_BITCAST:.*]] = f32[2,5,7,1]{3,2,1,0}
// CHECK-SAME: )", reshape_or_bitcast_, R"((%[[P1]])
// CHECK:  %[[DOT:.*]] = f32[2,5,32,1]{3,2,1,0} dot(%[[P0]], %[[RESHAPE_OR_BITCAST]]),
// CHECK-SAME: lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
// CHECK:  ROOT %[[ROOT:.*]] = f32[2,5,32]{2,1,0}
// CHECK-SAME: )", reshape_or_bitcast_, R"((%[[DOT]])
)");
  // clang-format on

  RunAndFilecheckHloRewrite(hlo, GemvRewriter(is_layout_sensitive_), expected);
}

TEST_P(GemvRewriterTest, DotNotRewriteVectorVectorMultiplication) {
  const char* hlo = R"(
  HloModule m

  ENTRY e {
    p0 = f32[7] parameter(0)
    p1 = f32[7] parameter(1)
    ROOT d = f32[] dot(p0, p1),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}
  })";

  RunAndFilecheckHloRewrite(hlo, GemvRewriter(is_layout_sensitive_),
                            /*expected=*/std::nullopt);
}

TEST_P(GemvRewriterTest, DotNotRewriteMatrixMatrixMultiplication) {
  const char* hlo = R"(
  HloModule m

  ENTRY e {
    p0 = f32[5,7] parameter(0)
    p1 = f32[7,32] parameter(1)
    ROOT d = f32[5,32] dot(p0, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  RunAndFilecheckHloRewrite(hlo, GemvRewriter(is_layout_sensitive_),
                            /*expected=*/std::nullopt);
}

TEST_P(GemvRewriterTest, DoNotRewriteDotsWithNonNormalizedLayout) {
  const char* hlo = R"(
  HloModule m

  ENTRY e {
    p0 = f32[5,32,7]{2,1,0} parameter(0)
    p1 = f32[5,7]{0,1} parameter(1)
    ROOT d = f32[5,32]{0,1} dot(p0, p1),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  GemvRewriter rewriter(is_layout_sensitive_);
  absl::StatusOr<bool> result = this->RunHloPass(&rewriter, module.get());
  if (is_layout_sensitive_) {
    EXPECT_FALSE(result.ok());
    EXPECT_THAT(result.status().message(),
                ::testing::HasSubstr("Layout is not normalized."));
  } else {
    // No rewrite when the layout is not normalized, but the pass should succeed
    // in this configuration.
    EXPECT_TRUE(result.ok());
    EXPECT_FALSE(result.value());
  }
}

INSTANTIATE_TEST_SUITE_P(
    GemvRewriterTestSuite, GemvRewriterTest, ::testing::Values(true, false),
    [](const ::testing::TestParamInfo<GemvRewriterTest::ParamType>& info) {
      return info.param ? "LayoutSensitive" : "NotLayoutSensitive";
    });

}  // namespace
}  // namespace xla

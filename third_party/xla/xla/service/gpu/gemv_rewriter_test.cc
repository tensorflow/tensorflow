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

#include "xla/service/gpu/gemv_rewriter.h"

#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

class GemvRewriterTest : public HloTestBase {};

TEST_F(GemvRewriterTest, RewriteMatrixVectorMultiplicationToGemm) {
  const char* hlo = R"(
  HloModule m

  ENTRY e {
    p0 = f32[32,7] parameter(0)
    p1 = f32[7] parameter(1)
    ROOT d = f32[32] dot(p0, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  const char* expected = R"()
// CHECK:  %[[P0:.*]] = f32[32,7]{1,0} parameter(0)
// CHECK:  %[[P1:.*]] = f32[7]{0} parameter(1)
// CHECK:  %[[BITCAST:.*]] = f32[7,1]{1,0} bitcast(%[[P1]])
// CHECK:  %[[DOT:.*]] = f32[32,1]{1,0} dot(%[[P0]], %[[BITCAST]]), lhs_contracting_dims={1}, rhs_contracting_dims={0}
// CHECK:  ROOT %[[ROOT:.*]] = f32[32]{0} bitcast(%[[DOT]])
})";

  RunAndFilecheckHloRewrite(hlo, GemvRewriter(), expected);
}

TEST_F(GemvRewriterTest, RewriteVectorMatrixMultiplicationToGemm) {
  const char* hlo = R"(
  HloModule m

  ENTRY e {
    p0 = f32[7] parameter(0)
    p1 = f32[7,32] parameter(1)
    ROOT d = f32[32] dot(p0, p1),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}
  })";

  const char* expected = R"()
// CHECK:  %[[P0:.*]] = f32[7]{0} parameter(0)
// CHECK:  %[[BITCAST:.*]] = f32[7,1]{1,0} bitcast(%[[P0]])
// CHECK:  %[[P1:.*]] = f32[7,32]{1,0} parameter(1)
// CHECK:  %[[DOT:.*]] = f32[1,32]{1,0} dot(%[[BITCAST]], %[[P1]]), lhs_contracting_dims={0}, rhs_contracting_dims={0}
// CHECK:  ROOT %[[ROOT:.*]].1 = f32[32]{0} bitcast(%[[DOT]])
})";

  RunAndFilecheckHloRewrite(hlo, GemvRewriter(), expected);
}

TEST_F(GemvRewriterTest, RewriteMatrixVectorMultiplicationWithBatch) {
  const char* hlo = R"(
  HloModule m

  ENTRY e {
    p0 = f32[2,5,32,7] parameter(0)
    p1 = f32[2,5,7] parameter(1)
    ROOT d = f32[2,5,32] dot(p0, p1),
      lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
      lhs_contracting_dims={3}, rhs_contracting_dims={2}
  })";

  const char* expected = R"()
// CHECK:  %[[P0:.*]] = f32[2,5,32,7]{3,2,1,0} parameter(0)
// CHECK:  %[[P1:.*]] = f32[2,5,7]{2,1,0} parameter(1)
// CHECK:  %[[BITCAST:.*]] = f32[2,5,7,1]{3,2,1,0} bitcast(%[[P1]])
// CHECK:  %[[DOT:.*]] = f32[2,5,32,1]{3,2,1,0} dot(%[[P0]], %[[BITCAST]]),
// CHECK-SAME: lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
// CHECK:  ROOT %[[ROOT:.*]] = f32[2,5,32]{2,1,0} bitcast(%[[DOT]])
})";

  RunAndFilecheckHloRewrite(hlo, GemvRewriter(), expected);
}

TEST_F(GemvRewriterTest, DotNotRewriteVectorVectorMultiplication) {
  const char* hlo = R"(
  HloModule m

  ENTRY e {
    p0 = f32[7] parameter(0)
    p1 = f32[7] parameter(1)
    ROOT d = f32[] dot(p0, p1),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}
  })";

  RunAndFilecheckHloRewrite(hlo, GemvRewriter(), /*expected=*/std::nullopt);
}

TEST_F(GemvRewriterTest, DotNotRewriteMatrixMatrixMultiplication) {
  const char* hlo = R"(
  HloModule m

  ENTRY e {
    p0 = f32[5,7] parameter(0)
    p1 = f32[7,32] parameter(1)
    ROOT d = f32[5,32] dot(p0, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  RunAndFilecheckHloRewrite(hlo, GemvRewriter(), /*expected=*/std::nullopt);
}

TEST_F(GemvRewriterTest, DoNotRewriteDotsWithNonNormalizedLayout) {
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
  GemvRewriter rewriter;
  absl::StatusOr<bool> result = this->RunHloPass(&rewriter, module.get());
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().message(), "Layout is not normalized.");
}

}  // namespace
}  // namespace xla::gpu

/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/dot_dimension_normalizer.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace {

using DotDimensionNormalizerTest = HloHardwareIndependentTestBase;

TEST_F(DotDimensionNormalizerTest, ConsecutiveContractingDims) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
 p0 = bf16[12,2,4] parameter(0)
 p1 = bf16[2,4,44] parameter(1)
 ROOT d = bf16[12,44] dot(p0, p1),
  lhs_batch_dims={}, lhs_contracting_dims={1,2},
  rhs_batch_dims={}, rhs_contracting_dims={0,1}
})";

  RunAndFilecheckHloRewrite(kHloText, DotDimensionNormalizer(), R"(
; CHECK: %[[R0:.*]] = bf16[12,8]{1,0} reshape(%p0)
; CHECK: %[[R1:.*]] = bf16[8,44]{1,0} reshape(%p1)
; CHECK: ROOT {{[^ ]+}} = bf16[12,44]{1,0} dot(%[[R0]], %[[R1]])
; CHECK-SAME: lhs_contracting_dims={1}
; CHECK-SAME: rhs_contracting_dims={0}
  )");
}

TEST_F(DotDimensionNormalizerTest, NonConsecutiveContractingDims) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
 p0 = bf16[2,12,4] parameter(0)
 p1 = bf16[2,4,44] parameter(1)
 ROOT d = bf16[12,44] dot(p0, p1),
  lhs_batch_dims={}, lhs_contracting_dims={0,2},
  rhs_batch_dims={}, rhs_contracting_dims={0,1}
})";

  RunAndFilecheckHloRewrite(kHloText, DotDimensionNormalizer(), R"(
; CHECK: %[[T0:.*]] = bf16[12,2,4]{2,0,1} transpose(%p0), dimensions={1,0,2}
; CHECK: %[[R0:.*]] = bf16[12,8]{{.*}} reshape(%[[T0]])
; CHECK: %[[R1:.*]] = bf16[8,44]{1,0} reshape(%p1)
; CHECK: ROOT {{[^ ]+}} = bf16[12,44]{1,0} dot(%[[R0]], %[[R1]])
; CHECK-SAME: lhs_contracting_dims={1}
; CHECK-SAME: rhs_contracting_dims={0}
  )");
}

TEST_F(DotDimensionNormalizerTest, UnsortedConsecutiveContractingDims) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
 p0 = bf16[4,5,518,20,10] parameter(0)
 p1 = bf16[4,5,10,20,6,7] parameter(1)
 ROOT d = bf16[4,5,518,6,7] dot(p0, p1),
  lhs_batch_dims={0,1}, lhs_contracting_dims={3,4},
  rhs_batch_dims={0,1}, rhs_contracting_dims={3,2}
})";

  RunAndFilecheckHloRewrite(kHloText, DotDimensionNormalizer(), R"(
; CHECK: %[[R0:.*]] = bf16[4,5,518,200]{{.*}} reshape(%p0)
; CHECK: %[[T1:.*]] = bf16[4,5,6,7,20,10]{{.*}} transpose(%p1), dimensions={0,1,4,5,3,2}
; CHECK: %[[R1:.*]] = bf16[4,5,6,7,200]{{.*}} reshape(%[[T1]])
; CHECK: ROOT {{[^ ]+}} = bf16[4,5,518,6,7]{{.*}} dot(%[[R0]], %[[R1]])
; CHECK-SAME: lhs_batch_dims={0,1}
; CHECK-SAME: lhs_contracting_dims={3}
; CHECK-SAME: rhs_batch_dims={0,1}
; CHECK-SAME: rhs_contracting_dims={4}
  )");
}

TEST_F(DotDimensionNormalizerTest, NoChangeWhenOneContractingDim) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
 p0 = bf16[12,2] parameter(0)
 p1 = bf16[2,44] parameter(1)
 ROOT d = bf16[12,44] dot(p0, p1),
  lhs_batch_dims={}, lhs_contracting_dims={1},
  rhs_batch_dims={}, rhs_contracting_dims={0}
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  ASSERT_OK_AND_ASSIGN(bool modified,
                       DotDimensionNormalizer().Run(module.get()));
  EXPECT_FALSE(modified);
}

TEST_F(DotDimensionNormalizerTest, FoldPreExistingReshape) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
 p0 = bf16[12,8] parameter(0)
 r0 = bf16[12,2,4] reshape(p0)
 p1 = bf16[2,4,44] parameter(1)
 ROOT d = bf16[12,44] dot(r0, p1),
  lhs_batch_dims={}, lhs_contracting_dims={1,2},
  rhs_batch_dims={}, rhs_contracting_dims={0,1}
})";

  RunAndFilecheckHloRewrite(kHloText, DotDimensionNormalizer(), R"(
; CHECK: %[[R1:.*]] = bf16[8,44]{{.*}} reshape(%p1)
; CHECK: ROOT {{[^ ]+}} = bf16[12,44]{{.*}} dot(%p0, %[[R1]])
; CHECK-SAME: lhs_contracting_dims={1}
; CHECK-SAME: rhs_contracting_dims={0}
  )");
}

TEST_F(DotDimensionNormalizerTest, FoldPreExistingTransposeAndReshape) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
 p0 = bf16[8,12] parameter(0)
 t0 = bf16[12,8] transpose(p0), dimensions={1,0}
 r0 = bf16[12,2,4] reshape(t0)
 p1 = bf16[2,4,44] parameter(1)
 ROOT d = bf16[12,44] dot(r0, p1),
  lhs_batch_dims={}, lhs_contracting_dims={1,2},
  rhs_batch_dims={}, rhs_contracting_dims={0,1}
})";

  RunAndFilecheckHloRewrite(kHloText, DotDimensionNormalizer(), R"(
; CHECK: %[[T0:.*]] = bf16[12,8]{{.*}} transpose(%p0), dimensions={1,0}
; CHECK: %[[R1:.*]] = bf16[8,44]{{.*}} reshape(%p1)
; CHECK: ROOT {{[^ ]+}} = bf16[12,44]{{.*}} dot(%[[T0]], %[[R1]])
; CHECK-SAME: lhs_contracting_dims={1}
; CHECK-SAME: rhs_contracting_dims={0}
  )");
}

}  // namespace
}  // namespace xla

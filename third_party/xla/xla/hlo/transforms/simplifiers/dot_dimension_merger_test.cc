/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/dot_dimension_merger.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using DotDimensionMergerTest = HloHardwareIndependentTestBase;

TEST_F(DotDimensionMergerTest, MergeConsecutiveBatchDimensions) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
 p0 = bf16[79,2,4,12,11] parameter(0)
 p1 = bf16[79,2,4,11,44] parameter(1)
 ROOT d = bf16[2,4,12,44] dot(p0, p1),
  lhs_batch_dims={1,2}, lhs_contracting_dims={0,4},
  rhs_batch_dims={1,2}, rhs_contracting_dims={0,3},
  metadata={op_name="testname"}
})";

  RunAndFilecheckHloRewrite(kHloText, DotDimensionMerger(), R"(
; CHECK: %[[R0:.*]] = bf16[79,8,12,11]{3,2,1,0} reshape(%p0)
; CHECK: %[[R1:.*]] = bf16[79,8,11,44]{3,2,1,0} reshape(%p1)
; CHECK: %[[DOT:.*]] = bf16[8,12,44]{2,1,0} dot(%[[R0]], %[[R1]])
; CHECK-SAME: lhs_batch_dims={1}
; CHECK-SAME: lhs_contracting_dims={0,3}
; CHECK-SAME: rhs_batch_dims={1}
; CHECK-SAME: rhs_contracting_dims={0,2}
; CHECK-NEXT: ROOT {{[^ ]+}} = bf16[2,4,12,44]{3,2,1,0} reshape(%[[DOT]])
; CHECK-SAME: metadata={op_name="testname"}
  )");
}

TEST_F(DotDimensionMergerTest,
       MergeConsecutiveBatchDimensionsNonDefaultLayouts) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
 p0 = bf16[79,2,4,12,11]{4,0,3,2,1} parameter(0)
 p1 = bf16[79,2,4,11,44]{3,0,4,2,1} parameter(1)
 ROOT d = bf16[2,4,12,44]{3,1,0,2} dot(p0, p1),
  lhs_batch_dims={1,2}, lhs_contracting_dims={0,4},
  rhs_batch_dims={1,2}, rhs_contracting_dims={0,3},
  metadata={op_name="testname"}
})";

  RunAndFilecheckHloRewrite(kHloText, DotDimensionMerger(), R"(
; CHECK: %[[R0:.*]] = bf16[79,8,12,11]{3,0,2,1} reshape(%p0)
; CHECK: %[[R1:.*]] = bf16[79,8,11,44]{2,0,3,1} reshape(%p1)
; CHECK: %[[DOT:.*]] = bf16[8,12,44]{2,0,1} dot(%[[R0]], %[[R1]])
; CHECK-SAME: lhs_batch_dims={1}
; CHECK-SAME: lhs_contracting_dims={0,3}
; CHECK-SAME: rhs_batch_dims={1}
; CHECK-SAME: rhs_contracting_dims={0,2}
; CHECK-NEXT: ROOT {{[^ ]+}} = bf16[2,4,12,44]{3,1,0,2} reshape(%[[DOT]])
; CHECK-SAME: metadata={op_name="testname"}
  )");
}

TEST_F(DotDimensionMergerTest, SkipPhysicallyNonConsecutiveBatchDimensions) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
 p0 = bf16[2,4,12,13]{3,1,2,0} parameter(0)
 p1 = bf16[2,4,13,55]{3,2,1,0} parameter(1)
 ROOT d = bf16[2,4,12,55]{3,2,1,0} dot(p0, p1),
  lhs_batch_dims={0,1}, lhs_contracting_dims={3},
  rhs_batch_dims={0,1}, rhs_contracting_dims={2}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          DotDimensionMerger().Run(module.get()));
  EXPECT_FALSE(modified);
}

TEST_F(DotDimensionMergerTest, SkipUnsortedBatchDimensions) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
 p0 = bf16[4,2,12,13] parameter(0)
 p1 = bf16[2,4,13,55] parameter(1)
 ROOT d = bf16[2,4,12,55] dot(p0, p1),
  lhs_batch_dims={1,0}, lhs_contracting_dims={3},
  rhs_batch_dims={0,1}, rhs_contracting_dims={2}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          DotDimensionMerger().Run(module.get()));
  EXPECT_FALSE(modified);
}

TEST_F(DotDimensionMergerTest, SkipLogicallyNonConsecutiveBatchDimensions) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
 p0 = bf16[2,12,4,13] parameter(0)
 p1 = bf16[2,4,13,55] parameter(1)
 ROOT d = bf16[2,4,12,55] dot(p0, p1),
  lhs_batch_dims={0,2}, lhs_contracting_dims={3},
  rhs_batch_dims={0,1}, rhs_contracting_dims={2}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(bool modified,
                          DotDimensionMerger().Run(module.get()));
  EXPECT_FALSE(modified);
}

TEST_F(DotDimensionMergerTest, SparseDotUpdatesDescriptor) {
  const std::string kHloText = R"(
HloModule m

ENTRY e {
 p0 = bf16[3,4,5,6,16] parameter(0)
 p1 = bf16[3,4,5,32,6] parameter(1)
 meta = u16[3,4,5,6,2] parameter(2)
 ROOT d = bf16[4,5,6,6] dot(p0, p1, meta), sparsity=L.4@2:4,
  lhs_batch_dims={1,2}, lhs_contracting_dims={0,4},
  rhs_batch_dims={1,2}, rhs_contracting_dims={0,3}
})";

  RunAndFilecheckHloRewrite(kHloText, DotDimensionMerger(), R"(
; CHECK: %[[R0:.*]] = bf16[3,20,6,16]{3,2,1,0} reshape(%p0)
; CHECK: %[[R1:.*]] = bf16[3,20,32,6]{3,2,1,0} reshape(%p1)
; CHECK: %[[R2:.*]] = u16[3,20,6,2]{3,2,1,0} reshape(%meta)
; CHECK: %[[DOT:.*]] = bf16[20,6,6]{2,1,0} dot(%[[R0]], %[[R1]], %[[R2]])
; CHECK-SAME: lhs_batch_dims={1}
; CHECK-SAME: lhs_contracting_dims={0,3}
; CHECK-SAME: rhs_batch_dims={1}
; CHECK-SAME: rhs_contracting_dims={0,2}
; CHECK-SAME: sparsity=L.3@2:4
; CHECK-NEXT: ROOT {{.+}} = bf16[4,5,6,6]{3,2,1,0} reshape(%[[DOT]])
  )");
}

}  // namespace
}  // namespace xla

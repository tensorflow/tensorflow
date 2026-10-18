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

#include "xla/backends/gpu/transforms/gpu_transpose_folding.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;

class GpuTransposeFoldingTest : public HloHardwareIndependentTestBase {
 protected:
  void ExpectFolded(absl::string_view hlo, absl::string_view expected) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    EXPECT_THAT(GpuTransposeFolding(kHopper).Run(module.get()),
                IsOkAndHolds(true));
    EXPECT_THAT(RunFileCheck(module->ToString(), expected), IsOkAndHolds(true));
  }

  void ExpectNotFolded(absl::string_view hlo) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    EXPECT_THAT(GpuTransposeFolding(kHopper).Run(module.get()),
                IsOkAndHolds(false));
  }

  const se::GpuComputeCapability kHopper{se::CudaComputeCapability::Hopper()};
};

TEST_F(GpuTransposeFoldingTest, FoldsTranspose) {
  ExpectFolded(R"(
    HloModule m
    ENTRY e {
      x = f32[2,3]{1,0} parameter(0)
      y = f32[2,3]{1,0} parameter(1)
      t = f32[3,2]{1,0} transpose(y), dimensions={1,0}
      ROOT dot = f32[2,2]{1,0} dot(x, t), lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    })",
               R"(
    // CHECK: %[[X:.*]] = {{.*}} parameter(0)
    // CHECK: %[[Y:.*]] = {{.*}} parameter(1)
    // CHECK: dot(%[[X]], %[[Y]]), lhs_contracting_dims={1}, rhs_contracting_dims={1}
    )");
}

TEST_F(GpuTransposeFoldingTest, FoldsTransposeBitcastChainThatIsIdentity) {
  // transpose + layout-changing bitcast is logically the identity; only the
  // layout differs. This is what layout normalization produces from a
  // layout-changing copy.
  ExpectFolded(R"(
    HloModule m
    ENTRY e {
      p = f32[8,64,32]{2,1,0} parameter(0)
      y = f32[8,64,6]{2,1,0} parameter(1)
      t = f32[8,32,64]{2,1,0} transpose(p), dimensions={0,2,1}
      b = f32[8,64,32]{1,2,0} bitcast(t)
      ROOT dot = f32[8,32,6]{2,1,0} dot(b, y), lhs_batch_dims={0},
        lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    })",
               R"(
    // CHECK: %[[P:.*]] = {{.*}} parameter(0)
    // CHECK: %[[Y:.*]] = {{.*}} parameter(1)
    // CHECK-NOT: transpose
    // CHECK: dot(%[[P]], %[[Y]]), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    )");
}

TEST_F(GpuTransposeFoldingTest, FoldsLayoutChangingBitcastThatIsTranspose) {
  // b is logically transpose(p, {0,2,1}).
  ExpectFolded(R"(
    HloModule m
    ENTRY e {
      p = f32[8,64,32]{2,1,0} parameter(0)
      y = f32[8,64,6]{2,1,0} parameter(1)
      b = f32[8,32,64]{1,2,0} bitcast(p)
      ROOT dot = f32[8,32,6]{2,1,0} dot(b, y), lhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    })",
               R"(
    // CHECK: %[[P:.*]] = {{.*}} parameter(0)
    // CHECK: %[[Y:.*]] = {{.*}} parameter(1)
    // CHECK: dot(%[[P]], %[[Y]]), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    )");
}

TEST_F(GpuTransposeFoldingTest, FoldsTransposeKeepingAdjacentDimsTogether) {
  // Dimensions 2 and 3 of p stay adjacent through the transpose.
  ExpectFolded(R"(
    HloModule m
    ENTRY e {
      p = f32[4,8,16,32]{3,2,1,0} parameter(0)
      y = f32[4,8,6]{2,1,0} parameter(1)
      t = f32[4,16,32,8]{3,2,1,0} transpose(p), dimensions={0,2,3,1}
      ROOT dot = f32[4,16,32,6]{3,2,1,0} dot(t, y), lhs_batch_dims={0},
        lhs_contracting_dims={3}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    })",
               R"(
    // CHECK: %[[P:.*]] = {{.*}} parameter(0)
    // CHECK: %[[Y:.*]] = {{.*}} parameter(1)
    // CHECK: dot(%[[P]], %[[Y]]), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    )");
}

TEST_F(GpuTransposeFoldingTest, DoesNotFoldBitcastThatIsReshape) {
  // Same dimensions as a transpose, but the bitcast is a reshape.
  ExpectNotFolded(R"(
    HloModule m
    ENTRY e {
      p = f32[8,64,32]{2,1,0} parameter(0)
      y = f32[8,64,6]{2,1,0} parameter(1)
      b = f32[8,32,64]{2,1,0} bitcast(p)
      ROOT dot = f32[8,32,6]{2,1,0} dot(b, y), lhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    })");
}

TEST_F(GpuTransposeFoldingTest, DoesNotFoldIfBatchDimBecomesMinor) {
  ExpectNotFolded(R"(
    HloModule m
    ENTRY e {
      p = f32[64,32,8]{2,1,0} parameter(0)
      y = f32[8,64,6]{2,1,0} parameter(1)
      t = f32[8,64,32]{2,1,0} transpose(p), dimensions={2,0,1}
      ROOT dot = f32[8,32,6]{2,1,0} dot(t, y), lhs_batch_dims={0},
        lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    })");
}

TEST_F(GpuTransposeFoldingTest, DoesNotFoldIfNonContractingDimsReorder) {
  // Folding would swap the order of the non-contracting dims in the output.
  ExpectNotFolded(R"(
    HloModule m
    ENTRY e {
      p = f32[8,16,32]{2,1,0} parameter(0)
      y = f32[8,6]{1,0} parameter(1)
      t = f32[32,16,8]{2,1,0} transpose(p), dimensions={2,1,0}
      ROOT dot = f32[32,16,6]{2,1,0} dot(t, y), lhs_contracting_dims={2},
        rhs_contracting_dims={0}
    })");
}

TEST_F(GpuTransposeFoldingTest, DoesNotFoldTransposeOfDegenerateDim) {
  // Moving a size-1 dimension is a reshape, not a transpose: the dimension
  // can't be tracked back to its category. Conservatively left alone.
  ExpectNotFolded(R"(
    HloModule m
    ENTRY e {
      p = f32[8,1,6]{2,1,0} parameter(0)
      y = f32[1,6,5]{2,1,0} parameter(1)
      t = f32[1,8,6]{2,1,0} transpose(p), dimensions={1,0,2}
      ROOT dot = f32[1,8,5]{2,1,0} dot(t, y), lhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    })");
}

TEST_F(GpuTransposeFoldingTest, DoesNotFoldIfS8ContractingDimBecomesMajor) {
  // This is what layout normalization produces from the copy that layout
  // assignment inserts to make the contracting dimension of `p` minor.
  ExpectNotFolded(R"(
    HloModule m
    ENTRY e {
      x = s8[12,4]{1,0} parameter(0)
      p = s8[4,8]{1,0} parameter(1)
      t = s8[8,4]{1,0} transpose(p), dimensions={1,0}
      b = s8[4,8]{0,1} bitcast(t)
      ROOT dot = s32[12,8]{1,0} dot(x, b), lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    })");
}

TEST_F(GpuTransposeFoldingTest, FoldsS8IfContractingDimStaysMinor) {
  ExpectFolded(R"(
    HloModule m
    ENTRY e {
      x = s8[12,4]{1,0} parameter(0)
      p = s8[8,4]{1,0} parameter(1)
      t = s8[4,8]{1,0} transpose(p), dimensions={1,0}
      ROOT dot = s32[12,8]{1,0} dot(x, t), lhs_contracting_dims={1},
        rhs_contracting_dims={0}
    })",
               R"(
    // CHECK: %[[X:.*]] = {{.*}} parameter(0)
    // CHECK: %[[P:.*]] = {{.*}} parameter(1)
    // CHECK: dot(%[[X]], %[[P]]), lhs_contracting_dims={1}, rhs_contracting_dims={1}
    )");
}

TEST_F(GpuTransposeFoldingTest, FoldsConvolutionOperands) {
  ExpectFolded(R"(
    HloModule m
    ENTRY e {
      x = f32[3,2,4,5]{3,2,1,0} parameter(0)
      y = f32[1,2,3,6]{3,2,1,0} parameter(1)
      tx = f32[2,3,5,4]{3,2,1,0} transpose(x), dimensions={1,0,3,2}
      ty = f32[3,6,1,2]{3,2,1,0} transpose(y), dimensions={2,3,0,1}
      ROOT conv = f32[2,6,5,3]{3,2,1,0} convolution(tx, ty),
        window={size=1x2}, dim_labels=bf01_io01->bf01
    })",
               R"(
    // CHECK: %[[X:.*]] = {{.*}} parameter(0)
    // CHECK: %[[Y:.*]] = {{.*}} parameter(1)
    // CHECK: convolution(%[[X]], %[[Y]]), window={size=1x2}, dim_labels=fb10_01io->bf01
    )");
}

}  // namespace
}  // namespace xla::gpu

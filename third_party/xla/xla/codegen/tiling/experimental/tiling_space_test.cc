/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/tiling/experimental/tiling_space.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"

namespace xla::gpu::experimental {
namespace {

using ::mlir::MLIRContext;

MATCHER_P(MatchString, tiling_space_string, "") {
  return ExplainMatchResult(
      true, ApproximateMatch(tiling_space_string, arg.ToString()),
      result_listener);
}

class TilingSpaceTest : public HloHardwareIndependentTestBase {
 public:
  HloInstruction* ParseAndGetRoot(absl::string_view hlo_string) {
    auto module_or = ParseAndReturnVerifiedModule(hlo_string);
    CHECK_OK(module_or);
    module_ = std::move(module_or.value());
    return module_->entry_computation()->root_instruction();
  }

  MLIRContext mlir_context_;
  std::unique_ptr<VerifiedHloModule> module_;
};

TEST_F(TilingSpaceTest, SingleOutputParallelDim) {
  auto root = ParseAndGetRoot(R"(
      HloModule m
      ENTRY e {
        p0 = f32[1000, 10] parameter(0)
        ROOT a0 = f32[1000, 10] exponential(p0)
      }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  auto tiling_space = TilingSpace::Create(*fusion_adaptor, &mlir_context_);
  EXPECT_THAT(*tiling_space, MatchString(R"(
    Dimensions:
        0 type: parallel size: 1000 dim ID:0
          hlo: %a0 = f32[1000,10]{1,0} exponential(%p0)
        1 type: parallel size: 10 dim ID:1
          hlo: %a0 = f32[1000,10]{1,0} exponential(%p0)
    Root tiles:
      0 root tile:
           offsets [tid_0 * ts_0, tid_1 * ts_1] sizes [ts_0, ts_1]
           strides [1, 1] upper bounds [1000, 10]
  )"));
}

TEST_F(TilingSpaceTest, SingleOutputContractionDim) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      p0 = bf16[2304,16,768]{2,1,0} parameter(0)
      p1 = bf16[16,16,768] parameter(1)
      ROOT dot = bf16[16,2304,16] dot(p0, p1),
          lhs_batch_dims={1}, lhs_contracting_dims={2},
          rhs_batch_dims={1}, rhs_contracting_dims={2}
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  auto tiling_space = TilingSpace::Create(*fusion_adaptor, &mlir_context_);
  EXPECT_THAT(*tiling_space, MatchString(R"(
    Dimensions:
      0 type: parallel size: 16 dim ID:0
        hlo: %dot = bf16[16,2304,16]{2,1,0} dot(%p0, %p1), lhs_batch_dims={1},
        lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={2}
      1 type: parallel size: 2304 dim ID:1
        hlo: %dot = bf16[16,2304,16]{2,1,0} dot(%p0, %p1), lhs_batch_dims={1},
        lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={2}
      2 type: parallel size: 16 dim ID:2
        hlo: %dot = bf16[16,2304,16]{2,1,0} dot(%p0, %p1), lhs_batch_dims={1},
        lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={2}
      3 type: sequential size: 768 dim ID:3
        hlo: %dot = bf16[16,2304,16]{2,1,0} dot(%p0, %p1), lhs_batch_dims={1},
        lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={2}
    Root tiles:
      0 root tile:
           offsets [tid_0 * ts_0, tid_1 * ts_1, tid_2 * ts_2]
           sizes [ts_0, ts_1, ts_2]
           strides [1, 1, 1]
           upper bounds [16, 2304, 16]
  )"));
}

TEST_F(TilingSpaceTest, SingleOutputReductionDim) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    max {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT max = f32[] maximum(p0, p1)
    }
    ENTRY e {
      p0 = f32[150,20,10,50] parameter(0)
      p1 = f32[] constant(-inf)
      ROOT reduce = f32[150,10] reduce(p0, p1), dimensions={3,1}, to_apply=max
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  auto tiling_space = TilingSpace::Create(*fusion_adaptor, &mlir_context_);
  EXPECT_THAT(*tiling_space, MatchString(R"(
    Dimensions:
      0 type: parallel size: 150 dim ID:0
        hlo: %reduce = f32[150,10]{1,0} reduce(%p0.1, %p1.1), dimensions={3,1},
        to_apply=%max
      1 type: parallel size: 10 dim ID:1
        hlo: %reduce = f32[150,10]{1,0} reduce(%p0.1, %p1.1), dimensions={3,1},
        to_apply=%max
      2 type: sequential size: 50 dim ID:2
        hlo: %reduce = f32[150,10]{1,0} reduce(%p0.1, %p1.1), dimensions={3,1},
        to_apply=%max
      3 type: sequential size: 20 dim ID:3
        hlo: %reduce = f32[150,10]{1,0} reduce(%p0.1, %p1.1), dimensions={3,1},
        to_apply=%max
    Root tiles:
      0 root tile:
           offsets [tid_0 * ts_0, tid_1 * ts_1] sizes [ts_0, ts_1]
           strides [1, 1] upper bounds [150, 10]
  )"));
}

TEST_F(TilingSpaceTest, VariadicReduce) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    min {
      tmp_0 = f32[] parameter(0)
      tmp_1 = f32[] parameter(2)
      tmp_2 = s32[] parameter(1)
      tmp_3 = s32[] parameter(3)
      cmp = pred[] compare(tmp_0, tmp_1), direction=GE
      select1 = f32[] select(cmp, tmp_0, tmp_1)
      select2 = s32[] select(cmp, tmp_2, tmp_3)
      ROOT tmp_4 = (f32[], s32[]) tuple(select1, select2)
    }
    ENTRY e {
      p0 = f32[256,10] parameter(0)
      p0_init = f32[] constant(-inf)
      p1 = s32[256,10] parameter(1)
      p1_init = s32[] constant(0)
      ROOT reduce = (f32[10], s32[10]) reduce(p0, p1, p0_init, p1_init),
        dimensions={0}, to_apply=min
    }

  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);

  auto tiling_space = TilingSpace::Create(*fusion_adaptor, &mlir_context_);
  EXPECT_THAT(*tiling_space, MatchString(R"(
    Dimensions:
      0 type: parallel size: 10 dim ID:0 hlo:
        %reduce = (f32[10]{0}, s32[10]{0}) reduce(%p0, %p1, %p0_init, %p1_init),
        dimensions={0}, to_apply=%min
      1 type: sequential size: 256 dim ID:1 hlo:
        %reduce = (f32[10]{0}, s32[10]{0}) reduce(%p0, %p1, %p0_init, %p1_init),
        dimensions={0}, to_apply=%min
    Root tiles:
      0 root tile:
        offsets [tid_0 * ts_0] sizes [ts_0] strides [1] upper bounds [10]
      1 root tile:
        offsets [tid_0 * ts_0] sizes [ts_0] strides [1] upper bounds [10]
  )"));
}

TEST_F(TilingSpaceTest, DynamicSlice) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      %src = s32[2,2,258] parameter(0)
      %of1 = s32[] parameter(1)
      %of2 = s32[] parameter(2)
      %of3 = s32[] parameter(3)
      ROOT %ds = s32[1,2,32] dynamic-slice(s32[2,2,258] %src,
        s32[] %of1, s32[] %of2, s32[] %of3),
        dynamic_slice_sizes={1, 2, 32}
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);

  auto tiling_space = TilingSpace::Create(*fusion_adaptor, &mlir_context_);
  EXPECT_THAT(*tiling_space, MatchString(R"(
    Dimensions:
        0 type: parallel size: 1 dim ID:0
          hlo: %ds = s32[1,2,32]{2,1,0} dynamic-slice(%src, %of1, %of2, %of3),
          dynamic_slice_sizes={1,2,32}
        1 type: parallel size: 2 dim ID:1
          hlo: %ds = s32[1,2,32]{2,1,0} dynamic-slice(%src, %of1, %of2, %of3),
          dynamic_slice_sizes={1,2,32}
        2 type: parallel size: 32 dim ID:2
          hlo: %ds = s32[1,2,32]{2,1,0} dynamic-slice(%src, %of1, %of2, %of3),
          dynamic_slice_sizes={1,2,32}
    Runtime variables:
        0 bounds: [0, 1] hlo: %of1 = s32[] parameter(1)
        1 bounds: [0, 0] hlo: %of2 = s32[] parameter(2)
        2 bounds: [0, 226] hlo: %of3 = s32[] parameter(3)
    Root tiles:
      0 root tile:
           offsets [tid_0 * ts_0, tid_1 * ts_1, tid_2 * ts_2]
           sizes [ts_0, ts_1, ts_2] strides [1, 1, 1] upper bounds [1, 2, 32]
  )"));
}

TEST_F(TilingSpaceTest, TwoOutputsParallelDims) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = f32[10,8] parameter(0)
      p1 = f32[10,8] parameter(1)
      p2 = f32[11,9] parameter(2)
      p3 = f32[11,9] parameter(3)
      add = f32[10,8] add(p0, p1)
      mul = f32[11,9] multiply(p2, p3)
      ROOT t = (f32[10,8], f32[11,9]) tuple(add, mul)
    }

    ENTRY e {
      p0 = f32[10,8] parameter(0)
      p1 = f32[10,8] parameter(1)
      p2 = f32[11,9] parameter(2)
      p3 = f32[11,9] parameter(3)
      ROOT fusion = (f32[10,8], f32[11,9]) fusion(p0, p1, p2, p3),
        kind=kLoop, calls=f
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  auto tiling_space = TilingSpace::Create(*fusion_adaptor, &mlir_context_);
  EXPECT_THAT(*tiling_space, MatchString(R"(
    Dimensions:
        0 type: parallel size: 10 dim ID:0
          hlo: %add = f32[10,8]{1,0} add(%p0, %p1)
        1 type: parallel size: 8 dim ID:1
          hlo: %add = f32[10,8]{1,0} add(%p0, %p1)
        2 type: parallel size: 11 dim ID:0
          hlo: %mul = f32[11,9]{1,0} multiply(%p2, %p3)
        3 type: parallel size: 9 dim ID:1
          hlo: %mul = f32[11,9]{1,0} multiply(%p2, %p3)
    Root tiles:
        0 root tile:
             offsets [tid_0 * ts_0, tid_1 * ts_1] sizes [ts_0, ts_1]
             strides [1, 1] upper bounds [10, 8]
        1 root tile:
             offsets [tid_0 * ts_0, tid_1 * ts_1] sizes [ts_0, ts_1]
             strides [1, 1] upper bounds [11, 9]
  )"));
}

}  // namespace
}  // namespace xla::gpu::experimental

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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map_serialization.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/xla.pb.h"

namespace xla::gpu::experimental {
namespace {

using ::absl_testing::StatusIs;
using ::mlir::MLIRContext;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

MATCHER_P(MatchString, expected, "") {
  const absl::string_view expected_string = expected;
  const std::string actual_string = arg.ToString();
  const auto [expected_index, actual_index] =
      FindApproximateMismatch(expected_string, actual_string);
  const bool matches = expected_index == expected_string.size() &&
                       actual_index == actual_string.size();
  if (!matches) {
    *result_listener << GetMismatchReport(expected_index, actual_index,
                                          expected_string, actual_string);
  }
  return matches;
}

class TilingSpaceTest : public HloHardwareIndependentTestBase {
 public:
  TilingSpaceTest() { RegisterSymbolicExprStorage(&mlir_context_); }

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
  ASSERT_OK_AND_ASSIGN(auto tiling_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));
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
  ASSERT_OK_AND_ASSIGN(auto tiling_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));
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
  ASSERT_OK_AND_ASSIGN(auto tiling_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));
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

TEST_F(TilingSpaceTest, SingleOutputScanDim) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      add = f32[] add(p0, p1)
      ROOT tuple = (f32[], f32[]) tuple(add, add)
    }
    fused_computation {
      p0 = f32[150] parameter(0)
      p1 = f32[] constant(0.0)
      scan = (f32[150], f32[]) scan(p0, p1), dimensions={0}, num_carries=1, is_associative=false, to_apply=add
      ROOT get-tuple-element = f32[150] get-tuple-element(scan), index=0
    }
    ENTRY e {
      p0 = f32[150] parameter(0)
      ROOT fusion = f32[150] fusion(p0), kind=kLoop, calls=fused_computation
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  ASSERT_OK_AND_ASSIGN(auto tiling_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));
  EXPECT_THAT(*tiling_space, MatchString(R"(
    Dimensions:
      0 type: sequential size: 150 dim ID:0
        hlo: %scan = (f32[150]{0}, f32[]) scan(%p0.1, %p1.1), dimensions={0}, num_carries=1, is_associative=false, to_apply=%add
    Root tiles:
      0 root tile:
           offsets [tid_0 * ts_0] sizes [ts_0]
           strides [1] upper bounds [150]
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

  ASSERT_OK_AND_ASSIGN(auto tiling_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));
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

  ASSERT_OK_AND_ASSIGN(auto tiling_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));
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
  HloInstruction* root = ParseAndGetRoot(R"(
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
  root->GetModule()
      ->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_unsupported_enable_triton_multi_output_fusion(true);
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  ASSERT_OK_AND_ASSIGN(auto tiling_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));
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
           offsets [tid_2 * ts_2, tid_3 * ts_3] sizes [ts_2, ts_3]
           strides [1, 1] upper bounds [11, 9]
  )"));
}

TEST_F(TilingSpaceTest, TwoOutputsEqualShapesParallelDims) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = f32[10,8] parameter(0)
      p1 = f32[10,8] parameter(1)
      p2 = f32[10,8] parameter(2)
      p3 = f32[10,8] parameter(3)
      add = f32[10,8] add(p0, p1)
      mul = f32[10,8] multiply(p2, p3)
      ROOT t = (f32[10,8], f32[10,8]) tuple(add, mul)
    }

    ENTRY e {
      p0 = f32[10,8] parameter(0)
      p1 = f32[10,8] parameter(1)
      p2 = f32[10,8] parameter(2)
      p3 = f32[10,8] parameter(3)
      ROOT fusion = (f32[10,8], f32[10,8]) fusion(p0, p1, p2, p3),
        kind=kLoop, calls=f
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  EXPECT_THAT(
      TilingSpace::Create(*fusion_adaptor, &mlir_context_),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("multiple roots")));
}

TEST_F(TilingSpaceTest, IsIndexWiseVariadic) {
  ParseAndGetRoot(R"(
    HloModule m
    add_pair {
      a = f32[] parameter(0)
      b = s32[] parameter(1)
      c = f32[] parameter(2)
      d = s32[] parameter(3)
      add_f = f32[] add(a, c)
      add_s = s32[] add(b, d)
      ROOT t = (f32[], s32[]) tuple(add_f, add_s)
    }
    ENTRY e {
      p0 = f32[1,8] parameter(0)
      p1 = s32[1] parameter(1)
      p2 = f32[4,8] parameter(2)
      p3 = s32[4,8] parameter(3)
      c0 = f32[] constant(0)
      c1 = s32[] constant(0)
      single_ag = f32[4,8] all-gather(p0), replica_groups={{0,1,2,3}},
        dimensions={0}
      variadic_ag = (f32[4,8], s32[4]) all-gather(p0, p1),
        replica_groups={{0,1,2,3}}, dimensions={0}
      variadic_reduce = (f32[4], s32[4]) reduce(p2, p3, c0, c1),
        dimensions={1}, to_apply=add_pair
      gte = f32[4,8] get-tuple-element(variadic_ag), index=0
      ROOT t = (f32[4,8], f32[4,8]) tuple(single_ag, gte)
    }
  )");
  HloComputation* entry = module_->entry_computation();
  EXPECT_TRUE(
      IsIndexWiseVariadic(*entry->GetInstructionWithName("variadic_ag")));
  EXPECT_FALSE(
      IsIndexWiseVariadic(*entry->GetInstructionWithName("single_ag")));
  EXPECT_FALSE(
      IsIndexWiseVariadic(*entry->GetInstructionWithName("variadic_reduce")));
  EXPECT_FALSE(IsIndexWiseVariadic(*entry->GetInstructionWithName("gte")));
  EXPECT_FALSE(IsIndexWiseVariadic(*entry->root_instruction()));
}

// An index-wise variadic instruction must be decomposed into GTEs + a tuple
// root; a bare tuple-producing variadic root is rejected.
TEST_F(TilingSpaceTest, BareVariadicAllGatherRootIsRejected) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = f32[1,2,8,16] parameter(0)
      p1 = s32[1,2] parameter(1)
      ROOT ag = (f32[4,2,8,16], s32[4,2]) all-gather(p0, p1),
        replica_groups={{0,1,2,3}}, dimensions={0}
    }

    ENTRY e {
      p0 = f32[1,2,8,16] parameter(0)
      p1 = s32[1,2] parameter(1)
      ROOT fusion = (f32[4,2,8,16], s32[4,2]) fusion(p0, p1), kind=kCustom,
        calls=f
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  EXPECT_THAT(TilingSpace::Create(*fusion_adaptor, &mlir_context_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported root shape")));
}

// A tuple-shaped index-wise variadic instruction consumed through GTE roots
// gets independent tile dimensions for every output even though their shapes
// differ. The dimensions of the outputs are laid out back to back in the
// ordered dimension list of the instruction. The peer coupling between the
// outputs is recovered later by the scheduler.
TEST_F(TilingSpaceTest, VariadicAllGatherGetsPerOutputDimensions) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = f32[1,2,8,16] parameter(0)
      p1 = s32[1,2] parameter(1)
      ag = (f32[4,2,8,16], s32[4,2]) all-gather(p0, p1),
        replica_groups={{0,1,2,3}}, dimensions={0}
      gte0 = f32[4,2,8,16] get-tuple-element(ag), index=0
      gte1 = s32[4,2] get-tuple-element(ag), index=1
      ROOT t = (f32[4,2,8,16], s32[4,2]) tuple(gte0, gte1)
    }

    ENTRY e {
      p0 = f32[1,2,8,16] parameter(0)
      p1 = s32[1,2] parameter(1)
      ROOT fusion = (f32[4,2,8,16], s32[4,2]) fusion(p0, p1), kind=kCustom,
        calls=f
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  ASSERT_OK_AND_ASSIGN(auto tiling_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));
  const HloInstruction* ag =
      root->fused_instructions_computation()->GetInstructionWithName("ag");
  EXPECT_TRUE(tiling_space->HasPerOutputTiles(
      HloInstructionAdaptor(*ag, fusion_adaptor.get())));
  EXPECT_TRUE(tiling_space->HasPerOutputTiles());
  EXPECT_EQ(tiling_space->num_dimensions(), 6);
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(tiling_space->GetDimensionInfo(*ag, i).id, TiledDimId(i));
  }
  EXPECT_THAT(*tiling_space, MatchString(R"(
    Dimensions:
        0 type: parallel size: 4 dim ID:0
          hlo: %gte0 = f32[4,2,8,16]{3,2,1,0} get-tuple-element(%ag), index=0
        1 type: parallel size: 2 dim ID:1
          hlo: %gte0 = f32[4,2,8,16]{3,2,1,0} get-tuple-element(%ag), index=0
        2 type: parallel size: 8 dim ID:2
          hlo: %gte0 = f32[4,2,8,16]{3,2,1,0} get-tuple-element(%ag), index=0
        3 type: parallel size: 16 dim ID:3
          hlo: %gte0 = f32[4,2,8,16]{3,2,1,0} get-tuple-element(%ag), index=0
        4 type: parallel size: 4 dim ID:0
          hlo: %gte1 = s32[4,2]{1,0} get-tuple-element(%ag), index=1
        5 type: parallel size: 2 dim ID:1
          hlo: %gte1 = s32[4,2]{1,0} get-tuple-element(%ag), index=1
    Root tiles:
      0 root tile:
           offsets [tid_0 * ts_0, tid_1 * ts_1, tid_2 * ts_2, tid_3 * ts_3]
           sizes [ts_0, ts_1, ts_2, ts_3]
           strides [1, 1, 1, 1]
           upper bounds [4, 2, 8, 16]
      1 root tile:
           offsets [tid_4 * ts_4, tid_5 * ts_5]
           sizes [ts_4, ts_5]
           strides [1, 1]
           upper bounds [4, 2]
  )"));
}

TEST_F(TilingSpaceTest,
       InteriorVariadicAllGatherPropagatesDimensionsFromRoots) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = f32[2,16] parameter(0)
      p1 = s32[4] parameter(1)
      ag = (f32[8,16], s32[16]) all-gather(p0, p1),
        replica_groups={{0,1,2,3}}, dimensions={0}
      gte0 = f32[8,16] get-tuple-element(ag), index=0
      gte1 = s32[16] get-tuple-element(ag), index=1
      c0 = bf16[8,16] convert(gte0)
      c1 = f32[16] convert(gte1)
      ROOT t = (bf16[8,16], f32[16]) tuple(c0, c1)
    }

    ENTRY e {
      p0 = f32[2,16] parameter(0)
      p1 = s32[4] parameter(1)
      ROOT fusion = (bf16[8,16], f32[16]) fusion(p0, p1), kind=kCustom,
        calls=f
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  ASSERT_OK_AND_ASSIGN(auto tiling_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));
  const HloInstruction* ag =
      root->fused_instructions_computation()->GetInstructionWithName("ag");
  EXPECT_EQ(tiling_space->GetDimensionInfo(*ag, 0).id, TiledDimId(0));
  EXPECT_EQ(tiling_space->GetDimensionInfo(*ag, 1).id, TiledDimId(1));
  EXPECT_EQ(tiling_space->GetDimensionInfo(*ag, 2).id, TiledDimId(2));
}

TEST_F(TilingSpaceTest,
       VariadicAllGatherWithDotProducersGetsSequentialDimensions) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    f {
      lhs0 = f32[2,8] parameter(0)
      rhs0 = f32[8,16] parameter(1)
      lhs1 = f32[4,8] parameter(2)
      rhs1 = f32[8,32] parameter(3)
      dot0 = f32[2,16] dot(lhs0, rhs0),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      dot1 = f32[4,32] dot(lhs1, rhs1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ag = (f32[8,16], f32[16,32]) all-gather(dot0, dot1),
        replica_groups={{0,1,2,3}}, dimensions={0}
      gte0 = f32[8,16] get-tuple-element(ag), index=0
      gte1 = f32[16,32] get-tuple-element(ag), index=1
      ROOT t = (f32[8,16], f32[16,32]) tuple(gte0, gte1)
    }

    ENTRY e {
      lhs0 = f32[2,8] parameter(0)
      rhs0 = f32[8,16] parameter(1)
      lhs1 = f32[4,8] parameter(2)
      rhs1 = f32[8,32] parameter(3)
      ROOT fusion = (f32[8,16], f32[16,32]) fusion(lhs0, rhs0, lhs1, rhs1),
        kind=kCustom, calls=f
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  ASSERT_OK_AND_ASSIGN(auto tiling_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));
  EXPECT_EQ(tiling_space->num_dimensions(), 6);
  EXPECT_EQ(tiling_space->num_parallel_dimensions(), 4);
}

class TilingSpaceSameShapeMultiOutputTest : public TilingSpaceTest {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = TilingSpaceTest::GetDebugOptionsForTest();
    debug_options
        .set_xla_gpu_experimental_enable_same_shape_multi_output_fusion(true);
    return debug_options;
  }
};

TEST_F(TilingSpaceSameShapeMultiOutputTest,
       TwoOutputsEqualShapesDuplicateRoots) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = f32[10,8] parameter(0)
      ROOT t = (f32[10,8], f32[10,8]) tuple(p0, p0)
    }

    ENTRY e {
      p0 = f32[10,8] parameter(0)
      ROOT fusion = (f32[10,8], f32[10,8]) fusion(p0), kind=kLoop, calls=f
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  ASSERT_OK_AND_ASSIGN(auto tiling_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));
  EXPECT_THAT(*tiling_space, MatchString(R"(
    Dimensions:
        0 type: parallel size: 10 dim ID:0
          hlo: %p0 = f32[10,8]{1,0} parameter(0)
        1 type: parallel size: 8 dim ID:1
          hlo: %p0 = f32[10,8]{1,0} parameter(0)
    Root tiles:
      0 root tile:
           offsets [tid_0 * ts_0, tid_1 * ts_1] sizes [ts_0, ts_1]
           strides [1, 1] upper bounds [10, 8]
      1 root tile:
           offsets [tid_0 * ts_0, tid_1 * ts_1] sizes [ts_0, ts_1]
           strides [1, 1] upper bounds [10, 8]
  )"));
}

TEST_F(TilingSpaceSameShapeMultiOutputTest, TwoOutputsEqualShapesParallelDims) {
  HloInstruction* root = ParseAndGetRoot(R"(
    HloModule m
    f {
      p0 = f32[10,8] parameter(0)
      p1 = f32[10,8] parameter(1)
      p2 = f32[10,8] parameter(2)
      p3 = f32[10,8] parameter(3)
      add = f32[10,8] add(p0, p1)
      mul = f32[10,8] multiply(p2, p3)
      ROOT t = (f32[10,8], f32[10,8]) tuple(add, mul)
    }

    ENTRY e {
      p0 = f32[10,8] parameter(0)
      p1 = f32[10,8] parameter(1)
      p2 = f32[10,8] parameter(2)
      p3 = f32[10,8] parameter(3)
      ROOT fusion = (f32[10,8], f32[10,8]) fusion(p0, p1, p2, p3),
        kind=kLoop, calls=f
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  ASSERT_OK_AND_ASSIGN(auto tiling_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));
  EXPECT_THAT(*tiling_space, MatchString(R"(
    Dimensions:
        0 type: parallel size: 10 dim ID:0
          hlo: %add = f32[10,8]{1,0} add(%p0, %p1)
        1 type: parallel size: 8 dim ID:1
          hlo: %add = f32[10,8]{1,0} add(%p0, %p1)
    Root tiles:
      0 root tile:
           offsets [tid_0 * ts_0, tid_1 * ts_1] sizes [ts_0, ts_1]
           strides [1, 1] upper bounds [10, 8]
      1 root tile:
           offsets [tid_0 * ts_0, tid_1 * ts_1] sizes [ts_0, ts_1]
           strides [1, 1] upper bounds [10, 8]
  )"));
}

class TilingSpaceSimplifyExpressionsTest : public TilingSpaceTest {
 public:
  void SetUp() override {
    TilingSpaceTest::SetUp();
    HloInstruction* root = ParseAndGetRoot(R"(
        HloModule m
        ENTRY e {
          p0 = f32[100, 10] parameter(0)
          ROOT a0 = f32[100, 10] exponential(p0)
        }
    )");

    auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
    ASSERT_OK_AND_ASSIGN(tiling_space_,
                         TilingSpace::Create(*fusion_adaptor, &mlir_context_));

    // Assign concrete tile sizes of [16, 2].
    // Dimension 0 (100) / 16 = 7 blocks (tid_0 in [0, 6]).
    // Dimension 1 (10) / 2 = 5 blocks (tid_1 in [0, 4]).
    CHECK_OK(tiling_space_->AssignTileSizes({16, 2}));
  }

  SymbolicExpr ParseExpr(absl::string_view expr_str) {
    return ParseSymbolicExpr(expr_str, &mlir_context_, /*num_dims=*/2);
  }

  std::unique_ptr<TilingSpace> tiling_space_;
};

TEST_F(TilingSpaceSimplifyExpressionsTest, ModRemovedIfLessThanDivisor) {
  EXPECT_THAT(tiling_space_->SimplifyExpressions({ParseExpr("(d0 * 8) mod 96")})
                  .expressions,
              ElementsAre(ParseExpr("d0 * 8")));
}

TEST_F(TilingSpaceSimplifyExpressionsTest, MultipleExpressionsSimplified) {
  EXPECT_THAT(tiling_space_
                  ->SimplifyExpressions({ParseExpr("(d0 * 8) mod 96"),
                                         ParseExpr("(d1 * 2) / 4"),
                                         ParseExpr("d0 * 16 + 500")})
                  .expressions,
              ElementsAre(ParseExpr("d0 * 8"), ParseExpr("d1 / 2"),
                          ParseExpr("d0 * 16 + 500")));
}

TEST_F(TilingSpaceSimplifyExpressionsTest, DimTileSimplify) {
  DimTile dt{ParseExpr("(d0 * 8) mod 96"), ParseExpr("(d1 * 2) / 4"),
             ParseExpr("d0 * 16 + 500"),
             ParseExpr("(d0 * 16 + d1 * 2) mod 200")};
  dt.Simplify(*tiling_space_);
  EXPECT_THAT(dt, MatchString(R"(
    offset [v0 * 8],
    size [v1 / 2],
    stride [v0 * 16 + 500],
    upper bound [v0 * 16 + v1 * 2]
  )"));
}

TEST_F(TilingSpaceSimplifyExpressionsTest, SimplifyDimTiles) {
  llvm::SmallVector<DimTile> dim_tiles = {
      {ParseExpr("(d0 * 8) mod 96"), ParseExpr("(d1 * 2) / 4"),
       ParseExpr("d0 * 16 + 500"), ParseExpr("(d0 * 16 + d1 * 2) mod 200")},
      {ParseExpr("(d1 * 2) / 4"), ParseExpr("(d0 * 8) mod 96"),
       ParseExpr("d0 * 16 + 500"), ParseExpr("(d0 * 16 + d1 * 2) mod 200")}};
  SimplifyDimTiles(dim_tiles, *tiling_space_);
  EXPECT_THAT(dim_tiles[0], MatchString(R"(
    offset [v0 * 8],
    size [v1 / 2],
    stride [v0 * 16 + 500],
    upper bound [v0 * 16 + v1 * 2]
  )"));
  EXPECT_THAT(dim_tiles[1], MatchString(R"(
    offset [v1 / 2],
    size [v0 * 8],
    stride [v0 * 16 + 500],
    upper bound [v0 * 16 + v1 * 2]
  )"));

  llvm::SmallVector<DimTile> empty_dim_tiles;
  SimplifyDimTiles(empty_dim_tiles, *tiling_space_);
  EXPECT_TRUE(empty_dim_tiles.empty());
}

TEST_F(TilingSpaceSimplifyExpressionsTest, SimplifyDimTilesGroups) {
  llvm::SmallVector<DimTile> group1 = {
      {ParseExpr("(d0 * 8) mod 96"), ParseExpr("(d1 * 2) / 4"),
       ParseExpr("d0 * 16 + 500"), ParseExpr("(d0 * 16 + d1 * 2) mod 200")}};
  llvm::SmallVector<DimTile> group2 = {
      {ParseExpr("(d1 * 2) / 4"), ParseExpr("(d0 * 8) mod 96"),
       ParseExpr("d0 * 16 + 500"), ParseExpr("(d0 * 16 + d1 * 2) mod 200")}};
  SimplifyDimTiles({group1, group2}, *tiling_space_);
  EXPECT_THAT(group1[0], MatchString(R"(
    offset [v0 * 8],
    size [v1 / 2],
    stride [v0 * 16 + 500],
    upper bound [v0 * 16 + v1 * 2]
  )"));
  EXPECT_THAT(group2[0], MatchString(R"(
    offset [v1 / 2],
    size [v0 * 8],
    stride [v0 * 16 + 500],
    upper bound [v0 * 16 + v1 * 2]
  )"));
}

TEST_F(TilingSpaceSimplifyExpressionsTest, SimplifyDimTilesWithConstraints) {
  llvm::SmallVector<DimTile> dim_tiles = {
      {ParseExpr("d0"), ParseExpr("d1"), ParseExpr("1"),
       ParseExpr("(d0 * 16 + d1 * 2) mod 100")}};
  SimplifyDimTiles(dim_tiles, *tiling_space_,
                   {{ParseExpr("d0 * 16 + d1 * 2"), Interval{0, 50}}});
  EXPECT_THAT(dim_tiles[0], MatchString(R"(
    offset [v0],
    size [v1],
    stride [1],
    upper bound [v0 * 16 + v1 * 2]
  )"));
}

TEST_F(TilingSpaceSimplifyExpressionsTest,
       PointExpressionsSimplifiedToConstant) {
  EXPECT_THAT(tiling_space_->SimplifyExpressions({ParseExpr("(d1 * 2) / 10")})
                  .expressions,
              ElementsAre(ParseExpr("0")));
}

TEST_F(TilingSpaceSimplifyExpressionsTest,
       InfeasibleConstraintsReturnOriginalExpressions) {
  // d0 in [0, 6], so d0 * 16 <= 96 and the constraint is never satisfied.
  TilingSpace::SimplificationResult result = tiling_space_->SimplifyExpressions(
      {ParseExpr("(d0 * 8) mod 96")},
      {{ParseExpr("d0 * 16"), Interval{200, 300}}});
  EXPECT_TRUE(result.is_known_empty);
  EXPECT_THAT(result.expressions, ElementsAre(ParseExpr("(d0 * 8) mod 96")));
}

TEST_F(TilingSpaceSimplifyExpressionsTest,
       SimplifyDimTilesKeepsExpressionsForInfeasibleConstraints) {
  llvm::SmallVector<DimTile> dim_tiles = {
      {ParseExpr("(d0 * 8) mod 96"), ParseExpr("(d1 * 2) / 4"), ParseExpr("1"),
       ParseExpr("(d0 * 16 + d1 * 2) mod 200")}};
  llvm::SmallVector<DimTile> original = dim_tiles;
  SimplifyDimTiles(dim_tiles, *tiling_space_,
                   {{ParseExpr("d0 * 16"), Interval{200, 300}}});
  EXPECT_EQ(dim_tiles, original);
}

TEST_F(TilingSpaceTest, ClonePerformsDeepCopies) {
  auto root = ParseAndGetRoot(R"(
    HloModule m
    ENTRY e {
      src = s32[2,2,258] parameter(0)
      of1 = s32[] parameter(1)
      of2 = s32[] parameter(2)
      of3 = s32[] parameter(3)
      ROOT ds = s32[1,2,32] dynamic-slice(s32[2,2,258] src,
        s32[] of1, s32[] of2, s32[] of3),
        dynamic_slice_sizes={1, 2, 32}
    }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  ASSERT_OK_AND_ASSIGN(auto original_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));

  std::unique_ptr<TilingSpace> cloned_space = original_space->Clone();
  ASSERT_NE(cloned_space, nullptr);

  EXPECT_EQ(cloned_space->num_dimensions(), original_space->num_dimensions());
  EXPECT_EQ(cloned_space->num_parallel_dimensions(),
            original_space->num_parallel_dimensions());
  EXPECT_EQ(cloned_space->num_rt_vars(), original_space->num_rt_vars());
  EXPECT_EQ(cloned_space->mlir_context(), original_space->mlir_context());
  EXPECT_EQ(cloned_space->IsSymbolic(), original_space->IsSymbolic());

  // Dimensions deep copied.
  auto orig_dims = original_space->dimensions();
  auto cloned_dims = cloned_space->dimensions();
  ASSERT_EQ(orig_dims.size(), cloned_dims.size());
  for (size_t i = 0; i < orig_dims.size(); ++i) {
    EXPECT_EQ(orig_dims[i].id, cloned_dims[i].id);
    EXPECT_EQ(orig_dims[i].dimension_size, cloned_dims[i].dimension_size);
    EXPECT_EQ(orig_dims[i].type, cloned_dims[i].type);
    EXPECT_EQ(orig_dims[i].hlo, cloned_dims[i].hlo);
    EXPECT_EQ(orig_dims[i].dim_position, cloned_dims[i].dim_position);

    const auto& cloned_dim_ref = cloned_space->GetDimensionInfo(
        *cloned_dims[i].hlo, cloned_dims[i].dim_position);
    const auto& orig_dim_ref = original_space->GetDimensionInfo(
        *orig_dims[i].hlo, orig_dims[i].dim_position);

    EXPECT_NE(&cloned_dim_ref, &orig_dim_ref);
  }

  // RTVars deep copied.
  ASSERT_EQ(cloned_space->num_rt_vars(), 3);
  for (int64_t operand_id = 1; operand_id <= 3; ++operand_id) {
    auto orig_rt = original_space->GetRTVarInfo(*root, operand_id);
    auto cloned_rt = cloned_space->GetRTVarInfo(*root, operand_id);
    ASSERT_TRUE(orig_rt.has_value());
    ASSERT_TRUE(cloned_rt.has_value());
    EXPECT_EQ((*orig_rt)->id, (*cloned_rt)->id);
    EXPECT_EQ((*orig_rt)->bounds, (*cloned_rt)->bounds);
    EXPECT_EQ((*orig_rt)->hlo, (*cloned_rt)->hlo);
    EXPECT_NE(*orig_rt, *cloned_rt);
  }

  // Root tiles reference cloned space, not original space.
  ASSERT_EQ(cloned_space->tiled_roots().size(),
            original_space->tiled_roots().size());
  for (size_t i = 0; i < cloned_space->tiled_roots().size(); ++i) {
    EXPECT_EQ(&cloned_space->tiled_roots()[i].tiling_space(),
              cloned_space.get());
    EXPECT_NE(&cloned_space->tiled_roots()[i].tiling_space(),
              original_space.get());
  }
}

TEST_F(TilingSpaceTest, CloneAssignsIndependentTileSizes) {
  auto root = ParseAndGetRoot(R"(
      HloModule m
      ENTRY e {
        p0 = f32[1000, 10] parameter(0)
        ROOT a0 = f32[1000, 10] exponential(p0)
      }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  ASSERT_OK_AND_ASSIGN(auto original_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));

  std::unique_ptr<TilingSpace> cloned_space = original_space->Clone();
  ASSERT_NE(cloned_space, nullptr);

  EXPECT_TRUE(original_space->IsSymbolic());
  EXPECT_TRUE(cloned_space->IsSymbolic());

  // Assign tile sizes to cloned_space.
  EXPECT_OK(cloned_space->AssignTileSizes({16, 2}));
  EXPECT_FALSE(cloned_space->IsSymbolic());
  EXPECT_TRUE(original_space->IsSymbolic());

  auto cloned_dims = cloned_space->dimensions();
  EXPECT_EQ(cloned_dims[0].tile_size, 16);
  EXPECT_EQ(cloned_dims[1].tile_size, 2);

  auto orig_dims = original_space->dimensions();
  EXPECT_FALSE(orig_dims[0].tile_size.has_value());
  EXPECT_FALSE(orig_dims[1].tile_size.has_value());

  // Assign different tile sizes to original_space.
  EXPECT_OK(original_space->AssignTileSizes({32, 4}));
  EXPECT_FALSE(original_space->IsSymbolic());
  EXPECT_EQ(original_space->dimensions()[0].tile_size, 32);
  EXPECT_EQ(original_space->dimensions()[1].tile_size, 4);

  // Cloned space remains unaffected.
  EXPECT_EQ(cloned_space->dimensions()[0].tile_size, 16);
  EXPECT_EQ(cloned_space->dimensions()[1].tile_size, 2);
}

TEST_F(TilingSpaceTest, CloneIntoAnotherContextRebindsRootTiles) {
  auto root = ParseAndGetRoot(R"(
      HloModule m
      ENTRY e {
        p0 = f32[1000, 10] parameter(0)
        ROOT a0 = f32[1000, 10] exponential(p0)
      }
  )");
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
  ASSERT_OK_AND_ASSIGN(auto original_space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));

  mlir::MLIRContext target_context;
  std::unique_ptr<TilingSpace> cloned_space =
      original_space->Clone(&target_context);
  ASSERT_NE(cloned_space, nullptr);
  EXPECT_EQ(cloned_space->mlir_context(), &target_context);
  EXPECT_EQ(cloned_space->num_dimensions(), original_space->num_dimensions());

  for (const auto& root_tile : cloned_space->tiled_roots()) {
    for (const auto& dim_tile : root_tile.dim_tiles()) {
      EXPECT_EQ(dim_tile.size.GetContext(), &target_context);
      EXPECT_EQ(dim_tile.offset.GetContext(), &target_context);
      EXPECT_EQ(dim_tile.stride.GetContext(), &target_context);
      EXPECT_EQ(dim_tile.upper_bound.GetContext(), &target_context);
    }
  }

  EXPECT_TRUE(cloned_space->IsSymbolic());
  EXPECT_OK(cloned_space->AssignTileSizes({64, 2}));
  EXPECT_FALSE(cloned_space->IsSymbolic());
  EXPECT_EQ(cloned_space->dimensions()[0].tile_size, 64);
  EXPECT_EQ(cloned_space->dimensions()[1].tile_size, 2);
  EXPECT_TRUE(original_space->IsSymbolic());
}

TEST_F(TilingSpaceTest,
       ElementwiseChainRehashDoesNotInvalidateDimensionPointers) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m
    fused_computation {
      p0 = f32[2,4,8,16] parameter(0)
      p1 = f32[2,4,8,16] parameter(1)
      p2 = f32[2,4,8,16] parameter(2)
      p3 = f32[2,4,8,16] parameter(3)
      p4 = f32[2,4,8,16] parameter(4)
      p5 = f32[2,4,8,16] parameter(5)
      p6 = f32[2,4,8,16] parameter(6)
      p7 = f32[2,4,8,16] parameter(7)
      a0 = f32[2,4,8,16] add(p0, p1)
      a1 = f32[2,4,8,16] multiply(a0, p2)
      a2 = f32[2,4,8,16] subtract(a1, p3)
      a3 = f32[2,4,8,16] maximum(a2, p4)
      a4 = f32[2,4,8,16] minimum(a3, p5)
      a5 = f32[2,4,8,16] divide(a4, p6)
      a6 = f32[2,4,8,16] add(a5, p7)
      a7 = f32[2,4,8,16] multiply(a6, a0)
      a8 = f32[2,4,8,16] subtract(a7, a1)
      a9 = f32[2,4,8,16] maximum(a8, a2)
      a10 = f32[2,4,8,16] minimum(a9, a3)
      ROOT out = f32[2,4,8,16] add(a10, a4)
    }
    ENTRY main {
      p0 = f32[2,4,8,16] parameter(0)
      p1 = f32[2,4,8,16] parameter(1)
      p2 = f32[2,4,8,16] parameter(2)
      p3 = f32[2,4,8,16] parameter(3)
      p4 = f32[2,4,8,16] parameter(4)
      p5 = f32[2,4,8,16] parameter(5)
      p6 = f32[2,4,8,16] parameter(6)
      p7 = f32[2,4,8,16] parameter(7)
      ROOT fusion = f32[2,4,8,16] fusion(p0, p1, p2, p3, p4, p5, p6, p7),
        kind=kLoop, calls=fused_computation
    }
  )");
  ASSERT_OK(module.status());
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(
      module.value()->entry_computation()->root_instruction());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TilingSpace> space,
                       TilingSpace::Create(*fusion_adaptor, &mlir_context_));
  std::unique_ptr<TilingSpace> cloned = space->Clone();
  const HloComputation* comp = module.value()
                                   ->entry_computation()
                                   ->root_instruction()
                                   ->fused_instructions_computation();
  for (const HloInstruction* param : comp->parameter_instructions()) {
    for (int64_t dim = 0; dim < 4; ++dim) {
      EXPECT_EQ(cloned->GetDimensionInfo(*param, dim).id, TiledDimId(dim));
    }
  }
}

}  // namespace
}  // namespace xla::gpu::experimental

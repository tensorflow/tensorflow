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

#include "xla/backends/cpu/transforms/fusion_wrapper.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/cpu/target_machine_features_stub.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace cpu {
namespace {

class FusionWrapperTest : public HloHardwareIndependentTestBase {
 protected:
  TargetMachineFeaturesStub target_machine_features_{
      [](int64_t size_bytes) { return 16; }};
};

TEST_F(FusionWrapperTest, Scatter) {
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT sum = f32[] add(p0, p1)
    }
    ENTRY e {
      operand = f32[10,5] parameter(0)
      indices = s32[24,1] parameter(1)
      update = f32[24,2,3] parameter(2)
      ROOT scatter = f32[10,5] scatter(
          f32[10,5] operand,
          s32[24,1] indices,
          f32[24,2,3] update
        ),
        update_window_dims={1,2},
        inserted_window_dims={},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1,
        unique_indices=false,
        to_apply=add
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(m.get()));
  EXPECT_TRUE(changed);

  // A subsequent run should be a no-op -- the scatter is already fused.
  ASSERT_OK_AND_ASSIGN(changed, wrapper.Run(m.get()));
  EXPECT_FALSE(changed);
}

TEST_F(FusionWrapperTest, TransposeWrappedWithNewFusionEmitters) {
  // Standalone transposes are wrapped into loop fusions.
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    ENTRY e {
      p0 = f32[64,32] parameter(0)
      ROOT t = f32[32,64] transpose(p0), dimensions={1,0}
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(m.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kFusion);
}

TEST_F(FusionWrapperTest, DynamicUpdateSliceWrappedWithNewFusionEmitters) {
  // The MLIR fusion path has a dedicated dynamic-update-slice emitter with an
  // in-place check, so wrap standalone dynamic-update-slice when the new fusion
  // emitters are enabled.
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    ENTRY e {
      p0 = f32[64,64] parameter(0)
      p1 = f32[8,8] parameter(1)
      i = s32[] parameter(2)
      j = s32[] parameter(3)
      ROOT dus = f32[64,64] dynamic-update-slice(p0, p1, i, j)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(m.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kFusion);
}

TEST_F(FusionWrapperTest, PadWrappedWithNewFusionEmitters) {
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    ENTRY e {
      p0 = f32[4,4] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[10,8] pad(p0, p1), padding=1_2_1x2_2_0
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(m.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kFusion);
}

TEST_F(FusionWrapperTest, MissingElementalOpcodesWrappedWithNewFusionEmitters) {
  static constexpr absl::string_view kUnaryOpcodes[] = {
      "acos", "acosh", "asin", "asinh", "atanh", "cosh", "sinh"};
  for (absl::string_view op : kUnaryOpcodes) {
    std::string hlo_string = absl::StrFormat(R"(
    HloModule m
      ENTRY e {
        p0 = f32[64,32] parameter(0)
        ROOT r = f32[64,32] %s(p0)
      }
    )",
                                             op);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                         ParseAndReturnVerifiedModule(hlo_string));
    FusionWrapper wrapper;
    ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(m.get()));
    EXPECT_TRUE(changed) << "Failed for opcode: " << op;
    EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kFusion)
        << "Failed for opcode: " << op;
  }

  {
    static constexpr absl::string_view hlo_string = R"(
    HloModule m
      ENTRY e {
        p0 = s32[64,32] parameter(0)
        p1 = s32[64,32] parameter(1)
        ROOT r = s32[64,32] mulhi(p0, p1)
      }
    )";
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                         ParseAndReturnVerifiedModule(hlo_string));
    FusionWrapper wrapper;
    ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(m.get()));
    EXPECT_TRUE(changed) << "Failed for opcode: mulhi";
    EXPECT_EQ(m->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kFusion);
  }
}

TEST_F(FusionWrapperTest,
       CopyWithMismatchedLayoutsWrappedWithNewFusionEmitters) {
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    ENTRY e {
      p0 = f32[64,32]{0,1} parameter(0)
      ROOT copy = f32[64,32]{1,0} copy(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  EXPECT_TRUE(
      wrapper.MustWrapInstruction(*m->entry_computation()->root_instruction()));
}

TEST_F(FusionWrapperTest, CopyWithMatchingLayoutsNotWrapped) {
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    ENTRY e {
      p0 = f32[64,32]{0,1} parameter(0)
      ROOT copy = f32[64,32]{0,1} copy(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  EXPECT_FALSE(
      wrapper.MustWrapInstruction(*m->entry_computation()->root_instruction()));
}

TEST_F(FusionWrapperTest,
       LayoutChangingSubByteCopyWrappedWithNewFusionEmitters) {
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    ENTRY e {
      p0 = u2[20,20]{1,0:E(2)} parameter(0)
      ROOT copy = u2[20,20]{0,1:E(2)} copy(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  EXPECT_TRUE(
      wrapper.MustWrapInstruction(*m->entry_computation()->root_instruction()));
}

TEST_F(FusionWrapperTest, ConcatenateWithMismatchedLayoutsWrapped) {
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    ENTRY e {
      p0 = f32[64,32]{0,1} parameter(0)
      p1 = f32[64,32]{0,1} parameter(1)
      ROOT c = f32[128,32]{1,0} concatenate(p0, p1), dimensions={0}
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  EXPECT_TRUE(
      wrapper.MustWrapInstruction(*m->entry_computation()->root_instruction()));
}

TEST_F(FusionWrapperTest, ConcatenateWithMatchingLayoutsWrapped) {
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    ENTRY e {
      p0 = f32[64,32]{0,1} parameter(0)
      p1 = f32[64,32]{0,1} parameter(1)
      ROOT c = f32[128,32]{0,1} concatenate(p0, p1), dimensions={0}
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  EXPECT_TRUE(
      wrapper.MustWrapInstruction(*m->entry_computation()->root_instruction()));
}

TEST_F(FusionWrapperTest, NonEigenConvolutionWrappedWithNewFusionEmitters) {
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    ENTRY e {
      p0 = f32[1,8,8,3]{0,1,2,3} parameter(0)
      p1 = f32[3,3,3,16]{3,2,1,0} parameter(1)
      ROOT conv = f32[1,8,8,16]{3,2,1,0} convolution(p0, p1),
        window={size=3x3 pad=1_1x1_1},
        dim_labels=b01f_01io->b01f
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  EXPECT_TRUE(
      wrapper.MustWrapInstruction(*m->entry_computation()->root_instruction()));
}

TEST_F(FusionWrapperTest, EigenConvolutionNotWrapped) {
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    ENTRY e {
      p0 = f32[1,8,8,3]{3,2,1,0} parameter(0)
      p1 = f32[3,3,3,16]{3,2,1,0} parameter(1)
      ROOT conv = f32[1,8,8,16]{3,2,1,0} convolution(p0, p1),
        window={size=3x3 pad=1_1x1_1},
        dim_labels=b01f_01io->b01f
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  EXPECT_FALSE(
      wrapper.MustWrapInstruction(*m->entry_computation()->root_instruction()));
}

TEST_F(FusionWrapperTest, NonEigenConvolutionWrapped) {
  static constexpr absl::string_view hlo_string = R"(
  HloModule m
    ENTRY e {
      p0 = f32[3,3,64,64]{3,2,1,0} parameter(0)
      p1 = f32[672,7,7,64]{3,2,1,0} parameter(1)
      ROOT conv = f32[672,9,9,64]{3,2,1,0} convolution(p0, p1),
        window={size=7x7 pad=6_6x6_6},
        dim_labels=01bf_o01i->f01b
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  EXPECT_TRUE(
      wrapper.MustWrapInstruction(*m->entry_computation()->root_instruction()));
}

TEST_F(FusionWrapperTest, NaiveDotWrapped) {
  // Small matmuls are lowered to a naive loop nest, so they are routed to the
  // MLIR fusion emitter.
  static constexpr absl::string_view hlo_string = R"hlo(
  HloModule m
    ENTRY e {
      p0 = f32[3,3] parameter(0)
      p1 = f32[3,3] parameter(1)
      ROOT dot = f32[3,3] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

  // CHECK:       %[[COMPUTATION:.+]] ({{.*}}) -> f32[3,3] {
  // CHECK-NEXT:    %[[PARAM0:.+]] = f32[3,3]{1,0} parameter(0)
  // CHECK-NEXT:    %[[PARAM1:.+]] = f32[3,3]{1,0} parameter(1)
  // CHECK-NEXT:    ROOT %{{.+}} = f32[3,3]{1,0} dot(%[[PARAM0]], %[[PARAM1]]),
  // CHECK-SAME:      lhs_contracting_dims={1}, rhs_contracting_dims={0}

  // CHECK:       ENTRY %{{.+}} ({{.*}}) -> f32[3,3] {
  // CHECK-NEXT:    %[[P0:.+]] = f32[3,3]{1,0} parameter(0)
  // CHECK-NEXT:    %[[P1:.+]] = f32[3,3]{1,0} parameter(1)
  // CHECK-NEXT:    ROOT %{{.+}} = f32[3,3]{1,0} fusion(%[[P0]], %[[P1]]),
  // CHECK-SAME:      kind=kLoop, calls=%[[COMPUTATION]]
  )hlo";
  RunAndFilecheckHloRewrite(
      hlo_string, FusionWrapper(&target_machine_features_),
      [this](HloModule* module) {
        // A subsequent run should be a no-op -- the dot is already fused.
        FusionWrapper wrapper(&target_machine_features_);
        ASSERT_OK_AND_ASSIGN(bool changed, wrapper.Run(module));
        EXPECT_FALSE(changed);
      });
}

TEST_F(FusionWrapperTest, NaiveBatchDotWrapped) {
  // The strategy of a batch dot is the strategy of its inner dot.
  static constexpr absl::string_view hlo_string = R"hlo(
  HloModule m
    ENTRY e {
      p0 = f32[8,3,3] parameter(0)
      p1 = f32[8,3,3] parameter(1)
      ROOT dot = f32[8,3,3] dot(p0, p1),
        lhs_batch_dims={0}, lhs_contracting_dims={2},
        rhs_batch_dims={0}, rhs_contracting_dims={1}
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  EXPECT_TRUE(
      wrapper.MustWrapInstruction(*m->entry_computation()->root_instruction()));
}

TEST_F(FusionWrapperTest, NaiveDotWithoutTargetMachineFeaturesNotWrapped) {
  static constexpr absl::string_view hlo_string = R"hlo(
  HloModule m
    ENTRY e {
      p0 = f32[3,3] parameter(0)
      p1 = f32[3,3] parameter(1)
      ROOT dot = f32[3,3] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper;
  EXPECT_FALSE(
      wrapper.MustWrapInstruction(*m->entry_computation()->root_instruction()));
}

TEST_F(FusionWrapperTest, MatrixVectorDotNotWrapped) {
  // Lowered to the tiled LLVM IR GEMV implementation.
  static constexpr absl::string_view hlo_string = R"hlo(
  HloModule m
    ENTRY e {
      p0 = f32[64,32] parameter(0)
      p1 = f32[32] parameter(1)
      ROOT dot = f32[64] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  EXPECT_FALSE(
      wrapper.MustWrapInstruction(*m->entry_computation()->root_instruction()));
}

TEST_F(FusionWrapperTest, MatrixMatrixDotNotWrapped) {
  // Lowered to the tiled LLVM IR GEMM implementation or to an Eigen call.
  for (int64_t size : {16, 512}) {
    std::string hlo_string =
        absl::StrFormat(R"hlo(
    HloModule m
      ENTRY e {
        p0 = f32[%d,%d] parameter(0)
        p1 = f32[%d,%d] parameter(1)
        ROOT dot = f32[%d,%d] dot(p0, p1),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
      }
    )hlo",
                        size, size, size, size, size, size);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                         ParseAndReturnVerifiedModule(hlo_string));
    FusionWrapper wrapper(&target_machine_features_);
    EXPECT_FALSE(wrapper.MustWrapInstruction(
        *m->entry_computation()->root_instruction()))
        << "Failed for size: " << size;
  }
}

TEST_F(FusionWrapperTest, NaiveDotWithUnsupportedAlgorithmNotWrapped) {
  static constexpr absl::string_view hlo_string = R"hlo(
  HloModule m
    ENTRY e {
      p0 = f32[3,3] parameter(0)
      p1 = f32[3,3] parameter(1)
      ROOT dot = f32[3,3] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0},
        algorithm=dot_tf32_tf32_f32
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo_string));
  FusionWrapper wrapper(&target_machine_features_);
  EXPECT_FALSE(
      wrapper.MustWrapInstruction(*m->entry_computation()->root_instruction()));
}

}  // namespace
}  // namespace cpu
}  // namespace xla

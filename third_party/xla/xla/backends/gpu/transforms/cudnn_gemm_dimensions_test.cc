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

#include "xla/backends/gpu/transforms/cudnn_gemm_dimensions.h"

#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::Pointee;

class CuDnnGemmDimensionsTest : public HloHardwareIndependentTestBase {
 protected:
  // Analyzes the entry computation of `hlo_text`, which contains a dot.
  absl::StatusOr<CuDnnGemmDimensions> Analyze(absl::string_view hlo_text) {
    ABSL_ASSIGN_OR_RETURN(module_, ParseAndReturnVerifiedModule(hlo_text));
    const HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
        *module_->entry_computation(),
        {HloOpcode::kDot, HloOpcode::kScaledDot});
    if (dot == nullptr) {
      return absl::InvalidArgumentError("No dot found.");
    }
    return CuDnnGemmDimensions::Create(*dot);
  }

  const HloInstruction& Instruction(absl::string_view name) {
    return *FindInstruction(module_.get(), name);
  }

  std::unique_ptr<VerifiedHloModule> module_;
};

template <typename SizesMatcher, typename StridesMatcher>
auto HasSizesAndStrides(SizesMatcher sizes, StridesMatcher strides) {
  return Pointee(
      AllOf(Field("sizes", &CudnnTensorLayout::sizes, sizes),
            Field("strides", &CudnnTensorLayout::strides, strides),
            Field("slices", &CudnnTensorLayout::slices, Eq(std::nullopt))));
}

TEST_F(CuDnnGemmDimensionsTest, PlainDot) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  p0 = f32[64,64] parameter(0)
  p1 = f32[64,64] parameter(1)
  ROOT d = f32[64,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p0")),
      HasSizesAndStrides(ElementsAre(1, 64, 64), ElementsAre(4096, 64, 1)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p1")),
      HasSizesAndStrides(ElementsAre(1, 64, 64), ElementsAre(4096, 64, 1)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("d")),
      HasSizesAndStrides(ElementsAre(1, 64, 64), ElementsAre(4096, 64, 1)));
}

TEST_F(CuDnnGemmDimensionsTest, ScaledDot) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  lhs = f8e4m3fn[256,128] parameter(0)
  rhs = f8e4m3fn[384,128] parameter(1)
  lhs_scale = f8e8m0fnu[256,4] parameter(2)
  rhs_scale = f8e8m0fnu[384,4] parameter(3)
  ROOT d = f32[256,384] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})"));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("lhs")),
      HasSizesAndStrides(ElementsAre(1, 256, 128), ElementsAre(32768, 128, 1)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("rhs")),
      HasSizesAndStrides(ElementsAre(1, 128, 384), ElementsAre(49152, 1, 128)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("d")),
      HasSizesAndStrides(ElementsAre(1, 256, 384), ElementsAre(98304, 384, 1)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("lhs_scale")),
      HasSizesAndStrides(ElementsAre(1, 256, 4), ElementsAre(1024, 4, 1)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("rhs_scale")),
      HasSizesAndStrides(ElementsAre(1, 4, 384), ElementsAre(1536, 1, 4)));
}

TEST_F(CuDnnGemmDimensionsTest, VectorTensorMultiplication) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  p0 = bf16[64,1] parameter(0)
  p1 = s8[64,128] parameter(1)
  p1c = bf16[64,128] convert(p1)
  ROOT out = bf16[1,128] dot(p0, p1c),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p0")),
      HasSizesAndStrides(ElementsAre(1, 1, 64), ElementsAre(64, 64, 1)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p1")),
      HasSizesAndStrides(ElementsAre(1, 64, 128), ElementsAre(8192, 128, 1)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("out")),
      HasSizesAndStrides(ElementsAre(1, 1, 128), ElementsAre(128, 128, 1)));
}

TEST_F(CuDnnGemmDimensionsTest, TensorVectorMultiplication) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  p0 = bf16[64,256] parameter(0)
  p1 = s8[64,1] parameter(1)
  p1c = bf16[64,1] convert(p1)
  ROOT out = bf16[256,1] dot(p0, p1c),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p0")),
      HasSizesAndStrides(ElementsAre(1, 256, 64), ElementsAre(16384, 1, 256)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p1")),
      HasSizesAndStrides(ElementsAre(1, 64, 1), ElementsAre(64, 1, 64)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("out")),
      HasSizesAndStrides(ElementsAre(1, 256, 1), ElementsAre(256, 1, 256)));
}

TEST_F(CuDnnGemmDimensionsTest, NonDefaultLayout) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  p0 = f32[32,64] parameter(0)
  p1 = f32[64,128]{0,1} parameter(1)
  ROOT out = f32[32,128]{0,1} dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p0")),
      HasSizesAndStrides(ElementsAre(1, 32, 64), ElementsAre(2048, 64, 1)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p1")),
      HasSizesAndStrides(ElementsAre(1, 64, 128), ElementsAre(8192, 1, 64)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("out")),
      HasSizesAndStrides(ElementsAre(1, 32, 128), ElementsAre(4096, 1, 32)));
}

TEST_F(CuDnnGemmDimensionsTest, BatchedDotWithCopyAndEpilogue) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  p0 = bf16[96,512,64]{1,2,0} parameter(0)
  cp = bf16[96,512,64]{2,1,0} copy(p0)
  p1 = bf16[96,64,512]{2,1,0} parameter(1)
  d = bf16[96,512,512]{2,1,0} dot(cp, p1),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
  p2 = bf16[96,512,512] parameter(2)
  ROOT s = bf16[96,512,512] subtract(p2, d)
})"));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p0")),
      HasSizesAndStrides(ElementsAre(96, 512, 64), ElementsAre(32768, 1, 512)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p1")),
      HasSizesAndStrides(ElementsAre(96, 64, 512), ElementsAre(32768, 512, 1)));
  EXPECT_THAT(dims.TensorLayout(Instruction("p2")),
              HasSizesAndStrides(ElementsAre(96, 512, 512),
                                 ElementsAre(262144, 512, 1)));
  EXPECT_THAT(dims.TensorLayout(Instruction("s")),
              HasSizesAndStrides(ElementsAre(96, 512, 512),
                                 ElementsAre(262144, 512, 1)));
  EXPECT_EQ(dims.TensorLayout(Instruction("cp")), nullptr);
  EXPECT_EQ(dims.TensorLayout(Instruction("d")), nullptr);
}

TEST_F(CuDnnGemmDimensionsTest, LayoutThatIsNotItsOwnInverse) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  p0 = bf16[96,512,64]{0,2,1} parameter(0)
  p1 = bf16[96,64,32] parameter(1)
  ROOT d = bf16[96,512,32] dot(p0, p1),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
})"));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p0")),
      HasSizesAndStrides(ElementsAre(96, 512, 64), ElementsAre(1, 6144, 96)));
}

TEST_F(CuDnnGemmDimensionsTest, SeveralBatchAndNonContractingDimensions) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  p0 = f32[8,2,4,16,32] parameter(0)
  p1 = f32[8,2,32,64] parameter(1)
  ROOT d = f32[8,2,4,16,64] dot(p0, p1),
    lhs_batch_dims={0,1}, lhs_contracting_dims={4},
    rhs_batch_dims={0,1}, rhs_contracting_dims={2}
})"));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p0")),
      HasSizesAndStrides(ElementsAre(16, 64, 32), ElementsAre(2048, 32, 1)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p1")),
      HasSizesAndStrides(ElementsAre(16, 32, 64), ElementsAre(2048, 64, 1)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("d")),
      HasSizesAndStrides(ElementsAre(16, 64, 64), ElementsAre(4096, 64, 1)));
}

TEST_F(CuDnnGemmDimensionsTest, BatchDimensionIsNotTheMajorMostOne) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  p0 = f32[16,8,32] parameter(0)
  p1 = f32[8,32,24] parameter(1)
  ROOT d = f32[8,16,24] dot(p0, p1),
    lhs_batch_dims={1}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
})"));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p0")),
      HasSizesAndStrides(ElementsAre(8, 16, 32), ElementsAre(32, 256, 1)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p1")),
      HasSizesAndStrides(ElementsAre(8, 32, 24), ElementsAre(768, 24, 1)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("d")),
      HasSizesAndStrides(ElementsAre(8, 16, 24), ElementsAre(384, 24, 1)));
}

TEST_F(CuDnnGemmDimensionsTest, BroadcastToMinorDimension) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  p0 = f16[16,32,128] parameter(0)
  p1 = f16[16,128,64] parameter(1)
  p2 = f16[16,32] parameter(2)
  p2b = f16[16,32,128] broadcast(p2), dimensions={0,1}
  a = f16[16,32,128] add(p0, p2b)
  ROOT r = f16[16,32,64] dot(a, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
})"));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p2")),
      HasSizesAndStrides(ElementsAre(16, 32, 1), ElementsAre(32, 1, 32)));
}

TEST_F(CuDnnGemmDimensionsTest, BroadcastToBatchDimension) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  p0 = bf16[32,128] parameter(0)
  p0b = bf16[5,32,128] broadcast(p0), dimensions={1,2}
  p1 = bf16[5,128,64] parameter(1)
  ROOT r = f32[5,32,64] dot(p0b, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
})"));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p0")),
      HasSizesAndStrides(ElementsAre(1, 32, 128), ElementsAre(1, 128, 1)));
}

TEST_F(CuDnnGemmDimensionsTest, BroadcastOfScalarParameters) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  x = f8e4m3fn[16,32] parameter(0)
  y = f8e4m3fn[32,16] parameter(1)
  dot = f32[16,16] dot(x, y), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  x_scale = f32[] parameter(2)
  y_scale = f32[] parameter(3)
  combined_scale = f32[] multiply(x_scale, y_scale)
  scale_bcast = f32[16,16] broadcast(combined_scale), dimensions={}
  ROOT out = f32[16,16] multiply(dot, scale_bcast)
})"));
  EXPECT_THAT(dims.TensorLayout(Instruction("x_scale")),
              HasSizesAndStrides(ElementsAre(1, 1, 1), ElementsAre(1, 1, 1)));
  EXPECT_THAT(dims.TensorLayout(Instruction("y_scale")),
              HasSizesAndStrides(ElementsAre(1, 1, 1), ElementsAre(1, 1, 1)));
}

TEST_F(CuDnnGemmDimensionsTest, ClampWithScalarBounds) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  d = f32[32,64] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  lo = f32[] parameter(2)
  hi = f32[] parameter(3)
  ROOT c = f32[32,64] clamp(lo, d, hi)
})"));
  EXPECT_THAT(dims.TensorLayout(Instruction("lo")),
              HasSizesAndStrides(ElementsAre(1, 1, 1), ElementsAre(1, 1, 1)));
  EXPECT_THAT(dims.TensorLayout(Instruction("hi")),
              HasSizesAndStrides(ElementsAre(1, 1, 1), ElementsAre(1, 1, 1)));
}

TEST_F(CuDnnGemmDimensionsTest, ScalarDotBroadcastByClampIsNotSupported) {
  EXPECT_THAT(Analyze(R"(
e {
  p0 = f32[64] parameter(0)
  p1 = f32[64] parameter(1)
  p2 = f32[16,16] parameter(2)
  hi = f32[] parameter(3)
  d = f32[] dot(p0, p1), lhs_contracting_dims={0}, rhs_contracting_dims={0}
  ROOT out = f32[16,16] clamp(d, p2, hi)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Implicit broadcasts of the dot")));
}

TEST_F(CuDnnGemmDimensionsTest, BroadcastInEpilogueAfterBitcast) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  p0 = f32[64,32] parameter(0)
  p1 = f32[32,48] parameter(1)
  d = f32[64,48] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bc = f32[4,16,48] bitcast(d)
  p2 = f32[48] parameter(2)
  p2b = f32[4,16,48] broadcast(p2), dimensions={2}
  ROOT a = f32[4,16,48] add(bc, p2b)
})"));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p2")),
      HasSizesAndStrides(ElementsAre(1, 1, 48), ElementsAre(48, 48, 1)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("a")),
      HasSizesAndStrides(ElementsAre(1, 64, 48), ElementsAre(3072, 48, 1)));
}

TEST_F(CuDnnGemmDimensionsTest, ConsistentDiamondAfterDot) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  p0 = f32[64,32] parameter(0)
  p1 = f32[32,48] parameter(1)
  d = f32[64,48] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bc = f32[4,16,48] bitcast(d)
  n = f32[4,16,48] negate(bc)
  p2 = f32[4,16,48] parameter(2)
  m = f32[4,16,48] multiply(n, p2)
  ROOT a = f32[4,16,48] add(bc, m)
})"));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("p2")),
      HasSizesAndStrides(ElementsAre(1, 64, 48), ElementsAre(3072, 48, 1)));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("a")),
      HasSizesAndStrides(ElementsAre(1, 64, 48), ElementsAre(3072, 48, 1)));
}

TEST_F(CuDnnGemmDimensionsTest, InconsistentDiamondAfterDotIsNotSupported) {
  // d + transpose(d) combines elements at different positions of the dot
  // result, which cannot be expressed as a cuDNN graph.
  EXPECT_THAT(
      Analyze(R"(
e {
  p0 = f32[64,32] parameter(0)
  p1 = f32[32,64] parameter(1)
  d = f32[64,64] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  t = f32[64,64] transpose(d), dimensions={1,0}
  ROOT a = f32[64,64] add(d, t)
})"),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("several ways")));
}

TEST_F(CuDnnGemmDimensionsTest, SameParameterInBothOperandsIsNotSupported) {
  EXPECT_THAT(
      Analyze(R"(
e {
  p0 = f32[64,64] parameter(0)
  ROOT d = f32[64,64] dot(p0, p0),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("several ways")));
}

TEST_F(CuDnnGemmDimensionsTest, UnsupportedInstructionIsRejected) {
  EXPECT_THAT(Analyze(R"(
add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT a = f32[] add(x, y)
}

e {
  p0 = f32[16,32,4] parameter(0)
  c0 = f32[] constant(0)
  r0 = f32[16,32] reduce(p0, c0), dimensions={2}, to_apply=add
  p1 = f32[32,8] parameter(1)
  ROOT d = f32[16,8] dot(r0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Unsupported instruction")));
}

TEST_F(CuDnnGemmDimensionsTest, MultipleOutputsAreNotSupported) {
  EXPECT_THAT(
      Analyze(R"(
e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  d = f32[32,64] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  n = f32[32,64] negate(d)
  ROOT t = (f32[32,64], f32[32,64]) tuple(d, n)
})"),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("single output")));
}

TEST_F(CuDnnGemmDimensionsTest, UnusedParameterIsRejected) {
  EXPECT_THAT(Analyze(R"(
e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  p2 = f32[32,64] parameter(2)
  ROOT d = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("does not contribute to the output")));
}

TEST_F(CuDnnGemmDimensionsTest, ReduceAfterTheDotIsRejected) {
  EXPECT_THAT(Analyze(R"(
add {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT a = f32[] add(x, y)
}

e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  d = f32[32,64] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c0 = f32[] constant(0)
  ROOT r = f32[32] reduce(d, c0), dimensions={1}, to_apply=add
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Unsupported instruction")));
}

TEST_F(CuDnnGemmDimensionsTest, ParameterThatIsNotAStridedViewIsRejected) {
  // r0 reads the elements of p0 in an order that no strides can describe.
  EXPECT_THAT(Analyze(R"(
e {
  p0 = f32[3,2,16] parameter(0)
  t0 = f32[2,3,16] transpose(p0), dimensions={1,0,2}
  r0 = f32[3,2,16] reshape(t0)
  p1 = f32[3,16,8] parameter(1)
  ROOT d = f32[3,2,8] dot(r0, p1),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Not a strided view")));
}

TEST_F(CuDnnGemmDimensionsTest, CanonicalSizesOfUnaryOfBroadcast) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  p0 = f32[16,32] parameter(0)
  c = f32[] constant(1)
  cb = f32[16,32] broadcast(c), dimensions={}
  n = f32[16,32] negate(cb)
  a = f32[16,32] add(p0, n)
  p1 = f32[32,8] parameter(1)
  ROOT d = f32[16,8] dot(a, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  EXPECT_THAT(dims.CanonicalSizes(Instruction("n")),
              Pointee(ElementsAre(1, 16, 32)));
}

TEST_F(CuDnnGemmDimensionsTest, MDimensionWithThreeSegmentsIsNotSupported) {
  EXPECT_THAT(Analyze(R"(
e {
  p0 = f32[2,3,4,5] parameter(0)
  t0 = f32[4,3,2,5] transpose(p0), dimensions={2,1,0,3}
  r0 = f32[24,5] reshape(t0)
  p1 = f32[5,16] parameter(1)
  ROOT d = f32[24,16] dot(r0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("not a single strided dimension")));
}

TEST_F(CuDnnGemmDimensionsTest, WorkspaceIsIgnored) {
  ASSERT_OK_AND_ASSIGN(CuDnnGemmDimensions dims, Analyze(R"(
e {
  a = f32[32,96] parameter(0)
  b = f32[96,64] parameter(1)
  d = f32[32,64] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  c = s8[1024] custom-call(), custom_call_target="__nop"
  ROOT t = (f32[32,64], s8[1024]) tuple(d, c)
})"));
  EXPECT_THAT(
      dims.TensorLayout(Instruction("d")),
      HasSizesAndStrides(ElementsAre(1, 32, 64), ElementsAre(2048, 64, 1)));
  EXPECT_EQ(dims.TensorLayout(Instruction("c")), nullptr);
  EXPECT_EQ(dims.CanonicalSizes(Instruction("c")), nullptr);
}

}  // namespace
}  // namespace xla::gpu

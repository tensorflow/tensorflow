/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/dot_operand_converter.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/tensor_float_32_utils.h"

namespace xla::gpu {
namespace {

namespace op = ::xla::testing::opcode_matchers;

using ::testing::HasSubstr;
using ::testing::Not;

class DotOperandConverterTest : public HloPjRtGpuTestBase {
 public:
  void TestConvert(bool left_less_precise, PrimitiveType lhs_type,
                   PrimitiveType rhs_type, PrimitiveType result_type) {
    absl::string_view module_tmpl = R"(
  HloModule module

  ENTRY main {
    p0 = $0[2,3]{1,0} parameter(0)
    p1 = $1[3,2]{1,0} parameter(1)
    ROOT dot = $2[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                         rhs_contracting_dims={0}
  })";
    auto module_string = absl::Substitute(
        module_tmpl, primitive_util::LowercasePrimitiveTypeName(lhs_type),
        primitive_util::LowercasePrimitiveTypeName(rhs_type),
        primitive_util::LowercasePrimitiveTypeName(result_type));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(module_string));
    ASSERT_OK_AND_ASSIGN(bool upcasted,
                         DotOperandConverter().Run(module.get()));
    EXPECT_TRUE(upcasted);
    if (left_less_precise) {
      auto original_lhs = op::Parameter(0);
      auto upcasted_lhs =
          AllOf(op::Convert(original_lhs),
                op::Shape(absl::Substitute(
                    "$0[2,3]{1,0}",
                    primitive_util::LowercasePrimitiveTypeName(rhs_type))));
      EXPECT_THAT(
          module->entry_computation()->root_instruction(),
          AllOf(op::Dot(upcasted_lhs, op::Parameter(1)),
                op::Shape(absl::Substitute(
                    "$0[2,2]{1,0}",
                    primitive_util::LowercasePrimitiveTypeName(result_type)))));
    } else {
      auto original_rhs = op::Parameter(1);
      auto upcasted_rhs =
          AllOf(op::Convert(original_rhs),
                op::Shape(absl::Substitute(
                    "$0[3,2]{1,0}",
                    primitive_util::LowercasePrimitiveTypeName(lhs_type))));
      EXPECT_THAT(
          module->entry_computation()->root_instruction(),
          AllOf(op::Dot(op::Parameter(0), upcasted_rhs),
                op::Shape(absl::Substitute(
                    "$0[2,2]{1,0}",
                    primitive_util::LowercasePrimitiveTypeName(result_type)))));
    }
  }
};

TEST_F(DotOperandConverterTest, ConvertsLeftAndRight) {
  TestConvert(/*left_less_precise=*/true, S8, BF16, F32);
  TestConvert(/*left_less_precise=*/false, BF16, S8, F32);
}

TEST_F(DotOperandConverterTest, ConvertsSmallIntAndF32ToBf16ByDefault) {
  for (PrimitiveType int_type : {S4, U4, S8, U8}) {
    for (bool int_on_lhs : {true, false}) {
      for (PrimitiveType out_type : {F32, BF16}) {
        PrimitiveType lhs_type = int_on_lhs ? int_type : F32;
        PrimitiveType rhs_type = int_on_lhs ? F32 : int_type;
        SCOPED_TRACE(absl::StrCat(
            "lhs=", primitive_util::LowercasePrimitiveTypeName(lhs_type),
            ", rhs=", primitive_util::LowercasePrimitiveTypeName(rhs_type),
            ", out=", primitive_util::LowercasePrimitiveTypeName(out_type)));
        absl::string_view module_tmpl = R"(
        HloModule module

        ENTRY main {
          p0 = $0[2,3]{1,0} parameter(0)
          p1 = $1[3,2]{1,0} parameter(1)
          ROOT dot = $2[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                                rhs_contracting_dims={0}
        })";
        auto out_type_str =
            primitive_util::LowercasePrimitiveTypeName(out_type);
        auto module_string = absl::Substitute(
            module_tmpl, primitive_util::LowercasePrimitiveTypeName(lhs_type),
            primitive_util::LowercasePrimitiveTypeName(rhs_type), out_type_str);
        ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                             ParseAndReturnVerifiedModule(module_string));
        ASSERT_OK_AND_ASSIGN(bool changed,
                             DotOperandConverter().Run(module.get()));
        EXPECT_TRUE(changed);
        auto converted_lhs =
            AllOf(op::Convert(op::Parameter(0)), op::Shape("bf16[2,3]{1,0}"));
        auto converted_rhs =
            AllOf(op::Convert(op::Parameter(1)), op::Shape("bf16[3,2]{1,0}"));
        EXPECT_THAT(
            module->entry_computation()->root_instruction(),
            AllOf(op::Dot(converted_lhs, converted_rhs),
                  op::Shape(absl::Substitute("$0[2,2]{1,0}", out_type_str))));
      }
    }
  }
}

TEST_F(DotOperandConverterTest,
       ClearsElementSizeInBitsWhenConvertingPackedSmallIntToBf16) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s4[2,4]{1,0:E(4)} parameter(0)
    p1 = f32[4,2]{1,0} parameter(1)
    ROOT dot = f32[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotOperandConverter().Run(module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  auto converted_lhs =
      AllOf(op::Convert(op::Parameter(0)), op::Shape("bf16[2,4]{1,0}"));
  auto converted_rhs =
      AllOf(op::Convert(op::Parameter(1)), op::Shape("bf16[4,2]{1,0}"));
  EXPECT_THAT(root, AllOf(op::Dot(converted_lhs, converted_rhs),
                          op::Shape("f32[2,2]{1,0}")));
  EXPECT_EQ(root->operand(0)->shape().layout().element_size_in_bits(), 0);
}

TEST_F(DotOperandConverterTest, PreservesF32ForSmallIntWithHighPrecision) {
  for (absl::string_view operand_precision :
       {"{HIGH, HIGH}", "{DEFAULT, HIGH}", "{HIGH, DEFAULT}"}) {
    SCOPED_TRACE(operand_precision);
    absl::string_view module_tmpl = R"(
    HloModule module

    ENTRY main {
      p0 = s8[2,3]{1,0} parameter(0)
      p1 = f32[3,2]{1,0} parameter(1)
      ROOT dot = f32[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                            rhs_contracting_dims={0},
                                            operand_precision=$0
    })";
    auto module_string = absl::Substitute(module_tmpl, operand_precision);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(module_string));
    ASSERT_OK_AND_ASSIGN(bool changed, DotOperandConverter().Run(module.get()));
    EXPECT_TRUE(changed);
    auto upcasted_lhs =
        AllOf(op::Convert(op::Parameter(0)), op::Shape("f32[2,3]{1,0}"));
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                AllOf(op::Dot(upcasted_lhs, op::Parameter(1)),
                      op::Shape("f32[2,2]{1,0}")));
  }
}

TEST_F(DotOperandConverterTest, PreservesF32ForSmallIntWithExplicitAlgorithm) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[2,3]{1,0} parameter(0)
    p1 = f32[3,2]{1,0} parameter(1)
    ROOT dot = f32[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0},
                                          algorithm=dot_tf32_tf32_f32
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotOperandConverter().Run(module.get()));
  EXPECT_TRUE(changed);
  auto upcasted_lhs =
      AllOf(op::Convert(op::Parameter(0)), op::Shape("f32[2,3]{1,0}"));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Dot(upcasted_lhs, op::Parameter(1)),
                    op::Shape("f32[2,2]{1,0}")));
}

TEST_F(DotOperandConverterTest,
       UpcastsSmallIntToF32WhenOutputTypeIsNotF32OrBf16) {
  TestConvert(/*left_less_precise=*/true, S8, F32, F64);
}

TEST_F(DotOperandConverterTest,
       PreservesF32ForSmallIntWhenTensorFloat32Disabled) {
  bool orig_tf32 = tsl::tensor_float_32_execution_enabled();
  tsl::enable_tensor_float_32_execution(false);
  absl::Cleanup restore_tf32 = [orig_tf32] {
    tsl::enable_tensor_float_32_execution(orig_tf32);
  };
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[2,3]{1,0} parameter(0)
    p1 = f32[3,2]{1,0} parameter(1)
    ROOT dot = f32[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotOperandConverter().Run(module.get()));
  EXPECT_TRUE(changed);
  auto upcasted_lhs =
      AllOf(op::Convert(op::Parameter(0)), op::Shape("f32[2,3]{1,0}"));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Dot(upcasted_lhs, op::Parameter(1)),
                    op::Shape("f32[2,2]{1,0}")));
}

TEST_F(DotOperandConverterTest,
       PreservesF32ForSmallIntOnTargetsWithoutBf16DotSupport) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[2,3]{1,0} parameter(0)
    p1 = f32[3,2]{1,0} parameter(1)
    ROOT dot = f32[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
  })";
  for (se::GpuComputeCapability cc :
       {se::GpuComputeCapability(se::CudaComputeCapability::Volta()),
        se::GpuComputeCapability(se::RocmComputeCapability("gfx908")),
        se::GpuComputeCapability(se::RocmComputeCapability("gfx90a")),
        se::GpuComputeCapability(se::RocmComputeCapability("gfx1030")),
        se::GpuComputeCapability(se::RocmComputeCapability("gfx1100")),
        se::GpuComputeCapability(se::RocmComputeCapability("gfx1200"))}) {
    SCOPED_TRACE(cc.ToString());
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(module_string));
    ASSERT_OK_AND_ASSIGN(bool changed,
                         DotOperandConverter(cc).Run(module.get()));
    EXPECT_TRUE(changed);
    auto upcasted_lhs =
        AllOf(op::Convert(op::Parameter(0)), op::Shape("f32[2,3]{1,0}"));
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                AllOf(op::Dot(upcasted_lhs, op::Parameter(1)),
                      op::Shape("f32[2,2]{1,0}")));
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> mi300_module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(
      bool mi300_changed,
      DotOperandConverter(
          se::GpuComputeCapability(se::RocmComputeCapability("gfx942")))
          .Run(mi300_module.get()));
  EXPECT_TRUE(mi300_changed);
  auto converted_lhs =
      AllOf(op::Convert(op::Parameter(0)), op::Shape("bf16[2,3]{1,0}"));
  auto converted_rhs =
      AllOf(op::Convert(op::Parameter(1)), op::Shape("bf16[3,2]{1,0}"));
  EXPECT_THAT(
      mi300_module->entry_computation()->root_instruction(),
      AllOf(op::Dot(converted_lhs, converted_rhs), op::Shape("f32[2,2]{1,0}")));
}

TEST_F(DotOperandConverterTest, PreservesF32ForS16AndF32) {
  TestConvert(/*left_less_precise=*/true, S16, F32, F32);
}

TEST_F(DotOperandConverterTest, NoConvertHappensWithSameTypes) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[2,3]{1,0} parameter(0)
    p1 = s8[3,2]{1,0} parameter(1)
    ROOT dot = bf16[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                         rhs_contracting_dims={0}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool upcasted, DotOperandConverter().Run(module.get()));
  EXPECT_FALSE(upcasted);
}

TEST_F(DotOperandConverterTest, NoConvertFromF8toF8) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f8e4m3fn[2,3]{1,0} parameter(0)
    p1 = f8e5m2[3,2]{1,0} parameter(1)
    ROOT dot = bf16[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                         rhs_contracting_dims={0}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool upcasted, DotOperandConverter().Run(module.get()));
  EXPECT_FALSE(upcasted);
}

TEST_F(DotOperandConverterTest, NoConvertFromF8FNUZtoF8FNUZ) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f8e4m3fnuz[2,3]{1,0} parameter(0)
    p1 = f8e5m2fnuz[3,2]{1,0} parameter(1)
    ROOT dot = bf16[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                           rhs_contracting_dims={0}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool upcasted, DotOperandConverter().Run(module.get()));
  EXPECT_FALSE(upcasted);
}

TEST_F(DotOperandConverterTest, CompilerOptimizesUsingDotOperandConverter) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[2,3]{1,0} parameter(0)
    p1 = bf16[3,2]{1,0} parameter(1)
    ROOT dot = bf16[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                         rhs_contracting_dims={0}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       GetOptimizedModule(module_string));
}

TEST_F(DotOperandConverterTest,
       CompilerConvertsMultiAxisSmallIntAndF32DotToBf16) {
  if (!DotOperandConverter::SupportsBf16Dot(
          device_description().gpu_compute_capability())) {
    GTEST_SKIP() << "Requires BF16 dot support.";
  }
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[16,32,128]{2,1,0} parameter(0)
    p1 = s8[128,128]{1,0} parameter(1)
    ROOT dot = f32[16,32,128]{2,1,0} dot(p0, p1), lhs_contracting_dims={2},
                                                  rhs_contracting_dims={0}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       GetOptimizedModule(module_string));
  EXPECT_THAT(module->ToString(), HasSubstr("bf16[128,128]"));
  EXPECT_THAT(module->ToString(), Not(HasSubstr("f32[128,128]")));
}

TEST_F(DotOperandConverterTest, DemotesUpcastS8AndBf16DotToBf16) {
  // Mirrors `dot(convert<f32>(s8), convert<f32>(reduce-precision(bf16)))`.
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[64,8,128]{2,1,0} parameter(0)
    p1 = bf16[32,8,128]{2,1,0} parameter(1)
    c0 = f32[64,8,128]{2,1,0} convert(p0), metadata={op_name="upcast_s8"}
    rp = bf16[32,8,128]{2,1,0} reduce-precision(p1), exponent_bits=8,
                                                     mantissa_bits=7
    c1 = f32[32,8,128]{2,1,0} convert(rp)
    ROOT dot = f32[8,64,32]{2,1,0} dot(c0, c1), lhs_batch_dims={1},
        lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={2}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotOperandConverter().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Dot(AllOf(op::Convert(op::Parameter(0)),
                                  op::Shape("bf16[64,8,128]{2,1,0}")),
                            op::ReducePrecision(op::Parameter(1))),
                    op::Shape("f32[8,64,32]{2,1,0}")));
  // The new convert replaces `c0`, so it inherits its metadata rather than the
  // parameter's.
  EXPECT_EQ(module->entry_computation()
                ->root_instruction()
                ->operand(0)
                ->metadata()
                .op_name(),
            "upcast_s8");
}

TEST_F(DotOperandConverterTest, DemotesUpcastBf16AndSmallIntDotsToBf16) {
  // BF16 has an 8-bit significand (7 explicit mantissa bits plus the implicit
  // one), so every integer with magnitude <= 2^8 is exact; this covers all
  // 4-bit and 8-bit integers (|x| <= 255).
  for (PrimitiveType int_type : {S4, U4, S8, U8}) {
    for (PrimitiveType out_type : {F32, BF16}) {
      SCOPED_TRACE(absl::StrCat(
          "int=", primitive_util::LowercasePrimitiveTypeName(int_type),
          ", out=", primitive_util::LowercasePrimitiveTypeName(out_type)));
      absl::string_view module_tmpl = R"(
      HloModule module

      ENTRY main {
        p0 = bf16[2,3]{1,0} parameter(0)
        p1 = $0[3,2]{1,0} parameter(1)
        c0 = f32[2,3]{1,0} convert(p0)
        c1 = f32[3,2]{1,0} convert(p1)
        ROOT dot = $1[2,2]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                             rhs_contracting_dims={0}
      })";
      std::string out_type_str(
          primitive_util::LowercasePrimitiveTypeName(out_type));
      ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<HloModule> module,
          ParseAndReturnVerifiedModule(absl::Substitute(
              module_tmpl, primitive_util::LowercasePrimitiveTypeName(int_type),
              out_type_str)));
      ASSERT_OK_AND_ASSIGN(bool changed,
                           DotOperandConverter().Run(module.get()));
      EXPECT_TRUE(changed);
      EXPECT_THAT(
          module->entry_computation()->root_instruction(),
          AllOf(op::Dot(op::Parameter(0), AllOf(op::Convert(op::Parameter(1)),
                                                op::Shape("bf16[3,2]{1,0}"))),
                op::Shape(absl::Substitute("$0[2,2]{1,0}", out_type_str))));
    }
  }
}

TEST_F(DotOperandConverterTest,
       DemotesF32ReducedToBf16PrecisionAndLooksThroughDataMovement) {
  // exponent_bits=8, mantissa_bits=7 is exactly the BF16 format (8 exponent
  // bits, 7 explicit mantissa bits), so the reduce-precision output is
  // BF16-exact. HIGH precision still allows TF32 inputs.
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[3,2]{1,0} parameter(0)
    p1 = s8[2,3]{1,0} parameter(1)
    rp = f32[3,2]{1,0} reduce-precision(p0), exponent_bits=8, mantissa_bits=7
    t0 = f32[2,3]{1,0} transpose(rp), dimensions={1,0}
    c1 = f32[2,3]{1,0} convert(p1)
    t1 = f32[3,2]{1,0} transpose(c1), dimensions={1,0}
    ROOT dot = f32[2,2]{1,0} dot(t0, t1), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0},
                                          operand_precision={HIGH, DEFAULT}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotOperandConverter().Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Dot(AllOf(op::Transpose(AllOf(op::Convert(op::ReducePrecision()),
                                        op::Shape("bf16[3,2]{1,0}"))),
                    op::Shape("bf16[2,3]{1,0}")),
              AllOf(op::Transpose(AllOf(op::Convert(op::Parameter(1)),
                                        op::Shape("bf16[2,3]{1,0}"))),
                    op::Shape("bf16[3,2]{1,0}"))));
}

TEST_F(DotOperandConverterTest, DoesNotDemoteDotWithNonBf16ExactOperand) {
  // Each `rhs` can produce values that BF16 cannot represent exactly:
  // - an arbitrary F32 value (negate does not reduce precision);
  // - s16 integers up to 2^15, which need up to 15 significand bits (> 8);
  // - f16 values, which have 10 explicit mantissa bits (> 7);
  // - reduce-precision to 8 or 10 explicit mantissa bits (> 7); 8 is the
  //   off-by-one boundary of the 7-bit BF16 mantissa;
  // - a bitcast that reinterprets s8 bytes as F32 bit patterns: the s8 values
  //   are BF16-exact, but the reinterpreted F32 values are not.
  for (absl::string_view rhs :
       {"p1 = f32[3,2]{1,0} parameter(1)\n c1 = f32[3,2]{1,0} negate(p1)",
        "p1 = s16[3,2]{1,0} parameter(1)\n c1 = f32[3,2]{1,0} convert(p1)",
        "p1 = f16[3,2]{1,0} parameter(1)\n c1 = f32[3,2]{1,0} convert(p1)",
        "p1 = f32[3,2]{1,0} parameter(1)\n"
        " c1 = f32[3,2]{1,0} reduce-precision(p1), exponent_bits=8, "
        "mantissa_bits=8",
        "p1 = f32[3,2]{1,0} parameter(1)\n"
        " c1 = f32[3,2]{1,0} reduce-precision(p1), exponent_bits=8, "
        "mantissa_bits=10",
        "p1 = s8[3,2,4]{2,1,0} parameter(1)\n"
        " c1 = f32[3,2]{1,0} bitcast(p1)"}) {
    SCOPED_TRACE(rhs);
    absl::string_view module_tmpl = R"(
    HloModule module

    ENTRY main {
      p0 = s8[2,3]{1,0} parameter(0)
      c0 = f32[2,3]{1,0} convert(p0)
      $0
      ROOT dot = f32[2,2]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                            rhs_contracting_dims={0}
    })";
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseAndReturnVerifiedModule(absl::Substitute(module_tmpl, rhs)));
    ASSERT_OK_AND_ASSIGN(bool changed, DotOperandConverter().Run(module.get()));
    EXPECT_FALSE(changed);
  }
}

TEST_F(DotOperandConverterTest, DoesNotDemoteDotWithoutTf32Precision) {
  // HIGHEST precision requests IEEE F32 inputs (no TF32), and an explicit
  // algorithm pins the operand types, so neither dot is demoted.
  for (absl::string_view precision :
       {"operand_precision={HIGHEST, DEFAULT}",
        "operand_precision={DEFAULT, HIGHEST}", "algorithm=dot_f32_f32_f32",
        "algorithm=dot_tf32_tf32_f32"}) {
    SCOPED_TRACE(precision);
    absl::string_view module_tmpl = R"(
    HloModule module

    ENTRY main {
      p0 = s8[2,3]{1,0} parameter(0)
      p1 = bf16[3,2]{1,0} parameter(1)
      c0 = f32[2,3]{1,0} convert(p0)
      c1 = f32[3,2]{1,0} convert(p1)
      ROOT dot = f32[2,2]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                            rhs_contracting_dims={0}, $0
    })";
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseAndReturnVerifiedModule(absl::Substitute(module_tmpl, precision)));
    ASSERT_OK_AND_ASSIGN(bool changed, DotOperandConverter().Run(module.get()));
    EXPECT_FALSE(changed);
  }
}

TEST_F(DotOperandConverterTest,
       DoesNotDemoteWhenTensorFloat32DisabledOrBf16DotUnsupported) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[2,3]{1,0} parameter(0)
    p1 = bf16[3,2]{1,0} parameter(1)
    c0 = f32[2,3]{1,0} convert(p0)
    c1 = f32[3,2]{1,0} convert(p1)
    ROOT dot = f32[2,2]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
  })";
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(module_string));
    ASSERT_OK_AND_ASSIGN(bool changed,
                         DotOperandConverter(se::CudaComputeCapability::Volta())
                             .Run(module.get()));
    EXPECT_FALSE(changed);
  }
  bool orig_tf32 = tsl::tensor_float_32_execution_enabled();
  tsl::enable_tensor_float_32_execution(false);
  absl::Cleanup restore_tf32 = [orig_tf32] {
    tsl::enable_tensor_float_32_execution(orig_tf32);
  };
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool changed, DotOperandConverter().Run(module.get()));
  EXPECT_FALSE(changed);
}

constexpr absl::string_view kUpcastS8Bf16DotTemplate = R"(
  HloModule module

  ENTRY main {
    p0 = s8[512,4,1024]{2,1,0} parameter(0)
    p1 = bf16[32,4,1024]{2,1,0} parameter(1)
    c0 = f32[512,4,1024]{2,1,0} convert(p0)
    c1 = f32[32,4,1024]{2,1,0} convert(p1)
    ROOT dot = f32[4,512,32]{2,1,0} dot(c0, c1), lhs_batch_dims={1},
        lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={2}$0
  })";

TEST_F(DotOperandConverterTest, CompilerDemotesUpcastS8AndBf16DotToBf16) {
  if (!DotOperandConverter::SupportsBf16Dot(
          device_description().gpu_compute_capability())) {
    GTEST_SKIP() << "Requires BF16 dot support.";
  }
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      GetOptimizedModule(absl::Substitute(kUpcastS8Bf16DotTemplate, "")));
  // Split-K may reshape the contracting dimension, so only match the leading
  // non-contracting and batch dimensions of each operand.
  const std::string optimized = module->ToString();
  EXPECT_THAT(optimized, HasSubstr("bf16[512,4,"));
  EXPECT_THAT(optimized, Not(HasSubstr("f32[512,4,")));
  EXPECT_THAT(optimized, Not(HasSubstr("f32[32,4,")));
}

TEST_F(DotOperandConverterTest,
       CompilerDemotesUpcastThroughDataMovementToBf16) {
  if (!DotOperandConverter::SupportsBf16Dot(
          device_description().gpu_compute_capability())) {
    GTEST_SKIP() << "Requires BF16 dot support.";
  }
  // The demotion recreates the transpose and broadcast in BF16 on top of
  // `convert<bf16>(s8)` and the BF16 parameter, so no F32 copy of either
  // operand survives the pipeline.
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[4,1024,512]{2,1,0} parameter(0)
    p1 = bf16[32,1024]{1,0} parameter(1)
    c0 = f32[4,1024,512]{2,1,0} convert(p0)
    t0 = f32[512,4,1024]{2,1,0} transpose(c0), dimensions={2,0,1}
    c1 = f32[32,1024]{1,0} convert(p1)
    b1 = f32[32,4,1024]{2,1,0} broadcast(c1), dimensions={0,2}
    ROOT dot = f32[4,512,32]{2,1,0} dot(t0, b1), lhs_batch_dims={1},
        lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={2}
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       GetOptimizedModule(module_string));
  const std::string optimized = module->ToString();
  EXPECT_THAT(optimized, Not(HasSubstr("f32[4,1024,512]")));
  EXPECT_THAT(optimized, Not(HasSubstr("f32[512,4,")));
  EXPECT_THAT(optimized, Not(HasSubstr("f32[32,1024]")));
  EXPECT_THAT(optimized, Not(HasSubstr("f32[32,4,")));
}

TEST_F(DotOperandConverterTest, DemotedDotMatchesIeeeF32Dot) {
  if (!DotOperandConverter::SupportsBf16Dot(
          device_description().gpu_compute_capability())) {
    GTEST_SKIP() << "Requires BF16 dot support.";
  }
  // The HIGHEST-precision dot is not demoted and runs as an IEEE F32 dot. Both
  // dots form the same exact products (8-bit x 8-bit significands fit in F32's
  // 24 bits) and differ only in the order and rounding of the F32 accumulation
  // over the 1024-element contracting dimension, hence the small tolerance
  // (absolute 1e-3, relative 1e-5) instead of a BF16-level one.
  EXPECT_TRUE(RunAndCompareTwoModules(
      absl::Substitute(kUpcastS8Bf16DotTemplate, ""),
      absl::Substitute(kUpcastS8Bf16DotTemplate,
                       ", operand_precision={HIGHEST, HIGHEST}"),
      ErrorSpec{1e-3, 1e-5}));
}

}  // namespace
}  // namespace xla::gpu

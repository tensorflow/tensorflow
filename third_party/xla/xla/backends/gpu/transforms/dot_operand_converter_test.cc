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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
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
  EXPECT_THAT(module->ToString(), ::testing::HasSubstr("bf16[128,128]"));
  EXPECT_THAT(module->ToString(),
              ::testing::Not(::testing::HasSubstr("f32[128,128]")));
}

}  // namespace
}  // namespace xla::gpu

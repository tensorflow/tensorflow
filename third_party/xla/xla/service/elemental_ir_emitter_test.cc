/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/elemental_ir_emitter.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using std::nullopt;

class ElementalIrEmitterExecutionTest : public HloTestBase {
 protected:
  void RunTest(const std::string& hlo_text, absl::Span<Literal* const> args) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text, config));
    EXPECT_TRUE(RunAndCompareNoHloPasses(std::move(module), args, nullopt));
  }

  void RunTypeConversionTest(absl::string_view hlo_text) {
    HloModuleConfig config;
    auto debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_cpu_fast_math_honor_nans(true);
    debug_options.set_xla_cpu_fast_math_honor_infs(true);
    config.set_debug_options(debug_options);
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo_text, config));
    EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{(0.)}));
  }
};

class ElementalIrEmitterExecutionTestWithoutFastMinMax
    : public ElementalIrEmitterExecutionTest {
 protected:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options =
        ElementalIrEmitterExecutionTest::GetDebugOptionsForTest();
    debug_options.set_xla_cpu_enable_fast_min_max(false);
    debug_options.set_xla_gpu_enable_fast_min_max(false);
    return debug_options;
  }
};

XLA_TEST_F(ElementalIrEmitterExecutionTest, DotFusion) {
  const std::string hlo_text = R"(
HloModule FusedDot

fused_computation {
  arg0 = s32[1,2,1]{2,1,0} parameter(0)
  reshape.lhs = s32[2,1]{1,0} reshape(arg0)
  arg1 = s32[1,2,1]{2,1,0} parameter(1)
  reshape.rhs = s32[2,1]{1,0} reshape(arg1)
  ROOT dot = s32[1,1]{1,0} dot(reshape.lhs, reshape.rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY main {
  entry_arg0 = s32[1,2,1]{2,1,0} parameter(0)
  entry_arg1 = s32[1,2,1]{2,1,0} parameter(1)
  ROOT fusion = s32[1,1]{1,0} fusion(entry_arg0, entry_arg1), kind=kLoop, calls=fused_computation
}
)";

  Literal lhs = LiteralUtil::CreateR3<int32_t>({{{1}, {2}}});
  Literal rhs = LiteralUtil::CreateR3<int32_t>({{{3}, {4}}});
  RunTest(hlo_text, {&lhs, &rhs});
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ScalarDotFusion) {
  const char* hlo_text = R"(
HloModule ScalarDotFusion

fused_computation {
  arg0 = s32[2,2]{1,0} parameter(0)
  reshape.lhs = s32[4]{0} reshape(arg0)
  arg1 = s32[2,2]{1,0} parameter(1)
  reshape.rhs = s32[4]{0} reshape(arg1)
  ROOT dot = s32[] dot(reshape.lhs, reshape.rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
}

ENTRY main {
  entry_arg0 = s32[2,2]{1,0} parameter(0)
  entry_arg1 = s32[2,2]{1,0} parameter(1)
  ROOT fusion = s32[] fusion(entry_arg0, entry_arg1), kind=kLoop, calls=fused_computation
}
)";

  Literal lhs = LiteralUtil::CreateR2<int32_t>({{1, 2}, {3, 4}});
  Literal rhs = LiteralUtil::CreateR2<int32_t>({{10, 20}, {30, 40}});
  RunTest(hlo_text, {&lhs, &rhs});
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, BatchDot) {
  const char* hlo_text = R"(
HloModule BatchDot

fused_computation.1 {
  param_0 = f64[1,1,8]{2,1,0} parameter(0)
  r.1 = f64[2,4]{1,0} reshape(param_0)
  param_1 = f64[1,2,2,2,1]{4,3,2,1,0} parameter(1)
  r.2 = f64[2,4,1]{2,1,0} reshape(param_1)
  ROOT dot = f64[2,1]{1,0} dot(r.1, r.2), lhs_batch_dims={0},
                                          lhs_contracting_dims={1},
                                          rhs_batch_dims={0},
                                          rhs_contracting_dims={1}
}

ENTRY resampler_Resampler.49 {
  p0 = f64[1,1,8]{2,1,0} parameter(0)
  p1 = f64[1,2,2,2,1]{4,3,2,1,0} parameter(1)
  ROOT f = f64[2,1]{1,0} fusion(p0, p1), kind=kLoop, calls=fused_computation.1
}
)";

  HloModuleConfig config;
  auto debug_options = GetDebugOptionsForTest();
  // Disable the layout assignment pass because it would throw away the layouts
  // in the fusion computation, but not recreate them.
  debug_options.add_xla_disable_hlo_passes("layout-assignment");
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{4e-3, 4e-3}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTest,
           DivideComplexNumbersWithInfiniteNormRhs) {
  constexpr char hlo_text[] = R"(
    HloModule DivideComplexNumbers
    ENTRY DivideComplexNumbers {
      constant.1 = c64[8]{0} constant({
        (1, 1),     (1, inf),   (1, inf),   (nan, 1),
        (inf, inf), (inf, nan), (nan, nan), (1, 2)})
      real = f32[8]{0} constant({nan, nan, inf, inf, inf, 1, inf, 3})
      imag = f32[8]{0} constant({inf, inf, inf, inf, 1, inf, inf, 4})
      complex.2 = c64[8]{0} complex(real, imag)
      ROOT divide.1 = c64[8]{0} divide(constant.1, complex.2)
    }
  )";
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_cpu_fast_math_honor_nans(true);
  debug_options.set_xla_cpu_fast_math_honor_infs(true);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{(0.)}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTest,
           DivideComplexNumbersWithFiniteNormRhs) {
  constexpr char hlo_text[] = R"(
    HloModule DivideComplexNumbers
    ENTRY DivideComplexNumbers {
      constant.1 = c64[5]{0} constant({
        (1, inf), (inf, 1), (inf, nan), (inf, inf), (nan, inf)})
      real = f32[5]{0} constant({1, 1, 1, 1, 1})
      imag = f32[5]{0} constant({1, 1, 1, 1, 1})
      complex.2 = c64[5]{0} complex(real, imag)
      ROOT divide.1 = c64[5]{0} divide(constant.1, complex.2)
    }
  )";
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_cpu_fast_math_honor_nans(true);
  debug_options.set_xla_cpu_fast_math_honor_infs(true);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{(0.)}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTest,
           DivideComplexNumbersWithZeroNormRhs) {
  constexpr char hlo_text[] = R"(
    HloModule DivideComplexNumbers
    ENTRY DivideComplexNumbers {
      constant.1 = c64[9]{0} constant({
        (1, 1),     (1, nan), (1, inf),   (inf, inf), (inf, 1),
        (inf, nan), (nan, 1), (nan, inf), (nan, nan)})
      real = f32[9]{0} constant({0, 0, 0, 0, 0, 0, 0, 0, 0})
      imag = f32[9]{0} constant({0, 0, 0, 0, 0, 0, 0, 0, 0})
      complex.2 = c64[9]{0} complex(real, imag)
      ROOT divide.1 = c64[9]{0} divide(constant.1, complex.2)
    }
  )";
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_cpu_fast_math_honor_nans(true);
  debug_options.set_xla_cpu_fast_math_honor_infs(true);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{(0.)}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertFloatsToBF16) {
  RunTypeConversionTest(R"(
    HloModule convertToBF16
    ENTRY ConvertToBF16
        (f16_ f16[], f32_ f32[], f64_ f64[]) -> (bf16[], bf16[], bf16[]) {
      f16_ = f16[] parameter(0)
      f32_ = f32[] parameter(1)
      f64_ = f64[] parameter(2)
      converted_f16 = bf16[] convert(f16[] f16_)
      converted_f32 = bf16[] convert(f32[] f32_)
      converted_f64 = bf16[] convert(f64[] f64_)
      ROOT tuple = (bf16[], bf16[], bf16[]) tuple(converted_f16, converted_f32,
                                                  converted_f64)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertSignedToBF16) {
  RunTypeConversionTest(R"(
    HloModule convertToBF16
    ENTRY ConvertToBF16 (s8_ s8[], s16_ s16[], s32_ s32[], s64_ s64[]) ->
        (bf16[], bf16[], bf16[], bf16[]) {
      s8_ = s8[] parameter(0)
      s16_ = s16[] parameter(1)
      s32_ = s32[] parameter(2)
      s64_ = s64[] parameter(3)
      converted_s8 = bf16[] convert(s8[] s8_)
      converted_s16 = bf16[] convert(s16[] s16_)
      converted_s32 = bf16[] convert(s32[] s32_)
      converted_s64 = bf16[] convert(s64[] s64_)
      ROOT tuple = (bf16[], bf16[], bf16[], bf16[]) tuple(
          converted_s8, converted_s16, converted_s32, converted_s64)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertUnsignedToBF16) {
  RunTypeConversionTest(R"(
    HloModule convertToBF16
    ENTRY ConvertToBF16 (u8_ u8[], u16_ u16[], u32_ u32[], u64_ u64[]) ->
        (bf16[], bf16[], bf16[], bf16[]) {
      u8_ = u8[] parameter(0)
      u16_ = u16[] parameter(1)
      u32_ = u32[] parameter(2)
      u64_ = u64[] parameter(3)
      converted_u8 = bf16[] convert(u8[] u8_)
      converted_u16 = bf16[] convert(u16[] u16_)
      converted_u32 = bf16[] convert(u32[] u32_)
      converted_u64 = bf16[] convert(u64[] u64_)
      ROOT tuple = (bf16[], bf16[], bf16[], bf16[]) tuple(
          converted_u8, converted_u16, converted_u32, converted_u64)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertBF16ToFloat) {
  RunTypeConversionTest(R"(
    HloModule convertFromBF16
    ENTRY ConvertFromBF16
        (to_f16 bf16[], to_f32 bf16[], to_f64 bf16[]) -> (f16[], f32[], f64[]) {
      to_f16 = bf16[] parameter(0)
      to_f32 = bf16[] parameter(1)
      to_f64 = bf16[] parameter(2)
      f16_ = f16[] convert(bf16[] to_f16)
      f32_ = f32[] convert(bf16[] to_f32)
      f64_ = f64[] convert(bf16[] to_f64)
      ROOT tuple = (f16[], f32[], f64[]) tuple(f16_, f32_, f64_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertBF16ToSigned) {
  RunTypeConversionTest(R"(
    HloModule convertFromBF16
    ENTRY ConvertFromBF16(to_s8 bf16[], to_s16 bf16[], to_s32 bf16[],
                          to_s64 bf16[]) -> (s8[], s16[], s32[], s64[]) {
      to_s8 = bf16[] parameter(0)
      to_s16 = bf16[] parameter(1)
      to_s32 = bf16[] parameter(2)
      to_s64 = bf16[] parameter(3)
      s8_ = s8[] convert(bf16[] to_s8)
      s16_ = s16[] convert(bf16[] to_s16)
      s32_ = s32[] convert(bf16[] to_s32)
      s64_ = s64[] convert(bf16[] to_s64)
      ROOT tuple = (s8[], s16[], s32[], s64[]) tuple(s8_, s16_, s32_, s64_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertBF16ToUnsigned) {
  RunTypeConversionTest(R"(
    HloModule convertFromBF16
    ENTRY ConvertFromBF16(to_u8 bf16[], to_u16 bf16[], to_u32 bf16[],
                          to_u64 bf16[]) -> (u8[], u16[], u32[], u64[]) {
      to_u8 = bf16[] parameter(0)
      to_u16 = bf16[] parameter(1)
      to_u32 = bf16[] parameter(2)
      to_u64 = bf16[] parameter(3)
      u8_ = u8[] convert(bf16[] to_u8)
      u16_ = u16[] convert(bf16[] to_u16)
      u32_ = u32[] convert(bf16[] to_u32)
      u64_ = u64[] convert(bf16[] to_u64)
      ROOT tuple = (u8[], u16[], u32[], u64[]) tuple(u8_, u16_, u32_, u64_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertBF16ToComplex) {
  RunTypeConversionTest(R"(
    HloModule convertFromBF16
    ENTRY ConvertFromBF16
        (to_c64 bf16[], to_c128 bf16[]) -> (c64[], c128[]) {
      to_c64 = bf16[] parameter(0)
      to_c128 = bf16[] parameter(1)
      c64_ = c64[] convert(bf16[] to_c64)
      c128_ = c128[] convert(bf16[] to_c128)
      ROOT tuple = (c64[], c128[]) tuple(c64_, c128_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, CompareBF16) {
  constexpr char hlo_text[] = R"(
  HloModule compareBF16
  ENTRY main {
    p0 = bf16[4] parameter(0)
    p1 = bf16[4] parameter(1)
    ROOT cmp = pred[4] compare(p0, p1), direction=LT
})";

  Literal lhs = LiteralUtil::CreateR1<float>({1, 2, 3, 4});
  Literal rhs = LiteralUtil::CreateR1<float>({4, 3, 2, 1});
  lhs = LiteralUtil::ConvertF32ToBF16(lhs);
  rhs = LiteralUtil::ConvertF32ToBF16(rhs);
  RunTest(hlo_text, {&lhs, &rhs});
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, IotaBF16) {
  constexpr char hlo_text[] = R"(
  HloModule IotaBF16
  ENTRY main {
    ROOT iota_ = bf16[4] iota(), iota_dimension=0
  }
  )";

  RunTest(hlo_text, {});
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, BatchDotBF16) {
  const char* const hlo_text = R"(
  HloModule matmul

  ENTRY main {
    x = bf16[8,16] parameter(0)
    y = bf16[8,16,32] parameter(1)
    ROOT dot = bf16[8,32] dot(x, y), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }
  )";
  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{1e-5, 1e-5}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertFloatsToF8E4FNUZ) {
  RunTypeConversionTest(R"(
    HloModule convertToF8E4FNUZ
    ENTRY ConvertToF8E4FNUZ
        (f16_ f16[], f32_ f32[], f64_ f64[], bf16_ bf16[]) -> (f8e4m3fnuz[], f8e4m3fnuz[], f8e4m3fnuz[], f8e4m3fnuz[]) {
      f16_ = f16[] parameter(0)
      f32_ = f32[] parameter(1)
      f64_ = f64[] parameter(2)
      bf16_ = bf16[] parameter(3)
      converted_f16 = f8e4m3fnuz[] convert(f16[] f16_)
      converted_f32 = f8e4m3fnuz[] convert(f32[] f32_)
      converted_f64 = f8e4m3fnuz[] convert(f64[] f64_)
      converted_bf16 = f8e4m3fnuz[] convert(bf16[] bf16_)
      ROOT tuple = (f8e4m3fnuz[], f8e4m3fnuz[], f8e4m3fnuz[], f8e4m3fnuz[]) tuple(
          converted_f16, converted_f32, converted_f64, converted_bf16)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertSignedToF8E4FNUZ) {
  RunTypeConversionTest(R"(
    HloModule convertToF8E4FNUZ
    ENTRY ConvertToF8E4FNUZ (s8_ s8[], s16_ s16[], s32_ s32[], s64_ s64[]) ->
        (f8e4m3fnuz[], f8e4m3fnuz[], f8e4m3fnuz[], f8e4m3fnuz[]) {
      s8_ = s8[] parameter(0)
      s16_ = s16[] parameter(1)
      s32_ = s32[] parameter(2)
      s64_ = s64[] parameter(3)
      converted_s8 = f8e4m3fnuz[] convert(s8[] s8_)
      converted_s16 = f8e4m3fnuz[] convert(s16[] s16_)
      converted_s32 = f8e4m3fnuz[] convert(s32[] s32_)
      converted_s64 = f8e4m3fnuz[] convert(s64[] s64_)
      ROOT tuple = (f8e4m3fnuz[], f8e4m3fnuz[], f8e4m3fnuz[], f8e4m3fnuz[]) tuple(
          converted_s8, converted_s16, converted_s32, converted_s64)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertUnsignedToF8E4FNUZ) {
  RunTypeConversionTest(R"(
    HloModule convertToF8E4FNUZ
    ENTRY ConvertToF8E4FNUZ (u8_ u8[], u16_ u16[], u32_ u32[], u64_ u64[]) ->
        (f8e4m3fnuz[], f8e4m3fnuz[], f8e4m3fnuz[], f8e4m3fnuz[]) {
      u8_ = u8[] parameter(0)
      u16_ = u16[] parameter(1)
      u32_ = u32[] parameter(2)
      u64_ = u64[] parameter(3)
      converted_u8 = f8e4m3fnuz[] convert(u8[] u8_)
      converted_u16 = f8e4m3fnuz[] convert(u16[] u16_)
      converted_u32 = f8e4m3fnuz[] convert(u32[] u32_)
      converted_u64 = f8e4m3fnuz[] convert(u64[] u64_)
      ROOT tuple = (f8e4m3fnuz[], f8e4m3fnuz[], f8e4m3fnuz[], f8e4m3fnuz[]) tuple(
          converted_u8, converted_u16, converted_u32, converted_u64)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertF8E4FNUZToFloat) {
  RunTypeConversionTest(R"(
    HloModule convertFromF8E4FNUZ
    ENTRY ConvertFromF8E4FNUZ
        (to_f16 f8e4m3fnuz[], to_f32 f8e4m3fnuz[], to_f64 f8e4m3fnuz[], to_bf16 f8e4m3fnuz[]) -> (f16[], f32[], f64[], bf16[]) {
      to_f16 = f8e4m3fnuz[] parameter(0)
      to_f32 = f8e4m3fnuz[] parameter(1)
      to_f64 = f8e4m3fnuz[] parameter(2)
      to_bf16 = f8e4m3fnuz[] parameter(3)
      f16_ = f16[] convert(f8e4m3fnuz[] to_f16)
      f32_ = f32[] convert(f8e4m3fnuz[] to_f32)
      f64_ = f64[] convert(f8e4m3fnuz[] to_f64)
      bf16_ = bf16[] convert(f8e4m3fnuz[] to_f64)
      ROOT tuple = (f16[], f32[], f64[], bf16[]) tuple(f16_, f32_, f64_, bf16_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertF8E4FNUZToSigned) {
  RunTypeConversionTest(R"(
    HloModule convertFromF8E4FNUZ
    ENTRY ConvertFromF8E4FNUZ(to_s8 f8e4m3fnuz[], to_s16 f8e4m3fnuz[], to_s32 f8e4m3fnuz[],
                          to_s64 f8e4m3fnuz[]) -> (s8[], s16[], s32[], s64[]) {
      to_s8 = f8e4m3fnuz[] parameter(0)
      to_s16 = f8e4m3fnuz[] parameter(1)
      to_s32 = f8e4m3fnuz[] parameter(2)
      to_s64 = f8e4m3fnuz[] parameter(3)
      s8_ = s8[] convert(f8e4m3fnuz[] to_s8)
      s16_ = s16[] convert(f8e4m3fnuz[] to_s16)
      s32_ = s32[] convert(f8e4m3fnuz[] to_s32)
      s64_ = s64[] convert(f8e4m3fnuz[] to_s64)
      ROOT tuple = (s8[], s16[], s32[], s64[]) tuple(s8_, s16_, s32_, s64_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertF8E4FNUZToUnsigned) {
  RunTypeConversionTest(R"(
    HloModule convertFromF8E4FNUZ
    ENTRY ConvertFromF8E4FNUZ(to_u8 f8e4m3fnuz[], to_u16 f8e4m3fnuz[], to_u32 f8e4m3fnuz[],
                          to_u64 f8e4m3fnuz[]) -> (u8[], u16[], u32[], u64[]) {
      to_u8 = f8e4m3fnuz[] parameter(0)
      to_u16 = f8e4m3fnuz[] parameter(1)
      to_u32 = f8e4m3fnuz[] parameter(2)
      to_u64 = f8e4m3fnuz[] parameter(3)
      u8_ = u8[] convert(f8e4m3fnuz[] to_u8)
      u16_ = u16[] convert(f8e4m3fnuz[] to_u16)
      u32_ = u32[] convert(f8e4m3fnuz[] to_u32)
      u64_ = u64[] convert(f8e4m3fnuz[] to_u64)
      ROOT tuple = (u8[], u16[], u32[], u64[]) tuple(u8_, u16_, u32_, u64_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertF8E4FNUZToComplex) {
  RunTypeConversionTest(R"(
    HloModule convertFromF8E4FNUZ
    ENTRY ConvertFromF8E4FNUZ
        (to_c64 f8e4m3fnuz[], to_c128 f8e4m3fnuz[]) -> (c64[], c128[]) {
      to_c64 = f8e4m3fnuz[] parameter(0)
      to_c128 = f8e4m3fnuz[] parameter(1)
      c64_ = c64[] convert(f8e4m3fnuz[] to_c64)
      c128_ = c128[] convert(f8e4m3fnuz[] to_c128)
      ROOT tuple = (c64[], c128[]) tuple(c64_, c128_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, CompareF8E4FNUZ) {
  constexpr char hlo_text[] = R"(
  HloModule compareF8E4FNUZ
  ENTRY main {
    p0 = f8e4m3fnuz[4] parameter(0)
    p1 = f8e4m3fnuz[4] parameter(1)
    ROOT cmp = pred[4] compare(p0, p1), direction=LT
})";

  Literal lhs = LiteralUtil::CreateR1<float>({1, 2, 3, 4});
  Literal rhs = LiteralUtil::CreateR1<float>({4, 3, 2, 1});
  lhs = LiteralUtil::ConvertF32ToF8E4M3FNUZ(lhs);
  rhs = LiteralUtil::ConvertF32ToF8E4M3FNUZ(rhs);
  RunTest(hlo_text, {&lhs, &rhs});
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, IotaF8E4FNUZ) {
  constexpr char hlo_text[] = R"(
  HloModule IotaF8E4FNUZ
  ENTRY main {
    ROOT iota_ = f8e4m3fnuz[4] iota(), iota_dimension=0
  }
  )";

  RunTest(hlo_text, {});
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertFloatsToF8E5FNUZ) {
  RunTypeConversionTest(R"(
    HloModule convertToF8E5FNUZ
    ENTRY ConvertToF8E5FNUZ
        (f16_ f16[], f32_ f32[], f64_ f64[], bf16_ bf16[]) -> (f8e5m2fnuz[], f8e5m2fnuz[], f8e5m2fnuz[], f8e5m2fnuz[]) {
      f16_ = f16[] parameter(0)
      f32_ = f32[] parameter(1)
      f64_ = f64[] parameter(2)
      bf16_ = bf16[] parameter(3)
      converted_f16 = f8e5m2fnuz[] convert(f16[] f16_)
      converted_f32 = f8e5m2fnuz[] convert(f32[] f32_)
      converted_f64 = f8e5m2fnuz[] convert(f64[] f64_)
      converted_bf16 = f8e5m2fnuz[] convert(bf16[] bf16_)
      ROOT tuple = (f8e5m2fnuz[], f8e5m2fnuz[], f8e5m2fnuz[], f8e5m2fnuz[]) tuple(
          converted_f16, converted_f32, converted_f64, converted_bf16)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertSignedToF8E5FNUZ) {
  RunTypeConversionTest(R"(
    HloModule convertToF8E5FNUZ
    ENTRY ConvertToF8E5FNUZ (s8_ s8[], s16_ s16[], s32_ s32[], s64_ s64[]) ->
        (f8e5m2fnuz[], f8e5m2fnuz[], f8e5m2fnuz[], f8e5m2fnuz[]) {
      s8_ = s8[] parameter(0)
      s16_ = s16[] parameter(1)
      s32_ = s32[] parameter(2)
      s64_ = s64[] parameter(3)
      converted_s8 = f8e5m2fnuz[] convert(s8[] s8_)
      converted_s16 = f8e5m2fnuz[] convert(s16[] s16_)
      converted_s32 = f8e5m2fnuz[] convert(s32[] s32_)
      converted_s64 = f8e5m2fnuz[] convert(s64[] s64_)
      ROOT tuple = (f8e5m2fnuz[], f8e5m2fnuz[], f8e5m2fnuz[], f8e5m2fnuz[]) tuple(
          converted_s8, converted_s16, converted_s32, converted_s64)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertUnsignedToF8E5FNUZ) {
  RunTypeConversionTest(R"(
    HloModule convertToF8E5FNUZ
    ENTRY ConvertToF8E5FNUZ (u8_ u8[], u16_ u16[], u32_ u32[], u64_ u64[]) ->
        (f8e5m2fnuz[], f8e5m2fnuz[], f8e5m2fnuz[], f8e5m2fnuz[]) {
      u8_ = u8[] parameter(0)
      u16_ = u16[] parameter(1)
      u32_ = u32[] parameter(2)
      u64_ = u64[] parameter(3)
      converted_u8 = f8e5m2fnuz[] convert(u8[] u8_)
      converted_u16 = f8e5m2fnuz[] convert(u16[] u16_)
      converted_u32 = f8e5m2fnuz[] convert(u32[] u32_)
      converted_u64 = f8e5m2fnuz[] convert(u64[] u64_)
      ROOT tuple = (f8e5m2fnuz[], f8e5m2fnuz[], f8e5m2fnuz[], f8e5m2fnuz[]) tuple(
          converted_u8, converted_u16, converted_u32, converted_u64)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertF8E5FNUZToFloat) {
  RunTypeConversionTest(R"(
    HloModule convertFromF8E5FNUZ
    ENTRY ConvertFromF8E5FNUZ
        (to_f16 f8e5m2fnuz[], to_f32 f8e5m2fnuz[], to_f64 f8e5m2fnuz[]) -> (f16[], f32[], f64[]) {
      to_f16 = f8e5m2fnuz[] parameter(0)
      to_f32 = f8e5m2fnuz[] parameter(1)
      to_f64 = f8e5m2fnuz[] parameter(2)
      f16_ = f16[] convert(f8e5m2fnuz[] to_f16)
      f32_ = f32[] convert(f8e5m2fnuz[] to_f32)
      f64_ = f64[] convert(f8e5m2fnuz[] to_f64)
      ROOT tuple = (f16[], f32[], f64[]) tuple(f16_, f32_, f64_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertF8E5FNUZToSigned) {
  RunTypeConversionTest(R"(
    HloModule convertFromF8E5FNUZ
    ENTRY ConvertFromF8E5FNUZ(to_s8 f8e5m2fnuz[], to_s16 f8e5m2fnuz[], to_s32 f8e5m2fnuz[],
                          to_s64 f8e5m2fnuz[]) -> (s8[], s16[], s32[], s64[]) {
      to_s8 = f8e5m2fnuz[] parameter(0)
      to_s16 = f8e5m2fnuz[] parameter(1)
      to_s32 = f8e5m2fnuz[] parameter(2)
      to_s64 = f8e5m2fnuz[] parameter(3)
      s8_ = s8[] convert(f8e5m2fnuz[] to_s8)
      s16_ = s16[] convert(f8e5m2fnuz[] to_s16)
      s32_ = s32[] convert(f8e5m2fnuz[] to_s32)
      s64_ = s64[] convert(f8e5m2fnuz[] to_s64)
      ROOT tuple = (s8[], s16[], s32[], s64[]) tuple(s8_, s16_, s32_, s64_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertF8E5FNUZToUnsigned) {
  RunTypeConversionTest(R"(
    HloModule convertFromF8E5FNUZ
    ENTRY ConvertFromF8E5FNUZ(to_u8 f8e5m2fnuz[], to_u16 f8e5m2fnuz[], to_u32 f8e5m2fnuz[],
                          to_u64 f8e5m2fnuz[]) -> (u8[], u16[], u32[], u64[]) {
      to_u8 = f8e5m2fnuz[] parameter(0)
      to_u16 = f8e5m2fnuz[] parameter(1)
      to_u32 = f8e5m2fnuz[] parameter(2)
      to_u64 = f8e5m2fnuz[] parameter(3)
      u8_ = u8[] convert(f8e5m2fnuz[] to_u8)
      u16_ = u16[] convert(f8e5m2fnuz[] to_u16)
      u32_ = u32[] convert(f8e5m2fnuz[] to_u32)
      u64_ = u64[] convert(f8e5m2fnuz[] to_u64)
      ROOT tuple = (u8[], u16[], u32[], u64[]) tuple(u8_, u16_, u32_, u64_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, ConvertF8E5FNUZToComplex) {
  RunTypeConversionTest(R"(
    HloModule convertFromF8E5FNUZ
    ENTRY ConvertFromF8E5FNUZ
        (to_c64 f8e5m2fnuz[], to_c128 f8e5m2fnuz[]) -> (c64[], c128[]) {
      to_c64 = f8e5m2fnuz[] parameter(0)
      to_c128 = f8e5m2fnuz[] parameter(1)
      c64_ = c64[] convert(f8e5m2fnuz[] to_c64)
      c128_ = c128[] convert(f8e5m2fnuz[] to_c128)
      ROOT tuple = (c64[], c128[]) tuple(c64_, c128_)
    }
  )");
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, CompareF8E5FNUZ) {
  constexpr char hlo_text[] = R"(
  HloModule compareF8E5FNUZ
  ENTRY main {
    p0 = f8e5m2fnuz[4] parameter(0)
    p1 = f8e5m2fnuz[4] parameter(1)
    ROOT cmp = pred[4] compare(p0, p1), direction=LT
})";

  Literal lhs = LiteralUtil::CreateR1<float>({1, 2, 3, 4});
  Literal rhs = LiteralUtil::CreateR1<float>({4, 3, 2, 1});
  lhs = LiteralUtil::ConvertF32ToF8E5M2FNUZ(lhs);
  rhs = LiteralUtil::ConvertF32ToF8E5M2FNUZ(rhs);
  RunTest(hlo_text, {&lhs, &rhs});
}

XLA_TEST_F(ElementalIrEmitterExecutionTest, IotaF8E5FNUZ) {
  constexpr char hlo_text[] = R"(
  HloModule IotaF8E5FNUZ
  ENTRY main {
    ROOT iota_ = f8e5m2fnuz[4] iota(), iota_dimension=0
  }
  )";

  RunTest(hlo_text, {});
}

XLA_TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax,
           MinimumHandlesNaNsOnTheLeft) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  neg1 = f32[] constant(-1)
  neg1s = f32[5,5] broadcast(neg1), dimensions={}
  nans = f32[5,5] sqrt(neg1s)
  ROOT min = f32[5,5] minimum(nans, neg1s)
})";

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

// TODO(b/324385428): Failing on GPU at head due to an LLVM integrate. Re-enable
// once this has been fixed.
XLA_TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax,
           DISABLED_MinimumHandlesNaNsOnTheRight) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  neg1 = f32[] constant(-1)
  neg1s = f32[5,5] broadcast(neg1), dimensions={}
  nans = f32[5,5] sqrt(neg1s)
  ROOT min = f32[5,5] minimum(neg1s, nans)
})";

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax,
           MaximumHandlesNaNsOnTheLeft) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  neg1 = f32[] constant(-1)
  neg1s = f32[5,5] broadcast(neg1), dimensions={}
  nans = f32[5,5] sqrt(neg1s)
  ROOT max = f32[5,5] maximum(nans, neg1s)
})";

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax,
           MaximumHandlesNaNsOnTheRight) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  neg1 = f32[] constant(-1)
  neg1s = f32[5,5] broadcast(neg1), dimensions={}
  nans = f32[5,5] sqrt(neg1s)
  ROOT max = f32[5,5] maximum(neg1s, nans)
})";

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax,
           MinimumReturnsLHS) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  zero = f32[] constant(0)
  zeros = f32[5,5] broadcast(zero), dimensions={}
  one = f32[] constant(1)
  ones = f32[5,5] broadcast(one), dimensions={}
  ROOT min = f32[5,5] minimum(zeros, ones)
})";

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3,
                                                /*arel=*/1e-3}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax,
           MinimumReturnsRHS) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  zero = f32[] constant(0)
  zeros = f32[5,5] broadcast(zero), dimensions={}
  one = f32[] constant(1)
  ones = f32[5,5] broadcast(one), dimensions={}
  ROOT min = f32[5,5] minimum(ones, zeros)
})";

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3,
                                                /*arel=*/1e-3}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax,
           MaximumReturnsLHS) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  zero = f32[] constant(0)
  zeros = f32[5,5] broadcast(zero), dimensions={}
  one = f32[] constant(1)
  ones = f32[5,5] broadcast(one), dimensions={}
  ROOT max = f32[5,5] maximum(ones, zeros)
})";

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3,
                                                /*arel=*/1e-3}));
}

XLA_TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax,
           MaximumReturnsRHS) {
  constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  zero = f32[] constant(0)
  zeros = f32[5,5] broadcast(zero), dimensions={}
  one = f32[] constant(1)
  ones = f32[5,5] broadcast(one), dimensions={}
  ROOT max = f32[5,5] maximum(zeros, ones)
})";

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3,
                                                /*arel=*/1e-3}));
}

class ElementalIrEmitterInternalTest : public HloTestBase {};

XLA_TEST_F(ElementalIrEmitterInternalTest, SparseDotIsUnsupported) {
  constexpr absl::string_view kHloText = R"(
HloModule test

ENTRY main {
  lhs = f16[5,16] parameter(0)
  rhs = f16[32,10] parameter(1)
  meta = u16[5,2] parameter(2)
  ROOT dot = f32[5,10] dot(lhs, rhs, meta),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@2:4
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloText));
  HloInstruction* root = module->entry_computation()->root_instruction();

  llvm::LLVMContext llvm_context;
  llvm::Module llvm_module("", llvm_context);
  llvm::IRBuilder<> builder(llvm_context);
  ElementalIrEmitterForTests emitter(&llvm_module, &builder);

  llvm_ir::IrArray::Index test_index{builder.getInt64Ty()};
  auto result = emitter.TestElementalDot(root, test_index);
  EXPECT_FALSE(result.ok());
}

}  // namespace
}  // namespace xla

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
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/types.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using std::nullopt;

struct EmitReducePrecisionIrTestCase {
  float input;
  std::string expected_res;
};

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
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        ElementalIrEmitterExecutionTest::GetDebugOptionsForTest();
    debug_options.set_xla_cpu_enable_fast_min_max(false);
    debug_options.set_xla_gpu_enable_fast_min_max(false);
    return debug_options;
  }
};

template <typename T>
class ElementalIrEmitterExecutionTypedTest
    : public ElementalIrEmitterExecutionTest {
 protected:
  const std::string& TypeName() {
    return primitive_util::LowercasePrimitiveTypeName(
        primitive_util::NativeToPrimitiveType<T>());
  }
};

using FloatTypes =
    ::testing::Types<bfloat16, tsl::float4_e2m1fn, tsl::float8_e3m4,
                     tsl::float8_e4m3, tsl::float8_e4m3b11fnuz,
                     tsl::float8_e4m3fn, tsl::float8_e4m3fnuz, tsl::float8_e5m2,
                     tsl::float8_e5m2fnuz, tsl::float8_e8m0fnu>;

TYPED_TEST_SUITE(ElementalIrEmitterExecutionTypedTest, FloatTypes);

TEST_F(ElementalIrEmitterExecutionTest, DotFusion) {
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

TEST_F(ElementalIrEmitterExecutionTest, EmitReducePrecisionIR_F16ToF8e5m2) {
  llvm::LLVMContext llvm_context;
  llvm::IRBuilder<> builder(llvm_context);
  llvm::IRBuilderBase* b = &builder;
  llvm::Type* f16_type = b->getHalfTy();

  float inf = std::numeric_limits<float>::infinity();
  float qnan = std::numeric_limits<float>::quiet_NaN();
  float snan = std::numeric_limits<float>::signaling_NaN();

  EmitReducePrecisionIrTestCase test_cases[] = {
      // clang-format off
      {0.0, "half 0xH0000"},
      {0x1.0p-14, "half 0xH0400"},
      {0.250, "half 0xH3400"},
      {1.0, "half 0xH3C00"},
      {0x1.2p0, "half 0xH3C00"},
      {0x1.Cp15, "half 0xH7B00"},
      {-0x1.Cp15, "half 0xHFB00"},
      {0x1.Dp15, "half 0xH7B00"},
      {0x1.Ep15, "half 0xH7C00"},
      {0x1.0p16, "half 0xH7C00"},
      {inf, "half 0xH7C00"},
      {-inf, "half 0xHFC00"},
      {qnan, "half 0xH7E00"},
      {-qnan, "half 0xHFE00"},
      {snan, "half 0xH7F00"},
      {-snan, "half 0xHFF00"},
      // clang-format on
  };

  for (auto tc : test_cases) {
    llvm::Value* c0 = llvm::ConstantFP::get(f16_type, tc.input);

    absl::StatusOr<llvm::Value*> f16_reduced_statusor = EmitReducePrecisionIR(
        /*src_ty=*/F16, c0,
        /*dest_exponent_bits=*/primitive_util::ExponentWidth(F8E5M2),
        /*dest_mantissa_bits=*/primitive_util::SignificandWidth(F8E5M2) - 1,
        /*quiet_nans=*/true, b);
    CHECK(f16_reduced_statusor.ok());
    llvm::Value* f16_reduced = f16_reduced_statusor.value();

    std::string res = llvm_ir::DumpToString(f16_reduced);
    EXPECT_EQ(res, tc.expected_res) << "Wrong result for input " << tc.input;
  }
}

TEST_F(ElementalIrEmitterExecutionTest, EmitReducePrecisionIR_F16ToF8e4m3) {
  llvm::LLVMContext llvm_context;
  llvm::IRBuilder<> builder(llvm_context);
  llvm::IRBuilderBase* b = &builder;
  llvm::Type* f16_type = b->getHalfTy();

  float inf = std::numeric_limits<float>::infinity();
  float qnan = std::numeric_limits<float>::quiet_NaN();
  float snan = std::numeric_limits<float>::signaling_NaN();

  EmitReducePrecisionIrTestCase test_cases[] = {
      // clang-format off
      {0.0, "half 0xH0000"},
      {0x1.0p-6, "half 0xH2400"},
      {0.125, "half 0xH3000"},
      {1.0, "half 0xH3C00"},
      {0x1.1p0, "half 0xH3C00"},
      {0x1.Ep7, "half 0xH5B80"},
      {-0x1.Ep7, "half 0xHDB80"},
      {0x1.E8p7, "half 0xH5B80"},
      {0x1.Fp7, "half 0xH7C00"},
      {0x1.0p8, "half 0xH7C00"},
      {inf, "half 0xH7C00"},
      {-inf, "half 0xHFC00"},
      {qnan, "half 0xH7E00"},
      {-qnan, "half 0xHFE00"},
      {snan, "half 0xH7E00"},
      {-snan, "half 0xHFE00"},
      // clang-format on
  };

  for (auto tc : test_cases) {
    llvm::Value* c0 = llvm::ConstantFP::get(f16_type, tc.input);

    absl::StatusOr<llvm::Value*> f16_reduced_statusor = EmitReducePrecisionIR(
        /*src_ty=*/F16, c0,
        /*dest_exponent_bits=*/4,
        /*dest_mantissa_bits=*/3,
        /*quiet_nans=*/true, b);
    CHECK(f16_reduced_statusor.ok());
    llvm::Value* f16_reduced = f16_reduced_statusor.value();

    std::string res = llvm_ir::DumpToString(f16_reduced);
    EXPECT_EQ(res, tc.expected_res) << "Wrong result for input " << tc.input;
  }
}

TEST_F(ElementalIrEmitterExecutionTest, EmitReducePrecisionIR_F16ToF8e3m4) {
  llvm::LLVMContext llvm_context;
  llvm::IRBuilder<> builder(llvm_context);
  llvm::IRBuilderBase* b = &builder;
  llvm::Type* f16_type = b->getHalfTy();

  float inf = std::numeric_limits<float>::infinity();
  float qnan = std::numeric_limits<float>::quiet_NaN();
  float snan = std::numeric_limits<float>::signaling_NaN();

  EmitReducePrecisionIrTestCase test_cases[] = {
      // clang-format off
      {0.0, "half 0xH0000"},
      {0x1.0p-2, "half 0xH3400"},
      {0.5, "half 0xH3800"},
      {1.0, "half 0xH3C00"},
      {0x1.08p0, "half 0xH3C00"},
      {0x1.Fp3, "half 0xH4BC0"},
      {-0x1.Fp3, "half 0xHCBC0"},
      {0x1.F4p3, "half 0xH4BC0"},
      {0x1.F8p3, "half 0xH7C00"},
      {0x1.0p4, "half 0xH7C00"},
      {inf, "half 0xH7C00"},
      {-inf, "half 0xHFC00"},
      {qnan, "half 0xH7E00"},
      {-qnan, "half 0xHFE00"},
      {snan, "half 0xH7E00"},
      {-snan, "half 0xHFE00"},
      // clang-format on
  };

  for (auto tc : test_cases) {
    llvm::Value* c0 = llvm::ConstantFP::get(f16_type, tc.input);

    absl::StatusOr<llvm::Value*> f16_reduced_statusor = EmitReducePrecisionIR(
        /*src_ty=*/F16, c0,
        /*dest_exponent_bits=*/3,
        /*dest_mantissa_bits=*/4,
        /*quiet_nans=*/true, b);
    CHECK(f16_reduced_statusor.ok());
    llvm::Value* f16_reduced = f16_reduced_statusor.value();

    std::string res = llvm_ir::DumpToString(f16_reduced);
    EXPECT_EQ(res, tc.expected_res) << "Wrong result for input " << tc.input;
  }
}

TEST_F(ElementalIrEmitterExecutionTest, EmitReducePrecisionIR_F16ToF8e4m3fn) {
  llvm::LLVMContext llvm_context;
  llvm::IRBuilder<> builder(llvm_context);
  llvm::IRBuilderBase* b = &builder;
  llvm::Type* f16_type = b->getHalfTy();

  float inf = std::numeric_limits<float>::infinity();

  EmitReducePrecisionIrTestCase test_cases[] = {
      // clang-format off
      {0.0, "half 0xH0000"},
      {0x1.0p-6, "half 0xH2400"},
      {0.125, "half 0xH3000"},
      {1.0, "half 0xH3C00"},
      {0x1.1p0, "half 0xH3C00"},
      {0x1.Cp8, "half 0xH5F00"},
      {-0x1.Cp8, "half 0xHDF00"},
      {0x1.Dp8, "half 0xH5F00"},
      {0x1.Ep8, "half 0xH5F80"},
      {0x1.0p9, "half 0xH6000"},
      {inf, "half 0xH7C00"},
      {-inf, "half 0xHFC00"},
      // clang-format on
  };

  for (auto tc : test_cases) {
    llvm::Value* c0 = llvm::ConstantFP::get(f16_type, tc.input);

    // Truncate the mantissa to 3 bits. ReducePrecision cannot deal with
    // f8E4M3FN's NaN representations, so don't use ReducePrecision to handle
    // exponent reduction.
    absl::StatusOr<llvm::Value*> f16_reduced_statusor = EmitReducePrecisionIR(
        /*src_ty=*/F16, c0,
        /*dest_exponent_bits=*/5,
        /*dest_mantissa_bits=*/3,
        /*quiet_nans=*/false, b);
    CHECK(f16_reduced_statusor.ok());
    llvm::Value* f16_reduced = f16_reduced_statusor.value();

    std::string res = llvm_ir::DumpToString(f16_reduced);
    EXPECT_EQ(res, tc.expected_res) << "Wrong result for input " << tc.input;
  }
}

TEST_F(ElementalIrEmitterExecutionTest, ScalarDotFusion) {
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

TEST_F(ElementalIrEmitterExecutionTest, BatchDot) {
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

TEST_F(ElementalIrEmitterExecutionTest,
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

TEST_F(ElementalIrEmitterExecutionTest, DivideComplexNumbersWithFiniteNormRhs) {
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

TEST_F(ElementalIrEmitterExecutionTest, DivideComplexNumbersWithZeroNormRhs) {
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

TYPED_TEST(ElementalIrEmitterExecutionTypedTest, ConvertFloatsToFloat) {
  auto tname = this->TypeName();
  const int n = 10;
  if (std::is_same<TypeParam, tsl::float8_e4m3fn>() ||
      std::is_same<TypeParam, tsl::float8_e4m3b11fnuz>()) {
    GTEST_SKIP() << "Skipping test for type " << tname;
  }
  const auto hlo_text =
      absl::StrReplaceAll(R"(
  HloModule m
  ENTRY main {
    f16_ = f16[$n] parameter(0)
    f32_ = f32[$n] parameter(1)
    f64_ = f64[$n] parameter(2)
    bf16_ = bf16[$n] parameter(3)
    converted_f16 = ${tname}[$n] convert(f16_)
    converted_f32 = ${tname}[$n] convert(f32_)
    converted_f64 = ${tname}[$n] convert(f64_)
    converted_bf16 = ${tname}[$n] convert(bf16_)
    ROOT tuple = (${tname}[$n], ${tname}[$n], ${tname}[$n], ${tname}[$n]) tuple(
        converted_f16, converted_f32, converted_f64, converted_bf16)
  }
  )",
                          {{"${tname}", tname}, {"$n", absl::StrCat(n)}});
  ElementalIrEmitterExecutionTest::RunTypeConversionTest(hlo_text);
}

TYPED_TEST(ElementalIrEmitterExecutionTypedTest, ConvertSignedToFloat) {
  auto tname = this->TypeName();
  const auto hlo_text = absl::StrReplaceAll(R"(
    HloModule m
    ENTRY main {
      s8_ = s8[] parameter(0)
      s16_ = s16[] parameter(1)
      s32_ = s32[] parameter(2)
      s64_ = s64[] parameter(3)
      converted_s8 = ${tname}[] convert(s8_)
      converted_s16 = ${tname}[] convert(s16_)
      converted_s32 = ${tname}[] convert(s32_)
      converted_s64 = ${tname}[] convert(s64_)
      ROOT tuple = (${tname}[], ${tname}[], ${tname}[], ${tname}[]) tuple(
          converted_s8, converted_s16, converted_s32, converted_s64)
    }
  )",
                                            {{"${tname}", tname}});
  ElementalIrEmitterExecutionTest::RunTypeConversionTest(hlo_text);
}

TYPED_TEST(ElementalIrEmitterExecutionTypedTest, ConvertUnsignedToFloat) {
  auto tname = this->TypeName();
  const auto hlo_text = absl::StrReplaceAll(R"(
    HloModule m
    ENTRY main {
      u8_ = u8[] parameter(0)
      u16_ = u16[] parameter(1)
      u32_ = u32[] parameter(2)
      u64_ = u64[] parameter(3)
      converted_u8 = ${tname}[] convert(u8_)
      converted_u16 = ${tname}[] convert(u16_)
      converted_u32 = ${tname}[] convert(u32_)
      converted_u64 = ${tname}[] convert(u64_)
      ROOT tuple = (${tname}[], ${tname}[], ${tname}[], ${tname}[]) tuple(
          converted_u8, converted_u16, converted_u32, converted_u64)
    }
  )",
                                            {{"${tname}", tname}});
  ElementalIrEmitterExecutionTest::RunTypeConversionTest(hlo_text);
}

TYPED_TEST(ElementalIrEmitterExecutionTypedTest, ConvertFloatToFloats) {
  auto tname = this->TypeName();
  const auto hlo_text = absl::StrReplaceAll(R"(
   HloModule m
    ENTRY main {
      to_f16 = ${tname}[] parameter(0)
      to_f32 = ${tname}[] parameter(1)
      to_f64 = ${tname}[] parameter(2)
      to_bf16 = ${tname}[] parameter(3)
      f16_ = f16[] convert(to_f16)
      f32_ = f32[] convert(to_f32)
      f64_ = f64[] convert(to_f64)
      bf16_ = bf16[] convert(to_f64)
      ROOT tuple = (f16[], f32[], f64[], bf16[]) tuple(f16_, f32_, f64_, bf16_)
    }
  )",
                                            {{"${tname}", tname}});
  ElementalIrEmitterExecutionTest::RunTypeConversionTest(hlo_text);
}

TYPED_TEST(ElementalIrEmitterExecutionTypedTest, ConvertFloatToSigned) {
  auto tname = this->TypeName();
  const auto hlo_text = absl::StrReplaceAll(R"(
    HloModule m
    ENTRY main {
      to_s8 = ${tname}[] parameter(0)
      to_s16 = ${tname}[] parameter(1)
      to_s32 = ${tname}[] parameter(2)
      to_s64 = ${tname}[] parameter(3)
      s8_ = s8[] convert(to_s8)
      s16_ = s16[] convert(to_s16)
      s32_ = s32[] convert(to_s32)
      s64_ = s64[] convert(to_s64)
      ROOT tuple = (s8[], s16[], s32[], s64[]) tuple(s8_, s16_, s32_, s64_)
    }
  )",
                                            {{"${tname}", tname}});
  ElementalIrEmitterExecutionTest::RunTypeConversionTest(hlo_text);
}

TYPED_TEST(ElementalIrEmitterExecutionTypedTest, ConvertFloatToUnsigned) {
  auto tname = this->TypeName();
  const auto hlo_text = absl::StrReplaceAll(R"(
    HloModule m
    ENTRY main {
      to_u8 = ${tname}[] parameter(0)
      to_u16 = ${tname}[] parameter(1)
      to_u32 = ${tname}[] parameter(2)
      to_u64 = ${tname}[] parameter(3)
      u8_ = u8[] convert(to_u8)
      u16_ = u16[] convert(to_u16)
      u32_ = u32[] convert(to_u32)
      u64_ = u64[] convert(to_u64)
      ROOT tuple = (u8[], u16[], u32[], u64[]) tuple(u8_, u16_, u32_, u64_)
    }
  )",
                                            {{"${tname}", tname}});
  ElementalIrEmitterExecutionTest::RunTypeConversionTest(hlo_text);
}

TYPED_TEST(ElementalIrEmitterExecutionTypedTest, ConvertFloatToComplex) {
  auto tname = this->TypeName();
  const auto hlo_text = absl::StrReplaceAll(R"(
    HloModule m
    ENTRY main {
      to_c64 = ${tname}[] parameter(0)
      to_c128 = ${tname}[] parameter(1)
      c64_ = c64[] convert(to_c64)
      c128_ = c128[] convert(to_c128)
      ROOT tuple = (c64[], c128[]) tuple(c64_, c128_)
    }
  )",
                                            {{"${tname}", tname}});
  ElementalIrEmitterExecutionTest::RunTypeConversionTest(hlo_text);
}

TYPED_TEST(ElementalIrEmitterExecutionTypedTest, CompareFloat) {
  auto tname = this->TypeName();
  if (std::is_same<TypeParam, tsl::float8_e4m3b11fnuz>()) {
    GTEST_SKIP() << "Skipping test for type " << tname;
  }
  const auto hlo_text = absl::StrReplaceAll(R"(
  HloModule m
  ENTRY main {
    p0 = ${tname}[4] parameter(0)
    p1 = ${tname}[4] parameter(1)
    ROOT cmp = pred[4] compare(p0, p1), direction=LT
})",
                                            {{"${tname}", tname}});
  Literal lhs = LiteralUtil::CreateR1<TypeParam>(
      {TypeParam(1.), TypeParam(2.), TypeParam(3.), TypeParam(4.)});
  Literal rhs = LiteralUtil::CreateR1<TypeParam>(
      {TypeParam(4.), TypeParam(4.), TypeParam(2.), TypeParam(1.)});
  ElementalIrEmitterExecutionTest::RunTest(hlo_text, {&lhs, &rhs});
}

TYPED_TEST(ElementalIrEmitterExecutionTypedTest, IotaFloat) {
  auto tname = this->TypeName();
  if (std::is_same<TypeParam, tsl::float8_e5m2>() ||
      std::is_same<TypeParam, tsl::float8_e4m3>() ||
      std::is_same<TypeParam, tsl::float8_e4m3fn>() ||
      std::is_same<TypeParam, tsl::float8_e4m3b11fnuz>() ||
      std::is_same<TypeParam, tsl::float8_e3m4>() ||
      std::is_same<TypeParam, tsl::float4_e2m1fn>() ||
      std::is_same<TypeParam, tsl::float8_e8m0fnu>()) {
    GTEST_SKIP() << "Skipping test for type " << tname;
  }
  const auto hlo_text = absl::StrReplaceAll(R"(
  HloModule m
  ENTRY main {
    ROOT iota_ = ${tname}[4] iota(), iota_dimension=0
  }
  )",
                                            {{"${tname}", tname}});
  ElementalIrEmitterExecutionTest::RunTest(hlo_text, {});
}

TYPED_TEST(ElementalIrEmitterExecutionTypedTest, BatchDotFloat) {
  auto tname = this->TypeName();
  if (std::is_same<TypeParam, tsl::float4_e2m1fn>() ||
      std::is_same<TypeParam, tsl::float8_e8m0fnu>()) {
    GTEST_SKIP() << "Skipping test for type " << tname;
  }
  const auto hlo_text = absl::StrReplaceAll(R"(
  HloModule matmul

  ENTRY main {
    x = ${tname}[8,16] parameter(0)
    y = ${tname}[8,16,32] parameter(1)
    ROOT dot = ${tname}[8,32] dot(x, y), lhs_batch_dims={0},
      rhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }
  )",
                                            {{"${tname}", tname}});
  HloModuleConfig config;
  DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      HloTestBase::ParseAndReturnVerifiedModule(hlo_text, config));
  EXPECT_TRUE(
      HloTestBase::RunAndCompare(std::move(module), ErrorSpec{1e-3, 1e-3}));
}

TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax,
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

TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax,
       MinimumHandlesNaNsOnTheRight) {
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

TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax,
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

TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax,
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

TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax, MinimumReturnsLHS) {
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

TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax, MinimumReturnsRHS) {
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

TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax, MaximumReturnsLHS) {
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

TEST_F(ElementalIrEmitterExecutionTestWithoutFastMinMax, MaximumReturnsRHS) {
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

TEST_F(ElementalIrEmitterInternalTest, SparseDotIsUnsupported) {
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

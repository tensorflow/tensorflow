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

#include "xla/service/llvm_ir/llvm_util.h"

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::llvm_ir {
namespace {

using std::nullopt;
struct EmitReducePrecisionIrTestCase {
  float input;
  std::string expected_res;
};

class EmitReducePrecisionIrExecutionTest : public HloTestBase {
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

TEST_F(EmitReducePrecisionIrExecutionTest, EmitReducePrecisionIR_F16ToF8e5m2) {
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

TEST_F(EmitReducePrecisionIrExecutionTest, EmitReducePrecisionIR_F16ToF8e4m3) {
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

TEST_F(EmitReducePrecisionIrExecutionTest, EmitReducePrecisionIR_F16ToF8e3m4) {
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

TEST_F(EmitReducePrecisionIrExecutionTest,
       EmitReducePrecisionIR_F16ToF8e4m3fn) {
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

class LLVMSPIRVTest : public HloTestBase {};

TEST_F(LLVMSPIRVTest, AddRangeMetadataTest) {
  llvm::LLVMContext llvm_context;
  llvm::IRBuilder<> builder(llvm_context);
  llvm::Triple spirv_triple("spirv64-unknown-unknown");
  auto llvm_module = std::make_unique<llvm::Module>("Module", llvm_context);
  llvm_module->setTargetTriple(spirv_triple);
  llvm::Value* p0 = builder.getInt64(2);

  auto SPIRVBuiltinOfType = [&llvm_context, &llvm_module](
                                llvm::Type* type, absl::string_view func_name) {
    llvm::FunctionType* func_type = llvm::FunctionType::get(
        type, {llvm::Type::getInt64Ty(llvm_context)}, false);
    return llvm_module->getOrInsertFunction(func_name, func_type);
  };

  auto EmitCallAndAddMDForType = [&](llvm::Type* type,
                                     absl::string_view func_name) {
    llvm::Instruction* call =
        builder.CreateCall(SPIRVBuiltinOfType(type, func_name), {p0});
    return AddRangeMetadata(5, 10, call, llvm_module.get());
  };

  auto GetMetadataValue = [](llvm::MDTuple* metadata_tuple, int index) {
    llvm::ValueAsMetadata* metadata = llvm::dyn_cast<llvm::ValueAsMetadata>(
        metadata_tuple->getOperand(index));
    return metadata->getValue();
  };

  llvm::Function* main_func = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(llvm_context), false),
      llvm::Function::ExternalLinkage, "main", *llvm_module);
  llvm::BasicBlock* entry_block =
      llvm::BasicBlock::Create(llvm_context, "entry", main_func);
  builder.SetInsertPoint(entry_block);

  llvm::Instruction* call_int = EmitCallAndAddMDForType(
      llvm::Type::getInt64Ty(llvm_context), "__spirv_builtin_func_i64");
  llvm::Instruction* call_f32 = EmitCallAndAddMDForType(
      llvm::Type::getFloatTy(llvm_context), "__spirv_builtin_func_float");

  EXPECT_NE(call_int->getMetadata(llvm::LLVMContext::MD_range), nullptr);
  // No metadata for non-int types
  EXPECT_EQ(call_f32->getMetadata(llvm::LLVMContext::MD_range), nullptr);
  llvm::MDTuple* metadata_tuple = llvm::dyn_cast<llvm::MDTuple>(
      call_int->getMetadata(llvm::LLVMContext::MD_range));

  // Metadata type must match Call type
  EXPECT_TRUE(GetMetadataValue(metadata_tuple, 0)->getType()->isIntegerTy(64));
  EXPECT_TRUE(GetMetadataValue(metadata_tuple, 1)->getType()->isIntegerTy(64));
}

}  // namespace

}  // namespace xla::llvm_ir

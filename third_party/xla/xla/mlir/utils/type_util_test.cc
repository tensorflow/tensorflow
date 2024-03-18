/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/mlir/utils/type_util.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/functional/function_ref.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/primitive_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// A pair of corresponding types
struct TypeUtilTestParam {
  xla::PrimitiveType xla_t;
  absl::FunctionRef<mlir::Type(mlir::Builder)> mlir_t;
};

inline std::string mlirTypeToString(mlir::Type type) {
  std::string result{};
  llvm::raw_string_ostream sstream(result);
  sstream << type;
  return result;
}

class TypeUtilTest : public ::testing::TestWithParam<TypeUtilTestParam> {};

TEST_P(TypeUtilTest, ConvertInvalidTypeTest) {
  mlir::MLIRContext context;
  mlir::Builder b(&context);

  EXPECT_EQ(ConvertMlirTypeToPrimitiveType(b.getIntegerType(17)),
            xla::PrimitiveType::PRIMITIVE_TYPE_INVALID);
}

TEST_P(TypeUtilTest, MLIRToPrimitiveTypeConversionTest) {
  mlir::MLIRContext context = mlir::MLIRContext();
  mlir::Builder b = mlir::Builder(&context);
  xla::PrimitiveType xla_type_expected = GetParam().xla_t;
  mlir::Type mlir_type = GetParam().mlir_t(b);
  xla::PrimitiveType xla_type_actual =
      ConvertMlirTypeToPrimitiveType(mlir_type);
  EXPECT_EQ(xla_type_actual, xla_type_expected)
      << "Expected: "
      << primitive_util::LowercasePrimitiveTypeName(xla_type_expected)
      << ". Actual: "
      << primitive_util::LowercasePrimitiveTypeName(xla_type_actual) << ".";
}

TEST_P(TypeUtilTest, PrimitiveTypeToMLIRTypeConversionTest) {
  mlir::MLIRContext context = mlir::MLIRContext();
  mlir::Builder b = mlir::Builder(&context);
  xla::PrimitiveType xla_type = GetParam().xla_t;
  mlir::Type mlir_type_expected = GetParam().mlir_t(b);
  TF_ASSERT_OK_AND_ASSIGN(mlir::Type mlir_type_actual,
                          ConvertPrimitiveTypeToMlirType(xla_type, b));
  EXPECT_EQ(mlir_type_actual, mlir_type_expected)
      << "Expected: " << mlirTypeToString(mlir_type_expected)
      << ". Actual: " << mlirTypeToString(mlir_type_actual) << ".";
}

TEST_P(TypeUtilTest, BidirectionalConversionTest) {
  mlir::MLIRContext context = mlir::MLIRContext();
  mlir::Builder b = mlir::Builder(&context);
  xla::PrimitiveType xla_type_expected = GetParam().xla_t;
  TF_ASSERT_OK_AND_ASSIGN(mlir::Type mlir_type_actual,
                          ConvertPrimitiveTypeToMlirType(xla_type_expected, b));
  xla::PrimitiveType xla_type_actual =
      ConvertMlirTypeToPrimitiveType(mlir_type_actual);
  EXPECT_EQ(xla_type_actual, xla_type_expected)
      << "Expected: "
      << primitive_util::LowercasePrimitiveTypeName(xla_type_expected)
      << ". Actual: "
      << primitive_util::LowercasePrimitiveTypeName(xla_type_actual)
      << ". Intermediate MLIR type: " << mlirTypeToString(mlir_type_actual)
      << ".";
}

INSTANTIATE_TEST_SUITE_P(
    Execute, TypeUtilTest,
    ::testing::ValuesIn(std::vector<TypeUtilTestParam>(
        {{PRED, [](mlir::Builder b) { return b.getI1Type(); }},
         {F8E5M2, [](mlir::Builder b) { return b.getFloat8E5M2Type(); }},
         {F8E4M3FN, [](mlir::Builder b) { return b.getFloat8E4M3FNType(); }},
         {F8E4M3B11FNUZ,
          [](mlir::Builder b) { return b.getFloat8E4M3B11FNUZType(); }},
         {F8E5M2FNUZ,
          [](mlir::Builder b) { return b.getFloat8E5M2FNUZType(); }},
         {F8E4M3FNUZ,
          [](mlir::Builder b) { return b.getFloat8E4M3FNUZType(); }},
         {F16, [](mlir::Builder b) { return b.getF16Type(); }},
         {BF16, [](mlir::Builder b) { return b.getBF16Type(); }},
         {F32, [](mlir::Builder b) { return b.getF32Type(); }},
         {F64, [](mlir::Builder b) { return b.getF64Type(); }},
         {U4, [](mlir::Builder b) { return b.getIntegerType(4, false); }},
         {U8, [](mlir::Builder b) { return b.getIntegerType(8, false); }},
         {U16, [](mlir::Builder b) { return b.getIntegerType(16, false); }},
         {U32, [](mlir::Builder b) { return b.getIntegerType(32, false); }},
         {U64, [](mlir::Builder b) { return b.getIntegerType(64, false); }},
         {S4,
          [](mlir::Builder b) {
            return mlir::IntegerType::get(b.getContext(), 4,
                                          mlir::IntegerType::Signless);
          }},
         {S8,
          [](mlir::Builder b) {
            return mlir::IntegerType::get(b.getContext(), 8,
                                          mlir::IntegerType::Signless);
          }},
         {S16,
          [](mlir::Builder b) {
            return mlir::IntegerType::get(b.getContext(), 16,
                                          mlir::IntegerType::Signless);
          }},
         {S32,
          [](mlir::Builder b) {
            return mlir::IntegerType::get(b.getContext(), 32,
                                          mlir::IntegerType::Signless);
          }},
         {S64,
          [](mlir::Builder b) {
            return mlir::IntegerType::get(b.getContext(), 64,
                                          mlir::IntegerType::Signless);
          }}})),
    [](const auto& info) {
      mlir::MLIRContext context;
      mlir::Builder b(&context);

      return absl::StrFormat(
          "xla_%s_mlir_%s",
          primitive_util::LowercasePrimitiveTypeName(info.param.xla_t),
          mlirTypeToString(info.param.mlir_t(b)));
    });

}  // namespace
}  // namespace xla

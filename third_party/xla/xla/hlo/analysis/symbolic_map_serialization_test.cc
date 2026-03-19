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

#include "xla/hlo/analysis/symbolic_map_serialization.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_test_utils.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"

namespace xla {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

struct SymbolicMapSerializationTest : public ::testing::Test {
  SymbolicMapSerializationTest() {
    RegisterSymbolicExprStorage(&ctx);
    v0 = CreateSymbolicVariable(0, &ctx);
    v1 = CreateSymbolicVariable(1, &ctx);
    v2 = CreateSymbolicVariable(2, &ctx);
    v3 = CreateSymbolicVariable(3, &ctx);
    v4 = CreateSymbolicVariable(4, &ctx);
  }

  mlir::MLIRContext ctx;
  SymbolicExpr v0;
  SymbolicExpr v1;
  SymbolicExpr v2;
  SymbolicExpr v3;
  SymbolicExpr v4;
};

TEST_F(SymbolicMapSerializationTest,
       PrintSymbolicExprWithDifferentNumDimensions) {
  SymbolicExpr expr = v0 * 2 + v1;

  EXPECT_THAT(expr.ToString(), MatchIndexingString("((v0 * 2) + v1)"));
  // Only symbols
  EXPECT_THAT(expr.ToString(0), MatchIndexingString("((s0 * 2) + s1)"));
  // One dimension and one symbol
  EXPECT_THAT(expr.ToString(1), MatchIndexingString("((d0 * 2) + s0)"));
  // Only dimensions
  EXPECT_THAT(expr.ToString(2), MatchIndexingString("((d0 * 2) + d1)"));
}

TEST_F(SymbolicMapSerializationTest, ParseSymbolicExprAndPrint) {
  const std::string kStringContainingAllOperators =
      "(((((v0 + 42) * max(min(v1, 2), 0)) floordiv 2) ceildiv 2) mod 5)";
  SymbolicExpr parsed_expr =
      ParseSymbolicExpr(kStringContainingAllOperators, &ctx);
  ASSERT_NE(parsed_expr, nullptr);
  EXPECT_THAT(parsed_expr.ToString(),
              MatchIndexingString(kStringContainingAllOperators));
}

TEST_F(SymbolicMapSerializationTest, ParseSymbolicExprAndPrint_Invalid) {
  absl::ScopedMockLog log(absl::MockLogDefault::kDisallowUnexpected);
  log.StartCapturingLogs();

  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                       "Unexpected end of expression at: \"\""));
  EXPECT_EQ(ParseSymbolicExpr("1 + ", &ctx), SymbolicExpr());

  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                       "Failed to parse expression at: \")\""));
  EXPECT_EQ(ParseSymbolicExpr("max(1, )", &ctx), SymbolicExpr());

  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                       "Missing parenthesis at: \"\""));
  EXPECT_EQ(ParseSymbolicExpr("(1 + 2", &ctx), SymbolicExpr());

  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                       "Failed to parse expression at: \"foo(3, 4)\""));
  EXPECT_EQ(ParseSymbolicExpr("foo(3, 4)", &ctx), SymbolicExpr());
}

TEST_F(SymbolicMapSerializationTest, ParseSymbolicExprAndAdvance_Invalid) {
  absl::ScopedMockLog log(absl::MockLogDefault::kDisallowUnexpected);
  log.StartCapturingLogs();

  // Invalid: Incomplete expression
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                       "Unexpected end of expression at: \"\""));
  absl::string_view expr_str = "1 + ";
  EXPECT_EQ(ParseSymbolicExprAndAdvance(&expr_str, &ctx), SymbolicExpr());
  // The expression string is not consumed because parsing failed.
  EXPECT_EQ(expr_str, "1 + ");
}

TEST_F(SymbolicMapSerializationTest, ParseSymbolicExprWithVariableMap) {
  llvm::DenseMap<llvm::StringRef, SymbolicExpr> variable_map;
  variable_map["foo"] = v0;
  // Purposely use a variable name that starts with a 'd' to test that the
  // dim/symbol parsing is not triggered when the variable map is provided.
  variable_map["dim_bar"] = v1;

  absl::string_view expr_str = "foo + dim_bar * 2";
  SymbolicExpr expr =
      ParseSymbolicExprAndAdvance(&expr_str, &ctx, variable_map);
  EXPECT_EQ(expr, v0 + v1 * 2);
  EXPECT_TRUE(expr_str.empty());

  absl::ScopedMockLog log(absl::MockLogDefault::kDisallowUnexpected);
  log.StartCapturingLogs();
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                       "Failed to parse expression at: \"baz\""));
  expr_str = "baz";
  EXPECT_EQ(ParseSymbolicExprAndAdvance(&expr_str, &ctx, variable_map),
            SymbolicExpr());
}

TEST_F(SymbolicMapSerializationTest, ParseSymbolicExprDimsAndSymbols) {
  EXPECT_EQ(ParseSymbolicExpr("d0", &ctx), v0);
  EXPECT_EQ(ParseSymbolicExpr("s0", &ctx, /*num_dims=*/2),
            CreateSymbolicVariable(2, &ctx));
  EXPECT_EQ(ParseSymbolicExpr("s0", &ctx, /*num_dims=*/0), v0);

  absl::ScopedMockLog log(absl::MockLogDefault::kDisallowUnexpected);
  log.StartCapturingLogs();
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, testing::_,
                       "Symbol cannot be parsed because number of dimensions "
                       "is not set. at: \"0\""));
  EXPECT_EQ(ParseSymbolicExpr("s0", &ctx), SymbolicExpr());
}

TEST_F(SymbolicMapSerializationTest, ParseSymbolicMap) {
  SymbolicMap id = ParseSymbolicMap("(d0) -> (d0)", &ctx);
  EXPECT_EQ(id.GetNumDims(), 1);
  EXPECT_EQ(id.GetNumSymbols(), 0);
  EXPECT_THAT(id.GetResults(), ElementsAre(v0));

  SymbolicMap empty = ParseSymbolicMap("()[] -> ()", &ctx);
  EXPECT_EQ(empty.GetNumDims(), 0);
  EXPECT_EQ(empty.GetNumSymbols(), 0);
  EXPECT_TRUE(empty.IsEmpty());

  SymbolicMap no_dims = ParseSymbolicMap("()[s0] -> (s0)", &ctx);
  EXPECT_EQ(no_dims.GetNumDims(), 0);
  EXPECT_EQ(no_dims.GetNumSymbols(), 1);
  EXPECT_THAT(no_dims.GetResults(), ElementsAre(v0));

  SymbolicMap no_symbols = ParseSymbolicMap("(d0)[] -> (d0)", &ctx);
  EXPECT_EQ(no_symbols.GetNumDims(), 1);
  EXPECT_EQ(no_symbols.GetNumSymbols(), 0);
  EXPECT_THAT(no_symbols.GetResults(), ElementsAre(v0));

  SymbolicMap no_results = ParseSymbolicMap("(d0)[s0] -> ()", &ctx);
  EXPECT_EQ(no_results.GetNumDims(), 1);
  EXPECT_EQ(no_results.GetNumSymbols(), 1);
  EXPECT_TRUE(no_results.IsEmpty());

  SymbolicMap map_with_constants =
      ParseSymbolicMap("(d0)[s0] -> (d0 * 2 + s0 - 5)", &ctx);
  EXPECT_EQ(map_with_constants.GetNumDims(), 1);
  EXPECT_EQ(map_with_constants.GetNumSymbols(), 1);
  EXPECT_THAT(map_with_constants.GetResults(), ElementsAre(v0 * 2 + v1 - 5));

  // Expressions with different naming convention.
  SymbolicMap map_with_different_naming = ParseSymbolicMap(
      "(d0, d1)[range, rt0, rt1] -> (d1, d0, range + rt0, rt1)", &ctx);
  EXPECT_EQ(map_with_different_naming.GetNumDims(), 2);
  EXPECT_EQ(map_with_different_naming.GetNumSymbols(), 3);
  EXPECT_THAT(map_with_different_naming.GetResults(),
              ElementsAre(v1, v0, v2 + v3, v4));
}

TEST_F(SymbolicMapSerializationTest, ParseSymbolicMap_Invalid) {
  absl::ScopedMockLog log(absl::MockLogDefault::kDisallowUnexpected);
  log.StartCapturingLogs();

  EXPECT_CALL(log,
              Log(absl::LogSeverity::kError, _, HasSubstr("missing `->`")));
  EXPECT_EQ(ParseSymbolicMap("(d0) (d0)", &ctx), SymbolicMap());

  // Invalid: Unbalanced parentheses
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                       HasSubstr("Failed to parse dimension list")));
  EXPECT_EQ(ParseSymbolicMap("(d0 -> (d0)", &ctx), SymbolicMap());

  // Invalid: Unbalanced brackets
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                       HasSubstr("Failed to parse symbol list")));
  EXPECT_EQ(ParseSymbolicMap("(d0)[s0 -> (d0)", &ctx), SymbolicMap());

  // Invalid: Missing parentheses around expression list.
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                       HasSubstr("Failed to parse expression list")));
  EXPECT_EQ(ParseSymbolicMap("(d0) -> d0", &ctx), SymbolicMap());
  ::testing::Mock::VerifyAndClearExpectations(&log);
}

TEST_F(SymbolicMapSerializationTest, ParseSymbolicMapAndAdvance_ConsumesAll) {
  absl::string_view map_str = "(d0) -> (d0)";
  SymbolicMap map = ParseSymbolicMapAndAdvance(&map_str, &ctx);
  EXPECT_EQ(map.ToString(), "(d0)[] -> (d0)");
  EXPECT_EQ(map_str, "");
}

TEST_F(SymbolicMapSerializationTest, ParseSymbolicMapAndAdvance_WithSuffix) {
  absl::string_view map_str = "(d0) -> (d0) domain: d0 in [0, 1]";
  SymbolicMap map = ParseSymbolicMapAndAdvance(&map_str, &ctx);
  EXPECT_EQ(map.ToString(), "(d0)[] -> (d0)");
  EXPECT_EQ(absl::StripLeadingAsciiWhitespace(map_str), "domain: d0 in [0, 1]");
}

TEST_F(SymbolicMapSerializationTest, ParseSymbolicMapAndAdvance_Invalid) {
  absl::ScopedMockLog log(absl::MockLogDefault::kDisallowUnexpected);
  log.StartCapturingLogs();

  // Invalid: Empty string
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                       HasSubstr("Failed to parse dimension list")));
  absl::string_view map_str = "";
  EXPECT_EQ(ParseSymbolicMapAndAdvance(&map_str, &ctx), SymbolicMap());

  // Invalid: Malformed map string
  EXPECT_CALL(log, Log(absl::LogSeverity::kError, _,
                       HasSubstr("Failed to parse expression list")));
  map_str = "(d0) -> d0";
  EXPECT_EQ(ParseSymbolicMapAndAdvance(&map_str, &ctx), SymbolicMap());
  // The map string is not consumed because parsing failed.
  EXPECT_EQ(map_str, "(d0) -> d0");
}

}  // namespace
}  // namespace xla

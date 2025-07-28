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

#include "xla/service/gpu/model/experimental/symbolic_expr.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/Support/MathExtras.h"

namespace xla {
namespace gpu {
namespace {

std::string GetBinaryOpString(SymbolicExprType type) {
  switch (type) {
    case SymbolicExprType::kAdd:
      return "+";
    case SymbolicExprType::kMul:
      return "*";
    case SymbolicExprType::kFloorDiv:
      return "floordiv";
    case SymbolicExprType::kCeilDiv:
      return "ceildiv";
    case SymbolicExprType::kMod:
      return "mod";
    case SymbolicExprType::kMax:
      return "max";
    case SymbolicExprType::kMin:
      return "min";
    default:
      LOG(FATAL) << "unknown binary operation on symbolic expressions";
  }
}

}  // namespace

namespace {
// Helper class to manage the state of the parser.
class Parser {
 public:
  Parser(absl::string_view str, SymbolicExprContext* context)
      : remaining_str_(str), context_(context) {}

  SymbolicExpr* Parse() {
    SymbolicExpr* expr = ParseExpression();
    SkipWhitespace();
    CHECK(remaining_str_.empty()) << "Did not parse entire string";
    return expr;
  }

 private:
  int64_t ParseNumber(std::string& error_msg) {
    size_t num_len = 0;
    if (!remaining_str_.empty() &&
        (isdigit(remaining_str_[0]) || remaining_str_[0] == '-')) {
      num_len = 1;
    }
    while (num_len < remaining_str_.size() &&
           isdigit(remaining_str_[num_len])) {
      num_len++;
    }
    CHECK(num_len > 0) << error_msg;
    int64_t number;
    CHECK(absl::SimpleAtoi(remaining_str_.substr(0, num_len), &number));
    remaining_str_.remove_prefix(num_len);
    return number;
  }

  // Handles lowest precedence operators: +
  SymbolicExpr* ParseExpression() {
    SymbolicExpr* lhs = ParseTerm();
    while (true) {
      SkipWhitespace();
      if (absl::ConsumePrefix(&remaining_str_, "+")) {
        lhs =
            context_->CreateBinaryOp(SymbolicExprType::kAdd, lhs, ParseTerm());
      } else {
        break;
      }
    }
    return lhs;
  }

  // Handles higher precedence operators: *, floordiv, ceildiv
  SymbolicExpr* ParseTerm() {
    SymbolicExpr* lhs = ParseFactor();
    while (true) {
      SkipWhitespace();
      if (absl::ConsumePrefix(&remaining_str_, "*")) {
        lhs = context_->CreateBinaryOp(SymbolicExprType::kMul, lhs,
                                       ParseFactor());
      } else if (absl::ConsumePrefix(&remaining_str_, "floordiv")) {
        lhs = context_->CreateBinaryOp(SymbolicExprType::kFloorDiv, lhs,
                                       ParseFactor());
      } else if (absl::ConsumePrefix(&remaining_str_, "ceildiv")) {
        lhs = context_->CreateBinaryOp(SymbolicExprType::kCeilDiv, lhs,
                                       ParseFactor());
      } else {
        break;
      }
    }
    return lhs;
  }

  // Attempts to parse a binary function call (e.g., "name(lhs, rhs)")
  // Returns the parsed expression, or nullptr if `func_name` does not match.
  SymbolicExpr* ParseBinaryFunction(SymbolicExprType type) {
    std::string func_name = GetBinaryOpString(type);
    if (!absl::ConsumePrefix(&remaining_str_, absl::StrCat(func_name, "("))) {
      return nullptr;
    }
    SymbolicExpr* lhs = ParseExpression();
    SkipWhitespace();
    CHECK(absl::ConsumePrefix(&remaining_str_, ","))
        << "Missing ',' in " << func_name << "()";
    SymbolicExpr* rhs = ParseExpression();
    SkipWhitespace();
    CHECK(absl::ConsumePrefix(&remaining_str_, ")"))
        << "Missing ')' in " << func_name << "()";
    return context_->CreateBinaryOp(type, lhs, rhs);
  }

  // Handles highest precedence items: numbers, variables, and functions.
  SymbolicExpr* ParseFactor() {
    SkipWhitespace();
    CHECK(!remaining_str_.empty()) << "Unexpected end of expression.";

    // Case 1:Function call like max( ... ) or min( ... )
    SymbolicExpr* expr = nullptr;
    if ((expr = ParseBinaryFunction(SymbolicExprType::kMax)) ||
        (expr = ParseBinaryFunction(SymbolicExprType::kMin))) {
      return expr;
    }
    // Case 2: Parenthesized subexpression
    if (absl::ConsumePrefix(&remaining_str_, "(")) {
      SymbolicExpr* expr = ParseExpression();
      SkipWhitespace();
      CHECK(absl::ConsumePrefix(&remaining_str_, ")")) << "Missing parenthesis";
      return expr;
    }
    // Case 3:Variable (e.g., "v123")
    // TODO(karupayun): Add support for variables that do not start with "v".
    if (absl::ConsumePrefix(&remaining_str_, "v")) {
      std::string error_msg = "Invalid variable format";
      int64_t var_id = ParseNumber(error_msg);
      return context_->CreateVariable(var_id);
    }
    // Case 4: Number
    std::string error_msg =
        absl::StrCat("Failed to parse expression: \"", remaining_str_, "\"");
    int64_t val = ParseNumber(error_msg);
    return context_->CreateConstant(val);
  }

  void SkipWhitespace() {
    remaining_str_ = absl::StripLeadingAsciiWhitespace(remaining_str_);
  }

  absl::string_view remaining_str_;
  SymbolicExprContext* context_;
};

}  // namespace

std::string SymbolicExpr::ToString() const {
  switch (type_) {
    case SymbolicExprType::kConstant:
      return std::to_string(GetValue());
    case SymbolicExprType::kVariable:
      return absl::StrCat("v", GetValue());
    case SymbolicExprType::kAdd:
    case SymbolicExprType::kMul:
    case SymbolicExprType::kFloorDiv:
    case SymbolicExprType::kCeilDiv:
    case SymbolicExprType::kMod: {
      auto bin_op_str = GetBinaryOpString(type_);
      return absl::StrCat("(", GetLHS()->ToString(), " ", bin_op_str, " ",
                          GetRHS()->ToString(), ")");
    }
    case SymbolicExprType::kMax:
    case SymbolicExprType::kMin: {
      auto bin_op_str = GetBinaryOpString(type_);
      return absl::StrCat(bin_op_str, "( ", GetLHS()->ToString(), ", ",
                          GetRHS()->ToString(), ")");
    }
    default:
      LOG(FATAL) << "unknown type on symbolic expressions";
  }
}

SymbolicExpr* SymbolicExprContext::CreateConstant(int64_t value) {
  absl::MutexLock lock(&mutex);
  expr_storage.emplace_back(SymbolicExpr(SymbolicExprType::kConstant, value));
  return &expr_storage.back();
}

SymbolicExpr* SymbolicExprContext::CreateVariable(int64_t var_id) {
  absl::MutexLock lock(&mutex);
  expr_storage.emplace_back(SymbolicExpr(SymbolicExprType::kVariable, var_id));
  return &expr_storage.back();
}

SymbolicExpr* SymbolicExprContext::CreateBinaryOp(SymbolicExprType type,
                                                  SymbolicExpr* lhs,
                                                  SymbolicExpr* rhs) {
  absl::MutexLock lock(&mutex);
  CHECK(type != SymbolicExprType::kConstant &&
        type != SymbolicExprType::kVariable && lhs != nullptr && rhs != nullptr)
      << "We expect a binary operation and two symbolic expressions as "
         "children.";
  expr_storage.emplace_back(SymbolicExpr(type, lhs, rhs));
  return &expr_storage.back();
}

int64_t SymbolicExpr::Evaluate(
    absl::Span<const int64_t> variable_values) const {
  int64_t lhs_value = 0, rhs_value = 0;
  if (lhs_ != nullptr) {
    lhs_value = lhs_->Evaluate(variable_values);
  }
  if (rhs_ != nullptr) {
    rhs_value = rhs_->Evaluate(variable_values);
  }
  switch (GetType()) {
    case SymbolicExprType::kConstant:
      return GetValue();
    case SymbolicExprType::kVariable: {
      int var_id = GetValue();
      CHECK(var_id >= 0 && var_id < variable_values.size())
          << "Evaluate has not provided a value for VariableID " << var_id
          << ".";
      return variable_values[var_id];
    }
    case SymbolicExprType::kAdd:
      return lhs_value + rhs_value;
    case SymbolicExprType::kMul:
      return lhs_value * rhs_value;
    case SymbolicExprType::kFloorDiv:
      return llvm::divideFloorSigned(lhs_value, rhs_value);
    case SymbolicExprType::kCeilDiv:
      return llvm::divideCeilSigned(lhs_value, rhs_value);
    case SymbolicExprType::kMod: {
      CHECK_NE(rhs_value, 0);
      // C++'s '%' can be negative. A true modulo is always non-negative.
      return (lhs_value % rhs_value + std::abs(rhs_value)) % rhs_value;
    }
    case SymbolicExprType::kMax:
      return std::max(lhs_value, rhs_value);
    case SymbolicExprType::kMin:
      return std::min(lhs_value, rhs_value);
    default:
      LOG(FATAL) << "Evaluate not implemented for this expression type.";
  }
}

SymbolicExpr* SymbolicExpr::ReplaceVariables(
    absl::Span<SymbolicExpr* const> substitutions,
    SymbolicExprContext* ctx) const {
  switch (GetType()) {
    case SymbolicExprType::kConstant:
      return ctx->CreateConstant(GetValue());
    case SymbolicExprType::kVariable: {
      const VariableID var_id = GetValue();
      if (var_id >= 0 && var_id < substitutions.size() &&
          substitutions[var_id] != nullptr) {
        return substitutions[var_id];
      }
      return ctx->CreateVariable(GetValue());
    }
    case SymbolicExprType::kAdd:
    case SymbolicExprType::kMul:
    case SymbolicExprType::kFloorDiv:
    case SymbolicExprType::kCeilDiv:
    case SymbolicExprType::kMod:
    case SymbolicExprType::kMax:
    case SymbolicExprType::kMin: {
      SymbolicExpr* new_lhs = GetLHS()->ReplaceVariables(substitutions, ctx);
      SymbolicExpr* new_rhs = GetRHS()->ReplaceVariables(substitutions, ctx);
      return ctx->CreateBinaryOp(GetType(), new_lhs, new_rhs);
    }
    default:
      LOG(FATAL) << "Substitute not implemented for this type.";
  }
}

SymbolicExpr* SymbolicExprContext::Parse(absl::string_view expr_str) {
  return Parser(expr_str, this).Parse();
}

}  // namespace gpu
}  // namespace xla

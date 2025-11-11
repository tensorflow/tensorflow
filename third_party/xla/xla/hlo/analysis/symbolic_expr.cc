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

#include "xla/hlo/analysis/symbolic_expr.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/const_init.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/StorageUniquer.h"
#include "mlir/Support/TypeID.h"

namespace xla {
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

// Helper class to manage the state of the parser.
class Parser {
 public:
  Parser(absl::string_view str, SymbolicExprContext* context)
      : remaining_str_(str), context_(context) {}

  SymbolicExpr Parse() {
    SymbolicExpr expr = ParseExpression();
    SkipWhitespace();
    CHECK(remaining_str_.empty()) << "Did not parse entire string";
    return expr;
  }

 private:
  int64_t ParseNumber(std::string& error_msg) {
    size_t num_len = 0;
    if (!remaining_str_.empty() &&
        (absl::ascii_isdigit(remaining_str_[0]) || remaining_str_[0] == '-')) {
      num_len = 1;
    }
    while (num_len < remaining_str_.size() &&
           absl::ascii_isdigit(remaining_str_[num_len])) {
      num_len++;
    }
    CHECK(num_len > 0) << error_msg;
    int64_t number;
    CHECK(absl::SimpleAtoi(remaining_str_.substr(0, num_len), &number));
    remaining_str_.remove_prefix(num_len);
    return number;
  }

  // Handles lowest precedence operators: +
  SymbolicExpr ParseExpression() {
    SymbolicExpr lhs = ParseTerm();
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
  SymbolicExpr ParseTerm() {
    SymbolicExpr lhs = ParseFactor();
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
  SymbolicExpr ParseBinaryFunction(SymbolicExprType type) {
    std::string func_name = GetBinaryOpString(type);
    if (!absl::ConsumePrefix(&remaining_str_, absl::StrCat(func_name, "("))) {
      return {};
    }
    SymbolicExpr lhs = ParseExpression();
    SkipWhitespace();
    CHECK(absl::ConsumePrefix(&remaining_str_, ","))
        << "Missing ',' in " << func_name << "()";
    SymbolicExpr rhs = ParseExpression();
    SkipWhitespace();
    CHECK(absl::ConsumePrefix(&remaining_str_, ")"))
        << "Missing ')' in " << func_name << "()";
    return context_->CreateBinaryOp(type, lhs, rhs);
  }

  // Handles highest precedence items: numbers, variables, and functions.
  SymbolicExpr ParseFactor() {
    SkipWhitespace();
    CHECK(!remaining_str_.empty()) << "Unexpected end of expression.";

    // Case 1:Function call like max( ... ) or min( ... )
    SymbolicExpr expr;
    if ((expr = ParseBinaryFunction(SymbolicExprType::kMax)) ||
        (expr = ParseBinaryFunction(SymbolicExprType::kMin))) {
      return expr;
    }
    // Case 2: Parenthesized subexpression
    if (absl::ConsumePrefix(&remaining_str_, "(")) {
      SymbolicExpr expr = ParseExpression();
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

// Returns {BASE, COEFF}, where expr is equivalent to BASE * COEFF.
std::pair<SymbolicExpr, int64_t> GetBaseAndCoeff(SymbolicExpr expr) {
  if (expr.GetType() == SymbolicExprType::kMul) {
    SymbolicExpr lhs = expr.GetLHS();
    SymbolicExpr rhs = expr.GetRHS();

    if (rhs.GetType() == SymbolicExprType::kConstant) {
      auto [base, coeff] = GetBaseAndCoeff(lhs);
      return {base, coeff * rhs.GetValue()};
    }
  }
  return {expr, 1};
}

// Helper function to recursively extract terms from an Add expression.
void ExtractTerms(SymbolicExpr expr,
                  llvm::SmallVector<std::pair<SymbolicExpr, int64_t>>& terms) {
  if (expr.GetType() == SymbolicExprType::kAdd) {
    ExtractTerms(expr.GetLHS(), terms);
    ExtractTerms(expr.GetRHS(), terms);
  } else {
    auto [base, coeff] = GetBaseAndCoeff(expr);
    terms.push_back({base, coeff});
  }
}

SymbolicExpr CanonicalizeAdd(SymbolicExpr lhs, SymbolicExpr rhs) {
  SymbolicExprContext* ctx = lhs.GetContext();

  // Flattening and term collection
  llvm::SmallVector<std::pair<SymbolicExpr, int64_t>> terms;
  ExtractTerms(lhs, terms);
  ExtractTerms(rhs, terms);

  absl::c_sort(terms,
               [](const auto& a, const auto& b) { return a.first < b.first; });

  llvm::SmallVector<SymbolicExpr> exprs;
  int64_t const_val = 0;

  for (int i = 0; i < terms.size(); ++i) {
    SymbolicExpr current_base = terms[i].first;
    int64_t current_coeff = terms[i].second;

    while (i + 1 < terms.size() && terms[i + 1].first == current_base) {
      current_coeff += terms[i + 1].second;
      i++;
    }

    if (current_coeff == 0) {
      continue;
    }

    if (current_base.GetType() == SymbolicExprType::kConstant) {
      const_val += current_base.GetValue() * current_coeff;
    } else {
      exprs.push_back((current_base * current_coeff).Canonicalize());
    }
  }

  // Add the combined constant term as an expression
  if (const_val != 0) {
    exprs.push_back(ctx->CreateConstant(const_val));
  }
  if (exprs.empty()) {
    return ctx->CreateConstant(0);
  }
  // Sort all terms, including the constant
  absl::c_sort(exprs);

  SymbolicExpr result = exprs[0];
  for (size_t i = 1; i < exprs.size(); ++i) {
    result = ctx->CreateBinaryOp(SymbolicExprType::kAdd, result, exprs[i]);
  }
  return result;
}

SymbolicExpr CanonicalizeMul(SymbolicExpr lhs, SymbolicExpr rhs) {
  SymbolicExprContext* ctx = lhs.GetContext();

  // Neutral Elements
  if (rhs.GetType() == SymbolicExprType::kConstant) {
    if (rhs.GetValue() == 0) {
      return rhs;  // x * 0 = 0
    }
    if (rhs.GetValue() == 1) {
      return lhs;  // x * 1 = x
    }
  }

  // Associativity: (X * C1) * C2 => X * (C1 * C2)
  if (lhs.GetType() == SymbolicExprType::kMul &&
      lhs.GetRHS().GetType() == SymbolicExprType::kConstant &&
      rhs.GetType() == SymbolicExprType::kConstant) {
    return (lhs.GetLHS() * (lhs.GetRHS().GetValue() * rhs.GetValue()))
        .Canonicalize();
  }

  // Distribute Mul over Add: (A + B) * C => (A * C) + (B * C)
  if (lhs.GetType() == SymbolicExprType::kAdd) {
    return ((lhs.GetLHS() * rhs) + (lhs.GetRHS() * rhs)).Canonicalize();
  }
  if (rhs.GetType() == SymbolicExprType::kAdd) {
    return ((lhs * rhs.GetLHS()) + (lhs * rhs.GetRHS())).Canonicalize();
  }

  return ctx->CreateBinaryOp(SymbolicExprType::kMul, lhs, rhs);
}

std::optional<int64_t> SubtractAndGetConstDiff(SymbolicExpr lhs,
                                               SymbolicExpr rhs) {
  SymbolicExpr diff = (lhs - rhs).Canonicalize();
  if (diff.GetType() == SymbolicExprType::kConstant) {
    return diff.GetValue();
  }
  return std::nullopt;
}

SymbolicExpr CanonicalizeMin(SymbolicExpr lhs, SymbolicExpr rhs) {
  SymbolicExprContext* ctx = lhs.GetContext();
  if (auto diff = SubtractAndGetConstDiff(lhs, rhs)) {  // min(X, X + k) = X
    return (diff.value() <= 0) ? lhs : rhs;
  }

  return ctx->CreateBinaryOp(SymbolicExprType::kMin, lhs, rhs);
}

SymbolicExpr CanonicalizeMax(SymbolicExpr lhs, SymbolicExpr rhs) {
  SymbolicExprContext* ctx = lhs.GetContext();
  if (auto diff = SubtractAndGetConstDiff(lhs, rhs)) {  // max(X, X + k) = X + k
    return (diff.value() >= 0) ? lhs : rhs;
  }

  return ctx->CreateBinaryOp(SymbolicExprType::kMax, lhs, rhs);
}

// Helper function to simplify (A * C1) op C2 using GCD.
SymbolicExpr TrySimplifyDivModByGCD(SymbolicExprType op_type, SymbolicExpr lhs,
                                    int64_t divisor) {
  if (lhs.GetType() != SymbolicExprType::kMul) {
    return SymbolicExpr();
  }
  SymbolicExpr mul_lhs = lhs.GetLHS();
  SymbolicExpr mul_rhs = lhs.GetRHS();

  // mul_lhs can't be a constant because lhs is already canonicalized
  // and constants are on the RHS.
  if (mul_rhs.GetType() != SymbolicExprType::kConstant) {
    return SymbolicExpr();
  }

  int64_t mul_rhs_val = mul_rhs.GetValue();
  int64_t gcd = std::gcd(std::abs(mul_rhs_val), std::abs(divisor));

  if (gcd <= 1) {  // common is never 0 because divisor is non-zero
    return SymbolicExpr();
  }

  SymbolicExpr new_lhs = mul_lhs * (mul_rhs_val / gcd);
  int64_t new_divisor = divisor / gcd;

  switch (op_type) {
    case SymbolicExprType::kFloorDiv: {
      return (new_lhs / new_divisor).Canonicalize();
    }
    case SymbolicExprType::kCeilDiv: {
      return new_lhs.ceilDiv(new_divisor).Canonicalize();
    }
    case SymbolicExprType::kMod: {
      // (A * C1) mod C2 = ((A * (C1 / g)) mod (C2 / g)) * g
      return (((mul_lhs * (mul_rhs_val / gcd)) % (divisor / gcd)) * gcd)
          .Canonicalize();
    }
    default: {
      LOG(FATAL) << "Unsupported op_type in TrySimplifyDivModByGCD";
    }
  }
}

// Simplifies (A + B) floordiv C = (A / C) + (B floordiv C) if A is a multiple
// of C.
SymbolicExpr SimplifyFloorDivAddOperand(SymbolicExpr a, SymbolicExpr b,
                                        int64_t div) {
  int64_t a_coeff;
  SymbolicExpr remaining_expr;

  if (a.GetType() == SymbolicExprType::kMul &&
      a.GetRHS().GetType() == SymbolicExprType::kConstant) {
    a_coeff = a.GetRHS().GetValue();
    remaining_expr = a.GetLHS();
  } else if (a.GetType() == SymbolicExprType::kConstant) {
    a_coeff = a.GetValue();
    remaining_expr = a.GetContext()->CreateConstant(1);
  } else {
    return SymbolicExpr();  // Cannot simplify
  }

  if (a_coeff % div != 0) {
    return SymbolicExpr();  // Cannot simplify
  }
  return (remaining_expr * (a_coeff / div) + b / div).Canonicalize();
}

SymbolicExpr CanonicalizeFloorDiv(SymbolicExpr lhs, SymbolicExpr rhs) {
  SymbolicExprContext* ctx = lhs.GetContext();

  if (lhs.GetType() == SymbolicExprType::kConstant && lhs.GetValue() == 0) {
    return lhs;  // 0 floordiv X => 0
  }

  if (rhs.GetType() == SymbolicExprType::kConstant) {
    int64_t divisor = rhs.GetValue();
    CHECK_NE(divisor, 0) << "Division by zero";
    if (divisor == 1) {
      return lhs;
    }
    if (divisor == -1) {
      return -lhs;
    }

    SymbolicExpr gcd_simplified =
        TrySimplifyDivModByGCD(SymbolicExprType::kFloorDiv, lhs, divisor);
    if (gcd_simplified) {
      return gcd_simplified;
    }

    // Distributivity for (A + C1) floordiv C2 where C1 % C2 == 0
    if (lhs.GetType() == SymbolicExprType::kAdd) {
      SymbolicExpr add_lhs = lhs.GetLHS();
      SymbolicExpr add_rhs = lhs.GetRHS();

      if (auto simplified =
              SimplifyFloorDivAddOperand(add_lhs, add_rhs, divisor)) {
        return simplified;
      }
      if (auto simplified =
              SimplifyFloorDivAddOperand(add_rhs, add_lhs, divisor)) {
        return simplified;
      }
    }
  }

  return ctx->CreateBinaryOp(SymbolicExprType::kFloorDiv, lhs, rhs);
}

SymbolicExpr CanonicalizeCeilDiv(SymbolicExpr lhs, SymbolicExpr rhs) {
  SymbolicExprContext* ctx = lhs.GetContext();

  if (lhs.GetType() == SymbolicExprType::kConstant && lhs.GetValue() == 0) {
    return lhs;  // 0 ceildiv X => 0
  }

  if (rhs.GetType() == SymbolicExprType::kConstant) {
    int64_t divisor = rhs.GetValue();
    CHECK_NE(divisor, 0) << "Division by zero";
    if (divisor == 1) {
      return lhs;
    }
    if (divisor == -1) {
      return -lhs;
    }

    SymbolicExpr gcd_simplified =
        TrySimplifyDivModByGCD(SymbolicExprType::kCeilDiv, lhs, divisor);
    if (gcd_simplified) {
      return gcd_simplified;
    }

    if (divisor > 0) {
      return ((lhs + divisor - 1).Canonicalize()).floorDiv(rhs).Canonicalize();
    }
    return (-(lhs.floorDiv(-divisor))).Canonicalize();
  }

  return ctx->CreateBinaryOp(SymbolicExprType::kCeilDiv, lhs, rhs);
}

SymbolicExpr CanonicalizeMod(SymbolicExpr lhs, SymbolicExpr rhs) {
  SymbolicExprContext* ctx = lhs.GetContext();

  if (lhs.GetType() == SymbolicExprType::kConstant && lhs.GetValue() == 0) {
    return lhs;  // 0 mod X => 0
  }

  if (lhs == rhs) {
    return ctx->CreateConstant(0);  // X mod X => 0
  }

  if (rhs.GetType() == SymbolicExprType::kConstant) {
    int64_t divisor = rhs.GetValue();
    CHECK_NE(divisor, 0) << "Modulo by zero";

    if (SymbolicExpr gcd_simplified =
            TrySimplifyDivModByGCD(SymbolicExprType::kMod, lhs, divisor)) {
      return gcd_simplified;
    }
  }

  // Fallback: A mod B = A - (A floordiv B) * B
  SymbolicExpr floor_div_expr = lhs.floorDiv(rhs).Canonicalize();
  if (floor_div_expr.GetType() == SymbolicExprType::kConstant &&
      floor_div_expr.GetValue() == 0) {
    return lhs;  // If A floordiv B is 0, then A mod B is A
  }
  SymbolicExpr product = (floor_div_expr * rhs).Canonicalize();
  return (lhs - product).Canonicalize();
}

}  // namespace

class SymbolicExprStorage : public mlir::StorageUniquer::BaseStorage {
 public:
  using KeyTy =
      std::tuple<SymbolicExprType, int64_t, SymbolicExpr, SymbolicExpr>;

  static SymbolicExprStorage* construct(
      mlir::StorageUniquer::StorageAllocator& allocator, const KeyTy& key) {
    SymbolicExprStorage* storage = allocator.allocate<SymbolicExprStorage>();
    SymbolicExprType type = std::get<0>(key);
    if (type == SymbolicExprType::kConstant ||
        type == SymbolicExprType::kVariable) {
      return new (storage) SymbolicExprStorage(type, std::get<1>(key));
    }
    return new (storage)
        SymbolicExprStorage(type, std::get<2>(key), std::get<3>(key));
  }

  bool operator==(const KeyTy& key) const {
    SymbolicExprType key_type = std::get<0>(key);
    if (type_ != key_type) {
      return false;
    }

    // Based on the type, compare the relevant fields.
    if (key_type == SymbolicExprType::kConstant ||
        key_type == SymbolicExprType::kVariable) {
      return value_ == std::get<1>(key);
    }
    return lhs_ == std::get<2>(key) && rhs_ == std::get<3>(key);
  }

 protected:
  friend class SymbolicExpr;
  friend class SymbolicExprContext;
  SymbolicExprType type_;
  int64_t value_ = 0;
  SymbolicExpr lhs_;
  SymbolicExpr rhs_;
  SymbolicExprContext* ctx_ = nullptr;

 private:
  SymbolicExprStorage(SymbolicExprType type, int64_t value)
      : type_(type), value_(value) {}
  SymbolicExprStorage(SymbolicExprType type, SymbolicExpr lhs, SymbolicExpr rhs)
      : type_(type), lhs_(lhs), rhs_(rhs) {}
};

SymbolicExprContext* SymbolicExpr::GetContext() const { return impl_->ctx_; }

SymbolicExprType SymbolicExpr::GetType() const { return impl_->type_; }

bool SymbolicExpr::IsBinaryOp() const {
  auto type = GetType();
  return type != SymbolicExprType::kConstant &&
         type != SymbolicExprType::kVariable;
}

SymbolicExpr SymbolicExpr::GetLHS() const { return impl_->lhs_; }

SymbolicExpr SymbolicExpr::GetRHS() const { return impl_->rhs_; }

int64_t SymbolicExpr::GetValue() const { return impl_->value_; }

bool SymbolicExpr::operator<(const SymbolicExpr& other) const {
  CHECK(*this && other);
  if (this == &other) {
    return false;
  }
  SymbolicExprType lhs_type = GetType();
  SymbolicExprType rhs_type = other.GetType();

  if (lhs_type != rhs_type) {
    return lhs_type < rhs_type;
  }

  switch (lhs_type) {
    case SymbolicExprType::kVariable:
    case SymbolicExprType::kConstant:
      return GetValue() < other.GetValue();
    case SymbolicExprType::kAdd:
    case SymbolicExprType::kMul:
    case SymbolicExprType::kFloorDiv:
    case SymbolicExprType::kCeilDiv:
    case SymbolicExprType::kMod:
    case SymbolicExprType::kMax:
    case SymbolicExprType::kMin:
      if (GetLHS() != other.GetLHS()) {
        return GetLHS() < other.GetLHS();
      }
      return GetRHS() < other.GetRHS();
    default:
      return GetImpl() < other.GetImpl();
  }
}

std::string SymbolicExpr::ToString(int64_t num_dims) const {
  switch (GetType()) {
    case SymbolicExprType::kConstant:
      return std::to_string(GetValue());
    case SymbolicExprType::kVariable: {
      int64_t var_id = GetValue();
      if (num_dims == -1) {
        return absl::StrCat("v", var_id);
      }
      // If num_dims is provided, then the first num_dims variables are
      // dimensions, and the rest are symbols.
      if (var_id < num_dims) {
        return absl::StrCat("d", var_id);
      }
      return absl::StrCat("s", var_id - num_dims);
    }
    case SymbolicExprType::kAdd:
    case SymbolicExprType::kMul:
    case SymbolicExprType::kFloorDiv:
    case SymbolicExprType::kCeilDiv:
    case SymbolicExprType::kMod: {
      auto bin_op_str = GetBinaryOpString(GetType());
      return absl::StrCat("(", GetLHS().ToString(num_dims), " ", bin_op_str,
                          " ", GetRHS().ToString(num_dims), ")");
    }
    case SymbolicExprType::kMax:
    case SymbolicExprType::kMin: {
      auto bin_op_str = GetBinaryOpString(GetType());
      return absl::StrCat(bin_op_str, "(", GetLHS().ToString(num_dims), ", ",
                          GetRHS().ToString(num_dims), ")");
    }
    default:
      LOG(FATAL) << "unknown type on symbolic expressions";
  }
}

int64_t SymbolicExpr::Evaluate(
    absl::Span<const int64_t> variable_values) const {
  int64_t lhs_value = GetLHS() ? GetLHS().Evaluate(variable_values) : 0;
  int64_t rhs_value = GetRHS() ? GetRHS().Evaluate(variable_values) : 0;
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

SymbolicExpr SymbolicExpr::ReplaceVariables(
    absl::Span<const SymbolicExpr> substitutions) const {
  SymbolicExprContext* ctx = GetContext();
  switch (GetType()) {
    case SymbolicExprType::kConstant:
      return *this;
    case SymbolicExprType::kVariable: {
      const VariableID var_id = GetValue();
      if (var_id >= 0 && var_id < substitutions.size() &&
          substitutions[var_id]) {
        return substitutions[var_id];
      }
      return *this;
    }
    case SymbolicExprType::kAdd:
    case SymbolicExprType::kMul:
    case SymbolicExprType::kFloorDiv:
    case SymbolicExprType::kCeilDiv:
    case SymbolicExprType::kMod:
    case SymbolicExprType::kMax:
    case SymbolicExprType::kMin: {
      SymbolicExpr new_lhs = GetLHS().ReplaceVariables(substitutions);
      SymbolicExpr new_rhs = GetRHS().ReplaceVariables(substitutions);
      if (new_lhs == GetLHS() && new_rhs == GetRHS()) {
        return *this;
      }
      return ctx->CreateBinaryOp(GetType(), new_lhs, new_rhs);
    }
    default:
      LOG(FATAL) << "Substitute not implemented for this type.";
  }
}

SymbolicExpr SymbolicExpr::ReplaceSymbols(
    absl::Span<const SymbolicExpr> sym_replacements, int64_t num_dims) const {
  return ReplaceDimsAndSymbols({}, sym_replacements, num_dims);
}

SymbolicExpr SymbolicExpr::ReplaceDimsAndSymbols(
    absl::Span<const SymbolicExpr> dim_replacements,
    absl::Span<const SymbolicExpr> symbol_replacements,
    int64_t num_dims) const {
  llvm::SmallVector<SymbolicExpr> replacements;
  replacements.append(dim_replacements.begin(), dim_replacements.end());
  for (int64_t i = dim_replacements.size(); i < num_dims; ++i) {
    replacements.push_back(GetContext()->CreateVariable(i));
  }
  replacements.append(symbol_replacements.begin(), symbol_replacements.end());
  return ReplaceVariables(replacements);
}

SymbolicExpr SymbolicExpr::Replace(SymbolicExpr expr,
                                   SymbolicExpr replacement) const {
  llvm::DenseMap<SymbolicExpr, SymbolicExpr> replacements;
  replacements[expr] = replacement;
  return Replace(replacements);
}

SymbolicExpr SymbolicExpr::Replace(
    const llvm::DenseMap<SymbolicExpr, SymbolicExpr>& replacements) const {
  auto it = replacements.find(*this);
  if (it != replacements.end()) {
    return it->second;
  }

  if (!IsBinaryOp()) {
    return *this;
  }

  SymbolicExprType type = GetType();
  SymbolicExpr lhs = GetLHS();
  SymbolicExpr rhs = GetRHS();
  SymbolicExpr new_lhs = lhs.Replace(replacements);
  SymbolicExpr new_rhs = rhs.Replace(replacements);

  if (new_lhs == lhs && new_rhs == rhs) {
    return *this;
  }
  return GetContext()->CreateBinaryOp(type, new_lhs, new_rhs);
}

void SymbolicExpr::GetUsedVariables(
    llvm::DenseSet<VariableID>& used_vars) const {
  if (!*this) {
    return;
  }

  switch (GetType()) {
    case SymbolicExprType::kConstant:
      return;
    case SymbolicExprType::kVariable:
      used_vars.insert(GetValue());
      return;
    case SymbolicExprType::kAdd:
    case SymbolicExprType::kMul:
    case SymbolicExprType::kFloorDiv:
    case SymbolicExprType::kCeilDiv:
    case SymbolicExprType::kMod:
    case SymbolicExprType::kMin:
    case SymbolicExprType::kMax:
      GetLHS().GetUsedVariables(used_vars);
      GetRHS().GetUsedVariables(used_vars);
      return;
  }
}

SymbolicExpr SymbolicExpr::Canonicalize() const {
  if (!*this) {
    return *this;
  }

  if (!IsBinaryOp()) {
    return *this;
  }

  SymbolicExprType type = GetType();
  SymbolicExpr lhs = this->GetLHS().Canonicalize();
  SymbolicExpr rhs = this->GetRHS().Canonicalize();

  // If both sides are constants, we can evaluate the expression.
  if (lhs.GetType() == SymbolicExprType::kConstant &&
      rhs.GetType() == SymbolicExprType::kConstant) {
    return GetContext()->CreateConstant(
        SymbolicExpr(GetContext()->CreateBinaryOp(type, lhs, rhs))
            .Evaluate({}));
  }

  // Assure constants are on the RHS for commutative operations.
  switch (type) {
    case SymbolicExprType::kAdd:
    case SymbolicExprType::kMul:
    case SymbolicExprType::kMin:
    case SymbolicExprType::kMax:
      if (rhs < lhs) {
        std::swap(lhs, rhs);
      }
      break;
    default:
      break;
  }

  switch (type) {
    case SymbolicExprType::kAdd:
      return CanonicalizeAdd(lhs, rhs);
    case SymbolicExprType::kMul:
      return CanonicalizeMul(lhs, rhs);
    case SymbolicExprType::kMin:
      return CanonicalizeMin(lhs, rhs);
    case SymbolicExprType::kMax:
      return CanonicalizeMax(lhs, rhs);
    case SymbolicExprType::kFloorDiv:
      return CanonicalizeFloorDiv(lhs, rhs);
    case SymbolicExprType::kCeilDiv:
      return CanonicalizeCeilDiv(lhs, rhs);
    case SymbolicExprType::kMod:
      return CanonicalizeMod(lhs, rhs);
    default:
      LOG(FATAL) << "Canonicalize not implemented for this expression type.";
  }
}

SymbolicExpr SymbolicExpr::operator+(int64_t v) const {
  return *this + GetContext()->CreateConstant(v);
}
SymbolicExpr SymbolicExpr::operator+(SymbolicExpr other) const {
  return GetContext()->CreateBinaryOp(SymbolicExprType::kAdd, *this, other);
}

SymbolicExpr SymbolicExpr::operator-() const {
  return (*this * GetContext()->CreateConstant(-1)).Canonicalize();
}
SymbolicExpr SymbolicExpr::operator-(int64_t v) const { return *this + (-v); }
SymbolicExpr SymbolicExpr::operator-(SymbolicExpr other) const {
  return *this + (-other);
}

SymbolicExpr SymbolicExpr::operator*(int64_t v) const {
  return *this * GetContext()->CreateConstant(v);
}
SymbolicExpr SymbolicExpr::operator*(SymbolicExpr other) const {
  return GetContext()->CreateBinaryOp(SymbolicExprType::kMul, *this, other);
}

SymbolicExpr SymbolicExpr::operator%(int64_t v) const {
  return this->operator%(GetContext()->CreateConstant(v));
}
SymbolicExpr SymbolicExpr::operator%(SymbolicExpr other) const {
  return GetContext()->CreateBinaryOp(SymbolicExprType::kMod, *this, other);
}

SymbolicExpr SymbolicExpr::floorDiv(int64_t v) const {
  return this->floorDiv(GetContext()->CreateConstant(v));
}
SymbolicExpr SymbolicExpr::floorDiv(SymbolicExpr other) const {
  return GetContext()->CreateBinaryOp(SymbolicExprType::kFloorDiv, *this,
                                      other);
}

SymbolicExpr SymbolicExpr::ceilDiv(int64_t v) const {
  return this->ceilDiv(GetContext()->CreateConstant(v));
}
SymbolicExpr SymbolicExpr::ceilDiv(SymbolicExpr other) const {
  return GetContext()->CreateBinaryOp(SymbolicExprType::kCeilDiv, *this, other);
}

SymbolicExpr SymbolicExpr::min(int64_t v) const {
  return this->min(GetContext()->CreateConstant(v));
}
SymbolicExpr SymbolicExpr::min(SymbolicExpr other) const {
  return GetContext()->CreateBinaryOp(SymbolicExprType::kMin, *this, other);
}

SymbolicExpr SymbolicExpr::max(int64_t v) const {
  return this->max(GetContext()->CreateConstant(v));
}
SymbolicExpr SymbolicExpr::max(SymbolicExpr other) const {
  return GetContext()->CreateBinaryOp(SymbolicExprType::kMax, *this, other);
}

static absl::Mutex& getSymbolicExprStorageMutex() {
  static absl::Mutex m(absl::kConstInit);
  return m;
}

SymbolicExprContext::SymbolicExprContext(mlir::MLIRContext* mlir_context)
    : mlir_context_(mlir_context) {
  CHECK(mlir_context != nullptr);
  absl::MutexLock lock(getSymbolicExprStorageMutex());
  auto* uniquer = &mlir_context_->getAffineUniquer();
  if (!uniquer->isParametricStorageInitialized(
          mlir::TypeID::get<SymbolicExprStorage>())) {
    uniquer->registerParametricStorageType<SymbolicExprStorage>();
  }
}

SymbolicExpr SymbolicExprContext::GetOrCreate(SymbolicExprType type,
                                              int64_t value, SymbolicExpr lhs,
                                              SymbolicExpr rhs) {
  auto initContext = [&](SymbolicExprStorage* storage) {
    storage->ctx_ = this;
  };
  return mlir_context_->getAffineUniquer().get<SymbolicExprStorage>(
      initContext, type, value, lhs, rhs);
}

SymbolicExpr SymbolicExprContext::CreateConstant(int64_t value) {
  return GetOrCreate(SymbolicExprType::kConstant, value, SymbolicExpr(),
                     SymbolicExpr());
}

SymbolicExpr SymbolicExprContext::CreateVariable(int64_t var_id) {
  return GetOrCreate(SymbolicExprType::kVariable, var_id, SymbolicExpr(),
                     SymbolicExpr());
}

SymbolicExpr SymbolicExprContext::CreateBinaryOp(SymbolicExprType type,
                                                 SymbolicExpr lhs,
                                                 SymbolicExpr rhs) {
  CHECK(type != SymbolicExprType::kConstant &&
        type != SymbolicExprType::kVariable && lhs && rhs)
      << "We expect a binary operation and two symbolic expressions as "
         "children.";
  return GetOrCreate(type, 0, lhs, rhs);
}

SymbolicExpr SymbolicExprContext::Parse(absl::string_view expr_str) {
  return Parser(expr_str, this).Parse();
}

bool SymbolicExprContext::operator==(const SymbolicExprContext& other) const {
  return mlir_context_ == other.mlir_context_;
}

void SymbolicExpr::Walk(
    const std::function<void(SymbolicExpr)>& callback) const {
  if (!*this) {
    return;
  }

  switch (GetType()) {
    case SymbolicExprType::kConstant:
    case SymbolicExprType::kVariable:
      break;
    case SymbolicExprType::kAdd:
    case SymbolicExprType::kMul:
    case SymbolicExprType::kFloorDiv:
    case SymbolicExprType::kCeilDiv:
    case SymbolicExprType::kMod:
    case SymbolicExprType::kMin:
    case SymbolicExprType::kMax:
      GetLHS().Walk(callback);
      GetRHS().Walk(callback);
      break;
  }
  callback(*this);
}

}  // namespace xla

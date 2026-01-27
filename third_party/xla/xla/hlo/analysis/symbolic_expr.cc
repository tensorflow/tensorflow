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
#include <iterator>
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
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
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

bool IsVariableCharacter(char c) { return absl::ascii_isalnum(c) || c == '_'; }

// Helper class to manage the state of the parser.
class Parser {
 public:
  Parser(absl::string_view str, mlir::MLIRContext* context,
         std::optional<int64_t> num_dims = std::nullopt)
      : remaining_str_(str), context_(context), num_dims_(num_dims) {}

  Parser(absl::string_view str, mlir::MLIRContext* context,
         const llvm::DenseMap<llvm::StringRef, SymbolicExpr>* variable_map)
      : remaining_str_(str), context_(context), variable_map_(variable_map) {}

  SymbolicExpr Parse() {
    SymbolicExpr expr = ParseExpression();
    SkipWhitespace();
    if (expr && !remaining_str_.empty()) {
      return ReportError("Did not parse entire string");
    }
    return expr;
  }

  SymbolicExpr ParsePartial() { return ParseExpression(); }

  absl::string_view GetRemainingStr() const { return remaining_str_; }

 private:
  // TODO: b/459357586 - Consider returning StatusOr instead of failing
  // gracefully by returning an empty SymbolicExpr.
  SymbolicExpr ReportError(absl::string_view msg) {
    LOG(ERROR) << msg << " at: \"" << remaining_str_ << "\"";
    return SymbolicExpr();
  }

  std::optional<int64_t> ParseNumber() {
    size_t num_len = 0;
    if (!remaining_str_.empty() &&
        (absl::ascii_isdigit(remaining_str_[0]) || remaining_str_[0] == '-')) {
      num_len = 1;
    }
    while (num_len < remaining_str_.size() &&
           absl::ascii_isdigit(remaining_str_[num_len])) {
      num_len++;
    }
    if (num_len == 0) {
      return std::nullopt;
    }
    int64_t number;
    if (!absl::SimpleAtoi(remaining_str_.substr(0, num_len), &number)) {
      return std::nullopt;
    }
    remaining_str_.remove_prefix(num_len);
    return number;
  }

  // Handles lowest precedence operators: +
  SymbolicExpr ParseExpression() {
    SymbolicExpr lhs = ParseTerm();
    if (!lhs) {
      return lhs;
    }
    while (true) {
      SkipWhitespace();
      if (absl::ConsumePrefix(&remaining_str_, "+")) {
        SymbolicExpr rhs = ParseTerm();
        if (!rhs) {
          return SymbolicExpr();
        }
        lhs =
            CreateSymbolicBinaryOp(SymbolicExprType::kAdd, lhs, rhs, context_);
      } else if (absl::ConsumePrefix(&remaining_str_, "-")) {
        SymbolicExpr rhs = ParseTerm();
        if (!rhs) {
          return SymbolicExpr();
        }
        lhs = lhs - rhs;
      } else {
        break;
      }
    }
    return lhs;
  }

  // Handles higher precedence operators: *, floordiv, ceildiv
  SymbolicExpr ParseTerm() {
    SymbolicExpr lhs = ParseFactor();
    if (!lhs) {
      return lhs;
    }
    while (true) {
      SkipWhitespace();
      if (absl::ConsumePrefix(&remaining_str_, "*")) {
        SymbolicExpr rhs = ParseFactor();
        if (!rhs) {
          return SymbolicExpr();
        }
        lhs =
            CreateSymbolicBinaryOp(SymbolicExprType::kMul, lhs, rhs, context_);
      } else if (absl::ConsumePrefix(&remaining_str_, "floordiv")) {
        SymbolicExpr rhs = ParseFactor();
        if (!rhs) {
          return SymbolicExpr();
        }
        lhs = CreateSymbolicBinaryOp(SymbolicExprType::kFloorDiv, lhs, rhs,
                                     context_);
      } else if (absl::ConsumePrefix(&remaining_str_, "ceildiv")) {
        SymbolicExpr rhs = ParseFactor();
        if (!rhs) {
          return SymbolicExpr();
        }
        lhs = CreateSymbolicBinaryOp(SymbolicExprType::kCeilDiv, lhs, rhs,
                                     context_);
      } else if (absl::ConsumePrefix(&remaining_str_, "mod")) {
        SymbolicExpr rhs = ParseFactor();
        if (!rhs) {
          return SymbolicExpr();
        }
        lhs =
            CreateSymbolicBinaryOp(SymbolicExprType::kMod, lhs, rhs, context_);
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
    if (!lhs) {
      return SymbolicExpr();
    }
    SkipWhitespace();
    if (!absl::ConsumePrefix(&remaining_str_, ",")) {
      return ReportError("Missing ',' in " + func_name + "()");
    }
    SymbolicExpr rhs = ParseExpression();
    if (!rhs) {
      return SymbolicExpr();
    }
    SkipWhitespace();
    if (!absl::ConsumePrefix(&remaining_str_, ")")) {
      return ReportError("Missing ')' in " + func_name + "()");
    }
    return CreateSymbolicBinaryOp(type, lhs, rhs, context_);
  }

  SymbolicExpr MaybeParseVariableFromMap() {
    auto it = absl::c_find_if_not(remaining_str_, IsVariableCharacter);
    size_t var_len = std::distance(remaining_str_.begin(), it);
    if (var_len > 0) {
      auto var_it =
          variable_map_->find(llvm::StringRef(remaining_str_.data(), var_len));
      if (var_it != variable_map_->end()) {
        remaining_str_.remove_prefix(var_len);
        return var_it->second;
      }
    }
    return SymbolicExpr();
  }

  SymbolicExpr MaybeParseDimAndSymbolVariables() {
    if (absl::ConsumePrefix(&remaining_str_, "v") ||
        absl::ConsumePrefix(&remaining_str_, "d")) {
      std::optional<int64_t> var_id = ParseNumber();
      if (!var_id.has_value()) {
        return ReportError("Cannot parse variable id");
      }
      return CreateSymbolicVariable(var_id.value(), context_);
    }
    if (absl::ConsumePrefix(&remaining_str_, "s")) {
      if (!num_dims_.has_value()) {
        return ReportError(
            "Symbol cannot be parsed because number of dimensions is not set.");
      }
      std::optional<int64_t> sym_id = ParseNumber();
      // We need to know the number of dimensions to determine a symbol id.
      if (!sym_id.has_value()) {
        return ReportError("Cannot parse symbol id after 's'");
      }
      return CreateSymbolicVariable(num_dims_.value() + sym_id.value(),
                                    context_);
    }
    return SymbolicExpr();
  }

  // Handles highest precedence items: numbers, variables, and functions.
  SymbolicExpr ParseFactor() {
    SkipWhitespace();
    if (remaining_str_.empty()) {
      return ReportError("Unexpected end of expression");
    }

    // Case 1:Function call like max( ... ) or min( ... )
    if (absl::StartsWith(remaining_str_, "max(")) {
      return ParseBinaryFunction(SymbolicExprType::kMax);
    }
    if (absl::StartsWith(remaining_str_, "min(")) {
      return ParseBinaryFunction(SymbolicExprType::kMin);
    }
    // Case 2: Parenthesized subexpression
    if (absl::ConsumePrefix(&remaining_str_, "(")) {
      SymbolicExpr expr = ParseExpression();
      if (!expr) {
        return SymbolicExpr();
      }
      SkipWhitespace();
      if (!absl::ConsumePrefix(&remaining_str_, ")")) {
        return ReportError("Missing parenthesis");
      }
      return expr;
    }

    // Case 3: Variables from map. If `variable_map_` is provided, variables in
    // it are checked before standard variables ('d', 's', 'v').
    if (variable_map_ != nullptr) {
      if (SymbolicExpr expr = MaybeParseVariableFromMap()) {
        return expr;
      }
    }

    // Case 4: Variable (e.g., "v123", "d0", "s0")
    char c = remaining_str_[0];
    if (c == 'v' || c == 'd' || c == 's') {
      return MaybeParseDimAndSymbolVariables();
    }

    // Case 5: Number
    std::optional<int64_t> val = ParseNumber();
    if (val.has_value()) {
      return CreateSymbolicConstant(val.value(), context_);
    }
    return ReportError("Failed to parse expression");
  }

  void SkipWhitespace() {
    remaining_str_ = absl::StripLeadingAsciiWhitespace(remaining_str_);
  }

  absl::string_view remaining_str_;
  mlir::MLIRContext* context_;
  std::optional<int64_t> num_dims_;
  const llvm::DenseMap<llvm::StringRef, SymbolicExpr>* variable_map_ = nullptr;
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

// TODO(b/459357586): Remove this function and use CanonicalizeAdd instead.
SymbolicExpr BasicAddSimplify(SymbolicExpr lhs, SymbolicExpr rhs) {
  if (rhs.GetType() == SymbolicExprType::kConstant && rhs.GetValue() == 0) {
    return lhs;
  }
  if (lhs.GetType() == SymbolicExprType::kConstant && lhs.GetValue() == 0) {
    return rhs;
  }
  return CreateSymbolicBinaryOp(SymbolicExprType::kAdd, lhs, rhs,
                                lhs.GetContext());
}

SymbolicExpr CanonicalizeAdd(SymbolicExpr lhs, SymbolicExpr rhs) {
  mlir::MLIRContext* ctx = lhs.GetContext();

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
    exprs.push_back(CreateSymbolicConstant(const_val, ctx));
  }
  if (exprs.empty()) {
    return CreateSymbolicConstant(0, ctx);
  }
  // Sort all terms, including the constant
  absl::c_sort(exprs);

  SymbolicExpr result = exprs[0];
  for (size_t i = 1; i < exprs.size(); ++i) {
    result =
        CreateSymbolicBinaryOp(SymbolicExprType::kAdd, result, exprs[i], ctx);
  }
  return result;
}

// Helper to simplify multiplication when the RHS is a constant.
SymbolicExpr SimplifyMulByConstantRHS(SymbolicExpr lhs, SymbolicExpr rhs) {
  if (rhs.GetType() != SymbolicExprType::kConstant) {
    return SymbolicExpr();
  }
  int64_t rhs_val = rhs.GetValue();
  mlir::MLIRContext* ctx = lhs.GetContext();

  if (rhs_val == 0) {
    return rhs;  // x * 0 = 0
  }
  if (rhs_val == 1) {
    return lhs;  // x * 1 = x
  }

  // Associativity: (X * C1) * C2 => X * (C1 * C2)
  if (lhs.GetType() == SymbolicExprType::kMul &&
      lhs.GetRHS().GetType() == SymbolicExprType::kConstant) {
    return CreateSymbolicBinaryOp(
        SymbolicExprType::kMul, lhs.GetLHS(),
        CreateSymbolicConstant(lhs.GetRHS().GetValue() * rhs_val, ctx), ctx);
  }
  return SymbolicExpr();
}

SymbolicExpr BasicMulSimplify(SymbolicExpr lhs, SymbolicExpr rhs) {
  mlir::MLIRContext* ctx = lhs.GetContext();

  // Try constant folding, neutral element simplification, and associativity.
  if (rhs.GetType() == SymbolicExprType::kConstant) {
    SymbolicExpr simplified = SimplifyMulByConstantRHS(lhs, rhs);
    if (simplified) {
      return simplified;
    }
  } else if (lhs.GetType() == SymbolicExprType::kConstant) {
    SymbolicExpr simplified = SimplifyMulByConstantRHS(rhs, lhs);
    if (simplified) {
      return simplified;
    }
  }

  return CreateSymbolicBinaryOp(SymbolicExprType::kMul, lhs, rhs, ctx);
}

SymbolicExpr CanonicalizeMul(SymbolicExpr lhs, SymbolicExpr rhs) {
  mlir::MLIRContext* ctx = lhs.GetContext();

  if (rhs.GetType() == SymbolicExprType::kConstant) {
    // Try constant folding, neutral element simplification, and associativity.
    SymbolicExpr simplified = SimplifyMulByConstantRHS(lhs, rhs);
    if (simplified) {
      if (simplified.GetType() == SymbolicExprType::kConstant ||
          simplified == lhs) {
        return simplified;
      }
      return simplified.Canonicalize();
    }
  }

  // Distribute Mul over Add: (A + B) * C => (A * C) + (B * C)
  if (lhs.GetType() == SymbolicExprType::kAdd) {
    return ((lhs.GetLHS() * rhs) + (lhs.GetRHS() * rhs)).Canonicalize();
  }
  if (rhs.GetType() == SymbolicExprType::kAdd) {
    return ((lhs * rhs.GetLHS()) + (lhs * rhs.GetRHS())).Canonicalize();
  }

  return CreateSymbolicBinaryOp(SymbolicExprType::kMul, lhs, rhs, ctx);
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
  mlir::MLIRContext* ctx = lhs.GetContext();
  if (auto diff = SubtractAndGetConstDiff(lhs, rhs)) {  // min(X, X + k) = X
    return (diff.value() <= 0) ? lhs : rhs;
  }

  return CreateSymbolicBinaryOp(SymbolicExprType::kMin, lhs, rhs, ctx);
}

SymbolicExpr CanonicalizeMax(SymbolicExpr lhs, SymbolicExpr rhs) {
  mlir::MLIRContext* ctx = lhs.GetContext();
  if (auto diff = SubtractAndGetConstDiff(lhs, rhs)) {  // max(X, X + k) = X + k
    return (diff.value() >= 0) ? lhs : rhs;
  }

  return CreateSymbolicBinaryOp(SymbolicExprType::kMax, lhs, rhs, ctx);
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
    remaining_expr = CreateSymbolicConstant(1, a.GetContext());
  } else {
    return SymbolicExpr();  // Cannot simplify
  }

  if (a_coeff % div != 0) {
    return SymbolicExpr();  // Cannot simplify
  }
  return (remaining_expr * (a_coeff / div) + b / div).Canonicalize();
}

SymbolicExpr CanonicalizeFloorDiv(SymbolicExpr lhs, SymbolicExpr rhs) {
  mlir::MLIRContext* ctx = lhs.GetContext();

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

  return CreateSymbolicBinaryOp(SymbolicExprType::kFloorDiv, lhs, rhs, ctx);
}

SymbolicExpr CanonicalizeCeilDiv(SymbolicExpr lhs, SymbolicExpr rhs) {
  mlir::MLIRContext* ctx = lhs.GetContext();

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

  return CreateSymbolicBinaryOp(SymbolicExprType::kCeilDiv, lhs, rhs, ctx);
}

SymbolicExpr CanonicalizeMod(SymbolicExpr lhs, SymbolicExpr rhs) {
  mlir::MLIRContext* ctx = lhs.GetContext();

  if (lhs.GetType() == SymbolicExprType::kConstant && lhs.GetValue() == 0) {
    return lhs;  // 0 mod X => 0
  }

  if (lhs == rhs) {
    return CreateSymbolicConstant(0, ctx);  // X mod X => 0
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
  friend SymbolicExpr GetOrCreateSymbolicExpr(SymbolicExprType type,
                                              int64_t value, SymbolicExpr lhs,
                                              SymbolicExpr rhs,
                                              mlir::MLIRContext* mlir_context);

  SymbolicExprType type_;
  int64_t value_ = 0;
  SymbolicExpr lhs_;
  SymbolicExpr rhs_;
  mlir::MLIRContext* mlir_context_ = nullptr;

 private:
  SymbolicExprStorage(SymbolicExprType type, int64_t value)
      : type_(type), value_(value) {}
  SymbolicExprStorage(SymbolicExprType type, SymbolicExpr lhs, SymbolicExpr rhs)
      : type_(type), lhs_(lhs), rhs_(rhs) {}
};

mlir::MLIRContext* SymbolicExpr::GetContext() const {
  return impl_->mlir_context_;
}

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
  mlir::MLIRContext* ctx = GetContext();
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
      return CreateSymbolicBinaryOp(GetType(), new_lhs, new_rhs, ctx);
    }
    default:
      LOG(FATAL) << "Substitute not implemented for this type.";
  }
}

SymbolicExpr SymbolicExpr::ReplaceSymbols(
    absl::Span<const SymbolicExpr> sym_replacements, int64_t num_dims) const {
  llvm::SmallVector<SymbolicExpr> dim_replacements;
  dim_replacements.reserve(num_dims);
  for (int64_t i = 0; i < num_dims; ++i) {
    dim_replacements.push_back(CreateSymbolicVariable(i, GetContext()));
  }
  return ReplaceDimsAndSymbols(dim_replacements, sym_replacements);
}

SymbolicExpr SymbolicExpr::ReplaceDimsAndSymbols(
    absl::Span<const SymbolicExpr> dim_replacements,
    absl::Span<const SymbolicExpr> symbol_replacements) const {
  llvm::SmallVector<SymbolicExpr> replacements;
  replacements.append(dim_replacements.begin(), dim_replacements.end());
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
  return CreateSymbolicBinaryOp(type, new_lhs, new_rhs, GetContext());
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
    return CreateSymbolicConstant(
        SymbolicExpr(CreateSymbolicBinaryOp(type, lhs, rhs, GetContext()))
            .Evaluate({}),
        GetContext());
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
  return *this + CreateSymbolicConstant(v, GetContext());
}
SymbolicExpr SymbolicExpr::operator+(SymbolicExpr other) const {
  // TODO(b/433693782): We should use our own canonicalization here instead of
  // relying on a similar one to AffineMap so tests do not fail.
  return BasicAddSimplify(*this, other);
}

SymbolicExpr SymbolicExpr::operator-() const {
  return (*this * CreateSymbolicConstant(-1, GetContext())).Canonicalize();
}
SymbolicExpr SymbolicExpr::operator-(int64_t v) const { return *this + (-v); }
SymbolicExpr SymbolicExpr::operator-(SymbolicExpr other) const {
  return *this + (-other);
}

SymbolicExpr SymbolicExpr::operator*(int64_t v) const {
  return *this * CreateSymbolicConstant(v, GetContext());
}
SymbolicExpr SymbolicExpr::operator*(SymbolicExpr other) const {
  // TODO(b/433693782): We should use our own canonicalization here instead of
  // relying on a similar one to AffineMap so tests do not fail.
  return BasicMulSimplify(*this, other);
}

SymbolicExpr SymbolicExpr::operator%(int64_t v) const {
  return this->operator%(CreateSymbolicConstant(v, GetContext()));
}
SymbolicExpr SymbolicExpr::operator%(SymbolicExpr other) const {
  return CreateSymbolicBinaryOp(SymbolicExprType::kMod, *this, other,
                                GetContext());
}

SymbolicExpr SymbolicExpr::floorDiv(int64_t v) const {
  return this->floorDiv(CreateSymbolicConstant(v, GetContext()));
}
SymbolicExpr SymbolicExpr::floorDiv(SymbolicExpr other) const {
  return CreateSymbolicBinaryOp(SymbolicExprType::kFloorDiv, *this, other,
                                GetContext());
}

SymbolicExpr SymbolicExpr::ceilDiv(int64_t v) const {
  return this->ceilDiv(CreateSymbolicConstant(v, GetContext()));
}
SymbolicExpr SymbolicExpr::ceilDiv(SymbolicExpr other) const {
  return CreateSymbolicBinaryOp(SymbolicExprType::kCeilDiv, *this, other,
                                GetContext());
}

SymbolicExpr SymbolicExpr::min(int64_t v) const {
  return this->min(CreateSymbolicConstant(v, GetContext()));
}
SymbolicExpr SymbolicExpr::min(SymbolicExpr other) const {
  return CreateSymbolicBinaryOp(SymbolicExprType::kMin, *this, other,
                                GetContext());
}

SymbolicExpr SymbolicExpr::max(int64_t v) const {
  return this->max(CreateSymbolicConstant(v, GetContext()));
}
SymbolicExpr SymbolicExpr::max(SymbolicExpr other) const {
  return CreateSymbolicBinaryOp(SymbolicExprType::kMax, *this, other,
                                GetContext());
}

static absl::Mutex& getSymbolicExprStorageMutex() {
  static absl::Mutex m(absl::kConstInit);
  return m;
}

void RegisterSymbolicExprStorage(mlir::MLIRContext* mlir_context) {
  CHECK(mlir_context != nullptr);
  auto* uniquer = &mlir_context->getAffineUniquer();
  {
    absl::MutexLock lock(getSymbolicExprStorageMutex());
    if (!uniquer->isParametricStorageInitialized(
            mlir::TypeID::get<SymbolicExprStorage>())) {
      uniquer->registerParametricStorageType<SymbolicExprStorage>();
    }
  }
}

SymbolicExpr GetOrCreateSymbolicExpr(SymbolicExprType type, int64_t value,
                                     SymbolicExpr lhs, SymbolicExpr rhs,
                                     mlir::MLIRContext* mlir_context) {
  auto* uniquer = &mlir_context->getAffineUniquer();
  auto initContext = [&](SymbolicExprStorage* storage) {
    storage->mlir_context_ = mlir_context;
  };
  return uniquer->get<SymbolicExprStorage>(initContext, type, value, lhs, rhs);
}

SymbolicExpr CreateSymbolicConstant(int64_t value,
                                    mlir::MLIRContext* mlir_context) {
  return GetOrCreateSymbolicExpr(SymbolicExprType::kConstant, value,
                                 SymbolicExpr(), SymbolicExpr(), mlir_context);
}

SymbolicExpr CreateSymbolicVariable(int64_t var_id,
                                    mlir::MLIRContext* mlir_context) {
  return GetOrCreateSymbolicExpr(SymbolicExprType::kVariable, var_id,
                                 SymbolicExpr(), SymbolicExpr(), mlir_context);
}

SymbolicExpr CreateSymbolicBinaryOp(SymbolicExprType type, SymbolicExpr lhs,
                                    SymbolicExpr rhs,
                                    mlir::MLIRContext* mlir_context) {
  CHECK(type != SymbolicExprType::kConstant &&
        type != SymbolicExprType::kVariable && lhs && rhs)
      << "We expect a binary operation and two symbolic expressions as "
         "children.";
  auto result = GetOrCreateSymbolicExpr(type, 0, lhs, rhs, mlir_context);
  // Basic constant folding.
  if (lhs.GetType() == SymbolicExprType::kConstant &&
      rhs.GetType() == SymbolicExprType::kConstant) {
    return CreateSymbolicConstant(result.Evaluate({}), mlir_context);
  }
  return result;
}

llvm::SmallVector<SymbolicExpr> CreateSymbolicConstantExprs(
    llvm::ArrayRef<int64_t> constants, mlir::MLIRContext* context) {
  llvm::SmallVector<SymbolicExpr> exprs;
  exprs.reserve(constants.size());
  for (int64_t constant : constants) {
    exprs.push_back(CreateSymbolicConstant(constant, context));
  }
  return exprs;
}
SymbolicExpr ParseSymbolicExpr(absl::string_view expr_str,
                               mlir::MLIRContext* mlir_context,
                               std::optional<int64_t> num_dims) {
  return ParseSymbolicExprAndAdvance(&expr_str, mlir_context, num_dims);
}

SymbolicExpr ParseSymbolicExprAndAdvance(absl::string_view* expr_str,
                                         mlir::MLIRContext* mlir_context,
                                         std::optional<int64_t> num_dims) {
  Parser parser(*expr_str, mlir_context, num_dims);
  SymbolicExpr expr = parser.ParsePartial();
  *expr_str = parser.GetRemainingStr();
  return expr;
}

SymbolicExpr ParseSymbolicExprAndAdvance(
    absl::string_view* expr_str, mlir::MLIRContext* mlir_context,
    const llvm::DenseMap<llvm::StringRef, SymbolicExpr>& variable_map) {
  Parser parser(*expr_str, mlir_context, &variable_map);
  SymbolicExpr expr = parser.ParsePartial();
  *expr_str = parser.GetRemainingStr();
  return expr;
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

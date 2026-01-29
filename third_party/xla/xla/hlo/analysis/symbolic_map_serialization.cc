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

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"

namespace xla {
namespace {

bool IsIdentifierCharacter(char c) {
  return absl::ascii_isalnum(c) || c == '_';
}

// Helper class to manage the state of the SymbolicExpr parser.
class SymbolicExprParserImpl {
 public:
  SymbolicExprParserImpl(absl::string_view str, mlir::MLIRContext* context,
                         std::optional<int64_t> num_dims = std::nullopt)
      : remaining_str_(str), context_(context), num_dims_(num_dims) {}

  SymbolicExprParserImpl(
      absl::string_view str, mlir::MLIRContext* context,
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
    auto it = absl::c_find_if_not(remaining_str_, IsIdentifierCharacter);
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

// Helper class to manage the state of the SymbolicMap parser.
class SymbolicMapParserImpl {
 public:
  SymbolicMapParserImpl(absl::string_view str, mlir::MLIRContext* context)
      : remaining_str_(str), context_(context) {}

  SymbolicMap Parse() {
    std::optional<int64_t> num_dims_opt =
        ParseArgList("(", ")", /*is_dim=*/true);
    if (!num_dims_opt.has_value()) {
      return ReportError(
          "Failed to parse dimension list in SymbolicMap string");
    }
    num_dims_ = num_dims_opt.value();

    std::optional<int64_t> num_symbols_opt = 0;
    SkipWhitespace();
    if (absl::StartsWith(remaining_str_, "[")) {
      num_symbols_opt = ParseArgList("[", "]", /*is_dim=*/false);
      if (!num_symbols_opt.has_value()) {
        return ReportError("Failed to parse symbol list in SymbolicMap string");
      }
    }
    num_symbols_ = num_symbols_opt.value();

    SkipWhitespace();
    if (!absl::ConsumePrefix(&remaining_str_, "->")) {
      return ReportError("Failed to parse SymbolicMap string: missing `->`");
    }

    auto exprs = ParseExprList();
    if (!exprs.has_value()) {
      return ReportError(
          "Failed to parse expression list in SymbolicMap string");
    }

    SkipWhitespace();
    if (!remaining_str_.empty()) {
      return ReportError(
          "Unexpected characters at the end of SymbolicMap string");
    }

    return SymbolicMap::Get(context_, num_dims_, num_symbols_, exprs.value());
  }

 private:
  // Logs the error message and returns an empty SymbolicMap similarly to the
  // ParseSymbolicExpr function.
  SymbolicMap ReportError(absl::string_view msg) {
    LOG(ERROR) << msg << " at: \"" << remaining_str_ << "\"";
    return SymbolicMap();
  }

  // Parses an identifier and removes it from remaining_str_. Returns nullopt if
  // parsing fails.
  std::optional<absl::string_view> ParseIdentifier() {
    auto it = absl::c_find_if_not(remaining_str_, IsIdentifierCharacter);
    size_t len = std::distance(remaining_str_.begin(), it);
    if (len == 0) {
      return std::nullopt;
    }
    absl::string_view name = remaining_str_.substr(0, len);
    remaining_str_.remove_prefix(len);
    return name;
  }

  // Parses a list of identifiers and removes it from remaining_str_. Returns a
  // count of the number of identifiers parsed. If is_dim is true, the
  // identifiers are interpreted as dimension indices, otherwise they are
  // interpreted as symbol indices.
  std::optional<int64_t> ParseArgList(absl::string_view open,
                                      absl::string_view close, bool is_dim) {
    SkipWhitespace();
    if (!absl::ConsumePrefix(&remaining_str_, open)) {
      return std::nullopt;
    }
    SkipWhitespace();
    if (absl::ConsumePrefix(&remaining_str_, close)) {
      return 0;
    }

    int64_t count = 0;
    while (true) {
      std::optional<absl::string_view> name = ParseIdentifier();
      if (!name.has_value()) {
        return std::nullopt;
      }

      // Update the variable map with the parsed identifier.
      if (is_dim) {
        variable_map_[llvm::StringRef(*name)] =
            CreateSymbolicVariable(count, context_);
      } else {
        variable_map_[llvm::StringRef(*name)] =
            CreateSymbolicVariable(num_dims_ + count, context_);
      }
      count++;

      SkipWhitespace();
      if (absl::ConsumePrefix(&remaining_str_, close)) {
        return count;
      }
      if (!absl::ConsumePrefix(&remaining_str_, ",")) {
        return std::nullopt;
      }
      SkipWhitespace();
    }
  }

  std::optional<llvm::SmallVector<SymbolicExpr>> ParseExprList() {
    SkipWhitespace();
    if (!absl::ConsumePrefix(&remaining_str_, "(")) {
      return std::nullopt;
    }

    llvm::SmallVector<SymbolicExpr> exprs;
    SkipWhitespace();
    if (absl::ConsumePrefix(&remaining_str_, ")")) {
      return exprs;
    }

    while (true) {
      SymbolicExpr expr =
          ParseSymbolicExprAndAdvance(&remaining_str_, context_, variable_map_);
      if (!expr) {
        return std::nullopt;
      }
      exprs.push_back(expr);
      SkipWhitespace();
      if (absl::ConsumePrefix(&remaining_str_, ")")) {
        return exprs;
      }
      if (!absl::ConsumePrefix(&remaining_str_, ",")) {
        return std::nullopt;
      }
      SkipWhitespace();
    }
  }

  void SkipWhitespace() {
    remaining_str_ = absl::StripLeadingAsciiWhitespace(remaining_str_);
  }

  absl::string_view remaining_str_;
  mlir::MLIRContext* context_;
  int64_t num_dims_ = 0;
  int64_t num_symbols_ = 0;
  llvm::DenseMap<llvm::StringRef, SymbolicExpr> variable_map_;
};

}  // namespace

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

void Print(SymbolicExpr expr, llvm::raw_ostream& os, int64_t num_dims) {
  switch (expr.GetType()) {
    case SymbolicExprType::kConstant:
      os << expr.GetValue();
      return;
    case SymbolicExprType::kVariable: {
      int64_t var_id = expr.GetValue();
      if (num_dims == -1) {
        os << "v" << var_id;
        return;
      }
      // If num_dims is provided, then the first num_dims variables are
      // dimensions, and the rest are symbols.
      if (var_id < num_dims) {
        os << "d" << var_id;
      } else {
        os << "s" << (var_id - num_dims);
      }
      return;
    }
    case SymbolicExprType::kAdd:
    case SymbolicExprType::kMul:
    case SymbolicExprType::kFloorDiv:
    case SymbolicExprType::kCeilDiv:
    case SymbolicExprType::kMod: {
      auto bin_op_str = GetBinaryOpString(expr.GetType());
      os << "(";
      Print(expr.GetLHS(), os, num_dims);
      os << " " << bin_op_str << " ";
      Print(expr.GetRHS(), os, num_dims);
      os << ")";
      return;
    }
    case SymbolicExprType::kMax:
    case SymbolicExprType::kMin: {
      auto bin_op_str = GetBinaryOpString(expr.GetType());
      os << bin_op_str << "(";
      Print(expr.GetLHS(), os, num_dims);
      os << ", ";
      Print(expr.GetRHS(), os, num_dims);
      os << ")";
      return;
    }
    default:
      LOG(FATAL) << "unknown type on symbolic expressions";
  }
}

void Print(const SymbolicMap& map, llvm::raw_ostream& os) {
  os << "(";
  for (int i = 0; i < map.GetNumDims(); ++i) {
    os << (i > 0 ? ", " : "") << "d" << i;
  }
  os << ")[";
  for (int i = 0; i < map.GetNumSymbols(); ++i) {
    os << (i > 0 ? ", " : "") << "s" << i;
  }
  os << "] -> (";
  for (int i = 0; i < map.GetResults().size(); ++i) {
    if (i > 0) {
      os << ", ";
    }
    Print(map.GetResult(i), os, map.GetNumDims());
  }
  os << ")";
}

SymbolicMap ParseSymbolicMap(absl::string_view serialized_symbolic_map,
                             mlir::MLIRContext* mlir_context) {
  return SymbolicMapParserImpl(serialized_symbolic_map, mlir_context).Parse();
}

SymbolicExpr ParseSymbolicExpr(absl::string_view expr_str,
                               mlir::MLIRContext* mlir_context,
                               std::optional<int64_t> num_dims) {
  return ParseSymbolicExprAndAdvance(&expr_str, mlir_context, num_dims);
}

SymbolicExpr ParseSymbolicExprAndAdvance(absl::string_view* expr_str,
                                         mlir::MLIRContext* mlir_context,
                                         std::optional<int64_t> num_dims) {
  SymbolicExprParserImpl parser(*expr_str, mlir_context, num_dims);
  SymbolicExpr expr = parser.ParsePartial();
  *expr_str = parser.GetRemainingStr();
  return expr;
}

SymbolicExpr ParseSymbolicExprAndAdvance(
    absl::string_view* expr_str, mlir::MLIRContext* mlir_context,
    const llvm::DenseMap<llvm::StringRef, SymbolicExpr>& variable_map) {
  SymbolicExprParserImpl parser(*expr_str, mlir_context, &variable_map);
  SymbolicExpr expr = parser.ParsePartial();
  *expr_str = parser.GetRemainingStr();
  return expr;
}

}  // namespace xla

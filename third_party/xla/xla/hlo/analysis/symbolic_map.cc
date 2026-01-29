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

#include "xla/hlo/analysis/symbolic_map.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"

namespace xla {
namespace {

bool IsIdentifierCharacter(char c) {
  return absl::ascii_isalnum(c) || c == '_';
}

llvm::SmallVector<SymbolicExpr> CreateVariableRange(mlir::MLIRContext* ctx,
                                                    int64_t n,
                                                    int64_t offset = 0) {
  llvm::SmallVector<SymbolicExpr> replacements;
  replacements.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    replacements.push_back(CreateSymbolicVariable(offset + i, ctx));
  }
  return replacements;
}

llvm::DenseSet<VariableID> GetUsedVariablesFromExpressions(
    const SymbolicMap& map) {
  llvm::DenseSet<VariableID> used_vars;
  for (const auto& expr : map.GetResults()) {
    expr.GetUsedVariables(used_vars);
  }
  return used_vars;
}

// Helper class to manage the state of the parser.
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
        variable_map_[llvm::StringRef(*name)] = CreateDimExpr(count, context_);
      } else {
        variable_map_[llvm::StringRef(*name)] =
            CreateSymbolExpr(count, num_dims_, context_);
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

SymbolicMap ParseSymbolicMap(absl::string_view serialized_symbolic_map,
                             mlir::MLIRContext* mlir_context) {
  return SymbolicMapParserImpl(serialized_symbolic_map, mlir_context).Parse();
}

SymbolicMap::SymbolicMap(mlir::MLIRContext* ctx, int64_t num_dimensions,
                         int64_t num_symbols,
                         llvm::SmallVector<SymbolicExpr> exprs)
    : ctx_(ctx),
      num_dimensions_(num_dimensions),
      num_symbols_(num_symbols),
      exprs_(std::move(exprs)) {}

/*static*/ SymbolicMap SymbolicMap::Get(mlir::MLIRContext* ctx,
                                        int64_t num_dimensions,
                                        int64_t num_symbols,
                                        llvm::SmallVector<SymbolicExpr> exprs) {
  return SymbolicMap(ctx, num_dimensions, num_symbols, std::move(exprs));
}

std::string SymbolicMap::ToString() const {
  std::string out = "(";
  for (int i = 0; i < GetNumDims(); ++i) {
    absl::StrAppend(&out, (i > 0 ? ", " : ""), "d", i);
  }
  out += ")[";
  for (int i = 0; i < GetNumSymbols(); ++i) {
    absl::StrAppend(&out, (i > 0 ? ", " : ""), "s", i);
  }
  out += "] -> (";

  absl::StrAppend(
      &out,
      absl::StrJoin(GetResults(), ", ", [&](std::string* s, const auto& expr) {
        absl::StrAppend(s, expr.ToString(GetNumDims()));
      }));
  out += ")";
  return out;
}

bool SymbolicMap::IsIdentity() const {
  if (num_dimensions_ != GetNumResults()) {
    return false;
  }
  for (int i = 0; i < num_dimensions_; ++i) {
    const auto& expr = exprs_[i];
    if (expr.GetType() != SymbolicExprType::kVariable || expr.GetValue() != i) {
      return false;
    }
  }
  return true;
}

bool SymbolicMap::IsConstant() const {
  for (const auto& expr : exprs_) {
    if (expr.GetType() != SymbolicExprType::kConstant) {
      return false;
    }
  }
  return true;
}

llvm::SmallVector<int64_t> SymbolicMap::GetConstantResults() const {
  CHECK(IsConstant()) << "Cannot get constant results from a non-constant map";
  llvm::SmallVector<int64_t> constants;
  constants.reserve(exprs_.size());
  for (const auto& expr : exprs_) {
    constants.push_back(expr.GetValue());
  }
  return constants;
}

SymbolicMap SymbolicMap::ReplaceDimsAndSymbols(
    absl::Span<const SymbolicExpr> dim_replacements,
    absl::Span<const SymbolicExpr> sym_replacements, int64_t num_result_dims,
    int64_t num_result_symbols) const {
  CHECK(dim_replacements.empty() || dim_replacements.size() == num_dimensions_);
  CHECK(sym_replacements.empty() || sym_replacements.size() == num_symbols_);

  llvm::SmallVector<SymbolicExpr> all_replacements;
  all_replacements.reserve(num_dimensions_ + num_symbols_);

  if (!dim_replacements.empty()) {
    absl::c_copy(dim_replacements, std::back_inserter(all_replacements));
  } else {
    for (int i = 0; i < num_dimensions_; ++i) {
      all_replacements.push_back(CreateSymbolicVariable(i, ctx_));
    }
  }

  if (!sym_replacements.empty()) {
    absl::c_copy(sym_replacements, std::back_inserter(all_replacements));
  } else {
    for (int i = 0; i < num_symbols_; ++i) {
      all_replacements.push_back(
          CreateSymbolicVariable(num_dimensions_ + i, ctx_));
    }
  }

  llvm::SmallVector<SymbolicExpr> new_exprs;
  new_exprs.reserve(exprs_.size());
  for (const auto& expr : exprs_) {
    new_exprs.push_back(expr.ReplaceVariables(all_replacements));
  }
  return SymbolicMap(ctx_, num_result_dims, num_result_symbols,
                     std::move(new_exprs));
}

SymbolicMap SymbolicMap::Compose(const SymbolicMap& other) const {
  CHECK_EQ(GetNumDims(), other.GetNumResults())
      << "Number of dimensions of this map must match number of results of "
         "other map";
  int64_t new_dims = other.GetNumDims();
  int64_t new_syms = GetNumSymbols() + other.GetNumSymbols();

  // We need to reindex the dimensions of the other map.
  auto other_dim_replacements = CreateVariableRange(ctx_, other.GetNumDims());
  int64_t offset = new_dims;
  auto this_symbol_replacements =
      CreateVariableRange(ctx_, GetNumSymbols(), offset);
  offset += GetNumSymbols();
  auto other_sym_replacements =
      CreateVariableRange(ctx_, other.GetNumSymbols(), offset);

  // First we reindex other map symbols.
  SymbolicMap updated_other = other.ReplaceDimsAndSymbols(
      other_dim_replacements, other_sym_replacements, new_dims, new_syms);

  // Then we compose the maps.
  return ReplaceDimsAndSymbols(updated_other.GetResults(),
                               this_symbol_replacements, new_dims, new_syms);
}

SymbolicMap SymbolicMap::GetSubMap(
    absl::Span<const size_t> result_indices) const {
  llvm::SmallVector<SymbolicExpr> sub_exprs;
  sub_exprs.reserve(result_indices.size());
  for (unsigned int index : result_indices) {
    CHECK_LT(index, exprs_.size()) << "Result index out of bounds";
    sub_exprs.push_back(exprs_[index]);
  }
  return SymbolicMap(ctx_, num_dimensions_, num_symbols_, std::move(sub_exprs));
}

SymbolicMap SymbolicMap::Replace(SymbolicExpr expr,
                                 SymbolicExpr replacement) const {
  llvm::SmallVector<SymbolicExpr> new_exprs;
  new_exprs.reserve(exprs_.size());
  bool changed = false;
  for (const auto& e : exprs_) {
    SymbolicExpr new_expr = e.Replace(expr, replacement);
    changed |= new_expr != e;
    new_exprs.push_back(std::move(new_expr));
  }

  if (!changed) {
    return *this;
  }
  return SymbolicMap(ctx_, num_dimensions_, num_symbols_, std::move(new_exprs));
}

bool SymbolicMap::operator==(const SymbolicMap& other) const {
  return (ctx_ == other.ctx_ && num_dimensions_ == other.num_dimensions_ &&
          num_symbols_ == other.num_symbols_ && exprs_ == other.exprs_);
}

llvm::SmallBitVector GetUnusedDimensionsBitVector(const SymbolicMap& map) {
  llvm::SmallBitVector unused_dims(map.GetNumDims(), true);
  if (map.IsEmpty() || map.GetNumDims() == 0) {
    return unused_dims;
  }

  llvm::DenseSet<VariableID> used_vars = GetUsedVariablesFromExpressions(map);
  for (int i = 0; i < map.GetNumDims(); ++i) {
    if (used_vars.contains(i)) {
      unused_dims[i] = false;
    }
  }
  return unused_dims;
}

llvm::SmallBitVector GetUnusedSymbolsBitVector(const SymbolicMap& map) {
  llvm::SmallBitVector unused_symbols(map.GetNumSymbols(), true);
  if (map.IsEmpty() || map.GetNumSymbols() == 0) {
    return unused_symbols;
  }

  llvm::DenseSet<VariableID> used_vars = GetUsedVariablesFromExpressions(map);
  int64_t num_dims = map.GetNumDims();
  for (int i = 0; i < map.GetNumSymbols(); ++i) {
    if (used_vars.contains(num_dims + i)) {
      unused_symbols[i] = false;
    }
  }
  return unused_symbols;
}

SymbolicMap CompressDims(const SymbolicMap& map,
                         const llvm::SmallBitVector& unused_dims) {
  CHECK_EQ(map.GetNumDims(), unused_dims.size());

  if (unused_dims.none()) {
    return map;
  }

  // Assert that all dimensions marked as unused are actually unused.
  llvm::SmallBitVector actual_unused_dims = GetUnusedDimensionsBitVector(map);
  for (int i = 0; i < map.GetNumDims(); ++i) {
    if (unused_dims[i]) {
      CHECK(actual_unused_dims[i])
          << "Attempting to compress a used dimension: " << i;
    }
  }

  int64_t new_num_dims = map.GetNumDims() - unused_dims.count();
  llvm::SmallVector<SymbolicExpr> dim_replacements(map.GetNumDims());

  int64_t current_new_dim_idx = 0;
  for (int i = 0; i < map.GetNumDims(); ++i) {
    if (!unused_dims[i]) {
      dim_replacements[i] =
          CreateSymbolicVariable(current_new_dim_idx++, map.GetContext());
    }
  }
  auto sym_replacements =
      CreateVariableRange(map.GetContext(), map.GetNumSymbols(), new_num_dims);

  return map.ReplaceDimsAndSymbols(dim_replacements, sym_replacements,
                                   new_num_dims, map.GetNumSymbols());
}

SymbolicMap CompressSymbols(const SymbolicMap& map,
                            const llvm::SmallBitVector& unused_symbols) {
  CHECK_EQ(map.GetNumSymbols(), unused_symbols.size());

  if (unused_symbols.none()) {
    return map;
  }

  // Assert that all symbols marked as unused are actually unused.
  llvm::SmallBitVector actual_unused_symbols = GetUnusedSymbolsBitVector(map);
  for (int i = 0; i < map.GetNumSymbols(); ++i) {
    if (unused_symbols[i]) {
      CHECK(actual_unused_symbols[i])
          << "Attempting to compress a used symbol: " << i;
    }
  }

  int64_t num_dims = map.GetNumDims();
  int64_t new_num_symbols = map.GetNumSymbols() - unused_symbols.count();

  auto dim_replacements = CreateVariableRange(map.GetContext(), num_dims);

  llvm::SmallVector<SymbolicExpr> sym_replacements(map.GetNumSymbols());
  int64_t current_new_sym_idx = 0;
  for (int i = 0; i < map.GetNumSymbols(); ++i) {
    if (!unused_symbols[i]) {
      sym_replacements[i] = CreateSymbolicVariable(
          num_dims + current_new_sym_idx++, map.GetContext());
    }
  }
  CHECK_EQ(current_new_sym_idx, new_num_symbols);

  return map.ReplaceDimsAndSymbols(dim_replacements, sym_replacements, num_dims,
                                   new_num_symbols);
}

}  // namespace xla

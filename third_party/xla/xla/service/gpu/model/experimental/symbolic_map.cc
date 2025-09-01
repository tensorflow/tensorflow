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

#include "xla/service/gpu/model/experimental/symbolic_map.h"

#include <cstdint>
#include <iterator>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"

namespace xla {
namespace gpu {

namespace {

llvm::SmallVector<SymbolicExpr> CreateVariableRange(SymbolicExprContext* ctx,
                                                    int64_t n,
                                                    int64_t offset = 0) {
  llvm::SmallVector<SymbolicExpr> replacements;
  replacements.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    replacements.push_back(ctx->CreateVariable(offset + i));
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

}  // namespace

SymbolicMap::SymbolicMap(SymbolicExprContext* ctx, int64_t num_dimensions,
                         int64_t num_symbols,
                         llvm::SmallVector<SymbolicExpr> exprs)
    : ctx_(ctx),
      num_dimensions_(num_dimensions),
      num_symbols_(num_symbols),
      exprs_(std::move(exprs)) {}

/*static*/ SymbolicMap SymbolicMap::Get(SymbolicExprContext* ctx,
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
  CHECK_EQ(dim_replacements.size(), num_dimensions_);
  CHECK_EQ(sym_replacements.size(), num_symbols_);

  llvm::SmallVector<SymbolicExpr> all_replacements;
  all_replacements.reserve(num_dimensions_ + num_symbols_);
  absl::c_copy(dim_replacements, std::back_inserter(all_replacements));
  absl::c_copy(sym_replacements, std::back_inserter(all_replacements));

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
  return ctx_ == other.ctx_ && num_dimensions_ == other.num_dimensions_ &&
         num_symbols_ == other.num_symbols_ && exprs_ == other.exprs_;
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

}  // namespace gpu
}  // namespace xla

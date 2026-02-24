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
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map_serialization.h"

namespace xla {
namespace {

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

}  // namespace

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
  std::string s;
  llvm::raw_string_ostream os(s);
  xla::Print(*this, os);
  return os.str();
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

bool SymbolicMap::IsFunctionOfDim(int64_t dim_id) const {
  for (const auto& expr : exprs_) {
    if (expr.IsFunctionOfVariable(dim_id)) {
      return true;
    }
  }
  return false;
}

bool SymbolicMap::IsFunctionOfSymbol(int64_t symbol_id) const {
  for (const auto& expr : exprs_) {
    if (expr.IsFunctionOfVariable(num_dimensions_ + symbol_id)) {
      return true;
    }
  }
  return false;
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

SymbolicMap SymbolicMap::Replace(
    const llvm::DenseMap<SymbolicExpr, SymbolicExpr>& map,
    int64_t numResultDims, int64_t numResultSyms) const {
  llvm::SmallVector<SymbolicExpr> new_exprs;
  new_exprs.reserve(exprs_.size());
  for (const auto& expr : exprs_) {
    new_exprs.push_back(expr.Replace(map));
  }
  return SymbolicMap(ctx_, numResultDims, numResultSyms, std::move(new_exprs));
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

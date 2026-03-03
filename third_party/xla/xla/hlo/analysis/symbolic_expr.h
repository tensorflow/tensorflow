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

#ifndef XLA_HLO_ANALYSIS_SYMBOLIC_EXPR_H_
#define XLA_HLO_ANALYSIS_SYMBOLIC_EXPR_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/StorageUniquer.h"

namespace xla {

class SymbolicExprStorage;

typedef int64_t VariableID;

enum class SymbolicExprType {
  kAdd,
  kMul,
  kMod,
  kFloorDiv,
  kCeilDiv,
  kMax,
  kMin,
  kVariable,
  kConstant,  // Constant should be the last type for the comparator.
  // TODO(karupayun): Add kIn operator.
  // kIn,  // 'var in [a, b]' .
};

class SymbolicExpr {
 public:
  using ImplType = SymbolicExprStorage;
  /* implicit */ SymbolicExpr(const ImplType* impl = nullptr) : impl_(impl) {}

  explicit operator bool() const { return impl_ != nullptr; }
  bool operator!() const { return impl_ == nullptr; }
  bool operator==(SymbolicExpr other) const { return impl_ == other.impl_; }
  bool operator!=(SymbolicExpr other) const { return !(*this == other); }
  bool operator<(const SymbolicExpr& other) const;

  mlir::MLIRContext* GetContext() const;
  SymbolicExprType GetType() const;
  bool IsBinaryOp() const;
  SymbolicExpr GetLHS() const;
  SymbolicExpr GetRHS() const;
  int64_t GetValue() const;
  // If num_dims is provided, then the first num_dims variables are dimensions,
  // and the rest are symbols. If variable names are provided, then they are
  // used instead of numbers.
  std::string ToString(std::optional<int64_t> num_dims = std::nullopt) const;
  std::string ToString(absl::Span<const std::string> var_names) const;
  std::string ToString(absl::Span<const std::string> dim_names,
                       absl::Span<const std::string> sym_names) const;
  int64_t Evaluate(absl::Span<const int64_t> variable_values) const;
  SymbolicExpr ReplaceVariables(
      absl::Span<const SymbolicExpr> substitutions) const;
  // TODO: b/459357586 - These methods are needed for IndexingMap, but
  // dimensions and symbols are SymbolicMap specific. We should remove them once
  // we have a better way to integrate SymbolicExpr with IndexingMap. It is
  // assuming that dimensions are the first (0...num_dims-1) variables and
  // symbols are the rest.
  SymbolicExpr ReplaceDims(absl::Span<const SymbolicExpr> replacements) const;
  SymbolicExpr ReplaceSymbols(absl::Span<const SymbolicExpr> replacements,
                              int64_t num_dims) const;
  SymbolicExpr ReplaceDimsAndSymbols(
      absl::Span<const SymbolicExpr> dim_replacements,
      absl::Span<const SymbolicExpr> symbol_replacements) const;

  SymbolicExpr Canonicalize() const;

  /// Sparse replace method. Replace `expr` by `replacement` and return the
  /// modified expression tree.
  SymbolicExpr Replace(SymbolicExpr expr, SymbolicExpr replacement) const;

  /// Sparse replace method. If `*this` appears in `map` replaces it by
  /// `map[*this]` and return the modified expression tree. Otherwise traverse
  /// `*this` and apply replace with `map` on its subexpressions.
  SymbolicExpr Replace(
      const llvm::DenseMap<SymbolicExpr, SymbolicExpr>& replacements) const;

  void GetUsedVariables(llvm::DenseSet<VariableID>& used_vars) const;

  // Returns true if this expression depends on the given variable.
  bool IsFunctionOfVariable(VariableID var_id) const;

  // Traverses the expression tree and calls the callback for each
  // subexpression in postorder.
  void Walk(const std::function<void(SymbolicExpr)>& callback) const;

  // Return true if the expression is a multiple of `factor`.
  bool IsMultipleOf(int64_t factor) const;

  SymbolicExpr operator+(int64_t v) const;
  SymbolicExpr operator+(SymbolicExpr other) const;
  SymbolicExpr operator-() const;
  SymbolicExpr operator-(int64_t v) const;
  SymbolicExpr operator-(SymbolicExpr other) const;
  SymbolicExpr operator*(int64_t v) const;
  SymbolicExpr operator*(SymbolicExpr other) const;
  SymbolicExpr operator/(int64_t v) const { return this->floorDiv(v); }
  SymbolicExpr operator/(SymbolicExpr other) const {
    return this->floorDiv(other);
  }
  SymbolicExpr operator%(int64_t v) const;
  SymbolicExpr operator%(SymbolicExpr other) const;
  SymbolicExpr floorDiv(int64_t v) const;
  SymbolicExpr floorDiv(SymbolicExpr other) const;
  SymbolicExpr ceilDiv(int64_t v) const;
  SymbolicExpr ceilDiv(SymbolicExpr other) const;
  SymbolicExpr min(int64_t v) const;
  SymbolicExpr min(SymbolicExpr other) const;
  SymbolicExpr max(int64_t v) const;
  SymbolicExpr max(SymbolicExpr other) const;

  const ImplType* GetImpl() const { return impl_; }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SymbolicExpr expr) {
    sink.Append(expr.ToString());
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const SymbolicExpr expr) {
    os << expr.ToString();
    return os;
  }

 private:
  const ImplType* impl_ = nullptr;
};

SymbolicExpr operator+(int64_t lhs, SymbolicExpr rhs);
SymbolicExpr operator*(int64_t lhs, SymbolicExpr rhs);

inline ::llvm::hash_code hash_value(SymbolicExpr expr) {
  return ::llvm::hash_value(expr.GetImpl());
}

template <typename H>
H AbslHashValue(H h, const SymbolicExpr& expr) {
  return H::combine(std::move(h), hash_value(expr));
}

// This method should be called once permlir::MLIRContext to register the
// SymbolicExprStorage type with themlir::MLIRContext's uniquifier. It should be
// called before any SymbolicExprs are created.
void RegisterSymbolicExprStorage(mlir::MLIRContext* mlir_context);

// Helpers to create SymbolicExprs.
SymbolicExpr CreateSymbolicConstant(int64_t value,
                                    mlir::MLIRContext* mlir_context);
SymbolicExpr CreateSymbolicVariable(int64_t var_id,
                                    mlir::MLIRContext* mlir_context);
SymbolicExpr CreateSymbolicBinaryOp(SymbolicExprType type, SymbolicExpr lhs,
                                    SymbolicExpr rhs,
                                    mlir::MLIRContext* mlir_context);
llvm::SmallVector<SymbolicExpr> CreateSymbolicConstantExprs(
    llvm::ArrayRef<int64_t> constants, mlir::MLIRContext* mlir_context);

}  // namespace xla

namespace llvm {

// SymbolicExpr hash just like pointers
template <>
struct DenseMapInfo<xla::SymbolicExpr> {
  static xla::SymbolicExpr getEmptyKey() {
    auto* pointer = llvm::DenseMapInfo<void*>::getEmptyKey();
    return xla::SymbolicExpr(static_cast<xla::SymbolicExprStorage*>(pointer));
  }
  static xla::SymbolicExpr getTombstoneKey() {
    auto* pointer = llvm::DenseMapInfo<void*>::getTombstoneKey();
    return xla::SymbolicExpr(static_cast<xla::SymbolicExprStorage*>(pointer));
  }
  static unsigned getHashValue(xla::SymbolicExpr val) {
    return hash_value(val);
  }
  static bool isEqual(xla::SymbolicExpr LHS, xla::SymbolicExpr RHS) {
    return LHS == RHS;
  }
};

}  // namespace llvm

#endif  // XLA_HLO_ANALYSIS_SYMBOLIC_EXPR_H_

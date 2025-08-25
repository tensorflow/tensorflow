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

#ifndef XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_EXPR_H_
#define XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_EXPR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/StorageUniquer.h"

namespace xla {
namespace gpu {

class SymbolicExprContext;
class SymbolicExprStorage;

typedef int64_t VariableID;

enum class SymbolicExprType {
  kConstant,
  kVariable,
  kAdd,
  kMul,
  kFloorDiv,
  kCeilDiv,
  kMod,
  kMax,
  kMin,
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

  SymbolicExprContext* GetContext() const;
  SymbolicExprType GetType() const;
  SymbolicExpr GetLHS() const;
  SymbolicExpr GetRHS() const;
  int64_t GetValue() const;
  std::string ToString() const;
  int64_t Evaluate(absl::Span<const int64_t> variable_values) const;
  SymbolicExpr ReplaceVariables(
      absl::Span<const SymbolicExpr> substitutions) const;
  SymbolicExpr Canonicalize() const;

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

 private:
  const ImplType* impl_ = nullptr;
};

inline ::llvm::hash_code hash_value(SymbolicExpr expr) {
  return ::llvm::hash_value(expr.GetImpl());
}

class SymbolicExprContext {
 public:
  SymbolicExprContext();
  SymbolicExpr Parse(absl::string_view expr_str);
  SymbolicExpr CreateConstant(int64_t value);
  SymbolicExpr CreateVariable(int64_t var_id);
  SymbolicExpr CreateBinaryOp(SymbolicExprType type, SymbolicExpr lhs,
                              SymbolicExpr rhs);

 private:
  SymbolicExpr GetOrCreate(SymbolicExprType type, int64_t value,
                           SymbolicExpr lhs, SymbolicExpr rhs);
  mlir::StorageUniquer uniquer_;
};

}  // namespace gpu
}  // namespace xla

namespace llvm {

// SymbolicExpr hash just like pointers
template <>
struct DenseMapInfo<xla::gpu::SymbolicExpr> {
  static xla::gpu::SymbolicExpr getEmptyKey() {
    auto* pointer = llvm::DenseMapInfo<void*>::getEmptyKey();
    return xla::gpu::SymbolicExpr(
        static_cast<xla::gpu::SymbolicExprStorage*>(pointer));
  }
  static xla::gpu::SymbolicExpr getTombstoneKey() {
    auto* pointer = llvm::DenseMapInfo<void*>::getTombstoneKey();
    return xla::gpu::SymbolicExpr(
        static_cast<xla::gpu::SymbolicExprStorage*>(pointer));
  }
  static unsigned getHashValue(xla::gpu::SymbolicExpr val) {
    return hash_value(val);
  }
  static bool isEqual(xla::gpu::SymbolicExpr LHS, xla::gpu::SymbolicExpr RHS) {
    return LHS == RHS;
  }
};

}  // namespace llvm

#endif  // XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_EXPR_H_

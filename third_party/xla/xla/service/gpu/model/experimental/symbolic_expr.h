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
#include <deque>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

namespace xla {
namespace gpu {

class SymbolicExprContext;
class SymbolicExpr;
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

class SymbolicExprStorage {
 public:
  using KeyTy = std::tuple<SymbolicExprType, int64_t, const SymbolicExpr*,
                           const SymbolicExpr*>;

  SymbolicExprStorage(SymbolicExprType type, int64_t val,
                      const SymbolicExpr* lhs, const SymbolicExpr* rhs)
      : type_(type), value_(val), lhs_(lhs), rhs_(rhs) {}

  bool isKeyEqual(const KeyTy& key) const {
    return key == KeyTy(type_, value_, lhs_, rhs_);
  }

  template <typename H>
  friend H AbslHashValue(H h, const KeyTy& key) {
    return H::combine(std::move(h), std::get<0>(key), std::get<1>(key),
                      std::get<2>(key), std::get<3>(key));
  }

  std::string ToString() const;
  int64_t Evaluate(absl::Span<const int64_t> variable_values) const;
  SymbolicExpr* ReplaceVariables(absl::Span<SymbolicExpr* const> substitutions,
                                 SymbolicExprContext* ctx) const;

 private:
  friend class SymbolicExpr;
  SymbolicExprType type_;
  int64_t value_;  // Value of the constant or id of the variable.
  const SymbolicExpr* lhs_;
  const SymbolicExpr* rhs_;
};

class SymbolicExpr {
 public:
  explicit SymbolicExpr(SymbolicExprStorage* storage) : storage_(storage) {}
  SymbolicExpr(SymbolicExpr&&) = default;
  SymbolicExprType GetType() const { return storage_->type_; }
  const SymbolicExpr* GetLHS() const { return storage_->lhs_; }
  const SymbolicExpr* GetRHS() const { return storage_->rhs_; }
  int64_t GetValue() const { return storage_->value_; }

  std::string ToString() const { return storage_->ToString(); }
  int64_t Evaluate(absl::Span<const int64_t> variable_values) const {
    return storage_->Evaluate(variable_values);
  }
  SymbolicExpr* ReplaceVariables(absl::Span<SymbolicExpr* const> substitutions,
                                 SymbolicExprContext* ctx) const {
    return storage_->ReplaceVariables(substitutions, ctx);
  }

  bool operator==(const SymbolicExpr& other) const {
    return storage_ == other.storage_;
  }

 private:
  SymbolicExprStorage* storage_;
};

// Maps a set of input variables to a set of output SymbolicExpr trees.
struct SymbolicMap {
  int64_t num_dimensions;
  int64_t num_ranges;
  int64_t num_symbols;
  std::vector<SymbolicExpr*> exprs;
};

class SymbolicExprContext {
 public:
  SymbolicExprContext() = default;
  SymbolicExpr* Parse(absl::string_view expr_str);
  SymbolicExpr* CreateConstant(int64_t value);
  SymbolicExpr* CreateVariable(int64_t var_id);
  SymbolicExpr* CreateBinaryOp(SymbolicExprType type, SymbolicExpr* lhs,
                               SymbolicExpr* rhs);

 private:
  SymbolicExpr* GetOrCreate(SymbolicExprStorage::KeyTy key);

  absl::Mutex mutex_;
  std::deque<SymbolicExprStorage> storage_pool_ ABSL_GUARDED_BY(mutex_);
  std::deque<SymbolicExpr> handle_pool_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<SymbolicExprStorage::KeyTy, SymbolicExpr*> cache_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_EXPR_H_

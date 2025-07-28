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
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

namespace xla {
namespace gpu {

class SymbolicExprContext;
struct SymbolicExpr;
struct SymbolicExprStorage;

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

// TODO(karupayun): This should be modified when implementing
// SymbolicExprStorage.
class SymbolicExpr {
 public:
  SymbolicExpr(SymbolicExpr&&) = default;
  std::string ToString() const;
  SymbolicExprType GetType() const { return type_; }
  SymbolicExpr* GetLHS() const { return lhs_; }
  SymbolicExpr* GetRHS() const { return rhs_; }
  int64_t GetValue() const { return value_; }
  int64_t Evaluate(absl::Span<const int64_t> variable_values) const;
  SymbolicExpr* ReplaceVariables(absl::Span<SymbolicExpr* const> substitutions,
                                 SymbolicExprContext* ctx) const;

 protected:
  friend class SymbolicExprContext;
  SymbolicExpr(SymbolicExprType type, SymbolicExpr* lhs, SymbolicExpr* rhs)
      : type_(type), lhs_(lhs), rhs_(rhs) {}
  SymbolicExpr(SymbolicExprType type, int64_t value)
      : type_(type), value_(value) {}

 private:
  SymbolicExprType type_;
  SymbolicExpr* lhs_ = nullptr;
  SymbolicExpr* rhs_ = nullptr;
  // Value of the constant or id of the variable.
  int64_t value_ = std::numeric_limits<int64_t>::min();
};

// Maps a set of input variables to a set of output SymbolicExpr trees.
struct SymbolicMap {
  int64_t num_dimensions;
  int64_t num_ranges;
  int64_t num_symbols;
  std::vector<SymbolicExpr*> exprs;
};

struct SymbolicExprContext {
 public:
  SymbolicExprContext() = default;
  SymbolicExpr* Parse(absl::string_view expr_str);
  SymbolicExpr* CreateConstant(int64_t value);
  SymbolicExpr* CreateVariable(int64_t var_id);
  SymbolicExpr* CreateBinaryOp(SymbolicExprType type, SymbolicExpr* lhs,
                               SymbolicExpr* rhs);

 private:
  absl::Mutex mutex;
  // TODO (karupayun): Implement SymbolicExprStorage.
  std::deque<SymbolicExpr> expr_storage ABSL_GUARDED_BY(mutex);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_EXPR_H_

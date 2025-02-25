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

#ifndef XLA_SERVICE_GPU_MODEL_CONSTRAINT_EXPRESSION_H_
#define XLA_SERVICE_GPU_MODEL_CONSTRAINT_EXPRESSION_H_

#include <cstdint>
#include <ostream>
#include <string>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "xla/hlo/analysis/indexing_map.h"

namespace xla {
namespace gpu {

// `ConstraintExpression` represents a "flat" constraint expression of the form
//   ((expr0 in interval0) && (expr1 in interval1)...) ||
//   ((expr{n} in interval{n}) &&...)...
//
// The underlying constraints are stored in a vector of vectors, such that each
// innermost vector represents the conjunction of some constraints, and the
// outermost vector represents the disjunction of all its elements
// (conjunctions). This representation is effective because `&&` (`And`) is
// distributive over `||` (`Or`), ensuring that we can always flatten any given
// `ConstraintExpression` in this way, and that we have reasonable combinators
// for `&&` and `||`.
//
// We store a boolean `is_satisfiable_` to indicate whether we expect that the
// constraints can be satisfied. When set to `false`, we expect the
// `ConstraintExpression` to be empty (bottom).
class ConstraintExpression {
 public:
  struct Constraint {
    mlir::AffineExpr expr;
    Interval interval;

    bool operator==(const Constraint& other) const {
      CHECK_EQ(expr.getContext(), other.expr.getContext())
          << "AffineExpr should be from the same MLIRContext.";
      return expr == other.expr && interval == other.interval;
    }
  };

 private:
  using ConjointConstraints = llvm::SmallVector<Constraint, 2>;
  explicit ConstraintExpression(bool is_satisfiable)
      : is_satisfiable_(is_satisfiable) {}

 public:
  // Constructs a `ConstraintExpression` from a single `Constraint`.
  explicit ConstraintExpression(const Constraint& constraint)
      : disjoint_conjoint_constraints_({{constraint}}) {}

  // Constructs a `ConstraintExpression` that is always satisfied.
  static ConstraintExpression GetAlwaysSatisfied() {
    return ConstraintExpression(true);
  }

  // Constructs a `ConstraintExpression` that is unsatisfiable.
  static ConstraintExpression GetUnsatisfiable() {
    return ConstraintExpression(false);
  }

  // Takes the conjunction of the constraints of `first` and `second`.
  friend ConstraintExpression operator&&(const ConstraintExpression& first,
                                         const ConstraintExpression& second);

  // Takes the disjunction of the constraints of `first` and `second`.
  friend ConstraintExpression operator||(const ConstraintExpression& first,
                                         const ConstraintExpression& second);

  // Whether the constraints can be satisfied.
  bool is_satisfiable() const { return is_satisfiable_; }

  // Returns `true` if the constraint expression is marked satisfiable and does
  // not contain any constraint.
  bool IsAlwaysSatisfied() const {
    return is_satisfiable_ && disjoint_conjoint_constraints_.empty();
  }

  // Returns `true` if the constraint expression is satisfied by the provided
  // dim_values, and `false` otherwise.  The caller is responsible for ensuring
  // that the number of provided dim_values is sufficient to verify the
  // constraints.
  bool IsSatisfiedBy(absl::Span<const int64_t> dim_values) const;

  std::string ToString() const;

  void Print(std::ostream& out) const;

  // Simplifies the constraint expression.
  //
  // We remove conjunctions that are always satisfied, and we remove
  // disjunctions that are unsatisfiable. If we can deduce that the whole
  // expression is unsatisfiable or always satisfied, than we change the whole
  // expression to the canonical form.
  //
  // E.g., if we find that one of the conjunctions is always satisfied, we don't
  // just throw away that part---we throw away everything and make the
  // ConstraintExpression canonically always satisfied.
  void Simplify();

 private:
  // This allows GUnit to print the expression.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ConstraintExpression& expr) {
    sink.Append(expr.ToString());
  }

  template <typename H>
  friend H AbslHashValue(H h, const Constraint& constraint) {
    llvm::hash_code expr_hash = mlir::hash_value(constraint.expr);
    return H::combine(std::move(h), static_cast<size_t>(expr_hash),
                      constraint.interval);
  }

  template <typename H>
  friend H AbslHashValue(H h, const ConjointConstraints& conjoint_constraints) {
    for (const auto& constraint : conjoint_constraints) {
      h = H::combine(std::move(h), constraint);
    }
    return h;
  }

  // When this is set to `false`, disjoint_conjoint_constraints_ must be empty.
  bool is_satisfiable_ = true;
  llvm::SmallVector<ConjointConstraints, 2> disjoint_conjoint_constraints_;
};

// Logical operators between `ConstraintExpression` and `Constraint`.
inline ConstraintExpression operator&&(
    const ConstraintExpression::Constraint& first,
    const ConstraintExpression& second) {
  return ConstraintExpression(first) && second;
}

inline ConstraintExpression operator&&(
    const ConstraintExpression& first,
    const ConstraintExpression::Constraint& second) {
  return first && ConstraintExpression(second);
}

inline ConstraintExpression operator&&(
    const ConstraintExpression::Constraint& first,
    const ConstraintExpression::Constraint& second) {
  return ConstraintExpression(first) && ConstraintExpression(second);
}

inline ConstraintExpression operator||(
    const ConstraintExpression::Constraint& first,
    const ConstraintExpression& second) {
  return ConstraintExpression(first) || second;
}

inline ConstraintExpression operator||(
    const ConstraintExpression& first,
    const ConstraintExpression::Constraint& second) {
  return first || ConstraintExpression(second);
}

inline ConstraintExpression operator||(
    const ConstraintExpression::Constraint& first,
    const ConstraintExpression::Constraint& second) {
  return ConstraintExpression(first) || ConstraintExpression(second);
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_CONSTRAINT_EXPRESSION_H_

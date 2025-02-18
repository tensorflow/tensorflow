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

#include "xla/service/gpu/model/constraint_expression.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/service/gpu/model/affine_map_evaluator.h"

namespace xla {
namespace gpu {
namespace {

using ::llvm::SmallVector;
using ::mlir::AffineConstantExpr;
using ::mlir::AffineExpr;
using Constraint = ConstraintExpression::Constraint;
using ConjointConstraints = llvm::SmallVector<Constraint, 2>;

// Tries to take the conjunction of `conjunction_1` and `conjunction_2`.
// Fails and returns `std::nullopt` if and only if the conjunction attempt
// results in an unsatisfiable constraint.
std::optional<ConjointConstraints> TryIntersectConjointConstraints(
    const ConjointConstraints& conjunction_1,
    const ConjointConstraints& conjunction_2) {
  if (conjunction_1.empty()) {
    return conjunction_2;
  }

  if (conjunction_2.empty()) {
    return conjunction_1;
  }

  ConjointConstraints result = conjunction_1;
  for (const auto& constraint : conjunction_2) {
    Constraint* result_it =
        llvm::find_if(result, [&](const Constraint& result_constraint) {
          return result_constraint.expr == constraint.expr;
        });
    const auto& [expr, interval] = constraint;
    if (result_it != result.end()) {
      auto& [result_expr, result_interval] = *result_it;
      result_interval = result_interval.Intersect(interval);
      if (!result_interval.IsFeasible()) {
        VLOG(1) << "Got two incompatible intervals for expression "
                << ToString(expr);
        return std::nullopt;
      }
    } else {
      result.push_back(Constraint{expr, interval});
    }
  }

  return result;
}

}  // namespace

ConstraintExpression operator&&(const ConstraintExpression& first,
                                const ConstraintExpression& second) {
  // When either one of the expressions is unsatisfiable, their conjunction is
  // necessarily unsatisfiable.
  if (!first.is_satisfiable() || !second.is_satisfiable()) {
    return ConstraintExpression::GetUnsatisfiable();
  }

  // Both first and second are satisfiable. Handle here explicitly the case
  // where one (or both) of the maps are trivially satisfied.
  if (first.IsAlwaysSatisfied()) {
    return second;
  }

  if (second.IsAlwaysSatisfied()) {
    return first;
  }

  // `IsAlwaysSatisfied()` is true if and only if the map holds literally no
  // useful information and is equivalent to a default-constructed
  // `ConstraintExpression`---one that is neither unsatisfiable, nor contains
  // any constraints. Therefore, we can assume below that both of the provided
  // `ConstraintExpression`s are satisfiable and each contain at least one
  // constraint.
  //
  // By distributivity, we have that:
  //     (conj0 || conj1 || ...) && (conj2 || conj3 || ...)
  //   = (conj0 && conj2 || conj0 && conj3 || ... ||
  //      conj1 && conj2 || conj1 && conj3 ...)
  // which allows us to construct the result by essentially taking the cartesian
  // product of the disjoint conjunctions of `first` with those of `second`.
  decltype(std::declval<ConstraintExpression>()
               .disjoint_conjoint_constraints_) conjunctions;
  for (const ConjointConstraints& conjunction_1 :
       first.disjoint_conjoint_constraints_) {
    for (const ConjointConstraints& conjunction_2 :
         second.disjoint_conjoint_constraints_) {
      std::optional<ConjointConstraints> maybe_conjunction =
          TryIntersectConjointConstraints(conjunction_1, conjunction_2);
      // We only add the resulting conjunction to the result
      // `ConstraintExpression` if it is satisfiable, since it is otherwise
      // redundant: (conj || false) == conj.
      if (maybe_conjunction.has_value()) {
        conjunctions.push_back(*maybe_conjunction);
      }
    }
  }

  // If all the resulting conjunctions are unsatisfiable, the result itself is
  // unsatisfiable: (false || false) == false.
  // In our case, this manifests as an empty list of constraints in the result.
  ConstraintExpression result(/*is_satisfiable=*/!conjunctions.empty());
  result.disjoint_conjoint_constraints_ = std::move(conjunctions);
  return result;
}

ConstraintExpression operator||(const ConstraintExpression& first,
                                const ConstraintExpression& second) {
  // If either one of the expressions is always satisfied, their disjunction is
  // always satisfied.
  if (first.IsAlwaysSatisfied() || second.IsAlwaysSatisfied()) {
    return ConstraintExpression::GetAlwaysSatisfied();
  }

  // When either one of the expressions is unsatisfiable, we can simply return
  // the other one.
  if (!first.is_satisfiable()) {
    return second;
  }
  if (!second.is_satisfiable()) {
    return first;
  }

  ConstraintExpression result = first;
  absl::c_copy(second.disjoint_conjoint_constraints_,
               std::back_inserter(result.disjoint_conjoint_constraints_));
  return result;
}

bool ConstraintExpression::IsSatisfiedBy(
    absl::Span<const int64_t> dim_values) const {
  if (disjoint_conjoint_constraints_.empty()) {
    return is_satisfiable_;
  }

  return absl::c_any_of(
      disjoint_conjoint_constraints_, [&](const auto& conjunction) {
        return absl::c_all_of(conjunction, [&](const Constraint& constraint) {
          int64_t value = EvaluateAffineExpr(constraint.expr, dim_values);
          return constraint.interval.Contains(value);
        });
      });
}

std::string ConstraintExpression::ToString() const {
  std::stringstream ss;
  Print(ss);
  return ss.str();
}

void ConstraintExpression::Print(std::ostream& out) const {
  if (IsAlwaysSatisfied()) {
    out << "always satisfied\n";
    return;
  }
  if (!is_satisfiable()) {
    out << "unsatisfiable\n";
    return;
  }

  // Accumulate constraints in a vector in order to put them in lexicographic
  // order and to get deterministic output.
  std::vector<std::string> conjunction_strings;
  conjunction_strings.reserve(disjoint_conjoint_constraints_.size());
  for (const auto& disjunction : disjoint_conjoint_constraints_) {
    std::vector<std::string> constraint_strings;
    constraint_strings.reserve(disjunction.size());
    for (const auto& [expr, interval] : disjunction) {
      constraint_strings.push_back(
          absl::StrCat(xla::ToString(expr), " in ", interval.ToString()));
    }
    std::sort(constraint_strings.begin(), constraint_strings.end());
    conjunction_strings.push_back(absl::StrJoin(constraint_strings, " && "));
  }
  std::sort(conjunction_strings.begin(), conjunction_strings.end());
  out << absl::StrJoin(conjunction_strings, " || ") << "\n";
}

namespace {

bool IsConstraintAlwaysSatisfied(AffineExpr expr, Interval interval) {
  if (AffineConstantExpr constant = mlir::dyn_cast<AffineConstantExpr>(expr)) {
    return interval.Contains(constant.getValue());
  }
  return false;
}

bool IsConstraintUnsatisfiable(AffineExpr expr, Interval interval) {
  if (!interval.IsFeasible()) {
    return true;
  }
  if (AffineConstantExpr constant = mlir::dyn_cast<AffineConstantExpr>(expr)) {
    return !interval.Contains(constant.getValue());
  }
  return false;
}

struct Unsatisfiable {};
struct AlwaysSatisfied {};

std::variant<Unsatisfiable, AlwaysSatisfied, ConjointConstraints>
SimplifyConjointConstraints(const ConjointConstraints& conjunction) {
  ConjointConstraints result;
  for (const auto& [expr, interval] : conjunction) {
    if (IsConstraintAlwaysSatisfied(expr, interval)) {
      continue;
    }
    if (IsConstraintUnsatisfiable(expr, interval)) {
      return Unsatisfiable{};
    }
    result.push_back(Constraint{expr, interval});
  }
  if (result.empty()) {
    return AlwaysSatisfied{};
  }

  // A comparator to canonicalize the order of constraints, so we can easily
  // check if two ConjointConstraints are equal. The order is arbitrary (doesn't
  // depend on the structure of the constraints) and can change between runs,
  // but is stable during a single execution. The printed version of the
  // constraints relies on sorting strings, so string representation will be
  // always the same.
  auto comp = [](const Constraint& a, const Constraint& b) {
    if (a.expr != b.expr) {
      // AffineExpr are deduplicated and stored as immutable objects in
      // MLIRContext. Comparing pointers gives us a fast and easy way to get
      // stable ordering.
      CHECK_EQ(a.expr.getContext(), b.expr.getContext())
          << "AffineExpr should be from the same MLIRContext.";
      return a.expr.getImpl() < b.expr.getImpl();
    }

    // Default comparison for intervals will return nullopt if intervals are
    // overlapping. Here we do strict ordering by comparing lower bounds first
    // and then upper bounds.
    return std::make_pair(a.interval.lower, a.interval.upper) <
           std::make_pair(b.interval.lower, b.interval.upper);
  };

  // Canonicalize constraints order and remove duplicates.
  std::sort(result.begin(), result.end(), comp);
  result.erase(std::unique(result.begin(), result.end()), result.end());

  return result;
}

}  // namespace

void ConstraintExpression::Simplify() {
  if (disjoint_conjoint_constraints_.empty()) {
    return;  // unsatisfiable or always satisfied.
  }

  // Find and remove redundant constraints.
  absl::flat_hash_set<ConjointConstraints> unique_conjunctions;
  for (const auto& conjunction : disjoint_conjoint_constraints_) {
    auto simplified_conjunction = SimplifyConjointConstraints(conjunction);
    if (std::holds_alternative<Unsatisfiable>(simplified_conjunction)) {
      continue;
    }
    if (std::holds_alternative<AlwaysSatisfied>(simplified_conjunction)) {
      return disjoint_conjoint_constraints_.clear();
    }
    unique_conjunctions.insert(
        std::get<ConjointConstraints>(simplified_conjunction));
  }
  disjoint_conjoint_constraints_.assign(unique_conjunctions.begin(),
                                        unique_conjunctions.end());
  // If all the conjunctions are unsatisfiable, the result is unsatisfiable.
  is_satisfiable_ = !disjoint_conjoint_constraints_.empty();
}

}  // namespace gpu
}  // namespace xla

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

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_test_utils.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ExplainMatchResult;

using Constraint = ConstraintExpression::Constraint;

MATCHER_P(MatchConstraintExpressionString, constraint_expression_string, "") {
  return ExplainMatchResult(
      true, ApproximateMatch(constraint_expression_string, arg.ToString()),
      result_listener);
}

class ConstraintExpressionTest : public IndexingTestBase {
 public:
  ConstraintExpression::Constraint GetConstraint(const std::string& string_expr,
                                                 int64_t lower, int64_t upper) {
    return {ParseAffineExpr(string_expr, &mlir_context_),
            Interval{lower, upper}};
  }
  ConstraintExpression Simplify(ConstraintExpression constraints) {
    constraints.Simplify();
    return constraints;
  }
};

TEST_F(ConstraintExpressionTest, CheckAlwaysSatisfied) {
  auto always_satisfied = ConstraintExpression::GetAlwaysSatisfied();
  EXPECT_TRUE(always_satisfied.is_satisfiable());
  EXPECT_TRUE(always_satisfied.IsAlwaysSatisfied());
  EXPECT_THAT(always_satisfied,
              MatchConstraintExpressionString("always satisfied"));
  EXPECT_TRUE(always_satisfied.IsSatisfiedBy({}));
  EXPECT_THAT(always_satisfied || GetConstraint("d0", 0, 0),
              MatchConstraintExpressionString("always satisfied"));
  EXPECT_THAT(always_satisfied && GetConstraint("d0", 0, 0),
              MatchConstraintExpressionString("d0 in [0, 0]"));
  EXPECT_THAT(always_satisfied || ConstraintExpression::GetUnsatisfiable(),
              MatchConstraintExpressionString("always satisfied"));
  EXPECT_THAT(ConstraintExpression::GetUnsatisfiable() || always_satisfied,
              MatchConstraintExpressionString("always satisfied"));
  EXPECT_THAT(Simplify(always_satisfied),
              MatchConstraintExpressionString("always satisfied"));
}

TEST_F(ConstraintExpressionTest, CheckUnsatisfiable) {
  auto unsatisfiable = ConstraintExpression::GetUnsatisfiable();
  EXPECT_FALSE(unsatisfiable.is_satisfiable());
  EXPECT_FALSE(unsatisfiable.IsAlwaysSatisfied());
  EXPECT_THAT(unsatisfiable, MatchConstraintExpressionString("unsatisfiable"));
  EXPECT_FALSE(unsatisfiable.IsSatisfiedBy({}));
  EXPECT_THAT(unsatisfiable || GetConstraint("d0", 0, 0),
              MatchConstraintExpressionString("d0 in [0, 0]"));
  EXPECT_THAT(unsatisfiable && GetConstraint("d0", 0, 0),
              MatchConstraintExpressionString("unsatisfiable"));
  EXPECT_THAT(unsatisfiable && ConstraintExpression::GetAlwaysSatisfied(),
              MatchConstraintExpressionString("unsatisfiable"));
  EXPECT_THAT(ConstraintExpression::GetAlwaysSatisfied() && unsatisfiable,
              MatchConstraintExpressionString("unsatisfiable"));
  EXPECT_THAT(Simplify(unsatisfiable),
              MatchConstraintExpressionString("unsatisfiable"));
}

TEST_F(ConstraintExpressionTest, PrettyPrintingTest) {
  ConstraintExpression constraints =
      GetConstraint("d2", 5, 6) ||
      (GetConstraint("d1", 3, 4) && GetConstraint("d0", 1, 2));
  EXPECT_THAT(constraints, MatchConstraintExpressionString(
                               "d0 in [1, 2] && d1 in [3, 4] || d2 in [5, 6]"));
}

TEST_F(ConstraintExpressionTest,
       ConjunctionOfConstraintsOnTheSameExpressionAreIntersected) {
  ConstraintExpression constraints{GetConstraint("d0", 0, 5)};
  EXPECT_THAT(constraints, MatchConstraintExpressionString("d0 in [0, 5]"));

  // Constraints are intersected.
  constraints = constraints && GetConstraint("d0", 3, 6);
  EXPECT_THAT(constraints, MatchConstraintExpressionString("d0 in [3, 5]"));

  // Empty intersection results in unsatisfiability.
  constraints = constraints && GetConstraint("d0", 7, 8);
  EXPECT_THAT(constraints, MatchConstraintExpressionString("unsatisfiable"));
}

TEST_F(
    ConstraintExpressionTest,
    CanSuccessfullyPerformConjunctionOfConstraintExpressionWithConjointConstraints) {  // NOLINT(whitespace/line_length)
  ConstraintExpression constraints = GetConstraint("d0", 0, 5) &&
                                     GetConstraint("d1", 0, 5) &&
                                     GetConstraint("d2", 0, 5);
  // Constraints can be merged without trouble, and hence the constraint
  // expression is satisfiable.
  EXPECT_TRUE(constraints.is_satisfiable());
  EXPECT_THAT(constraints, MatchConstraintExpressionString(
                               "d0 in [0, 5] && d1 in [0, 5] && d2 in [0, 5]"));
}

TEST_F(
    ConstraintExpressionTest,
    CorrectlyEliminatesConjunctionFromDisjunctionWhenItBecomesUnsatisfiable) {
  ConstraintExpression constraints =
      GetConstraint("d0", 0, 5) || GetConstraint("d1", 0, 5);
  EXPECT_THAT(constraints,
              MatchConstraintExpressionString("d0 in [0, 5] || d1 in [0, 5]"));

  // `conjunction_1` && `conjunction_3` is an unsatisfiable constraint. Taking
  // the conjunction of the existing constraint expression with `conjunction_3`
  // should therefore evict the unsatisfiable intersection of `conjunction_1`
  // and `conjunction_3` from the disjoint expression.
  constraints = constraints && GetConstraint("d0", 6, 6);

  EXPECT_THAT(constraints,
              MatchConstraintExpressionString("d0 in [6, 6] && d1 in [0, 5]"));

  // But becomes unsatisfiable if we eliminate the last remaining constraint by
  // constructing another unsatisfiable conjunction.
  constraints = constraints && GetConstraint("d0", 7, 7);
  EXPECT_THAT(constraints, MatchConstraintExpressionString("unsatisfiable"));
}

TEST_F(
    ConstraintExpressionTest,
    CanSuccessfullyPerformDisjunctionOfConstraintExpressionWithConjointConstraints) {  // NOLINT(whitespace/line_length)
  ConstraintExpression constraints =
      GetConstraint("d0", 0, 5) && GetConstraint("d1", 0, 5);
  constraints = constraints || GetConstraint("d2", 0, 5);
  EXPECT_TRUE(constraints.is_satisfiable());
  EXPECT_THAT(constraints, MatchConstraintExpressionString(
                               "d0 in [0, 5] && d1 in [0, 5] || d2 in [0, 5]"));
}

TEST_F(
    ConstraintExpressionTest,
    CanSuccessfullyPerformConjunctionOfConstraintExpressionWithConstraintExpression) {  // NOLINT(whitespace/line_length)
  // Construct the first `ConstraintExpression` to be of the form
  //   a || b.
  ConstraintExpression constraints_1 =
      GetConstraint("d0", 0, 5) || GetConstraint("d1", 0, 5);

  // Construct the second `ConstraintExpression` to be of the form
  //   c || d || e.
  ConstraintExpression constraints_2 = GetConstraint("d2", 0, 5) ||
                                       GetConstraint("d3", 0, 5) ||
                                       GetConstraint("d4", 0, 5);

  // Taking the conjunction of the two `ConstraintExpression`s should result in
  // a `ConstraintExpression` of the form
  //   a && c || a && d || a && e || b && c || b && d || b && e.
  ConstraintExpression result_constraint_expression =
      constraints_1 && constraints_2;

  EXPECT_TRUE(result_constraint_expression.is_satisfiable());
  // There are now six conjunctions in the disjoint expression, as described
  // above.
  EXPECT_THAT(
      result_constraint_expression,
      MatchConstraintExpressionString(
          "d0 in [0, 5] && d2 in [0, 5] || d0 in [0, 5] && d3 in [0, 5] || "
          "d0 in [0, 5] && d4 in [0, 5] || d1 in [0, 5] && d2 in [0, 5] || "
          "d1 in [0, 5] && d3 in [0, 5] || d1 in [0, 5] && d4 in [0, 5]"));

  // Lastly, make sure that the conjunction of an empty `ConstraintExpression`
  // with a non-empty one results in passing the non-empty one through, on both
  // sides.
  ConstraintExpression always_satisfied =
      ConstraintExpression::GetAlwaysSatisfied();
  EXPECT_THAT(always_satisfied && constraints_2,
              MatchConstraintExpressionString(
                  "d2 in [0, 5] || d3 in [0, 5] || d4 in [0, 5]"));
  EXPECT_THAT(constraints_2 && always_satisfied,
              MatchConstraintExpressionString(
                  "d2 in [0, 5] || d3 in [0, 5] || d4 in [0, 5]"));
}

TEST_F(
    ConstraintExpressionTest,
    CanSuccessfullyPerformDisjunctionOfConstraintExpressionWithConstraintExpression) {  // NOLINT(whitespace/line_length)
  // Construct the first `ConstraintExpression` to be of the form
  //   a || b.
  ConstraintExpression constraints_1 =
      GetConstraint("d0", 0, 5) || GetConstraint("d1", 0, 5);

  // Construct the second `ConstraintExpression` to be of the form
  //   c || d || e.
  ConstraintExpression constraints_2 = GetConstraint("d2", 0, 5) ||
                                       GetConstraint("d3", 0, 5) ||
                                       GetConstraint("d4", 0, 5);

  // Taking the disjunction of the two `ConstraintExpression`s should result in
  // a `ConstraintExpression` of the form
  //   a || b || c || d || e.
  ConstraintExpression result_constraint_expression =
      constraints_1 || constraints_2;

  EXPECT_TRUE(result_constraint_expression.is_satisfiable());
  // There are now five conjunctions in the disjoint expression, as described
  // above.
  EXPECT_THAT(result_constraint_expression,
              MatchConstraintExpressionString(
                  "d0 in [0, 5] || d1 in [0, 5] || d2 in [0, 5] || "
                  "d3 in [0, 5] || d4 in [0, 5]"));
}

TEST_F(ConstraintExpressionTest,
       CanSimplifyAlwaysSatisfiedContraintExpression) {
  ConstraintExpression constraints = GetConstraint("d0", 0, 1) ||
                                     GetConstraint("25", 0, 100) ||
                                     GetConstraint("1", 1, 1);
  constraints.Simplify();
  EXPECT_THAT(constraints, MatchConstraintExpressionString("always satisfied"));
}

TEST_F(ConstraintExpressionTest, CanSimplifyUnsatisfiableContraintExpression) {
  ConstraintExpression constraints =
      GetConstraint("d0", 0, -1) || GetConstraint("1", 2, 3);
  constraints.Simplify();
  EXPECT_THAT(constraints, MatchConstraintExpressionString("unsatisfiable"));
}

TEST_F(ConstraintExpressionTest,
       CanSimplifyAwayAlwaysSatisfiedPartOfConjunction) {
  EXPECT_THAT(Simplify(GetConstraint("d0", 0, 1) && GetConstraint("1", 1, 1) &&
                       GetConstraint("d1", 0, 1) && GetConstraint("2", 2, 3)),
              MatchConstraintExpressionString("d0 in [0, 1] && d1 in [0, 1]"));
}

TEST_F(ConstraintExpressionTest,
       CanSimplifyAwayUnsatisfiablePartOfDisjunction) {
  EXPECT_THAT(Simplify(GetConstraint("d0", 0, 1) ||
                       (GetConstraint("d1", 0, 1) && GetConstraint("1", 0, 0) &&
                        GetConstraint("d2", 0, 1))),
              MatchConstraintExpressionString("d0 in [0, 1]"));
}

TEST_F(ConstraintExpressionTest, SimplifyRemovesRedundantConstraints) {
  Constraint c0 = GetConstraint("d0", 0, 0);
  Constraint c1 = GetConstraint("d1", 1, 1);
  // We could simplify those contraints even further to `d0 in [0, 0]` by
  // checking that one conjunction is a subset of the other, but we don't do
  // that yet.
  EXPECT_THAT(
      Simplify((c0 && c1) || (c1 && c0) || c0 || (c1 && c0) || (c0 && c1)),
      MatchConstraintExpressionString(
          "d0 in [0, 0] || d0 in [0, 0] && d1 in [1, 1]"));
}

TEST_F(ConstraintExpressionTest, ConstraintSatisfactionIsEvaluatedCorrectly) {
  Constraint c0 = GetConstraint("d0 mod 6", 0, 0);
  Constraint c1 = GetConstraint("d1 mod 8", 0, 0);
  Constraint c2 = GetConstraint("d0 mod 13", 0, 0);
  ConstraintExpression constraints = (c0 && c1) || (c1 && c2);

  // Parameters {6, 8} satisfy these constraints.
  std::vector<int64_t> possible_tile_parameters({6, 8});
  EXPECT_TRUE(constraints.IsSatisfiedBy(possible_tile_parameters));

  // Parameters {13, 8} should also satisfy these constraints.
  std::vector<int64_t> other_possible_tile_parameters({13, 8});
  EXPECT_TRUE(constraints.IsSatisfiedBy(other_possible_tile_parameters));

  // However, tile sizes {6, 7} do not satisfy these constraints.
  std::vector<int64_t> impossible_tile_parameters({6, 7});
  EXPECT_FALSE(constraints.IsSatisfiedBy(impossible_tile_parameters));

  // Anything satisfies an always satisfied constraint expression.
  EXPECT_TRUE(ConstraintExpression::GetAlwaysSatisfied().IsSatisfiedBy(
      impossible_tile_parameters));

  // Nothing satisfies an unsatisfiable constraint expression.
  EXPECT_FALSE(ConstraintExpression::GetUnsatisfiable().IsSatisfiedBy(
      possible_tile_parameters));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/codegen/ir/xla_ops.h"

#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class XLAOpsTest : public HloTestBase {
 public:
  mlir::MLIRContext mlir_context_;
};

std::string VariableConstraintsToString(const IndexingMap& map) {
  std::string result;
  llvm::raw_string_ostream os(result);

  const auto& dim_names = GetDimVarNames(map);
  const auto& symbol_names = GetSymbolVarNames(map);
  auto constraints = GetConstraintsForVariables(map);
  for (const auto& [i, dim_constraints] :
       llvm::enumerate(constraints.constraints_for_dims)) {
    os << dim_names[i] << ": ";
    std::vector<std::string> constraint_strings;
    constraint_strings.reserve(dim_constraints.size());
    for (const auto& [expr, range] : dim_constraints) {
      constraint_strings.push_back(absl::StrCat(
          ToString(expr, dim_names, symbol_names), " in ", range.ToString()));
    }
    std::sort(constraint_strings.begin(), constraint_strings.end());
    if (constraint_strings.empty()) {
      constraint_strings.push_back("no constraints");
    }
    os << absl::StrJoin(constraint_strings, ", ");
    os << "\n";
  }
  for (const auto& [i, symbol_constraints] :
       llvm::enumerate(constraints.constraints_for_symbols)) {
    os << symbol_names[i] << ": ";
    std::vector<std::string> constraint_strings;
    constraint_strings.reserve(symbol_constraints.size());
    for (const auto& [expr, range] : symbol_constraints) {
      constraint_strings.push_back(absl::StrCat(
          ToString(expr, dim_names, symbol_names), " in ", range.ToString()));
    }
    std::sort(constraint_strings.begin(), constraint_strings.end());
    if (constraint_strings.empty()) {
      constraint_strings.push_back("no constraints");
    }
    os << absl::StrJoin(constraint_strings, ", ");
    os << "\n";
  }
  return result;
}

TEST_F(XLAOpsTest, GetConstraintsForVariables) {
  auto map = *ParseIndexingMap(R"(
    (x, y)[s0, w] -> (x + s0, y + w),
    domain: x in [0, 5],
    y in [0, 2],
    s0 in [0, 32],
    w in [0, 1024],
    y + w in [0, 4],
    y mod 32 in [0, 6],
    s0 + w in [0, 3],
    s0 mod 4 in [0, 1],
    w mod 4 in [0, 2],
  )",
                               &mlir_context_);
  EXPECT_EQ(VariableConstraintsToString(map),
            R"(x: no constraints
y: y + w in [0, 4], y mod 32 in [0, 6]
s0: s0 + w in [0, 3], s0 mod 4 in [0, 1]
w: s0 + w in [0, 3], w mod 4 in [0, 2], y + w in [0, 4]
)");
}

TEST_F(XLAOpsTest, GetConstraintsForVariablesEmpty) {
  auto map = *ParseIndexingMap(R"(
    (d0, d1)[s0, s1] -> (d0 + s0, d1 + s1),
    domain: d0 in [0, 5],
    d1 in [0, 2],
    s0 in [0, 32],
    s1 in [0, 1024],
  )",
                               &mlir_context_);
  EXPECT_EQ(VariableConstraintsToString(map),
            R"(d0: no constraints
d1: no constraints
s0: no constraints
s1: no constraints
)");
}

}  // namespace
}  // namespace xla

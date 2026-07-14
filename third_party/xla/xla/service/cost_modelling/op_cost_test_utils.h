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

#ifndef XLA_SERVICE_COST_MODELLING_OP_COST_TEST_UTILS_H_
#define XLA_SERVICE_COST_MODELLING_OP_COST_TEST_UTILS_H_

#include <string>

#include "absl/strings/str_join.h"
#include "xla/tsl/platform/test.h"

namespace xla {

MATCHER_P2(HasCalculationValue, expected_final_result,
           expected_final_result_source,
           absl::StrCat("has value ", expected_final_result, " from source ",
                        expected_final_result_source)) {
  if (!arg.HasValue()) {
    return false;
  }
  if (arg.Value() != expected_final_result) {
    return false;
  }
  if (arg.ValueSource() != expected_final_result_source) {
    return false;
  }

  return true;
}

MATCHER(MissingCalculationValue, "is missing a value") {
  return !arg.HasValue();
}

MATCHER_P(HasCalculatorMapValues, expected_calculator_values,
          absl::StrCat("has calculator values [",
                       absl::StrJoin(expected_calculator_values, ", ",
                                     [](std::string* out, const auto& pair) {
                                       absl::StrAppend(out, pair.first, ": ",
                                                       pair.second.ToString());
                                     }),
                       "]")) {
  return ::testing::Matches(::testing::UnorderedElementsAreArray(
      expected_calculator_values))(arg.GetCalculatorValueMap());
}

}  // namespace xla

#endif  // XLA_SERVICE_COST_MODELLING_OP_COST_TEST_UTILS_H_

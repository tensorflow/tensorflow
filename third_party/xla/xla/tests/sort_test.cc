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

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace {

class SortTest : public HloTestBase {};

XLA_TEST_F(SortTest, SortDim0) {
  absl::string_view hlo_text_module = R"(
    HloModule sort

    compare {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }

    ENTRY e {
      x = f32[32,64] parameter(0)
      ROOT sort = f32[32,64] sort(x), dimensions={0}, to_apply=compare
    }
  )";

  EXPECT_TRUE(RunAndCompare(hlo_text_module, ErrorSpec{0.0, 0.0}));
}

XLA_TEST_F(SortTest, SortDim1) {
  absl::string_view hlo_text_module = R"(
    HloModule sort

    compare {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }

    ENTRY e {
      x = f32[32,64] parameter(0)
      ROOT sort = f32[32,64] sort(x), dimensions={1}, to_apply=compare
    }
  )";

  EXPECT_TRUE(RunAndCompare(hlo_text_module, ErrorSpec{0.0, 0.0}));
}

XLA_TEST_F(SortTest, SortTwiceWithSameComparator) {
  absl::string_view hlo_text_module = R"(
    HloModule sort

    compare {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }

    ENTRY e {
      x = f32[32,64] parameter(0)
      y = f32[64,32] parameter(1)
      sort_x = f32[32,64] sort(x), dimensions={0}, to_apply=compare
      sort_y = f32[64,32] sort(y), dimensions={1}, to_apply=compare
      ROOT tuple = (f32[32,64], f32[64,32]) tuple(sort_x, sort_y)
    }
  )";

  EXPECT_TRUE(RunAndCompare(hlo_text_module, ErrorSpec{0.0, 0.0}));
}

class SortManyInputsTest : public SortTest,
                           public ::testing::WithParamInterface<int> {
 public:
  static std::string Name(const ::testing::TestParamInfo<int>& info) {
    auto num_inputs = info.param;
    return absl::StrFormat("Sort%dInputs", num_inputs);
  }
};

XLA_TEST_P(SortManyInputsTest, SortManyInputs) {
  int num_inputs = GetParam();
  absl::string_view hlo_text_module_template = R"(
    HloModule sort

    compare {
      ${COMPARE_DECLARATIONS}
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }

    ENTRY e {
      ${SORT_DECLARATIONS}
      ROOT sort = (${SORT_SHAPE}) sort(${SORT_PARAMS}), dimensions={0}, 
        to_apply=compare
    }
  )";

  // Prepare values for template substitutions.
  std::string sort_decls = "";
  std::vector<std::string> param_names;
  param_names.reserve(num_inputs * 2);
  for (int i = 0; i < num_inputs; ++i) {
    sort_decls += absl::StrFormat("p%d = f32[32,64] parameter(%d)\n", i, i);
    param_names.emplace_back(absl::StrCat("p", i));
  }
  std::string sort_params = absl::StrJoin(param_names, ", ");
  std::string sort_shape =
      absl::StrJoin(std::vector<std::string>(num_inputs, "f32[32,64]"), ",");
  std::string compare_decls = "";
  for (int i = 0; i < num_inputs * 2; ++i) {
    compare_decls += absl::StrFormat("p%d = f32[] parameter(%d)\n", i, i);
  }
  std::string compare_params = absl::StrJoin(param_names, ", ");

  // Finalize HLO text.
  std::string hlo_text_module = absl::StrReplaceAll(
      hlo_text_module_template, {{"${SORT_DECLARATIONS}", sort_decls},
                                 {"${SORT_SHAPE}", sort_shape},
                                 {"${SORT_PARAMS}", sort_params},
                                 {"${COMPARE_DECLARATIONS}", compare_decls}});
  EXPECT_TRUE(RunAndCompare(hlo_text_module, ErrorSpec{0.0, 0.0}));
}

INSTANTIATE_TEST_SUITE_P(ManyInputs, SortManyInputsTest,
                         ::testing::Values(17, 20), SortManyInputsTest::Name);

}  // namespace
}  // namespace xla

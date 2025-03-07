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

#include "xla/python/dlpack_strides.h"

#include <cstdint>
#include <tuple>
#include <vector>

#include "absl/types/span.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

typedef std::tuple<std::vector<int64_t>, std::vector<int64_t>,
                   std::vector<int64_t>>
    StridesToLayoutTestCase;

class DlpackStridesTestSuite
    : public testing::TestWithParam<StridesToLayoutTestCase> {};

TEST_P(DlpackStridesTestSuite, StridesToLayout) {
  auto [dims, strides, expected_layout] = GetParam();
  auto layout = StridesToLayout(absl::MakeSpan(dims), absl::MakeSpan(strides));
  EXPECT_TRUE(layout.ok());
  EXPECT_EQ(layout.value(), expected_layout);
}

INSTANTIATE_TEST_SUITE_P(StridesToLayout, DlpackStridesTestSuite,
                         testing::ValuesIn<StridesToLayoutTestCase>({
                             {{}, {}, {}},
                             {{2, 3, 4}, {12, 4, 1}, {2, 1, 0}},
                             {{2, 3, 4}, {1, 2, 6}, {0, 1, 2}},
                             {{2, 1, 3, 4}, {12, 12, 4, 1}, {3, 2, 1, 0}},
                             {{2, 1, 3, 4}, {12, 1, 4, 1}, {3, 2, 1, 0}},
                             {{1, 1}, {1, 100}, {1, 0}},
                             {{1, 1, 4}, {1, 100, 1}, {2, 1, 0}},
                             {{4, 1, 1}, {1, 100, 1}, {2, 1, 0}},
                             // When there is a unit dimension, but the other
                             // strides are not row-major, we choose to make
                             // the layout as close to row-major as possible.
                             {{2, 1, 3, 4}, {1, 2, 2, 6}, {0, 2, 1, 3}},
                         }));

}  // namespace
}  // namespace xla

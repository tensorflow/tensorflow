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

#include <utility>

#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"
#include "xla/tests/exhaustive/exhaustive_unary_test_definitions.h"
#include "tsl/platform/test.h"

namespace xla {
namespace exhaustive_op_test {
namespace {

#ifdef XLA_BACKEND_SUPPORTS_BFLOAT16
INSTANTIATE_TEST_SUITE_P(BF16, ExhaustiveBF16UnaryTest,
                         ::testing::Values(std::make_pair(0, 1 << 16)));
#else
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveBF16UnaryTest);
#endif

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
INSTANTIATE_TEST_SUITE_P(F16, ExhaustiveF16UnaryTest,
                         ::testing::Values(std::make_pair(0, 1 << 16)));
#else
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF16UnaryTest);
#endif

INSTANTIATE_TEST_SUITE_P(F32, ExhaustiveF32UnaryTest,
                         ::testing::ValuesIn(CreateExhaustiveF32Ranges()));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF64UnaryTest);

}  // namespace
}  // namespace exhaustive_op_test
}  // namespace xla

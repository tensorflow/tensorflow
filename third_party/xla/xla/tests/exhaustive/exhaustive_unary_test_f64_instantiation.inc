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

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF8E4M3FNUnaryTest);

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF8E5M2UnaryTest);

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveBF16UnaryTest);

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF16UnaryTest);

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF32UnaryTest);

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)
INSTANTIATE_TEST_SUITE_P(
    SpecialValues, ExhaustiveF64UnaryTest,
    ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()));

INSTANTIATE_TEST_SUITE_P(NormalValues, ExhaustiveF64UnaryTest,
                         ::testing::Values(GetNormals<double>(1000)));

// Tests a total of 4,000,000,000 inputs, with 16,000,000 inputs in each
// sub-test, to keep the peak memory usage low.
INSTANTIATE_TEST_SUITE_P(
    LargeAndSmallMagnitudeNormalValues, ExhaustiveF64UnaryTest,
    ::testing::ValuesIn(GetFpValuesForMagnitudeExtremeNormals<double>(
        4000000000ull, 16000000)));
#else
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF64UnaryTest);
#endif  // !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)

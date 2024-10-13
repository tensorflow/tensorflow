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

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF8E4M3FNBinaryTest);

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF8E5M2BinaryTest);

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveBF16BinaryTest);

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF16BinaryTest);

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF32BinaryTest);

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)
INSTANTIATE_TEST_SUITE_P(
    SpecialValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>())));

INSTANTIATE_TEST_SUITE_P(
    SpecialAndNormalValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>()),
        ::testing::Values(GetNormals<double>(1000))));

INSTANTIATE_TEST_SUITE_P(
    NormalAndSpecialValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(
        ::testing::Values(GetNormals<double>(1000)),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<double>())));

INSTANTIATE_TEST_SUITE_P(
    NormalAndNormalValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(::testing::Values(GetNormals<double>(1000)),
                       ::testing::Values(GetNormals<double>(1000))));

// Tests a total of 40000 ^ 2 inputs, with 2000 ^ 2 inputs in each sub-test.
// Similar to ExhaustiveF64BinaryTest, we use a smaller set of inputs for each
// for each sub-test comparing with the unary test to avoid timeout.
INSTANTIATE_TEST_SUITE_P(
    LargeAndSmallMagnitudeNormalValues, ExhaustiveF64BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<double>(40000, 2000)),
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<double>(40000, 2000))));
#else
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF64BinaryTest);
#endif  // !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT64)

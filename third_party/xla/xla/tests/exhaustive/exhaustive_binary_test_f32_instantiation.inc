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

INSTANTIATE_TEST_SUITE_P(
    SpecialValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>()),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>())));

INSTANTIATE_TEST_SUITE_P(
    SpecialAndNormalValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>()),
        ::testing::Values(GetNormals<float>(2000))));

INSTANTIATE_TEST_SUITE_P(
    NormalAndSpecialValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(
        ::testing::Values(GetNormals<float>(2000)),
        ::testing::ValuesIn(CreateFpValuesForBoundaryTest<float>())));

INSTANTIATE_TEST_SUITE_P(
    NormalAndNormalValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(::testing::Values(GetNormals<float>(2000)),
                       ::testing::Values(GetNormals<float>(2000))));

// Tests a total of 40000 ^ 2 inputs, with 2000 ^ 2 inputs in each sub-test.
// Comparing with the unary tests, the binary tests use a smaller set of inputs
// for each sub-test to avoid timeout because the implementation of ExpectNear
// more than 2x slower for binary test.
INSTANTIATE_TEST_SUITE_P(
    LargeAndSmallMagnitudeNormalValues, ExhaustiveF32BinaryTest,
    ::testing::Combine(
        ::testing::ValuesIn(GetFpValuesForMagnitudeExtremeNormals<float>(40000,
                                                                         2000)),
        ::testing::ValuesIn(
            GetFpValuesForMagnitudeExtremeNormals<float>(40000, 2000))));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF64BinaryTest);

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

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_F8E4M3FN)
INSTANTIATE_TEST_SUITE_P(F8E4M3FN, ExhaustiveF8E4M3FNBinaryTest,
                         ::testing::ValuesIn(CreateExhaustiveU16Ranges()));
#else
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF8E4M3FNBinaryTest);
#endif

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_F8E5M2)
INSTANTIATE_TEST_SUITE_P(F8E5M2, ExhaustiveF8E5M2BinaryTest,
                         ::testing::ValuesIn(CreateExhaustiveU16Ranges()));
#else
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF8E5M2BinaryTest);
#endif

INSTANTIATE_TEST_SUITE_P(BF16, ExhaustiveBF16BinaryTest,
                         ::testing::ValuesIn(CreateExhaustiveU32Ranges()));

#if !defined(XLA_BACKEND_DOES_NOT_SUPPORT_FLOAT16)
INSTANTIATE_TEST_SUITE_P(F16, ExhaustiveF16BinaryTest,
                         ::testing::ValuesIn(CreateExhaustiveU32Ranges()));
#else
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF16BinaryTest);
#endif

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF32BinaryTest);

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExhaustiveF64BinaryTest);

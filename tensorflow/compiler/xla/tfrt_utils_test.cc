// Copyright 2022 The TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/xla/tfrt_utils.h"

#include <array>

#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace {

using ::tfrt::DType;

struct TfrtToPrimitiveTypeTestCase {
  DType dtype;
  PrimitiveType primitive_type;
};

constexpr std::array<TfrtToPrimitiveTypeTestCase, 15>
    kTfrtToPrimitiveTypeTestCases = {{
        {DType::UI8, PrimitiveType::U8},
        {DType::UI16, PrimitiveType::U16},
        {DType::UI32, PrimitiveType::U32},
        {DType::UI64, PrimitiveType::U64},
        {DType::I1, PrimitiveType::PRED},
        {DType::I8, PrimitiveType::S8},
        {DType::I16, PrimitiveType::S16},
        {DType::I32, PrimitiveType::S32},
        {DType::I64, PrimitiveType::S64},
        {DType::F16, PrimitiveType::F16},
        {DType::F32, PrimitiveType::F32},
        {DType::F64, PrimitiveType::F64},
        {DType::BF16, PrimitiveType::BF16},
        {DType::Complex64, PrimitiveType::C64},
        {DType::Complex128, PrimitiveType::C128},
    }};

class TfrtToPrimitiveTypeTest
    : public ::testing::TestWithParam<TfrtToPrimitiveTypeTestCase> {};

TEST_P(TfrtToPrimitiveTypeTest, TfrtToPrimitiveType) {
  EXPECT_EQ(GetParam().primitive_type, TfrtToPrimitiveType(GetParam().dtype));
}

INSTANTIATE_TEST_SUITE_P(TfrtToPrimitiveType, TfrtToPrimitiveTypeTest,
                         ::testing::ValuesIn(kTfrtToPrimitiveTypeTestCases));

}  // namespace
}  // namespace xla

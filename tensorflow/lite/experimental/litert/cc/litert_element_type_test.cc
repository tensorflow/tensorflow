// Copyright 2024 Google LLC.
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

#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

namespace litert {

namespace {

template <typename T>
class ElementTypeTest : public ::testing::Test {
 public:
  size_t Size() const { return sizeof(T); }
};

TYPED_TEST_SUITE_P(ElementTypeTest);

TYPED_TEST_P(ElementTypeTest, TypeAndSize) {
  const size_t size = GetByteWidth<GetElementType<TypeParam>()>();
  EXPECT_EQ(size, this->Size());
}

REGISTER_TYPED_TEST_SUITE_P(ElementTypeTest, TypeAndSize);

using Types =
    ::testing::Types<bool, uint8_t, int8_t, int16_t, uint16_t, uint32_t,
                     int32_t, uint64_t, int64_t, float, double>;

INSTANTIATE_TYPED_TEST_SUITE_P(ElementTypeTestSuite, ElementTypeTest, Types);

}  // namespace
}  // namespace litert

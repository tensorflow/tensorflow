/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");;
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/function/trace_type/standard/primitive_types.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace trace_type {

TEST(PrimitiveTypesTest, LiteralIntType) {
  Literal<int> int_1 = Literal<int>(1);
  std::unique_ptr<TraceType> int_1_copy = int_1.clone();
  Literal<int> int_2 = Literal<int>(2);

  EXPECT_EQ(int_1.to_string(), "Int<1>");
  EXPECT_EQ(int_1_copy->to_string(), "Int<1>");
  EXPECT_EQ(int_2.to_string(), "Int<2>");

  EXPECT_EQ(int_1, *int_1_copy);
  EXPECT_EQ(int_1.hash(), int_1_copy->hash());
  EXPECT_NE(int_1, int_2);

  EXPECT_TRUE(int_1.is_subtype_of(*int_1_copy));
  EXPECT_FALSE(int_1.is_subtype_of(int_2));

  std::unique_ptr<TraceType> result =
      int_1.most_specific_common_supertype({&int_1});
  EXPECT_EQ(*result, int_1);
  EXPECT_EQ(int_1.most_specific_common_supertype({&int_2}), nullptr);
}

TEST(PrimitiveTypesTest, LiteralBoolType) {
  Literal<bool> bool_1 = Literal<bool>(true);
  std::unique_ptr<TraceType> bool_1_copy = bool_1.clone();
  Literal<bool> bool_2 = Literal<bool>(false);

  EXPECT_EQ(bool_1.to_string(), "Bool<True>");
  EXPECT_EQ(bool_1_copy->to_string(), "Bool<True>");
  EXPECT_EQ(bool_2.to_string(), "Bool<False>");

  EXPECT_EQ(bool_1, *bool_1_copy);
  EXPECT_EQ(bool_1.hash(), bool_1_copy->hash());
  EXPECT_NE(bool_1, bool_2);

  EXPECT_TRUE(bool_1.is_subtype_of(*bool_1_copy));
  EXPECT_FALSE(bool_1.is_subtype_of(bool_2));

  std::unique_ptr<TraceType> result =
      bool_1.most_specific_common_supertype({&bool_1});
  EXPECT_EQ(*result, bool_1);
  EXPECT_EQ(bool_1.most_specific_common_supertype({&bool_2}), nullptr);
}

TEST(PrimitiveTypesTest, LiteralStringType) {
  Literal<std::string> string_1 = Literal<std::string>("a");
  std::unique_ptr<TraceType> string_1_copy = string_1.clone();
  Literal<std::string> string_2 = Literal<std::string>("b");

  EXPECT_EQ(string_1.to_string(), "String<a>");
  EXPECT_EQ(string_1_copy->to_string(), "String<a>");
  EXPECT_EQ(string_2.to_string(), "String<b>");

  EXPECT_EQ(string_1, *string_1_copy);
  EXPECT_EQ(string_1.hash(), string_1_copy->hash());
  EXPECT_NE(string_1, string_2);

  EXPECT_TRUE(string_1.is_subtype_of(*string_1_copy));
  EXPECT_FALSE(string_1.is_subtype_of(string_2));

  std::unique_ptr<TraceType> result =
      string_1.most_specific_common_supertype({&string_1});
  EXPECT_EQ(*result, string_1);
  EXPECT_EQ(string_1.most_specific_common_supertype({&string_2}), nullptr);
}

}  // namespace trace_type
}  // namespace tensorflow

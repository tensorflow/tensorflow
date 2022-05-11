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

TEST(PrimitiveTypesTest, NoneType) {
  None none_1 = None();
  std::unique_ptr<TraceType> none_1_copy = none_1.clone();
  Literal<std::string> string = Literal<std::string>("b");

  EXPECT_EQ(none_1.to_string(), "None<>");
  EXPECT_EQ(none_1_copy->to_string(), "None<>");

  EXPECT_EQ(none_1, *none_1_copy);
  EXPECT_EQ(none_1.hash(), none_1_copy->hash());
  EXPECT_NE(none_1, string);

  EXPECT_TRUE(none_1.is_subtype_of(*none_1_copy));
  EXPECT_FALSE(none_1.is_subtype_of(string));

  std::unique_ptr<TraceType> result =
      none_1.most_specific_common_supertype({&none_1});
  EXPECT_EQ(*result, none_1);
  EXPECT_EQ(none_1.most_specific_common_supertype({&string}), nullptr);
}

TEST(PrimitiveTypesTest, Any) {
  Any any_1 = Any(std::make_unique<Literal<std::string>>("a"));
  std::unique_ptr<TraceType> any_1_copy = any_1.clone();
  Any any_2 = Any(absl::nullopt);

  EXPECT_EQ(any_1.to_string(), "Any<String<a>>");
  EXPECT_EQ(any_1_copy->to_string(), "Any<String<a>>");
  EXPECT_EQ(any_2.to_string(), "Any<Any>");

  EXPECT_EQ(any_1, *any_1_copy);
  EXPECT_EQ(any_1.hash(), any_1_copy->hash());
  EXPECT_NE(any_1, any_2);

  EXPECT_TRUE(any_1.is_subtype_of(*any_1_copy));
  EXPECT_TRUE(any_1.is_subtype_of(any_2));
  EXPECT_FALSE(any_2.is_subtype_of(any_1));

  std::unique_ptr<TraceType> result_1 =
      any_1.most_specific_common_supertype({&any_1});
  EXPECT_EQ(*result_1, any_1);
  EXPECT_EQ(any_1.most_specific_common_supertype({any_1.base().value()}),
            nullptr);
  std::unique_ptr<TraceType> result_2(
      any_1.most_specific_common_supertype({&any_2}));
  EXPECT_EQ(*result_2, any_2);
}

TEST(PrimitiveTypesTest, ProductOfLiterals) {
  std::vector<std::unique_ptr<TraceType>> elements;
  elements.push_back(std::make_unique<Literal<std::string>>("a"));
  elements.push_back(std::make_unique<Literal<int>>(33));
  elements.push_back(std::make_unique<Literal<bool>>(true));

  Product product_1 = Product(std::move(elements));
  std::unique_ptr<TraceType> product_1_copy = product_1.clone();

  std::vector<std::unique_ptr<TraceType>> elements_2;
  elements_2.push_back(std::make_unique<Literal<std::string>>("b"));
  elements_2.push_back(std::make_unique<Literal<int>>(34));
  elements_2.push_back(std::make_unique<Literal<bool>>(false));
  Product product_2 = Product(std::move(elements_2));

  EXPECT_EQ(product_1.to_string(), "Product<String<a>, Int<33>, Bool<True>>");
  EXPECT_EQ(product_1_copy->to_string(),
            "Product<String<a>, Int<33>, Bool<True>>");
  EXPECT_EQ(product_2.to_string(), "Product<String<b>, Int<34>, Bool<False>>");

  EXPECT_EQ(product_1, *product_1_copy);
  EXPECT_EQ(product_1.hash(), product_1_copy->hash());
  EXPECT_NE(product_1, product_2);

  EXPECT_TRUE(product_1.is_subtype_of(*product_1_copy));
  EXPECT_FALSE(product_1.is_subtype_of(product_2));

  std::unique_ptr<TraceType> result =
      product_1.most_specific_common_supertype({&product_1});
  EXPECT_EQ(*result, product_1);
  EXPECT_EQ(product_1.most_specific_common_supertype({&product_2}), nullptr);
}

TEST(PrimitiveTypesTest, ProductOfAny) {
  std::vector<std::unique_ptr<TraceType>> elements;
  elements.push_back(
      std::make_unique<Any>(std::make_unique<Literal<std::string>>("a")));
  elements.push_back(std::make_unique<Any>(std::make_unique<Literal<int>>(33)));

  Product product_1 = Product(std::move(elements));
  std::unique_ptr<TraceType> product_1_copy = product_1.clone();

  std::vector<std::unique_ptr<TraceType>> elements_2;
  elements_2.push_back(std::make_unique<Any>(absl::nullopt));
  elements_2.push_back(
      std::make_unique<Any>(std::make_unique<Literal<int>>(33)));
  Product product_2 = Product(std::move(elements_2));

  EXPECT_EQ(product_1.to_string(), "Product<Any<String<a>>, Any<Int<33>>>");
  EXPECT_EQ(product_1_copy->to_string(),
            "Product<Any<String<a>>, Any<Int<33>>>");
  EXPECT_EQ(product_2.to_string(), "Product<Any<Any>, Any<Int<33>>>");

  EXPECT_EQ(product_1, *product_1_copy);
  EXPECT_EQ(product_1.hash(), product_1_copy->hash());
  EXPECT_NE(product_1, product_2);

  EXPECT_TRUE(product_1.is_subtype_of(*product_1_copy));
  EXPECT_TRUE(product_1.is_subtype_of(product_2));
  EXPECT_FALSE(product_2.is_subtype_of(product_1));

  EXPECT_EQ(*product_1.most_specific_common_supertype({&product_1}), product_1);
  EXPECT_EQ(*product_1.most_specific_common_supertype({&product_2}), product_2);
}

TEST(PrimitiveTypesTest, RecordTypeLiterals) {
  std::vector<std::unique_ptr<TraceType>> keys;
  std::vector<std::unique_ptr<TraceType>> values;

  keys.push_back(std::make_unique<Literal<std::string>>("a"));
  values.push_back(std::make_unique<Literal<int>>(33));

  Record record_1 = Record(std::move(keys), std::move(values));
  std::unique_ptr<TraceType> record_1_copy = record_1.clone();

  std::vector<std::unique_ptr<TraceType>> keys_2;
  std::vector<std::unique_ptr<TraceType>> values_2;
  keys_2.push_back(std::make_unique<Literal<std::string>>("b"));
  values_2.push_back(std::make_unique<Literal<int>>(34));
  Record record_2 = Record(std::move(keys_2), std::move(values_2));

  EXPECT_EQ(record_1.to_string(), "Record<String<a>:Int<33>>");
  EXPECT_EQ(record_1_copy->to_string(), "Record<String<a>:Int<33>>");
  EXPECT_EQ(record_2.to_string(), "Record<String<b>:Int<34>>");

  EXPECT_EQ(record_1, *record_1_copy);
  EXPECT_EQ(record_1.hash(), record_1_copy->hash());
  EXPECT_NE(record_1, record_2);

  EXPECT_TRUE(record_1.is_subtype_of(*record_1_copy));
  EXPECT_FALSE(record_1.is_subtype_of(record_2));

  std::unique_ptr<TraceType> result =
      record_1.most_specific_common_supertype({&record_1});
  EXPECT_EQ(*result, record_1);
  EXPECT_EQ(record_1.most_specific_common_supertype({&record_2}), nullptr);
}

TEST(PrimitiveTypesTest, RecordTypeAnys) {
  std::vector<std::unique_ptr<TraceType>> keys;
  std::vector<std::unique_ptr<TraceType>> values;

  keys.push_back(std::make_unique<Literal<std::string>>("a"));
  values.push_back(std::make_unique<Any>(std::make_unique<Literal<int>>(33)));

  Record record_1 = Record(std::move(keys), std::move(values));
  std::unique_ptr<TraceType> record_1_copy = record_1.clone();

  std::vector<std::unique_ptr<TraceType>> keys_2;
  std::vector<std::unique_ptr<TraceType>> values_2;
  keys_2.push_back(std::make_unique<Literal<std::string>>("a"));
  values_2.push_back(std::make_unique<Any>(std::make_unique<Literal<int>>(34)));
  Record record_2 = Record(std::move(keys_2), std::move(values_2));

  EXPECT_EQ(record_1.to_string(), "Record<String<a>:Any<Int<33>>>");
  EXPECT_EQ(record_1_copy->to_string(), "Record<String<a>:Any<Int<33>>>");
  EXPECT_EQ(record_2.to_string(), "Record<String<a>:Any<Int<34>>>");

  EXPECT_EQ(record_1, *record_1_copy);
  EXPECT_EQ(record_1.hash(), record_1_copy->hash());
  EXPECT_NE(record_1, record_2);

  EXPECT_TRUE(record_1.is_subtype_of(*record_1_copy));
  EXPECT_FALSE(record_1.is_subtype_of(record_2));
  EXPECT_FALSE(record_2.is_subtype_of(record_1));

  EXPECT_EQ(*record_1.most_specific_common_supertype({&record_1}), record_1);
  EXPECT_EQ(*record_2.most_specific_common_supertype({&record_2}), record_2);

  std::vector<std::unique_ptr<TraceType>> keys_3;
  std::vector<std::unique_ptr<TraceType>> values_3;
  keys_3.push_back(std::make_unique<Literal<std::string>>("a"));
  values_3.push_back(std::make_unique<Any>(absl::nullopt));
  Record supertype = Record(std::move(keys_3), std::move(values_3));

  EXPECT_EQ(*record_1.most_specific_common_supertype({&record_2}), supertype);
}

}  // namespace trace_type
}  // namespace tensorflow

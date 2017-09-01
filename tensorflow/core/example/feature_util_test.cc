/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/example/feature_util.h"

#include <vector>

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

const float kTolerance = 1e-5;

TEST(GetFeatureValuesInt64Test, ReadsASingleValue) {
  Example example;
  (*example.mutable_features()->mutable_feature())["tag"]
      .mutable_int64_list()
      ->add_value(42);

  auto tag = GetFeatureValues<protobuf_int64>("tag", example);

  ASSERT_EQ(1, tag.size());
  EXPECT_EQ(42, tag.Get(0));
}

TEST(GetFeatureValuesInt64Test, WritesASingleValue) {
  Example example;

  GetFeatureValues<protobuf_int64>("tag", &example)->Add(42);

  ASSERT_EQ(1,
            example.features().feature().at("tag").int64_list().value_size());
  EXPECT_EQ(42, example.features().feature().at("tag").int64_list().value(0));
}

TEST(GetFeatureValuesInt64Test, CheckUntypedFieldExistence) {
  Example example;

  EXPECT_FALSE(ExampleHasFeature("tag", example));

  GetFeatureValues<protobuf_int64>("tag", &example)->Add(0);

  EXPECT_TRUE(ExampleHasFeature("tag", example));
}

TEST(GetFeatureValuesInt64Test, CheckTypedFieldExistence) {
  Example example;

  GetFeatureValues<float>("tag", &example)->Add(3.14);
  ASSERT_FALSE(ExampleHasFeature<protobuf_int64>("tag", example));

  GetFeatureValues<protobuf_int64>("tag", &example)->Add(42);

  EXPECT_TRUE(ExampleHasFeature<protobuf_int64>("tag", example));
  auto tag_ro = GetFeatureValues<protobuf_int64>("tag", example);
  ASSERT_EQ(1, tag_ro.size());
  EXPECT_EQ(42, tag_ro.Get(0));
}

TEST(GetFeatureValuesInt64Test, CopyIterableToAField) {
  Example example;
  std::vector<int> values{1, 2, 3};

  std::copy(values.begin(), values.end(),
            protobuf::RepeatedFieldBackInserter(
                GetFeatureValues<protobuf_int64>("tag", &example)));

  auto tag_ro = GetFeatureValues<protobuf_int64>("tag", example);
  ASSERT_EQ(3, tag_ro.size());
  EXPECT_EQ(1, tag_ro.Get(0));
  EXPECT_EQ(2, tag_ro.Get(1));
  EXPECT_EQ(3, tag_ro.Get(2));
}

TEST(GetFeatureValuesFloatTest, ReadsASingleValue) {
  Example example;
  (*example.mutable_features()->mutable_feature())["tag"]
      .mutable_float_list()
      ->add_value(3.14);

  auto tag = GetFeatureValues<float>("tag", example);

  ASSERT_EQ(1, tag.size());
  EXPECT_NEAR(3.14, tag.Get(0), kTolerance);
}

TEST(GetFeatureValuesFloatTest, WritesASingleValue) {
  Example example;

  GetFeatureValues<float>("tag", &example)->Add(3.14);

  ASSERT_EQ(1,
            example.features().feature().at("tag").float_list().value_size());
  EXPECT_NEAR(3.14,
              example.features().feature().at("tag").float_list().value(0),
              kTolerance);
}

TEST(GetFeatureValuesFloatTest, CheckTypedFieldExistence) {
  Example example;

  GetFeatureValues<protobuf_int64>("tag", &example)->Add(42);
  ASSERT_FALSE(ExampleHasFeature<float>("tag", example));

  GetFeatureValues<float>("tag", &example)->Add(3.14);

  EXPECT_TRUE(ExampleHasFeature<float>("tag", example));
  auto tag_ro = GetFeatureValues<float>("tag", example);
  ASSERT_EQ(1, tag_ro.size());
  EXPECT_NEAR(3.14, tag_ro.Get(0), kTolerance);
}

TEST(GetFeatureValuesStringTest, ReadsASingleValue) {
  Example example;
  (*example.mutable_features()->mutable_feature())["tag"]
      .mutable_bytes_list()
      ->add_value("FOO");

  auto tag = GetFeatureValues<string>("tag", example);

  ASSERT_EQ(1, tag.size());
  EXPECT_EQ("FOO", tag.Get(0));
}

TEST(GetFeatureValuesStringTest, WritesASingleValue) {
  Example example;

  *GetFeatureValues<string>("tag", &example)->Add() = "FOO";

  ASSERT_EQ(1,
            example.features().feature().at("tag").bytes_list().value_size());
  EXPECT_EQ("FOO",
            example.features().feature().at("tag").bytes_list().value(0));
}

TEST(GetFeatureValuesBytesTest, CheckTypedFieldExistence) {
  Example example;

  GetFeatureValues<protobuf_int64>("tag", &example)->Add(42);
  ASSERT_FALSE(ExampleHasFeature<string>("tag", example));

  *GetFeatureValues<string>("tag", &example)->Add() = "FOO";

  EXPECT_TRUE(ExampleHasFeature<string>("tag", example));
  auto tag_ro = GetFeatureValues<string>("tag", example);
  ASSERT_EQ(1, tag_ro.size());
  EXPECT_EQ("FOO", tag_ro.Get(0));
}

TEST(AppendFeatureValuesTest, FloatValuesFromContainer) {
  Example example;

  std::vector<double> values{1.1, 2.2, 3.3};
  AppendFeatureValues(values, "tag", &example);

  auto tag_ro = GetFeatureValues<float>("tag", example);
  ASSERT_EQ(3, tag_ro.size());
  EXPECT_NEAR(1.1, tag_ro.Get(0), kTolerance);
  EXPECT_NEAR(2.2, tag_ro.Get(1), kTolerance);
  EXPECT_NEAR(3.3, tag_ro.Get(2), kTolerance);
}

TEST(AppendFeatureValuesTest, FloatValuesUsingInitializerList) {
  Example example;

  AppendFeatureValues({1.1, 2.2, 3.3}, "tag", &example);

  auto tag_ro = GetFeatureValues<float>("tag", example);
  ASSERT_EQ(3, tag_ro.size());
  EXPECT_NEAR(1.1, tag_ro.Get(0), kTolerance);
  EXPECT_NEAR(2.2, tag_ro.Get(1), kTolerance);
  EXPECT_NEAR(3.3, tag_ro.Get(2), kTolerance);
}

TEST(AppendFeatureValuesTest, Int64ValuesUsingInitializerList) {
  Example example;

  std::vector<protobuf_int64> values{1, 2, 3};
  AppendFeatureValues(values, "tag", &example);

  auto tag_ro = GetFeatureValues<protobuf_int64>("tag", example);
  ASSERT_EQ(3, tag_ro.size());
  EXPECT_EQ(1, tag_ro.Get(0));
  EXPECT_EQ(2, tag_ro.Get(1));
  EXPECT_EQ(3, tag_ro.Get(2));
}

TEST(AppendFeatureValuesTest, StringValuesUsingInitializerList) {
  Example example;

  AppendFeatureValues({"FOO", "BAR", "BAZ"}, "tag", &example);

  auto tag_ro = GetFeatureValues<string>("tag", example);
  ASSERT_EQ(3, tag_ro.size());
  EXPECT_EQ("FOO", tag_ro.Get(0));
  EXPECT_EQ("BAR", tag_ro.Get(1));
  EXPECT_EQ("BAZ", tag_ro.Get(2));
}

TEST(AppendFeatureValuesTest, StringVariablesUsingInitializerList) {
  Example example;

  string string1("FOO");
  string string2("BAR");
  string string3("BAZ");

  AppendFeatureValues({string1, string2, string3}, "tag", &example);

  auto tag_ro = GetFeatureValues<string>("tag", example);
  ASSERT_EQ(3, tag_ro.size());
  EXPECT_EQ("FOO", tag_ro.Get(0));
  EXPECT_EQ("BAR", tag_ro.Get(1));
  EXPECT_EQ("BAZ", tag_ro.Get(2));
}

}  // namespace
}  // namespace tensorflow

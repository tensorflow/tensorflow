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

TEST(GetFeatureValuesInt64Test, ReadsASingleValueFromFeature) {
  Feature feature;
  feature.mutable_int64_list()->add_value(42);

  auto values = GetFeatureValues<protobuf_int64>(feature);

  ASSERT_EQ(1, values.size());
  EXPECT_EQ(42, values.Get(0));
}

TEST(GetFeatureValuesInt64Test, WritesASingleValue) {
  Example example;

  GetFeatureValues<protobuf_int64>("tag", &example)->Add(42);

  ASSERT_EQ(1,
            example.features().feature().at("tag").int64_list().value_size());
  EXPECT_EQ(42, example.features().feature().at("tag").int64_list().value(0));
}

TEST(GetFeatureValuesInt64Test, WritesASingleValueToFeature) {
  Feature feature;

  GetFeatureValues<protobuf_int64>(&feature)->Add(42);

  ASSERT_EQ(1, feature.int64_list().value_size());
  EXPECT_EQ(42, feature.int64_list().value(0));
}

TEST(GetFeatureValuesInt64Test, CheckUntypedFieldExistence) {
  Example example;
  ASSERT_FALSE(HasFeature("tag", example));

  GetFeatureValues<protobuf_int64>("tag", &example)->Add(0);

  EXPECT_TRUE(HasFeature("tag", example));
}

TEST(GetFeatureValuesInt64Test, CheckTypedFieldExistence) {
  Example example;

  GetFeatureValues<float>("tag", &example)->Add(3.14);
  ASSERT_FALSE(HasFeature<protobuf_int64>("tag", example));

  GetFeatureValues<protobuf_int64>("tag", &example)->Add(42);

  EXPECT_TRUE(HasFeature<protobuf_int64>("tag", example));
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

TEST(GetFeatureValuesFloatTest, ReadsASingleValueFromFeature) {
  Feature feature;
  feature.mutable_float_list()->add_value(3.14);

  auto values = GetFeatureValues<float>(feature);

  ASSERT_EQ(1, values.size());
  EXPECT_NEAR(3.14, values.Get(0), kTolerance);
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

TEST(GetFeatureValuesFloatTest, WritesASingleValueToFeature) {
  Feature feature;

  GetFeatureValues<float>(&feature)->Add(3.14);

  ASSERT_EQ(1, feature.float_list().value_size());
  EXPECT_NEAR(3.14, feature.float_list().value(0), kTolerance);
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
  ASSERT_FALSE(HasFeature<float>("tag", example));

  GetFeatureValues<float>("tag", &example)->Add(3.14);

  EXPECT_TRUE(HasFeature<float>("tag", example));
  auto tag_ro = GetFeatureValues<float>("tag", example);
  ASSERT_EQ(1, tag_ro.size());
  EXPECT_NEAR(3.14, tag_ro.Get(0), kTolerance);
}

TEST(GetFeatureValuesFloatTest, CheckTypedFieldExistenceForDeprecatedMethod) {
  Example example;

  GetFeatureValues<protobuf_int64>("tag", &example)->Add(42);
  ASSERT_FALSE(ExampleHasFeature<float>("tag", example));

  GetFeatureValues<float>("tag", &example)->Add(3.14);

  EXPECT_TRUE(ExampleHasFeature<float>("tag", example));
  auto tag_ro = GetFeatureValues<float>("tag", example);
  ASSERT_EQ(1, tag_ro.size());
  EXPECT_NEAR(3.14, tag_ro.Get(0), kTolerance);
}

TEST(GetFeatureValuesStringTest, ReadsASingleValueFromFeature) {
  Feature feature;
  feature.mutable_bytes_list()->add_value("FOO");

  auto values = GetFeatureValues<string>(feature);

  ASSERT_EQ(1, values.size());
  EXPECT_EQ("FOO", values.Get(0));
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

TEST(GetFeatureValuesStringTest, WritesASingleValueToFeature) {
  Feature feature;

  *GetFeatureValues<string>(&feature)->Add() = "FOO";

  ASSERT_EQ(1, feature.bytes_list().value_size());
  EXPECT_EQ("FOO", feature.bytes_list().value(0));
}

TEST(GetFeatureValuesStringTest, WritesASingleValue) {
  Example example;

  *GetFeatureValues<string>("tag", &example)->Add() = "FOO";

  ASSERT_EQ(1,
            example.features().feature().at("tag").bytes_list().value_size());
  EXPECT_EQ("FOO",
            example.features().feature().at("tag").bytes_list().value(0));
}

TEST(GetFeatureValuesStringTest, CheckTypedFieldExistence) {
  Example example;

  GetFeatureValues<protobuf_int64>("tag", &example)->Add(42);
  ASSERT_FALSE(HasFeature<string>("tag", example));

  *GetFeatureValues<string>("tag", &example)->Add() = "FOO";

  EXPECT_TRUE(HasFeature<string>("tag", example));
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

TEST(SequenceExampleTest, ReadsASingleValueFromContext) {
  SequenceExample se;
  (*se.mutable_context()->mutable_feature())["tag"]
      .mutable_int64_list()
      ->add_value(42);

  auto values = GetFeatureValues<protobuf_int64>("tag", se.context());

  ASSERT_EQ(1, values.size());
  EXPECT_EQ(42, values.Get(0));
}

TEST(SequenceExampleTest, WritesASingleValueToContext) {
  SequenceExample se;

  GetFeatureValues<protobuf_int64>("tag", se.mutable_context())->Add(42);

  ASSERT_EQ(1, se.context().feature().at("tag").int64_list().value_size());
  EXPECT_EQ(42, se.context().feature().at("tag").int64_list().value(0));
}

TEST(SequenceExampleTest, AppendFeatureValuesToContextSingleArg) {
  SequenceExample se;

  AppendFeatureValues({1.1, 2.2, 3.3}, "tag", se.mutable_context());

  auto tag_ro = GetFeatureValues<float>("tag", se.context());
  ASSERT_EQ(3, tag_ro.size());
  EXPECT_NEAR(1.1, tag_ro.Get(0), kTolerance);
  EXPECT_NEAR(2.2, tag_ro.Get(1), kTolerance);
  EXPECT_NEAR(3.3, tag_ro.Get(2), kTolerance);
}

TEST(SequenceExampleTest, CheckTypedFieldExistence) {
  SequenceExample se;

  GetFeatureValues<float>("tag", se.mutable_context())->Add(3.14);
  ASSERT_FALSE(HasFeature<protobuf_int64>("tag", se.context()));

  GetFeatureValues<protobuf_int64>("tag", se.mutable_context())->Add(42);

  EXPECT_TRUE(HasFeature<protobuf_int64>("tag", se.context()));
  auto tag_ro = GetFeatureValues<protobuf_int64>("tag", se.context());
  ASSERT_EQ(1, tag_ro.size());
  EXPECT_EQ(42, tag_ro.Get(0));
}

TEST(SequenceExampleTest, ReturnsExistingFeatureLists) {
  SequenceExample se;
  (*se.mutable_feature_lists()->mutable_feature_list())["tag"]
      .mutable_feature()
      ->Add();

  auto feature = GetFeatureList("tag", se);

  ASSERT_EQ(1, feature.size());
}

TEST(SequenceExampleTest, CreatesNewFeatureLists) {
  SequenceExample se;

  GetFeatureList("tag", &se)->Add();

  EXPECT_EQ(1, se.feature_lists().feature_list().at("tag").feature_size());
}

TEST(SequenceExampleTest, CheckFeatureListExistence) {
  SequenceExample se;
  ASSERT_FALSE(HasFeatureList("tag", se));

  GetFeatureList("tag", &se)->Add();

  ASSERT_TRUE(HasFeatureList("tag", se));
}

TEST(SequenceExampleTest, AppendFeatureValuesWithInitializerList) {
  SequenceExample se;

  AppendFeatureValues({1, 2, 3}, "ids", se.mutable_context());
  AppendFeatureValues({"cam1-0", "cam2-0"},
                      GetFeatureList("images", &se)->Add());
  AppendFeatureValues({"cam1-1", "cam2-2"},
                      GetFeatureList("images", &se)->Add());

  EXPECT_EQ(se.DebugString(),
            "context {\n"
            "  feature {\n"
            "    key: \"ids\"\n"
            "    value {\n"
            "      int64_list {\n"
            "        value: 1\n"
            "        value: 2\n"
            "        value: 3\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}\n"
            "feature_lists {\n"
            "  feature_list {\n"
            "    key: \"images\"\n"
            "    value {\n"
            "      feature {\n"
            "        bytes_list {\n"
            "          value: \"cam1-0\"\n"
            "          value: \"cam2-0\"\n"
            "        }\n"
            "      }\n"
            "      feature {\n"
            "        bytes_list {\n"
            "          value: \"cam1-1\"\n"
            "          value: \"cam2-2\"\n"
            "        }\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}\n");
}

TEST(SequenceExampleTest, AppendFeatureValuesWithVectors) {
  SequenceExample se;

  std::vector<float> readings{1.0, 2.5, 5.0};
  AppendFeatureValues(readings, GetFeatureList("movie_ratings", &se)->Add());

  EXPECT_EQ(se.DebugString(),
            "feature_lists {\n"
            "  feature_list {\n"
            "    key: \"movie_ratings\"\n"
            "    value {\n"
            "      feature {\n"
            "        float_list {\n"
            "          value: 1\n"
            "          value: 2.5\n"
            "          value: 5\n"
            "        }\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}\n");
}

}  // namespace
}  // namespace tensorflow

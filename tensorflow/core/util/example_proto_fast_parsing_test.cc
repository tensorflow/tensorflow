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

#include "tensorflow/core/util/example_proto_fast_parsing.h"

#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/util/example_proto_fast_parsing_test.pb.h"

namespace tensorflow {
namespace example {
namespace {

constexpr char kDenseInt64Key[] = "dense_int64";
constexpr char kDenseFloatKey[] = "dense_float";
constexpr char kDenseStringKey[] = "dense_string";

constexpr char kSparseInt64Key[] = "sparse_int64";
constexpr char kSparseFloatKey[] = "sparse_float";
constexpr char kSparseStringKey[] = "sparse_string";

std::string SerializedToReadable(std::string serialized) {
  std::string result;
  result += '"';
  for (char c : serialized)
    absl::StrAppend(&result, "\\x", absl::Hex(c, absl::kZeroPad2));
  result += '"';
  return result;
}

template <class T>
std::string Serialize(const T& example) {
  std::string serialized;
  example.SerializeToString(&serialized);
  return serialized;
}

// Tests that serialized gets parsed identically by TestFastParse(..)
// and the regular Example.ParseFromString(..).
void TestCorrectness(const std::string& serialized) {
  Example example;
  Example fast_example;
  EXPECT_TRUE(example.ParseFromString(serialized));
  example.DiscardUnknownFields();
  EXPECT_TRUE(TestFastParse(serialized, &fast_example));
  EXPECT_EQ(example.DebugString(), fast_example.DebugString());
  if (example.DebugString() != fast_example.DebugString()) {
    LOG(ERROR) << "Bad serialized: " << SerializedToReadable(serialized);
  }
}

// Fast parsing does not differentiate between EmptyExample and EmptyFeatures
// TEST(FastParse, EmptyExample) {
//   Example example;
//   TestCorrectness(example);
// }

TEST(FastParse, IgnoresPrecedingUnknownTopLevelFields) {
  ExampleWithExtras example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(13);
  example.set_extra1("some_str");
  example.set_extra2(123);
  example.set_extra3(234);
  example.set_extra4(345);
  example.set_extra5(4.56);
  example.add_extra6(5.67);
  example.add_extra6(6.78);
  (*example.mutable_extra7()->mutable_feature())["extra7"]
      .mutable_int64_list()
      ->add_value(1337);

  Example context;
  (*context.mutable_features()->mutable_feature())["zipcode"]
      .mutable_int64_list()
      ->add_value(94043);

  TestCorrectness(absl::StrCat(Serialize(example), Serialize(context)));
}

TEST(FastParse, IgnoresTrailingUnknownTopLevelFields) {
  Example example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(13);

  ExampleWithExtras context;
  (*context.mutable_features()->mutable_feature())["zipcode"]
      .mutable_int64_list()
      ->add_value(94043);
  context.set_extra1("some_str");
  context.set_extra2(123);
  context.set_extra3(234);
  context.set_extra4(345);
  context.set_extra5(4.56);
  context.add_extra6(5.67);
  context.add_extra6(6.78);
  (*context.mutable_extra7()->mutable_feature())["extra7"]
      .mutable_int64_list()
      ->add_value(1337);

  TestCorrectness(absl::StrCat(Serialize(example), Serialize(context)));
}

TEST(FastParse, SingleInt64WithContext) {
  Example example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(13);

  Example context;
  (*context.mutable_features()->mutable_feature())["zipcode"]
      .mutable_int64_list()
      ->add_value(94043);

  TestCorrectness(absl::StrCat(Serialize(example), Serialize(context)));
}

TEST(FastParse, DenseInt64WithContext) {
  Example example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(0);

  Example context;
  (*context.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(15);

  std::string serialized = Serialize(example) + Serialize(context);

  {
    Example deserialized;
    EXPECT_TRUE(deserialized.ParseFromString(serialized));
    EXPECT_EQ(deserialized.DebugString(), context.DebugString());
    // Whoa! Last EQ is very surprising, but standard deserialization is what it
    // is and Servo team requested to replicate this 'feature'.
    // In future we should return error.
  }
  TestCorrectness(serialized);
}

TEST(FastParse, NonPacked) {
  TestCorrectness(
      "\x0a\x0e\x0a\x0c\x0a\x03\x61\x67\x65\x12\x05\x1a\x03\x0a\x01\x0d");
}

TEST(FastParse, Packed) {
  TestCorrectness(
      "\x0a\x0d\x0a\x0b\x0a\x03\x61\x67\x65\x12\x04\x1a\x02\x08\x0d");
}

TEST(FastParse, ValueBeforeKeyInMap) {
  TestCorrectness("\x0a\x12\x0a\x10\x12\x09\x0a\x07\x0a\x05value\x0a\x03key");
}

TEST(FastParse, EmptyFeatures) {
  Example example;
  example.mutable_features();
  TestCorrectness(Serialize(example));
}

void TestCorrectnessJson(const std::string& json) {
  auto resolver = protobuf::util::NewTypeResolverForDescriptorPool(
      "type.googleapis.com", protobuf::DescriptorPool::generated_pool());
  std::string serialized;
  auto s = protobuf::util::JsonToBinaryString(
      resolver, "type.googleapis.com/tensorflow.Example", json, &serialized);
  EXPECT_TRUE(s.ok()) << s;
  delete resolver;
  TestCorrectness(serialized);
}

TEST(FastParse, JsonUnivalent) {
  TestCorrectnessJson(
      "{'features': {"
      "  'feature': {'age': {'int64_list': {'value': [0]} }}, "
      "  'feature': {'flo': {'float_list': {'value': [1.1]} }}, "
      "  'feature': {'byt': {'bytes_list': {'value': ['WW8='] }}}"
      "}}");
}

TEST(FastParse, JsonMultivalent) {
  TestCorrectnessJson(
      "{'features': {"
      "  'feature': {'age': {'int64_list': {'value': [0, 13, 23]} }}, "
      "  'feature': {'flo': {'float_list': {'value': [1.1, 1.2, 1.3]} }}, "
      "  'feature': {'byt': {'bytes_list': {'value': ['WW8=', 'WW8K'] }}}"
      "}}");
}

TEST(FastParse, SingleInt64) {
  Example example;
  (*example.mutable_features()->mutable_feature())["age"]
      .mutable_int64_list()
      ->add_value(13);
  TestCorrectness(Serialize(example));
}

static std::string ExampleWithSomeFeatures() {
  Example example;

  (*example.mutable_features()->mutable_feature())[""];

  (*example.mutable_features()->mutable_feature())["empty_bytes_list"]
      .mutable_bytes_list();
  (*example.mutable_features()->mutable_feature())["empty_float_list"]
      .mutable_float_list();
  (*example.mutable_features()->mutable_feature())["empty_int64_list"]
      .mutable_int64_list();

  BytesList* bytes_list =
      (*example.mutable_features()->mutable_feature())["bytes_list"]
          .mutable_bytes_list();
  bytes_list->add_value("bytes1");
  bytes_list->add_value("bytes2");

  FloatList* float_list =
      (*example.mutable_features()->mutable_feature())["float_list"]
          .mutable_float_list();
  float_list->add_value(1.0);
  float_list->add_value(2.0);

  Int64List* int64_list =
      (*example.mutable_features()->mutable_feature())["int64_list"]
          .mutable_int64_list();
  int64_list->add_value(3);
  int64_list->add_value(270);
  int64_list->add_value(86942);

  return Serialize(example);
}

TEST(FastParse, SomeFeatures) { TestCorrectness(ExampleWithSomeFeatures()); }

static void AddDenseFeature(const char* feature_name, DataType dtype,
                            PartialTensorShape shape, bool variable_length,
                            size_t elements_per_stride,
                            FastParseExampleConfig* out_config) {
  out_config->dense.emplace_back();
  auto& new_feature = out_config->dense.back();
  new_feature.feature_name = feature_name;
  new_feature.dtype = dtype;
  new_feature.shape = std::move(shape);
  new_feature.default_value = Tensor(dtype, {});
  new_feature.variable_length = variable_length;
  new_feature.elements_per_stride = elements_per_stride;
}

static void AddSparseFeature(const char* feature_name, DataType dtype,
                             FastParseExampleConfig* out_config) {
  out_config->sparse.emplace_back();
  auto& new_feature = out_config->sparse.back();
  new_feature.feature_name = feature_name;
  new_feature.dtype = dtype;
}

TEST(FastParse, StatsCollection) {
  const size_t kNumExamples = 13;
  std::vector<tstring> serialized(kNumExamples, ExampleWithSomeFeatures());

  FastParseExampleConfig config_dense;
  AddDenseFeature("bytes_list", DT_STRING, {2}, false, 2, &config_dense);
  AddDenseFeature("float_list", DT_FLOAT, {2}, false, 2, &config_dense);
  AddDenseFeature("int64_list", DT_INT64, {3}, false, 3, &config_dense);
  config_dense.collect_feature_stats = true;

  FastParseExampleConfig config_varlen;
  AddDenseFeature("bytes_list", DT_STRING, {-1}, true, 1, &config_varlen);
  AddDenseFeature("float_list", DT_FLOAT, {-1}, true, 1, &config_varlen);
  AddDenseFeature("int64_list", DT_INT64, {-1}, true, 1, &config_varlen);
  config_varlen.collect_feature_stats = true;

  FastParseExampleConfig config_sparse;
  AddSparseFeature("bytes_list", DT_STRING, &config_sparse);
  AddSparseFeature("float_list", DT_FLOAT, &config_sparse);
  AddSparseFeature("int64_list", DT_INT64, &config_sparse);
  config_sparse.collect_feature_stats = true;

  FastParseExampleConfig config_mixed;
  AddDenseFeature("bytes_list", DT_STRING, {2}, false, 2, &config_mixed);
  AddDenseFeature("float_list", DT_FLOAT, {-1}, true, 1, &config_mixed);
  AddSparseFeature("int64_list", DT_INT64, &config_mixed);
  config_mixed.collect_feature_stats = true;

  for (const FastParseExampleConfig& config :
       {config_dense, config_varlen, config_sparse, config_mixed}) {
    {
      Result result;
      TF_CHECK_OK(FastParseExample(config, serialized, {}, nullptr, &result));
      EXPECT_EQ(kNumExamples, result.feature_stats.size());
      for (const PerExampleFeatureStats& stats : result.feature_stats) {
        EXPECT_EQ(7, stats.features_count);
        EXPECT_EQ(7, stats.feature_values_count);
      }
    }

    {
      Result result;
      TF_CHECK_OK(FastParseSingleExample(config, serialized[0], &result));
      EXPECT_EQ(1, result.feature_stats.size());
      EXPECT_EQ(7, result.feature_stats[0].features_count);
      EXPECT_EQ(7, result.feature_stats[0].feature_values_count);
    }
  }
}

std::string RandStr(random::SimplePhilox* rng) {
  static const char key_char_lookup[] =
      "0123456789{}~`!@#$%^&*()"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
  auto len = 1 + rng->Rand32() % 200;
  std::string str;
  str.reserve(len);
  while (len-- > 0) {
    str.push_back(
        key_char_lookup[rng->Rand32() % (sizeof(key_char_lookup) /
                                         sizeof(key_char_lookup[0]))]);
  }
  return str;
}

void Fuzz(random::SimplePhilox* rng) {
  // Generate keys.
  auto num_keys = 1 + rng->Rand32() % 100;
  std::unordered_set<std::string> unique_keys;
  for (auto i = 0; i < num_keys; ++i) {
    unique_keys.emplace(RandStr(rng));
  }

  // Generate serialized example.
  Example example;
  std::string serialized_example;
  auto num_concats = 1 + rng->Rand32() % 4;
  std::vector<Feature::KindCase> feat_types(
      {Feature::kBytesList, Feature::kFloatList, Feature::kInt64List});
  std::vector<std::string> all_keys(unique_keys.begin(), unique_keys.end());
  while (num_concats--) {
    example.Clear();
    auto num_active_keys = 1 + rng->Rand32() % all_keys.size();

    // Generate features.
    for (auto i = 0; i < num_active_keys; ++i) {
      auto fkey = all_keys[rng->Rand32() % all_keys.size()];
      auto ftype_idx = rng->Rand32() % feat_types.size();
      auto num_features = 1 + rng->Rand32() % 5;
      switch (static_cast<Feature::KindCase>(feat_types[ftype_idx])) {
        case Feature::kBytesList: {
          BytesList* bytes_list =
              (*example.mutable_features()->mutable_feature())[fkey]
                  .mutable_bytes_list();
          while (num_features--) {
            bytes_list->add_value(RandStr(rng));
          }
          break;
        }
        case Feature::kFloatList: {
          FloatList* float_list =
              (*example.mutable_features()->mutable_feature())[fkey]
                  .mutable_float_list();
          while (num_features--) {
            float_list->add_value(rng->RandFloat());
          }
          break;
        }
        case Feature::kInt64List: {
          Int64List* int64_list =
              (*example.mutable_features()->mutable_feature())[fkey]
                  .mutable_int64_list();
          while (num_features--) {
            int64_list->add_value(rng->Rand64());
          }
          break;
        }
        default: {
          LOG(QFATAL);
          break;
        }
      }
    }
    serialized_example += example.SerializeAsString();
  }

  // Test correctness.
  TestCorrectness(serialized_example);
}

TEST(FastParse, FuzzTest) {
  const uint64_t seed = 1337;
  random::PhiloxRandom philox(seed);
  random::SimplePhilox rng(&philox);
  auto num_runs = 200;
  while (num_runs--) {
    LOG(INFO) << "runs left: " << num_runs;
    Fuzz(&rng);
  }
}

TEST(TestFastParseExample, Empty) {
  Result result;
  FastParseExampleConfig config;
  config.sparse.push_back({"test", DT_STRING});
  absl::Status status =
      FastParseExample(config, absl::Span<const tstring>(),
                       absl::Span<const tstring>(), nullptr, &result);
  EXPECT_TRUE(status.ok()) << status;
}

TEST(FastParse, OOB_Write_Vulnerability_NonPacked_FloatList) {
  FastParseExampleConfig config;
  AddDenseFeature("f", DT_FLOAT, {1}, false, 1, &config);

  auto encode_varint = [](uint32_t v, std::string* out) {
    while (v >= 0x80) {
      out->push_back((v & 0x7f) | 0x80);
      v >>= 7;
    }
    out->push_back(v);
  };

  std::string float_list_data;
  int num_elements = 10000;  // Large number to force crash
  for (int i = 0; i < num_elements; ++i) {
    float_list_data.push_back(13);  // kFixed32Tag(1)
    float v = 1.0f;
    const char* p = reinterpret_cast<const char*>(&v);
    float_list_data.append(p, 4);
  }

  std::string serialized_feature;
  serialized_feature.push_back(18);  // kDelimitedTag(2) for float_list
  encode_varint(float_list_data.size(), &serialized_feature);
  serialized_feature.append(float_list_data);

  std::string map_entry;
  map_entry.push_back(10);  // kDelimitedTag(1) for key
  map_entry.push_back(1);
  map_entry.push_back('f');
  map_entry.push_back(18);  // kDelimitedTag(2) for value
  encode_varint(serialized_feature.size(), &map_entry);
  map_entry.append(serialized_feature);

  std::string features_msg;
  features_msg.push_back(10);  // kDelimitedTag(1) for map entry
  encode_varint(map_entry.size(), &features_msg);
  features_msg.append(map_entry);

  std::string serialized_example;
  serialized_example.push_back(10);  // kDelimitedTag(1) for features
  encode_varint(features_msg.size(), &serialized_example);
  serialized_example.append(features_msg);

  Result result;
  std::vector<tstring> serialized_vec = {tstring(serialized_example)};
  absl::Status parse_status =
      FastParseExample(config, serialized_vec, {}, nullptr, &result);

  // We expect this to fail with INVALID_ARGUMENT due to size mismatch,
  // but WITHOUT crashing.
  EXPECT_FALSE(parse_status.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(parse_status));
}

TEST(FastParse, DenseFloat_TooManyElements_ReportsError) {
  FastParseExampleConfig config;
  AddDenseFeature("f", DT_FLOAT, {1}, false, 1, &config);

  auto encode_varint = [](uint32_t v, std::string* out) {
    while (v >= 0x80) {
      out->push_back((v & 0x7f) | 0x80);
      v >>= 7;
    }
    out->push_back(v);
  };

  std::string float_list_data;
  int num_elements = 5;  // Expecting 1, but providing 5
  for (int i = 0; i < num_elements; ++i) {
    float_list_data.push_back(13);  // kFixed32Tag(1)
    float v = 1.0f;
    const char* p = reinterpret_cast<const char*>(&v);
    float_list_data.append(p, 4);
  }

  std::string serialized_feature;
  serialized_feature.push_back(18);  // kDelimitedTag(2) for float_list
  encode_varint(float_list_data.size(), &serialized_feature);
  serialized_feature.append(float_list_data);

  std::string map_entry;
  map_entry.push_back(10);  // kDelimitedTag(1) for key
  map_entry.push_back(1);
  map_entry.push_back('f');
  map_entry.push_back(18);  // kDelimitedTag(2) for value
  encode_varint(serialized_feature.size(), &map_entry);
  map_entry.append(serialized_feature);

  std::string features_msg;
  features_msg.push_back(10);  // kDelimitedTag(1) for map entry
  encode_varint(map_entry.size(), &features_msg);
  features_msg.append(map_entry);

  std::string serialized_example;
  serialized_example.push_back(10);  // kDelimitedTag(1) for features
  encode_varint(features_msg.size(), &serialized_example);
  serialized_example.append(features_msg);

  Result result;
  std::vector<tstring> serialized_vec = {tstring(serialized_example)};
  absl::Status parse_status =
      FastParseExample(config, serialized_vec, {}, nullptr, &result);

  EXPECT_FALSE(parse_status.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(parse_status));
  EXPECT_NE(parse_status.ToString().find("Number of float values != expected"),
            std::string::npos);
}

TEST(FastParse, DenseFloat_TooFewElements_ReportsError) {
  FastParseExampleConfig config;
  // Expecting 3 elements per stride
  AddDenseFeature("f", DT_FLOAT, {3}, false, 3, &config);

  auto encode_varint = [](uint32_t v, std::string* out) {
    while (v >= 0x80) {
      out->push_back((v & 0x7f) | 0x80);
      v >>= 7;
    }
    out->push_back(v);
  };

  std::string float_list_data;
  int num_elements = 1;  // Providing only 1
  for (int i = 0; i < num_elements; ++i) {
    float_list_data.push_back(13);  // kFixed32Tag(1)
    float v = 1.0f;
    const char* p = reinterpret_cast<const char*>(&v);
    float_list_data.append(p, 4);
  }

  std::string serialized_feature;
  serialized_feature.push_back(18);  // kDelimitedTag(2) for float_list
  encode_varint(float_list_data.size(), &serialized_feature);
  serialized_feature.append(float_list_data);

  std::string map_entry;
  map_entry.push_back(10);  // kDelimitedTag(1) for key
  map_entry.push_back(1);
  map_entry.push_back('f');
  map_entry.push_back(18);  // kDelimitedTag(2) for value
  encode_varint(serialized_feature.size(), &map_entry);
  map_entry.append(serialized_feature);

  std::string features_msg;
  features_msg.push_back(10);  // kDelimitedTag(1) for map entry
  encode_varint(map_entry.size(), &features_msg);
  features_msg.append(map_entry);

  std::string serialized_example;
  serialized_example.push_back(10);  // kDelimitedTag(1) for features
  encode_varint(features_msg.size(), &serialized_example);
  serialized_example.append(features_msg);

  Result result;
  std::vector<tstring> serialized_vec = {tstring(serialized_example)};
  absl::Status parse_status =
      FastParseExample(config, serialized_vec, {}, nullptr, &result);

  EXPECT_FALSE(parse_status.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(parse_status));
  EXPECT_NE(parse_status.ToString().find("Number of float values != expected"),
            std::string::npos);
}

}  // namespace
}  // namespace example
}  // namespace tensorflow

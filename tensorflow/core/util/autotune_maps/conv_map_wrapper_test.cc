/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/autotune_maps/conv_map_wrapper.h"

#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "tensorflow/core/util/autotune_maps/autotune_map.pb.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.pb.h"

namespace tensorflow {
namespace {

ConvMapProto ThreeConvMapEntries() {
  ConvMapProto proto;

  auto r1 = proto.add_kv_pairs();
  r1->mutable_key()->set_batch(1);
  r1->mutable_key()->set_in_depths(2);
  r1->mutable_key()->set_out_depths(3);
  r1->mutable_value()->mutable_algorithm()->set_algo_id(4);

  auto r2 = proto.add_kv_pairs();
  r2->mutable_key()->set_batch(5);
  r2->mutable_key()->set_in_depths(6);
  r2->mutable_key()->set_out_depths(7);
  r2->mutable_value()->mutable_algorithm()->set_algo_id(8);

  auto r3 = proto.add_kv_pairs();
  r3->mutable_key()->set_batch(9);
  r3->mutable_key()->set_in_depths(10);
  r3->mutable_key()->set_out_depths(11);
  r3->mutable_value()->mutable_algorithm()->set_algo_id(12);

  return proto;
}

TEST(ConvMapWrapperTest, FullRoundTrip) {
  std::vector<ConvMapWrapper> wrappers =
      ConvMapWrapper::ConvMapToWrappers(ThreeConvMapEntries());

  std::vector<std::pair<ConvMapWrapper::OpaqueKey, ConvMapWrapper::OpaqueValue>>
      key_value_pairs;
  for (const auto& wrapper : wrappers) {
    key_value_pairs.emplace_back(wrapper.Key(), wrapper.Value());
  }

  std::vector<ConvMapWrapper> new_wrappers;
  for (const auto& [key, value] : key_value_pairs) {
    TF_ASSERT_OK_AND_ASSIGN(ConvMapWrapper wrapper,
                            ConvMapWrapper::FromKeyAndValue(key, value));
    new_wrappers.push_back(wrapper);
  }

  TF_ASSERT_OK_AND_ASSIGN(ConvMapProto round_tripped,
                          ConvMapWrapper::ConvMapFromWrappers(new_wrappers));
  EXPECT_EQ(round_tripped.kv_pairs_size(), 3);
  EXPECT_EQ(round_tripped.kv_pairs(0).key().batch(), 1);
  EXPECT_EQ(round_tripped.kv_pairs(0).key().in_depths(), 2);
  EXPECT_EQ(round_tripped.kv_pairs(0).key().out_depths(), 3);
  EXPECT_EQ(round_tripped.kv_pairs(0).value().algorithm().algo_id(), 4);
  EXPECT_EQ(round_tripped.kv_pairs(1).key().batch(), 5);
  EXPECT_EQ(round_tripped.kv_pairs(1).key().in_depths(), 6);
  EXPECT_EQ(round_tripped.kv_pairs(1).key().out_depths(), 7);
  EXPECT_EQ(round_tripped.kv_pairs(1).value().algorithm().algo_id(), 8);
  EXPECT_EQ(round_tripped.kv_pairs(2).key().batch(), 9);
  EXPECT_EQ(round_tripped.kv_pairs(2).key().in_depths(), 10);
  EXPECT_EQ(round_tripped.kv_pairs(2).key().out_depths(), 11);
  EXPECT_EQ(round_tripped.kv_pairs(2).value().algorithm().algo_id(), 12);
}

TEST(ConvMapWrapperTest, DeterministicSerialization) {
  std::vector<ConvMapWrapper> wrappers =
      ConvMapWrapper::ConvMapToWrappers(ThreeConvMapEntries());
  std::vector<ConvMapWrapper::OpaqueKey> keys;
  std::vector<ConvMapWrapper::OpaqueValue> values;
  for (const auto& wrapper : wrappers) {
    keys.push_back(wrapper.Key());
    values.push_back(wrapper.Value());
  }

  const int kNumIterations = 100;
  for (int i = 0; i < kNumIterations; ++i) {
    std::vector<ConvMapWrapper> test_wrappers =
        ConvMapWrapper::ConvMapToWrappers(ThreeConvMapEntries());
    std::vector<ConvMapWrapper::OpaqueKey> test_keys;
    std::vector<ConvMapWrapper::OpaqueValue> test_values;
    for (const auto& test_wrapper : test_wrappers) {
      test_keys.push_back(test_wrapper.Key());
      test_values.push_back(test_wrapper.Value());
    }
    EXPECT_EQ(keys, test_keys);
    EXPECT_EQ(values, test_values);
  }
}

}  // namespace
}  // namespace tensorflow

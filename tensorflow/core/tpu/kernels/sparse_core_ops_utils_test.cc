/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/sparse_core_ops_utils.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "xla/xla_data.pb.h"

namespace tensorflow {
namespace {

TEST(SetSparseCoreFrontendAttributesTest, Basic) {
  xla::FrontendAttributes attributes;
  ASSERT_OK(SetSparseCoreFrontendAttributes(&attributes,
                                            /*max_ids_per_partition=*/100,
                                            /*max_unique_ids_per_partition=*/20,
                                            /*num_sparsecores_per_device=*/4,
                                            /*vocab_size=*/4000,
                                            /*feature_width=*/64,
                                            /*input_size=*/128,
                                            /*table_name=*/"test_table"));
  const auto& attr_map = attributes.map();
  EXPECT_EQ(attr_map.at("_xla_compute_type"), "sparse");
  EXPECT_EQ(attr_map.at("_xla_sharding_strategy"), "mod");
  EXPECT_EQ(attr_map.at("_xla_pad_value"), absl::StrCat(kXlaPadValue));
  EXPECT_EQ(attr_map.at("_xla_max_ids_per_partition"), "100");
  EXPECT_EQ(attr_map.at("_xla_max_unique_ids_per_partition"), "20");
  EXPECT_EQ(attr_map.at("_xla_table_name"), "test_table");
  EXPECT_EQ(attr_map.at("_xla_vocab_size"), "1000");
  EXPECT_EQ(attr_map.at("_xla_feature_width"), "64");
  EXPECT_EQ(attr_map.at("_xla_sample_count"), "32");
}

TEST(SetSparseCoreFrontendAttributesTest, WithOptionals) {
  xla::FrontendAttributes attributes;
  ASSERT_OK(SetSparseCoreFrontendAttributes(
      &attributes, /*max_ids_per_partition=*/100,
      /*max_unique_ids_per_partition=*/20,
      /*num_sparsecores_per_device=*/4,
      /*vocab_size=*/4000,
      /*feature_width=*/64,
      /*input_size=*/128,
      /*table_name=*/"test_table",
      /*max_valency=*/16,
      /*quantization_config_low=*/-1.5f,
      /*quantization_config_high=*/1.5f,
      /*quantization_config_num_buckets=*/256));
  const auto& attr_map = attributes.map();
  EXPECT_EQ(attr_map.at("_xla_vocab_size"), "1000");
  EXPECT_EQ(attr_map.at("_xla_feature_width"), "64");
  EXPECT_EQ(attr_map.at("_xla_sample_count"), "32");
  EXPECT_EQ(attr_map.at("_xla_max_valency"), "16");
  EXPECT_EQ(attr_map.at("_xla_quantization_low_value"), "-1.5");
  EXPECT_EQ(attr_map.at("_xla_quantization_high_value"), "1.5");
  EXPECT_EQ(attr_map.at("_xla_quantization_num_buckets_value"), "256");
  EXPECT_EQ(attr_map.at("_xla_enable_full_hbm_sort"), "false");
}

TEST(ConvertSplitsAndBackTest, Split0) {
  const int max_division_level = 6;

  int64_t original_split = 0;
  std::vector<int> actual_buckets =
      ConvertBinarySplitsToBucketSplits(original_split, max_division_level);
  std::vector<int> expected_buckets = {};
  int64_t re_split =
      ConvertBucketSplitsToBinarySplits(expected_buckets, max_division_level);
  ASSERT_EQ(re_split, original_split);
}

TEST(ConvertSplitsAndBackTest, Split2) {
  const int max_division_level = 6;

  int64_t original_split = 2;
  std::vector<int> actual_buckets =
      ConvertBinarySplitsToBucketSplits(original_split, max_division_level);
  std::vector<int> expected_buckets = {16};
  int64_t re_split =
      ConvertBucketSplitsToBinarySplits(expected_buckets, max_division_level);
  ASSERT_EQ(re_split, original_split);
}

TEST(ConvertSplitsAndBackTest, Split3) {
  const int max_division_level = 6;

  int64_t original_split = 3;
  std::vector<int> actual_buckets =
      ConvertBinarySplitsToBucketSplits(original_split, max_division_level);
  std::vector<int> expected_buckets = {16, 32};
  int64_t re_split =
      ConvertBucketSplitsToBinarySplits(expected_buckets, max_division_level);
  ASSERT_EQ(re_split, original_split);
}

}  // namespace
}  // namespace tensorflow

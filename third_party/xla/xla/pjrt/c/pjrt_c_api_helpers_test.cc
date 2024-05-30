/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/pjrt/c/pjrt_c_api_helpers.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "xla/layout.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/statusor.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace pjrt {
namespace {

using ::testing::HasSubstr;

TEST(PjRtCApiHelperTest, ConvertValidPjRtValueType) {
  std::vector<int64_t> int64_list = {static_cast<int64_t>(1),
                                     static_cast<int64_t>(2)};
  absl::flat_hash_map<std::string, xla::PjRtValueType> original_cpp_map = {
      {"string", "v1"},
      {"int64", static_cast<int64_t>(1)},
      {"int64_list", int64_list},
      {"float", static_cast<float>(1.0)}};

  TF_ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_map,
                          ConvertToPjRtNamedValueList(original_cpp_map));
  auto converted_back_cpp_map =
      ConvertFromPjRtNamedValueList(c_map.data(), c_map.size());

  EXPECT_THAT(converted_back_cpp_map,
              testing::UnorderedElementsAreArray(original_cpp_map));
}

TEST(PjRtCApiHelperTest, ValidOptionNameAndPjRtValueTypeIndex) {
  const auto expected = absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
      {"string", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
      {"int64", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
  });
  absl::flat_hash_map<std::string, xla::PjRtValueType> valid_map = {
      {"string", static_cast<std::string>("v1")},
      {"int64", static_cast<int64_t>(1)}};

  TF_EXPECT_OK(ValidateCreateOptions(valid_map, expected));
}

TEST(PjRtCApiHelperTest, InvalidOptionName) {
  const auto expected = absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
      {"string", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
      {"int64", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
  });
  absl::flat_hash_map<std::string, xla::PjRtValueType> invalid_map = {
      {"invalid", "v1"}};

  auto status = ValidateCreateOptions(invalid_map, expected);

  EXPECT_NE(status, absl::OkStatus());
  EXPECT_THAT(status.message(),
              HasSubstr("Unexpected option name passed to PJRT_Client_Create"));
}

TEST(PjRtCApiHelperTest, InvalidOptionTypeIndex) {
  const auto expected = absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
      {"string", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
      {"int64", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
  });
  absl::flat_hash_map<std::string, xla::PjRtValueType> invalid_map = {
      {"string", static_cast<int64_t>(1)}};

  auto status = ValidateCreateOptions(invalid_map, expected);

  EXPECT_NE(status, absl::OkStatus());
  EXPECT_THAT(status.message(),
              HasSubstr("Option passed to PJRT_Client_Create with name string "
                        "has type index 2 but expected type index is 0"));
}

TEST(PjRtCApiHelperTest, Callback) {
  auto kv_store = std::make_shared<xla::InMemoryKeyValueStore>();

  auto kv_callback_data = ConvertToCKeyValueCallbacks(kv_store);
  auto converted_kv_store = ToCppKeyValueStore(
      kv_callback_data->c_kv_get, &kv_callback_data->kv_get_c_func,
      kv_callback_data->c_kv_put, &kv_callback_data->kv_put_c_func);

  auto s = converted_kv_store->Set("key", "value");
  TF_EXPECT_OK(s);

  auto v = converted_kv_store->Get("key", absl::Seconds(1));
  TF_EXPECT_OK(v.status());
  EXPECT_EQ(*v, "value");
}

TEST(PjRtCApiHelperTest, ConvertToCLayoutFromStrides) {
  std::vector<int64_t> strides = {4, 8};
  absl::StatusOr<BufferMemoryLayoutData> layout_data =
      ConvertToBufferMemoryLayoutData(strides);

  EXPECT_TRUE(layout_data.ok());
  EXPECT_EQ(
      layout_data->c_layout.type,
      PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Strides);
  EXPECT_EQ(layout_data->c_layout.strides.num_byte_strides, 2);
  EXPECT_EQ(layout_data->c_layout.strides.byte_strides[0], strides[0]);
  EXPECT_EQ(layout_data->c_layout.strides.byte_strides[1], strides[1]);
}

TEST(PjRtCApiHelperTest, ConvertToCLayoutFromLayoutNoTiles) {
  std::vector<int64_t> minor_to_major = {1, 0};
  xla::Layout layout(minor_to_major);

  TF_ASSERT_OK_AND_ASSIGN(BufferMemoryLayoutData layout_data,
                          ConvertToBufferMemoryLayoutData(layout));

  EXPECT_EQ(layout_data.c_layout.type,
            PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled);
  EXPECT_EQ(layout_data.c_layout.tiled.num_tiles, 0);
  PJRT_Buffer_MemoryLayout_Tiled tiled = layout_data.c_layout.tiled;
  EXPECT_EQ(tiled.minor_to_major_size, 2);
  EXPECT_EQ(tiled.minor_to_major[0], minor_to_major[0]);
  EXPECT_EQ(tiled.minor_to_major[1], minor_to_major[1]);
}

TEST(PjRtCApiHelperTest, ConvertToCLayoutFromLayoutWithTiles) {
  std::vector<int64_t> minor_to_major = {1, 0};
  xla::Layout layout(minor_to_major);
  std::vector<int64_t> tile_dims_1 = {2, 4};
  std::vector<int64_t> tile_dims_2 = {1};
  layout.mutable_tiles()->push_back(xla::Tile(tile_dims_1));
  layout.mutable_tiles()->push_back(xla::Tile(tile_dims_2));

  TF_ASSERT_OK_AND_ASSIGN(BufferMemoryLayoutData layout_data,
                          ConvertToBufferMemoryLayoutData(layout));

  EXPECT_EQ(layout_data.c_layout.type,
            PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled);
  PJRT_Buffer_MemoryLayout_Tiled tiled = layout_data.c_layout.tiled;
  EXPECT_EQ(tiled.minor_to_major_size, 2);
  EXPECT_EQ(tiled.minor_to_major[0], minor_to_major[0]);
  EXPECT_EQ(tiled.minor_to_major[1], minor_to_major[1]);
  EXPECT_EQ(tiled.num_tiles, 2);
  EXPECT_EQ(tiled.tile_dim_sizes[0], tile_dims_1.size());
  EXPECT_EQ(tiled.tile_dim_sizes[1], tile_dims_2.size());
  EXPECT_EQ(tiled.tile_dims[0], tile_dims_1[0]);
  EXPECT_EQ(tiled.tile_dims[1], tile_dims_1[1]);
  EXPECT_EQ(tiled.tile_dims[2], tile_dims_2[0]);
}

TEST(PjRtCApiHelperTest, ConvertFromCLayoutToLayout) {
  PJRT_Buffer_MemoryLayout c_layout;
  c_layout.type =
      PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled;
  std::vector<int64_t> minor_to_major = {1, 0};
  c_layout.tiled.minor_to_major_size = 2;
  c_layout.tiled.minor_to_major = minor_to_major.data();
  c_layout.tiled.num_tiles = 2;
  std::vector<size_t> tile_dim_sizes = {2, 1};
  c_layout.tiled.tile_dim_sizes = tile_dim_sizes.data();
  std::vector<int64_t> tile_dims = {2, 4, 1};
  c_layout.tiled.tile_dims = tile_dims.data();

  TF_ASSERT_OK_AND_ASSIGN(xla::Layout layout, ConvertToLayout(c_layout.tiled));

  EXPECT_EQ(layout.ToString(), "{1,0:T(2,4)(1)}");
}

TEST(PjRtCApiHelperTest, ConvertFromCLayoutToLayoutNoTile) {
  PJRT_Buffer_MemoryLayout c_layout;
  c_layout.type =
      PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled;
  c_layout.tiled.num_tiles = 0;
  std::vector<int64_t> minor_to_major = {1, 0};
  c_layout.tiled.minor_to_major_size = 2;
  c_layout.tiled.minor_to_major = minor_to_major.data();

  TF_ASSERT_OK_AND_ASSIGN(xla::Layout layout, ConvertToLayout(c_layout.tiled));

  EXPECT_EQ(layout.ToString(), "{1,0}");
}

}  // namespace
}  // namespace pjrt

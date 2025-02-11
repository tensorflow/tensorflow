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

#include "tensorflow/lite/experimental/litert/core/byte_code/byte_code_asset.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/byte_code/schema.h"
#include "tensorflow/lite/experimental/litert/test/test_macros.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAreArray;

using SharedByteCode = SharedByteCodeAsset::SharedByteCode;
using OwnedByteCode = UniqueByteCodeAsset::OwnedByteCode;

static constexpr uint8_t kTestBuffer[] = {1, 2, 3, 4, 5};
static constexpr absl::string_view kBackendId = "BAR_BACKEND";
static constexpr absl::string_view kEntryPoint = "BAR_ENTRY_POINT";
static constexpr absl::string_view kEntryPoint2 = "BAZ_ENTRY_POINT";

template <typename T>
T MakeTestBuffer() {
  auto res = T(new uint8_t[sizeof(kTestBuffer)]);
  std::copy(kTestBuffer, kTestBuffer + sizeof(kTestBuffer), res.get());
  return res;
}

TEST(SharedByteCodeAssetTest, Construct) {
  LiteRtOpT op;
  SharedByteCodeAsset asset(std::string(kBackendId), std::string(kEntryPoint),
                            MakeTestBuffer<SharedByteCode>(),
                            sizeof(kTestBuffer));

  EXPECT_EQ(asset.BackendId(), kBackendId);
  EXPECT_EQ(asset.ByteCode().Size(), sizeof(kTestBuffer));
  EXPECT_THAT(asset.ByteCode().Span(), ElementsAreArray(kTestBuffer));
}

TEST(SharedByteCodeAssetTest, Serialize) {
  SharedByteCodeAsset asset(std::string(kBackendId), std::string(kEntryPoint),
                            MakeTestBuffer<SharedByteCode>(),
                            sizeof(kTestBuffer));
  LiteRtOpT op1;
  LiteRtOpT op2;

  asset.AddCaller(&op1, std::string(kEntryPoint));
  asset.AddCaller(&op2, std::string(kEntryPoint2));

  LiteRtModelT model;
  LITERT_ASSERT_STATUS_OK(asset.Serialize(model));

  absl::string_view metadata_key;

  {
    auto exec_info = ExecInfoFromBuf(op1.CustomOptions());
    ASSERT_TRUE(exec_info);
    EXPECT_EQ(exec_info->backend_id, kBackendId);
    EXPECT_EQ(exec_info->entrypoint_name, kEntryPoint);
    metadata_key = exec_info->metadata_key;
  }

  {
    auto exec_info = ExecInfoFromBuf(op2.CustomOptions());
    ASSERT_TRUE(exec_info);
    EXPECT_EQ(exec_info->backend_id, kBackendId);
    EXPECT_EQ(exec_info->entrypoint_name, kEntryPoint2);
    ASSERT_EQ(metadata_key, exec_info->metadata_key);
  }

  auto metadata = model.FindMetadata(metadata_key);
  ASSERT_TRUE(metadata);

  EXPECT_EQ(metadata->Size(), sizeof(kTestBuffer));
  EXPECT_THAT(metadata->Span(), ElementsAreArray(kTestBuffer));
}

TEST(UniqueByteCodeAssetTest, Construct) {
  LiteRtOpT op;
  UniqueByteCodeAsset asset(
      std::string(kBackendId), &op, std::string(kEntryPoint),
      MakeTestBuffer<OwnedByteCode>(), sizeof(kTestBuffer));

  EXPECT_EQ(asset.BackendId(), kBackendId);
  EXPECT_EQ(asset.ByteCode().Size(), sizeof(kTestBuffer));
  EXPECT_THAT(asset.ByteCode().Span(), ElementsAreArray(kTestBuffer));
}

TEST(UniqueByteCodeAssetTest, Serialize) {
  LiteRtOpT op;
  UniqueByteCodeAsset asset(
      std::string(kBackendId), &op, std::string(kEntryPoint),
      MakeTestBuffer<OwnedByteCode>(), sizeof(kTestBuffer));
  LiteRtModelT model;
  LITERT_ASSERT_STATUS_OK(asset.Serialize(model));
  auto exec_info = ExecInfoFromBuf(op.CustomOptions());
  ASSERT_TRUE(exec_info);

  EXPECT_EQ(exec_info->backend_id, kBackendId);
  EXPECT_EQ(exec_info->entrypoint_name, kEntryPoint);

  auto metadata_key = exec_info->metadata_key;
  auto metadata = model.FindMetadata(metadata_key);
  ASSERT_TRUE(metadata);
  EXPECT_EQ(metadata->Size(), sizeof(kTestBuffer));
  EXPECT_THAT(metadata->Span(), ElementsAreArray(kTestBuffer));
}

}  // namespace
}  // namespace litert::internal

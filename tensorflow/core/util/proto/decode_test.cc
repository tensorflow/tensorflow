/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/proto/decode.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace internal {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

// Three packed varints: 1, 300, 2.
const uint8_t kPackedVarints[] = {0x01, 0xac, 0x02, 0x02};

TEST(ReadPackedFromArrayTest, ReadsAllValuesThatFit) {
  std::vector<int64_t> data(4, -1);
  int index = 1;
  absl::Status status = ReadPackedFromArray(
      kPackedVarints, sizeof(kPackedVarints), WireFormatLite::TYPE_INT64,
      /*field_number=*/1, DT_INT64, /*stride=*/1, static_cast<int>(data.size()),
      &index, data.data());
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(index, 4);
  EXPECT_EQ(data, (std::vector<int64_t>{-1, 1, 300, 2}));
}

TEST(ReadPackedFromArrayTest, RejectsValuesThatDoNotFit) {
  // Room for the first two values only; the element after them must not be
  // written.
  std::vector<int64_t> data(4, -1);
  int index = 0;
  absl::Status status = ReadPackedFromArray(
      kPackedVarints, sizeof(kPackedVarints), WireFormatLite::TYPE_INT64,
      /*field_number=*/1, DT_INT64, /*stride=*/1, /*data_size=*/2, &index,
      data.data());
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss,
                               HasSubstr("more values than expected")));
  EXPECT_EQ(index, 0);
  EXPECT_EQ(data, (std::vector<int64_t>{1, 300, -1, -1}));
}

TEST(ReadPackedFromArrayTest, RejectsFixedValuesThatDoNotFit) {
  const float values[] = {1.0f, 2.0f, 3.0f};
  std::vector<float> data(3, -1.0f);
  int index = 1;
  absl::Status status = ReadPackedFromArray(
      values, sizeof(values), WireFormatLite::TYPE_FLOAT, /*field_number=*/1,
      DT_FLOAT, /*stride=*/1, /*data_size=*/3, &index, data.data());
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kDataLoss,
                               HasSubstr("more values than expected")));
  EXPECT_EQ(index, 1);
  EXPECT_EQ(data, (std::vector<float>{-1.0f, 1.0f, 2.0f}));
}

TEST(ReadPackedFromArrayTest, NonRepeatedFieldKeepsLastValue) {
  // With stride 0 every value overwrites the same element, which then counts
  // as filled.
  std::vector<int64_t> data(1, -1);
  int index = 0;
  absl::Status status = ReadPackedFromArray(
      kPackedVarints, sizeof(kPackedVarints), WireFormatLite::TYPE_INT64,
      /*field_number=*/1, DT_INT64, /*stride=*/0, static_cast<int>(data.size()),
      &index, data.data());
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(index, 1);
  EXPECT_EQ(data[0], 2);
}

TEST(ReadPackedFromArrayTest, EmptyNonRepeatedFieldFillsNothing) {
  std::vector<int64_t> data(1, -1);
  int index = 0;
  absl::Status status = ReadPackedFromArray(
      kPackedVarints, /*buf_size=*/0, WireFormatLite::TYPE_INT64,
      /*field_number=*/1, DT_INT64, /*stride=*/0, static_cast<int>(data.size()),
      &index, data.data());
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(index, 0);
  EXPECT_EQ(data[0], -1);
}

}  // namespace
}  // namespace internal
}  // namespace tensorflow

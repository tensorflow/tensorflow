/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/name_utils.h"

#include "tensorflow/core/kernels/data/concatenate_dataset_op.h"
#include "tensorflow/core/kernels/data/range_dataset_op.h"
#include "tensorflow/core/kernels/data/shuffle_dataset_op.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

TEST(NameUtilsTest, DatasetDebugString) {
  EXPECT_EQ(name_utils::DatasetDebugString(ConcatenateDatasetOp::kDatasetType),
            "ConcatenateDatasetOp::Dataset");
  EXPECT_EQ(
      name_utils::DatasetDebugString(RangeDatasetOp::kDatasetType, 0, 10, 3),
      "RangeDatasetOp(0, 10, 3)::Dataset");
  EXPECT_EQ(name_utils::DatasetDebugString(ShuffleDatasetOp::kDatasetType,
                                           "FixedSeed", {"10", "1", "2"}),
            "ShuffleDatasetOp(10, 1, 2)::FixedSeedDataset");
}

TEST(NameUtilsTest, StrJoin) {
  EXPECT_EQ(name_utils::StrJoin("_"), "");
  EXPECT_EQ(name_utils::StrJoin("_", "A"), "A");
  EXPECT_EQ(name_utils::StrJoin("_", "A", 1, "B"), "A_1_B");
  EXPECT_EQ(name_utils::StrJoin(", ", "a", "b", "c"), "a, b, c");
}

}  // namespace
}  // namespace data
}  // namespace tensorflow

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

#include "tensorflow/core/kernels/sparse_utils.h"

#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/test.h"

namespace {

using tensorflow::DataType;
using tensorflow::int32;
using tensorflow::int64;
using tensorflow::Tensor;
using tensorflow::TTypes;
using tensorflow::uint16;
using tensorflow::uint32;
using tensorflow::uint64;
using tensorflow::sparse_utils::ContainsEmptyRows;
using tensorflow::sparse_utils::FindNextDenseRowStartIndex;
using tensorflow::sparse_utils::GetStartIndicesOfEachDenseRow;
using tensorflow::sparse_utils::ParseRowStartIndices;

TEST(SparseUtilsTest, GetStartIndicesOfEachDenseRow) {
  {
    int32 data[] = {0, 0, 1, 0, 4, 0, 6, 0, 7, 0, 8, 0, 10, 0, 12, 0};
    TTypes<int32>::ConstMatrix indices_mat(data, 8, 2);
    // indices_list = {0, 1, 4, 6, 7, 8, 10, 12};
    bool contains_empty_rows;
    EXPECT_TRUE(GetStartIndicesOfEachDenseRow<int32>(indices_mat,
                                                     &contains_empty_rows) ==
                std::vector<int32>({0, 1, 2, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8}));
    EXPECT_TRUE(contains_empty_rows);
  }
  {
    int32 data[] = {0, 0, 1, 0, 1, 0, 4, 0, 4, 0, 4, 0,  6, 0,  7,
                    0, 7, 0, 7, 0, 7, 0, 8, 0, 8, 0, 10, 0, 12, 0};
    TTypes<int32>::ConstMatrix indices_mat(data, 15, 2);
    // indices_list = {0, 1, 1, 4, 4, 4,  6, 7, 7, 7, 7, 8, 8, 10, 12};
    bool contains_empty_rows;
    EXPECT_TRUE(
        GetStartIndicesOfEachDenseRow<int32>(indices_mat,
                                             &contains_empty_rows) ==
        std::vector<int32>({0, 1, 3, 3, 3, 6, 6, 7, 11, 13, 13, 14, 14, 15}));
    EXPECT_TRUE(contains_empty_rows);
  }
  {
    int64 data[] = {3, 0};
    TTypes<int64>::ConstMatrix indices_mat(data, 1, 2);
    bool contains_empty_rows;
    EXPECT_TRUE(GetStartIndicesOfEachDenseRow<int64>(indices_mat,
                                                     &contains_empty_rows) ==
                std::vector<int64>({0, 0, 0, 0, 1}));
    EXPECT_TRUE(contains_empty_rows);
  }
  {
    uint32 data[] = {3, 0, 3, 0};
    TTypes<uint32>::ConstMatrix indices_mat(data, 2, 2);
    bool contains_empty_rows;
    EXPECT_TRUE(GetStartIndicesOfEachDenseRow<uint32>(indices_mat,
                                                      &contains_empty_rows) ==
                std::vector<uint32>({0, 0, 0, 0, 2}));
    EXPECT_TRUE(contains_empty_rows);
  }
  {
    uint16 data[] = {0, 0, 0, 0, 0, 0, 1, 0};
    TTypes<uint16>::ConstMatrix indices_mat(data, 4, 2);
    // indices_list = {0, 0, 0, 1};
    bool contains_empty_rows;
    EXPECT_TRUE(GetStartIndicesOfEachDenseRow<uint16>(indices_mat,
                                                      &contains_empty_rows) ==
                std::vector<uint16>({0, 3, 4}));
    EXPECT_FALSE(contains_empty_rows);
  }
  {
    uint64 data[] = {0, 0, 0, 0, 0, 0, 3, 0};
    TTypes<uint64>::ConstMatrix indices_mat(data, 4, 2);
    bool contains_empty_rows;
    // indices_list = {0, 0, 0, 3};
    EXPECT_TRUE(GetStartIndicesOfEachDenseRow<uint64>(indices_mat,
                                                      &contains_empty_rows) ==
                std::vector<uint64>({0, 3, 3, 3, 4}));
    EXPECT_TRUE(contains_empty_rows);
  }
}

TEST(SparseUtilsTest, ParseRowStartIndices) {
  {
    Tensor t(DataType::DT_INT32, {1});
    int indx = 0;
    for (const int32 v : {0}) {
      t.flat<int32>()(indx++) = v;
    }
    EXPECT_TRUE(ParseRowStartIndices<int32>(t, 1) ==
                std::vector<int32>({0, 1}));
  }
  {
    Tensor t(DataType::DT_INT64, {1});
    int indx = 0;
    for (const int64 v : {0}) {
      t.flat<int64>()(indx++) = v;
    }
    EXPECT_TRUE(ParseRowStartIndices<int64>(t, 2) ==
                std::vector<int64>({0, 2}));
  }
  {
    Tensor t(DataType::DT_UINT64, {2});
    int indx = 0;
    for (const uint64 v : {0, 3}) {
      t.flat<uint64>()(indx++) = v;
    }
    EXPECT_TRUE(ParseRowStartIndices<uint64>(t, 4) ==
                std::vector<uint64>({0, 3, 4}));
  }
  {
    Tensor t(DataType::DT_UINT16, {2});
    int indx = 0;
    for (const uint16 v : {0, 3}) {
      t.flat<uint16>()(indx++) = v;
    }
    EXPECT_TRUE(ParseRowStartIndices<uint16>(t, 4) ==
                std::vector<uint16>({0, 3, 4}));
  }
}

TEST(SparseUtilsTest, ContainsEmptyRows) {
  {
    int32 data[] = {0, 0, 1, 0, 4, 0, 6, 0, 7, 0, 8, 0, 10, 0, 12, 0};
    TTypes<int32>::ConstMatrix indices_mat(data, 8, 2);
    bool contains_empty_rows;
    const auto segment_indices =
        GetStartIndicesOfEachDenseRow<int32>(indices_mat, &contains_empty_rows);
    // indices_list = {0, 1, 4, 6, 7, 8, 10, 12};
    EXPECT_TRUE(ContainsEmptyRows(segment_indices));
  }
  {
    int64 data[] = {0, 0, 1, 0, 4, 0, 6, 0, 7, 0, 8, 0, 10, 0, 12, 0};
    TTypes<int64>::ConstMatrix indices_mat(data, 8, 2);
    bool contains_empty_rows;
    const auto segment_indices =
        GetStartIndicesOfEachDenseRow<int64>(indices_mat, &contains_empty_rows);
    // indices_list = {0, 1, 4, 6, 7, 8, 10, 12};
    EXPECT_TRUE(ContainsEmptyRows(segment_indices));
  }
  {
    int32 data[] = {1, 0, 1, 1, 2, 0, 2, 1, 2, 2, 3, 4};
    TTypes<int32>::ConstMatrix indices_mat(data, 6, 2);
    bool contains_empty_rows;
    const auto segment_indices =
        GetStartIndicesOfEachDenseRow<int32>(indices_mat, &contains_empty_rows);
    // indices_list = {1, 1, 2, 2, 2, 3};
    EXPECT_TRUE(ContainsEmptyRows(segment_indices));
  }
  {
    uint16 data[] = {1, 0, 1, 1, 2, 0, 2, 1, 2, 2, 3, 4};
    TTypes<uint16>::ConstMatrix indices_mat(data, 6, 2);
    bool contains_empty_rows;
    const auto segment_indices = GetStartIndicesOfEachDenseRow<uint16>(
        indices_mat, &contains_empty_rows);
    // indices_list = {1, 1, 2, 2, 2, 3};
    EXPECT_TRUE(ContainsEmptyRows(segment_indices));
  }
  {
    int32 data[] = {0, 0, 1, 0, 1, 1, 2, 0, 2, 1, 2, 2, 3, 4};
    TTypes<int32>::ConstMatrix indices_mat(data, 7, 2);
    bool contains_empty_rows;
    const auto segment_indices =
        GetStartIndicesOfEachDenseRow<int32>(indices_mat, &contains_empty_rows);
    // indices_list = {0, 1, 1, 2, 2, 2, 3};
    EXPECT_FALSE(ContainsEmptyRows(segment_indices));
  }
  {
    int64 data[] = {0, 0, 1, 0, 1, 1, 2, 0, 2, 1, 2, 2, 3, 4};
    TTypes<int64>::ConstMatrix indices_mat(data, 7, 2);
    bool contains_empty_rows;
    const auto segment_indices =
        GetStartIndicesOfEachDenseRow<int64>(indices_mat, &contains_empty_rows);
    // indices_list = {0, 1, 1, 2, 2, 2, 3};
    EXPECT_FALSE(ContainsEmptyRows(segment_indices));
  }
  {
    uint32 data[] = {0, 0, 0, 1, 0, 2, 2, 0, 2, 1, 2, 2, 3, 4};
    TTypes<uint32>::ConstMatrix indices_mat(data, 7, 2);
    bool contains_empty_rows;
    const auto segment_indices = GetStartIndicesOfEachDenseRow<uint32>(
        indices_mat, &contains_empty_rows);
    // indices_list = {0, 0, 0, 2, 2, 2, 3};
    EXPECT_TRUE(ContainsEmptyRows(segment_indices));
  }
  {
    int64 data[] = {0, 0, 0, 1, 0, 2, 2, 0, 2, 1, 2, 2, 3, 4};
    TTypes<int64>::ConstMatrix indices_mat(data, 7, 2);
    bool contains_empty_rows;
    const auto segment_indices =
        GetStartIndicesOfEachDenseRow<int64>(indices_mat, &contains_empty_rows);
    // indices_list = {0, 0, 0, 2, 2, 2, 3};
    EXPECT_TRUE(ContainsEmptyRows(segment_indices));
  }
  {
    uint64 data[] = {0, 0, 0, 1, 0, 2, 1, 0, 2, 1, 2, 2, 3, 4};
    TTypes<uint64>::ConstMatrix indices_mat(data, 7, 2);
    bool contains_empty_rows;
    const auto segment_indices = GetStartIndicesOfEachDenseRow<uint64>(
        indices_mat, &contains_empty_rows);
    // indices_list = {0, 0, 0, 1, 2, 2, 3};
    EXPECT_FALSE(ContainsEmptyRows(segment_indices));
  }
}

TEST(SparseUtilsTest, FindNextDenseRowStartIndex) {
  {
    int32 data[] = {0, 0, 1, 0, 4, 0, 6, 0, 7, 0, 8, 0, 10, 0, 12, 0};
    TTypes<int32>::ConstMatrix indices_mat(data, 8, 2);
    // indices_list = {0, 1, 4, 6, 7, 8, 10, 12};
    for (int32 i = 0; i < 8; ++i) {
      EXPECT_EQ(i + 1, FindNextDenseRowStartIndex<int32>(i, indices_mat));
    }
  }
  {
    uint16 data[] = {0, 0, 1, 0, 4, 0, 6, 0, 7, 0, 8, 0, 10, 0, 12, 0};
    TTypes<uint16>::ConstMatrix indices_mat(data, 8, 2);
    // indices_list = {0, 1, 4, 6, 7, 8, 10, 12};
    for (uint16 i = 0; i < 8; ++i) {
      EXPECT_EQ(i + 1, FindNextDenseRowStartIndex<uint16>(i, indices_mat));
    }
  }
  {
    int64 data[] = {0, 0, 1, 0, 1, 0, 4, 0, 4, 0, 4, 0,  6, 0,  7,
                    0, 7, 0, 7, 0, 7, 0, 8, 0, 8, 0, 10, 0, 12, 0};
    TTypes<int64>::ConstMatrix indices_mat(data, 15, 2);
    // indices_list = {0, 1, 1, 4, 4, 4,  6, 7, 7, 7, 7, 8, 8, 10, 12};
    EXPECT_EQ(3, FindNextDenseRowStartIndex<int64>(static_cast<int64>(1),
                                                   indices_mat));
    EXPECT_EQ(3, FindNextDenseRowStartIndex<int64>(static_cast<int64>(2),
                                                   indices_mat));
    EXPECT_EQ(6, FindNextDenseRowStartIndex<int64>(static_cast<int64>(3),
                                                   indices_mat));
    EXPECT_EQ(6, FindNextDenseRowStartIndex<int64>(static_cast<int64>(4),
                                                   indices_mat));
    EXPECT_EQ(14, FindNextDenseRowStartIndex<int64>(static_cast<int64>(13),
                                                    indices_mat));
    EXPECT_EQ(15, FindNextDenseRowStartIndex<int64>(static_cast<int64>(14),
                                                    indices_mat));
  }
}

}  // namespace

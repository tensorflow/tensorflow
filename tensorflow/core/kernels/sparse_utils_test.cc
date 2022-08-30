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

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace sparse_utils {
namespace {

using ::tensorflow::testing::StatusIs;
using ::testing::MatchesRegex;

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
    int64_t data[] = {3, 0};
    TTypes<int64_t>::ConstMatrix indices_mat(data, 1, 2);
    bool contains_empty_rows;
    EXPECT_TRUE(GetStartIndicesOfEachDenseRow<int64_t>(indices_mat,
                                                       &contains_empty_rows) ==
                std::vector<int64_t>({0, 0, 0, 0, 1}));
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
    for (const int32_t v : {0}) {
      t.flat<int32>()(indx++) = v;
    }
    EXPECT_TRUE(ParseRowStartIndices<int32>(t, 1) ==
                std::vector<int32>({0, 1}));
  }
  {
    Tensor t(DataType::DT_INT64, {1});
    int indx = 0;
    for (const int64_t v : {0}) {
      t.flat<int64_t>()(indx++) = v;
    }
    EXPECT_TRUE(ParseRowStartIndices<int64_t>(t, 2) ==
                std::vector<int64_t>({0, 2}));
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
    int64_t data[] = {0, 0, 1, 0, 4, 0, 6, 0, 7, 0, 8, 0, 10, 0, 12, 0};
    TTypes<int64_t>::ConstMatrix indices_mat(data, 8, 2);
    bool contains_empty_rows;
    const auto segment_indices = GetStartIndicesOfEachDenseRow<int64_t>(
        indices_mat, &contains_empty_rows);
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
    int64_t data[] = {0, 0, 1, 0, 1, 1, 2, 0, 2, 1, 2, 2, 3, 4};
    TTypes<int64_t>::ConstMatrix indices_mat(data, 7, 2);
    bool contains_empty_rows;
    const auto segment_indices = GetStartIndicesOfEachDenseRow<int64_t>(
        indices_mat, &contains_empty_rows);
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
    int64_t data[] = {0, 0, 0, 1, 0, 2, 2, 0, 2, 1, 2, 2, 3, 4};
    TTypes<int64_t>::ConstMatrix indices_mat(data, 7, 2);
    bool contains_empty_rows;
    const auto segment_indices = GetStartIndicesOfEachDenseRow<int64_t>(
        indices_mat, &contains_empty_rows);
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
    for (int32_t i = 0; i < 8; ++i) {
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
    int64_t data[] = {0, 0, 1, 0, 1, 0, 4, 0, 4, 0, 4, 0,  6, 0,  7,
                      0, 7, 0, 7, 0, 7, 0, 8, 0, 8, 0, 10, 0, 12, 0};
    TTypes<int64_t>::ConstMatrix indices_mat(data, 15, 2);
    // indices_list = {0, 1, 1, 4, 4, 4,  6, 7, 7, 7, 7, 8, 8, 10, 12};
    EXPECT_EQ(3, FindNextDenseRowStartIndex<int64_t>(static_cast<int64_t>(1),
                                                     indices_mat));
    EXPECT_EQ(3, FindNextDenseRowStartIndex<int64_t>(static_cast<int64_t>(2),
                                                     indices_mat));
    EXPECT_EQ(6, FindNextDenseRowStartIndex<int64_t>(static_cast<int64_t>(3),
                                                     indices_mat));
    EXPECT_EQ(6, FindNextDenseRowStartIndex<int64_t>(static_cast<int64_t>(4),
                                                     indices_mat));
    EXPECT_EQ(14, FindNextDenseRowStartIndex<int64_t>(static_cast<int64_t>(13),
                                                      indices_mat));
    EXPECT_EQ(15, FindNextDenseRowStartIndex<int64_t>(static_cast<int64_t>(14),
                                                      indices_mat));
  }
}

// Returns a shared random number generator.
::tensorflow::random::SimplePhilox& RandomPhilox() {
  // Safe initialization of static random generator.
  static auto* philox =
      new ::tensorflow::random::PhiloxRandom(tensorflow::testing::RandomSeed());
  static auto* rnd = new ::tensorflow::random::SimplePhilox(philox);
  return *rnd;
}

// Fills a tensor of indices with a unique set of random index tuples.
// The `SetType` must be a std::set-like type (e.g. flat_hash_set, btree_set)
// that is used to ensure uniqueness and governs the final index tuple order.
// For example, use a hash set for unordered indices, and sorted set for
// lexicographically ordered indices. The `shape` is used to ensure proper index
// bounds.
template <typename SetType>
void FillIndicesWithRandomTuples(const TensorShape& shape, Tensor& indices) {
  const int64_t nnz = indices.dim_size(0);
  const int64_t ndims = indices.dim_size(1);

  SetType indices_set;
  int64_t count = 0;
  // Generate nnz unique random tuples.
  while (count < nnz) {
    std::vector<int64_t> candidate(ndims);
    for (int64_t d = 0; d < ndims; ++d) {
      candidate[d] = RandomPhilox().Uniform64(shape.dim_size(d));
    }
    auto it = indices_set.insert(std::move(candidate));
    if (it.second) {
      ++count;
    }
  }

  // Copy index tuples from set into index tensor.
  auto indices_mat = indices.matrix<int64_t>();
  int64_t row = 0;
  for (const std::vector<int64_t>& idxs : indices_set) {
    for (int64_t col = 0; col < ndims; ++col) {
      indices_mat(row, col) = idxs[col];
    }
    ++row;
  }
}

// Populates components of a sparse random tensor with provided number of
// non-zeros `max_nnz` and tensor shape `shape`.
void GenerateRandomSparseTensor(int64_t max_nnz, const TensorShape& shape,
                                Tensor& output_indices, Tensor& output_values,
                                Tensor& output_shape) {
  const int64_t ndims = shape.dims();
  // We cannot generate more elements than the total in the tensor, so
  // potentially reduce nnz.
  const int64_t nnz = std::min(shape.num_elements(), max_nnz);
  output_indices = Tensor(DT_INT64, TensorShape({nnz, ndims}));
  output_values = Tensor(DT_FLOAT, TensorShape({nnz}));
  output_shape = Tensor(DT_INT64, TensorShape({ndims}));

  // Generate random unique unordered sparse indices.
  FillIndicesWithRandomTuples<absl::flat_hash_set<std::vector<int64_t>>>(
      shape, output_indices);

  auto values_vec = output_values.vec<float>();
  values_vec.setRandom();

  auto shape_vec = output_shape.vec<int64_t>();
  for (int i = 0; i < shape.dims(); ++i) {
    shape_vec(i) = shape.dim_size(i);
  }
}

TEST(ValidateSparseTensorTest, ValidSparseTensorPasses) {
  constexpr int kNumNonZeros = 1000;
  const TensorShape kTensorShapes[] = {
      {}, {3}, {4, 5}, {6, 7, 8}, {9, 10, 11, 12}};
  for (const TensorShape& tshape : kTensorShapes) {
    Tensor indices, values, shape;
    GenerateRandomSparseTensor(kNumNonZeros, tshape, indices, values, shape);
    TF_EXPECT_OK((ValidateSparseTensor(indices, values, shape)));
  }
}

TEST(ValidateSparseTensorTest, InvalidIndicesRankFails) {
  constexpr int kNumNonZeros = 1000;
  constexpr int kNumDims = 3;
  // Indices tensor must be rank 2, so try rank 0, 1, 3.
  const TensorShape kInvalidIndicesShapes[] = {
      {}, {kNumNonZeros}, {kNumNonZeros, kNumDims, 4}};
  for (const TensorShape& invalid_shape : kInvalidIndicesShapes) {
    const Tensor indices = Tensor(DT_INT64, invalid_shape);
    const Tensor values = Tensor(DT_FLOAT, TensorShape({kNumNonZeros}));
    const Tensor shape = Tensor(DT_INT64, TensorShape({kNumDims}));
    EXPECT_THAT((ValidateSparseTensor(indices, values, shape)),
                StatusIs(error::INVALID_ARGUMENT,
                         MatchesRegex("Sparse indices must be rank 2 .*")));
  }
}

TEST(ValidateSparseTensorTest, InvalidValuesRankFails) {
  constexpr int kNumNonZeros = 1000;
  constexpr int kNumDims = 3;
  // Values tensor must be rank 1, so try rank 0, 2.
  const TensorShape kInvalidValuesShapes[] = {{}, {kNumNonZeros, 2}};
  for (const TensorShape& invalid_shape : kInvalidValuesShapes) {
    const Tensor indices =
        Tensor(DT_INT64, TensorShape({kNumNonZeros, kNumDims}));
    const Tensor values = Tensor(DT_FLOAT, invalid_shape);
    const Tensor shape = Tensor(DT_INT64, TensorShape({kNumDims}));
    EXPECT_THAT((ValidateSparseTensor(indices, values, shape)),
                StatusIs(error::INVALID_ARGUMENT,
                         MatchesRegex("Sparse values must be rank 1 .*")));
  }
}

TEST(ValidateSparseTensorTest, InvalidShapeRankFails) {
  constexpr int kNumNonZeros = 1000;
  constexpr int kNumDims = 3;
  // Shape tensor must be rank 1, so try rank 0, 2.
  const TensorShape kInvalidShapeShapes[] = {{}, {kNumDims, 2}};
  for (const TensorShape& invalid_shape : kInvalidShapeShapes) {
    const Tensor indices =
        Tensor(DT_INT64, TensorShape({kNumNonZeros, kNumDims}));
    const Tensor values = Tensor(DT_FLOAT, TensorShape({kNumNonZeros}));
    const Tensor shape = Tensor(DT_INT64, invalid_shape);
    EXPECT_THAT((ValidateSparseTensor(indices, values, shape)),
                StatusIs(error::INVALID_ARGUMENT,
                         MatchesRegex("Sparse shape must be rank 1 .*")));
  }
}

TEST(ValidateSparseTensorTest, IncompatibleShapesFails) {
  constexpr int kNumNonZeros = 1000;
  constexpr int kNumDims = 3;

  const Tensor values = Tensor(DT_FLOAT, TensorShape({kNumNonZeros}));
  const Tensor shape = Tensor(DT_INT64, TensorShape({kNumDims}));

  // Indices and values must have the same size in dimension 0 (nnz).
  {
    const Tensor indices =
        Tensor(DT_INT64, TensorShape({kNumNonZeros + 1, kNumDims}));
    EXPECT_THAT((ValidateSparseTensor(indices, values, shape)),
                StatusIs(error::INVALID_ARGUMENT,
                         MatchesRegex("Number of elements in indices .* and "
                                      "values .* do not match")));
  }

  // Each index tuple must have the same size in dimension 1 as the dense
  // tensor shape (ndims).
  {
    const Tensor indices =
        Tensor(DT_INT64, TensorShape({kNumNonZeros, kNumDims + 1}));
    EXPECT_THAT(
        (ValidateSparseTensor(indices, values, shape)),
        StatusIs(error::INVALID_ARGUMENT,
                 MatchesRegex("Index rank .* and shape rank .* do not match")));
  }
}

}  // namespace
}  // namespace sparse_utils
}  // namespace tensorflow

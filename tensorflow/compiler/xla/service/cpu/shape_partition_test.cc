/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/shape_partition.h"

#include <algorithm>
#include <random>

#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace cpu {
namespace {

class ShapePartitionAssignerTest : public HloTestBase {
 protected:
  typedef std::vector<int64> Vec;

  void RunR2Test(const Shape& shape, int64 max_target_partition_count,
                 const std::vector<int64>* expected_partitions) {
    ShapePartitionAssigner assigner(shape);
    // Iterate through 1..max_target_partition_count.
    for (int64 i = 1; i <= max_target_partition_count; ++i) {
      std::vector<int64> actual_partitions =
          assigner.Run(/*target_partition_count=*/i);
      EXPECT_THAT(actual_partitions, expected_partitions[i - 1]);
    }
  }
};

TEST_F(ShapePartitionAssignerTest, Shape13WithLayout10) {
  std::vector<int64> expected_partitions[] = {{1} /* 1 */, {1, 2} /* 2 */};
  RunR2Test(ShapeUtil::MakeShapeWithLayout(F32, {1, 3}, {1, 0}), 2,
            expected_partitions);
}

TEST_F(ShapePartitionAssignerTest, Shape31WithLayout01) {
  std::vector<int64> expected_partitions[] = {
      {1} /* 1 */, {1, 2} /* 2 */
  };
  RunR2Test(ShapeUtil::MakeShapeWithLayout(F32, {3, 1}, {0, 1}), 2,
            expected_partitions);
}

TEST_F(ShapePartitionAssignerTest, Shape53WithLayout10) {
  std::vector<int64> expected_partitions[] = {{1} /* 1 */, {2} /* 2 */,
                                              {3} /* 3 */, {4} /* 4 */,
                                              {5} /* 5 */, {3, 2} /* 6 */};
  RunR2Test(ShapeUtil::MakeShapeWithLayout(F32, {5, 3}, {1, 0}), 6,
            expected_partitions);
}

TEST_F(ShapePartitionAssignerTest, Shape53WithLayout01) {
  std::vector<int64> expected_partitions[] = {
      {1} /* 1 */, {2} /* 2 */, {3} /* 3 */, {2, 2} /* 4 */};
  RunR2Test(ShapeUtil::MakeShapeWithLayout(F32, {5, 3}, {0, 1}), 4,
            expected_partitions);
}

TEST_F(ShapePartitionAssignerTest, Shape532WithLayout210) {
  std::vector<int64> expected_partitions[] = {
      {1} /* 1 */,     {2} /* 2 */,     {3} /* 3 */,     {4} /* 4 */,
      {5} /* 5 */,     {3, 2} /* 6 */,  {3, 2} /* 7 */,  {4, 2} /* 8 */,
      {3, 3} /* 9 */,  {3, 3} /* 10 */, {3, 3} /* 11 */, {4, 3} /* 12 */,
      {4, 3} /* 13 */, {4, 3} /* 14 */, {5, 3} /* 15 */, {4, 2, 2} /* 16 */};
  RunR2Test(ShapeUtil::MakeShapeWithLayout(F32, {5, 3, 2}, {2, 1, 0}), 16,
            expected_partitions);
}

TEST_F(ShapePartitionAssignerTest, Shape532WithLayout201) {
  std::vector<int64> expected_partitions[] = {
      {1} /* 1 */,     {2} /* 2 */,     {3} /* 3 */,     {2, 2} /* 4 */,
      {2, 2} /* 5 */,  {3, 2} /* 6 */,  {3, 2} /* 7 */,  {3, 2} /* 8 */,
      {3, 3} /* 9 */,  {3, 3} /* 10 */, {3, 3} /* 11 */, {3, 4} /* 12 */,
      {3, 4} /* 13 */, {3, 4} /* 14 */, {3, 5} /* 15 */, {3, 2, 2} /* 16 */};
  RunR2Test(ShapeUtil::MakeShapeWithLayout(F32, {5, 3, 2}, {2, 0, 1}), 16,
            expected_partitions);
}

class ShapePartitionIteratorTest : public HloTestBase {
 protected:
  typedef std::vector<std::pair<int64, int64>> Partition;
};

TEST_F(ShapePartitionIteratorTest, Shape53WithLayout10) {
  Shape shape = ShapeUtil::MakeShapeWithLayout(F32, {5, 3}, {1, 0});

  {
    ShapePartitionIterator iterator(shape, {1});
    EXPECT_EQ(1, iterator.GetTotalPartitionCount());
    EXPECT_TRUE(ContainersEqual(Partition({{0, 5}}), iterator.GetPartition(0)));
  }

  {
    ShapePartitionIterator iterator(shape, {2});
    EXPECT_EQ(2, iterator.GetTotalPartitionCount());
    EXPECT_TRUE(ContainersEqual(Partition({{0, 2}}), iterator.GetPartition(0)));
    EXPECT_TRUE(ContainersEqual(Partition({{2, 3}}), iterator.GetPartition(1)));
  }

  {
    ShapePartitionIterator iterator(shape, {3});
    EXPECT_EQ(3, iterator.GetTotalPartitionCount());
    EXPECT_TRUE(ContainersEqual(Partition({{0, 1}}), iterator.GetPartition(0)));
    EXPECT_TRUE(ContainersEqual(Partition({{1, 1}}), iterator.GetPartition(1)));
    EXPECT_TRUE(ContainersEqual(Partition({{2, 3}}), iterator.GetPartition(2)));
  }
}

TEST_F(ShapePartitionIteratorTest, Shape532WithLayout210) {
  Shape shape = ShapeUtil::MakeShapeWithLayout(F32, {5, 3, 2}, {2, 1, 0});

  {
    ShapePartitionIterator iterator(shape, {1, 1});
    EXPECT_EQ(1, iterator.GetTotalPartitionCount());
    EXPECT_TRUE(
        ContainersEqual(Partition({{0, 5}, {0, 3}}), iterator.GetPartition(0)));
  }

  {
    ShapePartitionIterator iterator(shape, {2, 2});
    EXPECT_EQ(4, iterator.GetTotalPartitionCount());
    EXPECT_TRUE(
        ContainersEqual(Partition({{0, 2}, {0, 1}}), iterator.GetPartition(0)));
    EXPECT_TRUE(
        ContainersEqual(Partition({{0, 2}, {1, 2}}), iterator.GetPartition(1)));
    EXPECT_TRUE(
        ContainersEqual(Partition({{2, 3}, {0, 1}}), iterator.GetPartition(2)));
    EXPECT_TRUE(
        ContainersEqual(Partition({{2, 3}, {1, 2}}), iterator.GetPartition(3)));
  }
}

class RandomShapePartitionIteratorTest : public HloTestBase {
 protected:
  typedef std::vector<std::pair<int64, int64>> Partition;
  RandomShapePartitionIteratorTest()
      : generator_(rd_()), distribution_(1, 10) {}

  std::vector<int64> RandR4Dims() { return {Rand(), Rand(), Rand(), Rand()}; }

  int64 Rand() { return distribution_(generator_); }

  std::random_device rd_;
  std::mt19937 generator_;
  std::uniform_int_distribution<int> distribution_;
};

TEST_F(RandomShapePartitionIteratorTest, RandomShapeAndPartitions) {
  // Choose random dimensions for R4 shape.
  Shape shape = ShapeUtil::MakeShapeWithLayout(F32, RandR4Dims(), {3, 2, 1, 0});
  // Choose random number of outer dimensions to partition.
  const int num_outer_dims_to_partition = 1 + (Rand() % 3);
  // Choose random outer dimension partition counts.
  std::vector<int64> dim_sizes(num_outer_dims_to_partition);
  std::vector<int64> dim_partition_counts(num_outer_dims_to_partition);
  int64 total_dim_size = 1;
  for (int i = 0; i < num_outer_dims_to_partition; ++i) {
    const int64 dimension = shape.layout().minor_to_major(
        shape.layout().minor_to_major_size() - 1 - i);
    dim_sizes[i] = shape.dimensions(dimension);
    total_dim_size *= dim_sizes[i];
    // Choose dimension partition count in [1, dim_size]
    const int64 dim_partition_count = 1 + Rand() % dim_sizes[i];
    dim_partition_counts[i] = dim_partition_count;
  }
  // Iterate through all partition: for each partition record covered
  // index ranges by dimension.
  std::vector<std::map<int64, int64>> ranges(num_outer_dims_to_partition);
  ShapePartitionIterator partition_iterator(shape, dim_partition_counts);
  const int64 partition_count = partition_iterator.GetTotalPartitionCount();
  for (int64 i = 0; i < partition_count; ++i) {
    const auto& dim_partition = partition_iterator.GetPartition(i);
    for (int dim = 0; dim < dim_partition.size(); ++dim) {
      ranges[dim].insert(
          std::make_pair(dim_partition[dim].first,
                         dim_partition[dim].first + dim_partition[dim].second));
    }
  }
  // Check that partitions cover entire dimension size range (for each
  // partitioned dimension).
  for (int i = 0; i < ranges.size(); ++i) {
    int64 expected_index = 0;
    for (auto& r : ranges[i]) {
      EXPECT_EQ(expected_index, r.first);
      expected_index = r.second;
    }
    EXPECT_EQ(expected_index, dim_sizes[i]);
  }
}

}  // namespace
}  // namespace cpu
}  // namespace xla

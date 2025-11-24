/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/ir/tile_assignment.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/array2d.h"

namespace xla {
namespace {

TEST(IotaTileAssignmentTest, Create) {
  // Test with dims only
  IotaTileAssignment iota1 = IotaTileAssignment::Create({2, 3});
  EXPECT_EQ(iota1.dims(), absl::MakeConstSpan(std::vector<int64_t>{2, 3}));
  EXPECT_EQ(iota1.reshape_dims(), absl::MakeConstSpan(std::vector<int64_t>{6}));
  EXPECT_EQ(iota1.transpose_perm(), absl::MakeConstSpan(std::vector<int>{0}));
  EXPECT_EQ(iota1.num_elements(), 6);

  // Test with reshape_dims and transpose_perm
  IotaTileAssignment iota2 = IotaTileAssignment::Create({2, 6}, {3, 4}, {0, 1});
  EXPECT_EQ(iota2.dims(), absl::MakeConstSpan(std::vector<int64_t>{2, 6}));
  EXPECT_EQ(iota2.reshape_dims(),
            absl::MakeConstSpan(std::vector<int64_t>{12}));
  EXPECT_EQ(iota2.transpose_perm(), absl::MakeConstSpan(std::vector<int>{0}));
  EXPECT_EQ(iota2.num_elements(), 12);

  // Test canonicalization: remove size one dims
  IotaTileAssignment iota3 = IotaTileAssignment::Create(
      {1, 3, 1, 4, 1, 5}, {1, 3, 1, 4, 1, 5}, {4, 3, 2, 5, 1, 0});
  EXPECT_EQ(iota3.dims(),
            absl::MakeConstSpan(std::vector<int64_t>{1, 3, 1, 4, 1, 5}));
  EXPECT_EQ(iota3.reshape_dims(),
            absl::MakeConstSpan(std::vector<int64_t>{3, 20}));
  EXPECT_EQ(iota3.transpose_perm(),
            absl::MakeConstSpan(std::vector<int>{1, 0}));

  // Test canonicalization: merge major to minor
  IotaTileAssignment iota4 =
      IotaTileAssignment::Create({2, 3, 4}, {2, 3, 4}, {0, 1, 2});
  EXPECT_EQ(iota4.reshape_dims(),
            absl::MakeConstSpan(std::vector<int64_t>{24}));
  EXPECT_EQ(iota4.transpose_perm(), absl::MakeConstSpan(std::vector<int>{0}));

  IotaTileAssignment iota5 =
      IotaTileAssignment::Create({2, 3, 4}, {2, 3, 4}, {1, 0, 2});
  EXPECT_EQ(iota5.reshape_dims(),
            absl::MakeConstSpan(std::vector<int64_t>{2, 3, 4}));
  EXPECT_EQ(iota5.transpose_perm(),
            absl::MakeConstSpan(std::vector<int>{1, 0, 2}));
}

TEST(IotaTileAssignmentTest, ToArray) {
  IotaTileAssignment iota = IotaTileAssignment::Create({2, 3}, {6}, {0});
  Array<int64_t> array = iota.ToArray();
  Array2D<int64_t> expected({{0, 1, 2}, {3, 4, 5}});
  EXPECT_EQ(array, expected);

  IotaTileAssignment iota2 = IotaTileAssignment::Create({3, 2}, {2, 3}, {1, 0});
  Array<int64_t> array2 = iota2.ToArray();
  Array2D<int64_t> expected2({{0, 3}, {1, 4}, {2, 5}});
  EXPECT_EQ(array2, expected2);

  IotaTileAssignment iota3 =
      IotaTileAssignment::Create({3, 4, 5}, {3, 4, 5}, {2, 0, 1});
  Array<int64_t> array3 = iota3.ToArray();

  Array<int64_t> expected3({3, 4, 5});
  expected3.FillIota(0);
  expected3.TransposeDimensions({2, 0, 1});
  expected3.Reshape({3, 4, 5});
  EXPECT_EQ(array3, expected3);
}

TEST(IotaTileAssignmentTest, Transpose) {
  IotaTileAssignment iota = IotaTileAssignment::Create({2, 3}, {6}, {0});

  // Noop transpose
  auto transposed1 = iota.Transpose({0, 1});
  EXPECT_TRUE(transposed1.has_value());
  EXPECT_EQ(transposed1->dims(),
            absl::MakeConstSpan(std::vector<int64_t>{2, 3}));
  EXPECT_EQ(transposed1->reshape_dims(),
            absl::MakeConstSpan(std::vector<int64_t>{6}));
  EXPECT_EQ(transposed1->transpose_perm(),
            absl::MakeConstSpan(std::vector<int>{0}));

  // Reshape transpose
  IotaTileAssignment iota2 = IotaTileAssignment::Create({2, 1, 3}, {6}, {0});
  auto transposed2 = iota2.Transpose({0, 2, 1});
  EXPECT_TRUE(transposed2.has_value());
  EXPECT_EQ(transposed2->dims(),
            absl::MakeConstSpan(std::vector<int64_t>{2, 3, 1}));
  EXPECT_EQ(transposed2->reshape_dims(),
            absl::MakeConstSpan(std::vector<int64_t>{6}));
  EXPECT_EQ(transposed2->transpose_perm(),
            absl::MakeConstSpan(std::vector<int>{0}));

  // Regular transpose
  IotaTileAssignment iota3 = IotaTileAssignment::Create({2, 3}, {2, 3}, {0, 1});
  auto transposed3 = iota3.Transpose({1, 0});
  EXPECT_TRUE(transposed3.has_value());
  EXPECT_EQ(transposed3->dims(),
            absl::MakeConstSpan(std::vector<int64_t>{3, 2}));
  EXPECT_EQ(transposed3->reshape_dims(),
            absl::MakeConstSpan(std::vector<int64_t>{2, 3}));
  EXPECT_EQ(transposed3->transpose_perm(),
            absl::MakeConstSpan(std::vector<int>{1, 0}));
}

TEST(IotaTileAssignmentTest, ValueAt) {
  IotaTileAssignment iota = IotaTileAssignment::Create({2, 3}, {6}, {0});
  EXPECT_EQ(iota.value_at({0, 0}), 0);
  EXPECT_EQ(iota.value_at({0, 1}), 1);
  EXPECT_EQ(iota.value_at({1, 0}), 3);
  EXPECT_EQ(iota.value_at({1, 2}), 5);

  IotaTileAssignment iota2 = IotaTileAssignment::Create({3, 2}, {2, 3}, {1, 0});
  EXPECT_EQ(iota2.value_at({0, 0}), 0);
  EXPECT_EQ(iota2.value_at({0, 1}), 3);
  EXPECT_EQ(iota2.value_at({1, 0}), 1);
  EXPECT_EQ(iota2.value_at({2, 1}), 5);
}

TEST(IotaTileAssignmentTest, ToString) {
  IotaTileAssignment iota = IotaTileAssignment::Create({2, 3});
  EXPECT_EQ(iota.ArrayToString(), "[6]");
  EXPECT_EQ(iota.ToString(), "[2,3]<=[6]");

  IotaTileAssignment iota2 = IotaTileAssignment::Create({2, 6}, {3, 4}, {0, 1});
  EXPECT_EQ(iota2.ArrayToString(), "[12]");
  EXPECT_EQ(iota2.ToString(), "[2,6]<=[12]");

  IotaTileAssignment iota3 =
      IotaTileAssignment::Create({3, 4, 5}, {3, 4, 5}, {2, 0, 1});
  EXPECT_EQ(iota3.ArrayToString(), "[12,5]T(1,0)");
  EXPECT_EQ(iota3.ToString(), "[3,4,5]<=[12,5]T(1,0)");
}

TEST(TileAssignmentTest, FromIota) {
  IotaTileAssignment iota = IotaTileAssignment::Create({2, 3});
  TileAssignment tile_assignment(iota);

  EXPECT_EQ(tile_assignment.ToString(), "devices=[2,3]<=[6]");
  EXPECT_EQ(tile_assignment.dimensions(),
            absl::MakeConstSpan(std::vector<int64_t>{2, 3}));
  EXPECT_EQ(tile_assignment.num_elements(), 6);
  EXPECT_EQ(tile_assignment({1, 1}), 4);
}

TEST(TileAssignmentTest, FromArray) {
  Array2D<int64_t> array({{0, 1}, {2, 3}});
  TileAssignment tile_assignment(std::make_shared<Array<int64_t>>(array));

  EXPECT_EQ(tile_assignment.ToString(), "devices=[2,2]0,1,2,3");
  EXPECT_EQ(tile_assignment.ArrayToString(), "0,1,2,3");
  EXPECT_EQ(tile_assignment.dimensions(),
            absl::MakeConstSpan(std::vector<int64_t>{2, 2}));
  EXPECT_EQ(tile_assignment.num_elements(), 4);
  EXPECT_EQ(tile_assignment({1, 1}), 3);
}

TEST(TileAssignmentTest, Equality) {
  IotaTileAssignment iota1 = IotaTileAssignment::Create({2, 3});
  IotaTileAssignment iota2 = IotaTileAssignment::Create({2, 3});
  IotaTileAssignment iota3 = IotaTileAssignment::Create({3, 2});

  TileAssignment ta1(iota1);
  TileAssignment ta2(iota2);
  TileAssignment ta3(iota3);

  EXPECT_EQ(ta1, ta2);
  EXPECT_NE(ta1, ta3);

  Array2D<int64_t> array1({{0, 1}, {2, 3}});
  Array2D<int64_t> array2({{0, 1}, {2, 3}});
  Array2D<int64_t> array3({{0, 1}, {2, 4}});

  TileAssignment ta4(std::make_shared<Array<int64_t>>(array1));
  TileAssignment ta5(std::make_shared<Array<int64_t>>(array2));
  TileAssignment ta6(std::make_shared<Array<int64_t>>(array3));

  EXPECT_EQ(ta4, ta5);
  EXPECT_NE(ta4, ta6);

  // Iota vs Array
  Array2D<int64_t> array_iota({{0, 1, 2}, {3, 4, 5}});
  TileAssignment ta7(std::make_shared<Array<int64_t>>(array_iota));
  EXPECT_EQ(ta1, ta7);
}

TEST(TileAssignmentTest, Reshape) {
  IotaTileAssignment iota = IotaTileAssignment::Create({2, 6});
  TileAssignment ta(iota);
  TileAssignment reshaped = ta.Reshape({3, 4});
  EXPECT_EQ(reshaped.dimensions(),
            absl::MakeConstSpan(std::vector<int64_t>{3, 4}));
  EXPECT_EQ(reshaped.num_elements(), 12);
  EXPECT_EQ(reshaped({1, 1}), 5);

  Array2D<int64_t> array({{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}});
  TileAssignment ta2(std::make_shared<Array<int64_t>>(array));
  TileAssignment reshaped2 = ta2.Reshape({6, 2});
  EXPECT_EQ(reshaped2.dimensions(),
            absl::MakeConstSpan(std::vector<int64_t>{6, 2}));
  EXPECT_EQ(reshaped2({1, 1}), 3);
}

TEST(TileAssignmentTest, Transpose) {
  IotaTileAssignment iota = IotaTileAssignment::Create({2, 3});
  TileAssignment ta(iota);
  TileAssignment transposed = ta.Transpose({1, 0});
  EXPECT_EQ(transposed.dimensions(),
            absl::MakeConstSpan(std::vector<int64_t>{3, 2}));
  EXPECT_EQ(transposed({1, 0}), 1);

  Array2D<int64_t> array({{0, 1}, {2, 3}});
  TileAssignment ta2(std::make_shared<Array<int64_t>>(array));
  TileAssignment transposed2 = ta2.Transpose({1, 0});
  EXPECT_EQ(transposed2.dimensions(),
            absl::MakeConstSpan(std::vector<int64_t>{2, 2}));
  EXPECT_EQ(transposed2({1, 0}), 1);
}

TEST(TileAssignmentTest, UsesDevice) {
  IotaTileAssignment iota = IotaTileAssignment::Create({2, 3});
  TileAssignment ta(iota);
  EXPECT_TRUE(ta.UsesDevice(0));
  EXPECT_TRUE(ta.UsesDevice(5));
  EXPECT_FALSE(ta.UsesDevice(6));

  Array2D<int64_t> array({{1, 2}, {3, 4}});
  TileAssignment ta2(std::make_shared<Array<int64_t>>(array));
  EXPECT_TRUE(ta2.UsesDevice(1));
  EXPECT_FALSE(ta2.UsesDevice(0));
}

TEST(TileAssignmentTest, MaterializeArray) {
  IotaTileAssignment iota = IotaTileAssignment::Create({2, 3});
  TileAssignment ta(iota);
  const Array<int64_t>& array = ta.array();
  Array2D<int64_t> expected({{0, 1, 2}, {3, 4, 5}});
  EXPECT_EQ(array, expected);

  std::shared_ptr<const Array<int64_t>> shared_array = ta.shared_array();
  EXPECT_EQ(*shared_array, expected);
}

TEST(TileAssignmentTest, Each) {
  IotaTileAssignment iota = IotaTileAssignment::Create({2, 2});
  TileAssignment ta(iota);
  std::vector<std::pair<std::vector<int64_t>, int64_t>> values;
  ta.Each([&](absl::Span<const int64_t> indices, int64_t value) {
    values.push_back({{indices.begin(), indices.end()}, value});
  });
  EXPECT_THAT(values, ::testing::ElementsAre(
                          ::testing::Pair(std::vector<int64_t>({0, 0}), 0),
                          ::testing::Pair(std::vector<int64_t>({0, 1}), 1),
                          ::testing::Pair(std::vector<int64_t>({1, 0}), 2),
                          ::testing::Pair(std::vector<int64_t>({1, 1}), 3)));
}

TEST(TileAssignmentTest, EachStatus) {
  IotaTileAssignment iota = IotaTileAssignment::Create({2, 2});
  TileAssignment ta(iota);
  absl::Status status =
      ta.EachStatus([&](absl::Span<const int64_t> indices, int64_t value) {
        if (value == 3) {
          return absl::InternalError("Test Error");
        }
        return absl::OkStatus();
      });
  EXPECT_FALSE(status.ok());
}

}  // namespace
}  // namespace xla

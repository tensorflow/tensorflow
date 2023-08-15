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

#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_util.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_schedule.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_live_range.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace op = xla::testing::opcode_matchers;

namespace {
using testing::ElementsAre;
using testing::ElementsAreArray;

xla::HloSharding CreateHloSharding(absl::Span<const int64_t> tile_dimensions,
                                   absl::Span<const int64_t> devices,
                                   bool replicate_on_last_dim = false) {
  xla::OpSharding proto;
  proto.set_type(xla::OpSharding::OTHER);
  for (auto i : tile_dimensions) {
    proto.add_tile_assignment_dimensions(i);
  }
  for (auto i : devices) {
    proto.add_tile_assignment_devices(i);
  }
  proto.set_replicate_on_last_tile_dim(replicate_on_last_dim);
  return xla::HloSharding::FromProto(proto).value();
}

std::vector<xla::HloInstruction*> GetInstructionVector(
    const xla::HloModule* module) {
  std::vector<xla::HloInstruction*> instructions;
  for (auto computation : module->computations()) {
    for (auto instruction : computation->instructions()) {
      instructions.push_back(instruction);
    }
  }
  return instructions;
}

TEST(UtilTest, GetDimensionMappingTest) {
  EXPECT_THAT(xla::spmd::GetDimensionMapping({0}, 3),
              ElementsAreArray({-1, 0, 1}));
  EXPECT_THAT(xla::spmd::GetDimensionMapping({1}, 3),
              ElementsAreArray({0, -1, 1}));
  EXPECT_THAT(xla::spmd::GetDimensionMapping({2}, 3),
              ElementsAreArray({0, 1, -1}));
  EXPECT_THAT(xla::spmd::GetDimensionMapping({0, 1}, 5),
              ElementsAreArray({-1, -1, 0, 1, 2}));
  EXPECT_THAT(xla::spmd::GetDimensionMapping({1, 2}, 5),
              ElementsAreArray({0, -1, -1, 1, 2}));
  EXPECT_THAT(xla::spmd::GetDimensionMapping({0, 2}, 5),
              ElementsAreArray({-1, 0, -1, 1, 2}));
  EXPECT_THAT(xla::spmd::GetDimensionMapping({0, 1, 2}, 5),
              ElementsAreArray({-1, -1, -1, 0, 1}));
}

TEST(UtilTest, IsDivisibleTest) {
  EXPECT_TRUE(xla::spmd::IsDivisible(4, 2));
  EXPECT_TRUE(xla::spmd::IsDivisible(1024, 256));
  EXPECT_TRUE(xla::spmd::IsDivisible(512, 128));
  EXPECT_FALSE(xla::spmd::IsDivisible(5, 2));
  EXPECT_FALSE(xla::spmd::IsDivisible(133, 3));
  EXPECT_FALSE(xla::spmd::IsDivisible(128, 5));
}

TEST(UtilTest, GetReplicaGroupsAlongOneDimensionDeviceMesh2x4) {
  // std::vector<std::vector<int>> replica_groups;
  xla::Array<int64_t> device_mesh({{1, 2, 3, 4}, {5, 6, 7, 8}});
  std::vector<std::vector<int64_t>> replica_groups_0 =
      xla::spmd::GetReplicaGroupsAlongOneDimension(device_mesh, 0);
  EXPECT_THAT(replica_groups_0,
              ElementsAre(ElementsAre(1, 5), ElementsAre(2, 6),
                          ElementsAre(3, 7), ElementsAre(4, 8)));

  std::vector<std::vector<int64_t>> replica_groups_1 =
      xla::spmd::GetReplicaGroupsAlongOneDimension(device_mesh, 1);
  EXPECT_THAT(replica_groups_1,
              ElementsAre(ElementsAre(1, 2, 3, 4), ElementsAre(5, 6, 7, 8)));
}

TEST(UtilTest, GetReplicaGroupsAlongOneDimensionDeviceMesh3x3) {
  xla::Array<int64_t> device_mesh({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::vector<std::vector<int64_t>> replica_groups_0 =
      xla::spmd::GetReplicaGroupsAlongOneDimension(device_mesh, 0);
  EXPECT_THAT(replica_groups_0,
              ElementsAre(ElementsAre(1, 4, 7), ElementsAre(2, 5, 8),
                          ElementsAre(3, 6, 9)));

  std::vector<std::vector<int64_t>> replica_groups_1 =
      xla::spmd::GetReplicaGroupsAlongOneDimension(device_mesh, 1);
  EXPECT_THAT(replica_groups_1,
              ElementsAre(ElementsAre(1, 2, 3), ElementsAre(4, 5, 6),
                          ElementsAre(7, 8, 9)));
}

TEST(UtilTest, GetReplicaGroupsAlongOneDimensionDeviceMesh2x2x2) {
  xla::Array<int64_t> device_mesh({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
  std::vector<std::vector<int64_t>> replica_groups_0 =
      xla::spmd::GetReplicaGroupsAlongOneDimension(device_mesh, 0);
  EXPECT_THAT(replica_groups_0,
              ElementsAre(ElementsAre(1, 5), ElementsAre(2, 6),
                          ElementsAre(3, 7), ElementsAre(4, 8)));

  std::vector<std::vector<int64_t>> replica_groups_1 =
      xla::spmd::GetReplicaGroupsAlongOneDimension(device_mesh, 1);
  EXPECT_THAT(replica_groups_1,
              ElementsAre(ElementsAre(1, 3), ElementsAre(2, 4),
                          ElementsAre(5, 7), ElementsAre(6, 8)));

  std::vector<std::vector<int64_t>> replica_groups_2 =
      xla::spmd::GetReplicaGroupsAlongOneDimension(device_mesh, 2);
  EXPECT_THAT(replica_groups_2,
              ElementsAre(ElementsAre(1, 2), ElementsAre(3, 4),
                          ElementsAre(5, 6), ElementsAre(7, 8)));
}

TEST(UtilTest, GetReplicaGroupsAlongOneDimensionDeviceMesh3x2x2) {
  xla::Array<int64_t> device_mesh(
      {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}});
  std::vector<std::vector<int64_t>> replica_groups_0 =
      xla::spmd::GetReplicaGroupsAlongOneDimension(device_mesh, 0);
  EXPECT_THAT(replica_groups_0,
              ElementsAre(ElementsAre(1, 5, 9), ElementsAre(2, 6, 10),
                          ElementsAre(3, 7, 11), ElementsAre(4, 8, 12)));

  std::vector<std::vector<int64_t>> replica_groups_1 =
      xla::spmd::GetReplicaGroupsAlongOneDimension(device_mesh, 1);
  EXPECT_THAT(
      replica_groups_1,
      ElementsAre(ElementsAre(1, 3), ElementsAre(2, 4), ElementsAre(5, 7),
                  ElementsAre(6, 8), ElementsAre(9, 11), ElementsAre(10, 12)));

  std::vector<std::vector<int64_t>> replica_groups_2 =
      xla::spmd::GetReplicaGroupsAlongOneDimension(device_mesh, 2);
  EXPECT_THAT(
      replica_groups_2,
      ElementsAre(ElementsAre(1, 2), ElementsAre(3, 4), ElementsAre(5, 6),
                  ElementsAre(7, 8), ElementsAre(9, 10), ElementsAre(11, 12)));
}

TEST(UtilTest, GetValuesAlongOneDimTest) {
  // Small arrays.
  EXPECT_THAT(
      xla::spmd::GetValuesAlongOneDim(xla::Array<int64_t>({{0, 1}, {2, 3}}), 0)
          .value(),
      ElementsAreArray({0, 2}));
  EXPECT_THAT(
      xla::spmd::GetValuesAlongOneDim(xla::Array<int64_t>({{0, 1}, {2, 3}}), 1)
          .value(),
      ElementsAreArray({0, 1}));
  EXPECT_THAT(xla::spmd::GetValuesAlongOneDim(
                  xla::Array<int64_t>({{0}, {1}, {2}, {3}}), 0)
                  .value(),
              ElementsAreArray({0, 1, 2, 3}));
  EXPECT_THAT(xla::spmd::GetValuesAlongOneDim(
                  xla::Array<int64_t>({{0}, {1}, {2}, {3}}), 1)
                  .value(),
              ElementsAreArray({0}));
  EXPECT_FALSE(xla::spmd::GetValuesAlongOneDim(
                   xla::Array<int64_t>({{0}, {1}, {2}, {3}}), 2)
                   .ok());
  // Larger arrays
  xla::Array<int64_t> array_2d({
      {0, 1, 2, 3, 4, 5, 6, 7, 8},
      {1, 2, 3, 4, 5, 6, 7, 8, 9},
      {2, 3, 4, 5, 6, 7, 8, 9, 10},
      {3, 4, 5, 6, 7, 8, 9, 10, 11},
  });
  EXPECT_THAT(xla::spmd::GetValuesAlongOneDim(array_2d, 0).value(),
              ElementsAreArray({0, 1, 2, 3}));
  EXPECT_THAT(xla::spmd::GetValuesAlongOneDim(array_2d, 1).value(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_FALSE(xla::spmd::GetValuesAlongOneDim(array_2d, 2).ok());

  // array_3d.dimensions = [4,2,4]
  xla::Array<int64_t> array_3d({
      {{0, 1, 2, 3}, {4, 5, 6, 7}},
      {{2, 3, 4, 5}, {6, 7, 8, 9}},
      {{3, 4, 5, 6}, {7, 8, 9, 10}},
      {{4, 5, 6, 7}, {8, 9, 10, 11}},
  });
  EXPECT_THAT(xla::spmd::GetValuesAlongOneDim(array_3d, 0).value(),
              ElementsAreArray({0, 2, 3, 4}));
  EXPECT_THAT(xla::spmd::GetValuesAlongOneDim(array_3d, 1).value(),
              ElementsAreArray({0, 4}));
  EXPECT_THAT(xla::spmd::GetValuesAlongOneDim(array_3d, 2).value(),
              ElementsAreArray({0, 1, 2, 3}));
  EXPECT_FALSE(xla::spmd::GetValuesAlongOneDim(array_3d, 3).ok());
}

TEST(UtilTest, CheckArithmeticSequenceTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto delta,
                          xla::spmd::CheckArithmeticSequence({0, 1, 2, 3}));
  EXPECT_EQ(delta, 1);
  EXPECT_FALSE(xla::spmd::CheckArithmeticSequence({0, 1, 2, 4}).ok());
  TF_ASSERT_OK_AND_ASSIGN(auto delta1, xla::spmd::CheckArithmeticSequence(
                                           {10, 11, 12, 13, 14, 15}));
  EXPECT_EQ(delta1, 1);
  TF_ASSERT_OK_AND_ASSIGN(
      auto delta2, xla::spmd::CheckArithmeticSequence({63, 65, 67, 69, 71}));
  EXPECT_EQ(delta2, 2);
}

TEST(UtilTest, GetTensorDimToMeshDim1D) {
  // 1D mesh, 1D sharding
  xla::Array<int64_t> device_mesh_1d({{0}, {1}, {2}, {3}});
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          3, CreateHloSharding({4, 1, 1}, {0, 1, 2, 3}, false), device_mesh_1d),
      ElementsAreArray({0, -1, -1}));

  // 2D mesh, 1D sharding
  xla::Array<int64_t> device_mesh({{0, 1}, {2, 3}});
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          2, CreateHloSharding({2, 1, 2}, {0, 1, 2, 3}, true), device_mesh),
      ElementsAreArray({0, -1}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          2, CreateHloSharding({2, 1, 2}, {0, 2, 1, 3}, true), device_mesh),
      ElementsAreArray({1, -1}));
  // 3D mesh, 1D sharding
  xla::Array<int64_t> device_mesh_3d({{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}});
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          2, CreateHloSharding({2, 1, 4}, {0, 1, 4, 5, 2, 3, 6, 7}, true),
          device_mesh_3d),
      ElementsAreArray({1, -1}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          3, CreateHloSharding({2, 1, 1, 4}, {0, 1, 2, 3, 4, 5, 6, 7}, true),
          device_mesh_3d),
      ElementsAreArray({0, -1, -1}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          3, CreateHloSharding({2, 1, 1, 4}, {0, 1, 4, 5, 2, 3, 6, 7}, true),
          device_mesh_3d),
      ElementsAreArray({1, -1, -1}));
}

TEST(UtilTest, GetTensorDimToMeshDim2D) {
  xla::Array<int64_t> device_mesh({{0, 1}, {2, 3}});
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          2, CreateHloSharding({2, 2}, {0, 1, 2, 3}, false), device_mesh),
      ElementsAreArray({0, 1}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          2, CreateHloSharding({2, 2}, {0, 2, 1, 3}, false), device_mesh),
      ElementsAreArray({1, 0}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          3, CreateHloSharding({1, 1, 2, 2}, {0, 1, 2, 3}, true), device_mesh),
      ElementsAreArray({-1, -1, 0}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          2, CreateHloSharding({1, 2, 2}, {0, 2, 1, 3}, true), device_mesh),
      ElementsAreArray({-1, 1}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          3, CreateHloSharding({2, 2, 1}, {0, 2, 1, 3}, false), device_mesh),
      ElementsAreArray({1, 0, -1}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          3, CreateHloSharding({2, 2, 1, 1}, {0, 1, 2, 3}, true), device_mesh),
      ElementsAreArray({0, 1, -1}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          3, CreateHloSharding({2, 2, 1, 1}, {0, 2, 1, 3}, true), device_mesh),
      ElementsAreArray({1, 0, -1}));
}

TEST(UtilTest, GetTensorDimToMeshDim3D) {
  xla::Array<int64_t> device_mesh({{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}});
  // 3D mesh, 1D sharding
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          2, CreateHloSharding({2, 1, 4}, {0, 1, 2, 3, 4, 5, 6, 7}, true),
          device_mesh),
      ElementsAreArray({0, -1}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          2, CreateHloSharding({2, 1, 4}, {0, 1, 4, 5, 2, 3, 6, 7}, true),
          device_mesh),
      ElementsAreArray({1, -1}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          2, CreateHloSharding({2, 1, 4}, {0, 2, 4, 6, 1, 3, 5, 7}, true),
          device_mesh),
      ElementsAreArray({2, -1}));
  // 3D mesh 2D sharding.
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          3, CreateHloSharding({2, 2, 1, 2}, {0, 1, 2, 3, 4, 5, 6, 7}, true),
          device_mesh),
      ElementsAreArray({0, 1, -1}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          3, CreateHloSharding({2, 2, 1, 2}, {0, 1, 4, 5, 2, 3, 6, 7}, true),
          device_mesh),
      ElementsAreArray({1, 0, -1}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          3, CreateHloSharding({2, 2, 1, 2}, {0, 2, 4, 6, 1, 3, 5, 7}, true),
          device_mesh),
      ElementsAreArray({2, 0, -1}));
  // 3D mesh, 3D sharding. Although 3D sharding is not supported yet, just
  // test to see whether this function works.
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          3, CreateHloSharding({2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7}, false),
          device_mesh),
      ElementsAreArray({0, 1, 2}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          2, CreateHloSharding({2, 2, 2}, {0, 1, 2, 3, 4, 5, 6, 7}, true),
          device_mesh),
      ElementsAreArray({0, 1}));
  EXPECT_THAT(
      xla::spmd::GetTensorDimToMeshDim(
          2, CreateHloSharding({2, 2, 2}, {0, 1, 4, 5, 2, 3, 6, 7}, true),
          device_mesh),
      ElementsAreArray({1, 0}));
}

TEST(UtilTest, Tile2x2Mesh) {
  xla::Array<int64_t> device_mesh({{0, 1}, {2, 3}});
  xla::HloSharding sharding_00 = xla::spmd::Tile(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {0}, {0}, device_mesh);
  EXPECT_THAT(sharding_00.tile_assignment().dimensions(), ElementsAre(2, 1, 2));
  EXPECT_THAT(sharding_00.tile_assignment().array(),
              ElementsAreArray({0, 1, 2, 3}));

  xla::HloSharding sharding_01 = xla::spmd::Tile(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {0}, {1}, device_mesh);
  EXPECT_THAT(sharding_01.tile_assignment().dimensions(), ElementsAre(2, 1, 2));
  EXPECT_THAT(sharding_01.tile_assignment().array(),
              ElementsAreArray({0, 2, 1, 3}));

  xla::HloSharding sharding_10 = xla::spmd::Tile(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {1}, {0}, device_mesh);
  EXPECT_THAT(sharding_10.tile_assignment().dimensions(), ElementsAre(1, 2, 2));
  EXPECT_THAT(sharding_10.tile_assignment().array(),
              ElementsAreArray({0, 1, 2, 3}));

  xla::HloSharding sharding_11 = xla::spmd::Tile(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {1}, {1}, device_mesh);
  EXPECT_THAT(sharding_11.tile_assignment().dimensions(), ElementsAre(1, 2, 2));
  EXPECT_THAT(sharding_11.tile_assignment().array(),
              ElementsAreArray({0, 2, 1, 3}));
}

TEST(UtilTest, Tile1x2x2Mesh) {
  xla::Array<int64_t> device_mesh({{{0, 1}, {2, 3}}});
  xla::HloSharding sharding_01 = xla::spmd::Tile(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {0}, {1}, device_mesh);
  EXPECT_THAT(sharding_01.tile_assignment().dimensions(), ElementsAre(2, 1, 2));
  EXPECT_THAT(sharding_01.tile_assignment().array(),
              ElementsAreArray({0, 1, 2, 3}));

  xla::HloSharding sharding_02 = xla::spmd::Tile(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {0}, {2}, device_mesh);
  EXPECT_THAT(sharding_02.tile_assignment().dimensions(), ElementsAre(2, 1, 2));
  EXPECT_THAT(sharding_02.tile_assignment().array(),
              ElementsAreArray({0, 2, 1, 3}));

  xla::HloSharding sharding_11 = xla::spmd::Tile(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {1}, {1}, device_mesh);
  EXPECT_THAT(sharding_11.tile_assignment().dimensions(), ElementsAre(1, 2, 2));
  EXPECT_THAT(sharding_11.tile_assignment().array(),
              ElementsAreArray({0, 1, 2, 3}));

  xla::HloSharding sharding_12 = xla::spmd::Tile(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {1}, {2}, device_mesh);
  EXPECT_THAT(sharding_12.tile_assignment().dimensions(), ElementsAre(1, 2, 2));
  EXPECT_THAT(sharding_12.tile_assignment().array(),
              ElementsAreArray({0, 2, 1, 3}));
}

TEST(UtilTest, Tile2x2x2Mesh) {
  xla::Array<int64_t> device_mesh({{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}});
  xla::HloSharding sharding_00 = xla::spmd::Tile(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {0}, {0}, device_mesh);
  EXPECT_THAT(sharding_00.tile_assignment().dimensions(), ElementsAre(2, 1, 4));
  EXPECT_THAT(sharding_00.tile_assignment().array(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7}));

  xla::HloSharding sharding_01 = xla::spmd::Tile(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {0}, {1}, device_mesh);
  EXPECT_THAT(sharding_01.tile_assignment().dimensions(), ElementsAre(2, 1, 4));
  EXPECT_THAT(sharding_01.tile_assignment().array(),
              ElementsAreArray({0, 1, 4, 5, 2, 3, 6, 7}));

  xla::HloSharding sharding_02 = xla::spmd::Tile(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {0}, {2}, device_mesh);
  EXPECT_THAT(sharding_02.tile_assignment().dimensions(), ElementsAre(2, 1, 4));
  EXPECT_THAT(sharding_02.tile_assignment().array(),
              ElementsAreArray({0, 2, 4, 6, 1, 3, 5, 7}));

  xla::HloSharding sharding_10 = xla::spmd::Tile(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {1}, {0}, device_mesh);
  EXPECT_THAT(sharding_10.tile_assignment().dimensions(), ElementsAre(1, 2, 4));
  EXPECT_THAT(sharding_10.tile_assignment().array(),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7}));

  xla::HloSharding sharding_11 = xla::spmd::Tile(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {1}, {1}, device_mesh);
  EXPECT_THAT(sharding_11.tile_assignment().dimensions(), ElementsAre(1, 2, 4));
  EXPECT_THAT(sharding_11.tile_assignment().array(),
              ElementsAreArray({0, 1, 4, 5, 2, 3, 6, 7}));
  EXPECT_TRUE(sharding_11.ReplicateOnLastTileDim());

  xla::HloSharding sharding_12 = xla::spmd::Tile(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {1}, {2}, device_mesh);
  EXPECT_THAT(sharding_12.tile_assignment().dimensions(), ElementsAre(1, 2, 4));
  EXPECT_THAT(sharding_12.tile_assignment().array(),
              ElementsAreArray({0, 2, 4, 6, 1, 3, 5, 7}));
  EXPECT_TRUE(sharding_12.ReplicateOnLastTileDim());
}

TEST(UtilTest, TransposeTest2D) {
  xla::Array<int64_t> array({{0, 1}, {2, 3}});
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(array, {1, 0}),
              ElementsAreArray({0, 2, 1, 3}));

  xla::Array<int64_t> wide_array({{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}});
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(wide_array, {1, 0}),
              ElementsAreArray({0, 5, 1, 6, 2, 7, 3, 8, 4, 9}));

  xla::Array<int64_t> long_array({{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}});
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(long_array, {1, 0}),
              ElementsAreArray({0, 2, 4, 6, 8, 1, 3, 5, 7, 9}));
}

TEST(UtilTest, TransposeTest3D) {
  xla::Array<int64_t> array({{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}});
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(array, {0, 1, 2}),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(array, {0, 2, 1}),
              ElementsAreArray({0, 2, 1, 3, 4, 6, 5, 7}));
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(array, {1, 0, 2}),
              ElementsAreArray({0, 1, 4, 5, 2, 3, 6, 7}));
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(array, {1, 2, 0}),
              ElementsAreArray({0, 4, 1, 5, 2, 6, 3, 7}));
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(array, {2, 0, 1}),
              ElementsAreArray({0, 2, 4, 6, 1, 3, 5, 7}));
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(array, {2, 1, 0}),
              ElementsAreArray({0, 4, 2, 6, 1, 5, 3, 7}));

  xla::Array<int64_t> wide_array(
      {{{0, 1, 2}, {3, 4, 5}}, {{6, 7, 8}, {9, 10, 11}}});
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(wide_array, {0, 1, 2}),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(wide_array, {0, 2, 1}),
              ElementsAreArray({0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11}));
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(wide_array, {2, 0, 1}),
              ElementsAreArray({0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11}));

  xla::Array<int64_t> long_array(
      {{{0, 1}, {2, 3}, {4, 5}}, {{6, 7}, {8, 9}, {10, 11}}});
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(long_array, {0, 1, 2}),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(long_array, {1, 2, 0}),
              ElementsAreArray({0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11}));
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(long_array, {2, 1, 0}),
              ElementsAreArray({0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11}));

  xla::Array<int64_t> deep_array(
      {{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}, {{8, 9}, {10, 11}}});
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(deep_array, {0, 1, 2}),
              ElementsAreArray({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(deep_array, {0, 2, 1}),
              ElementsAreArray({0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11}));
  EXPECT_THAT(xla::spmd::Transpose<int64_t>(deep_array, {2, 0, 1}),
              ElementsAreArray({0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11}));
}

TEST(UtilTest, ReshardingCostMixedMeshShapeTest) {
  // Mesh shapes are {1, 4, 2} and {1, 8, 1} (do not support {1, 1, 8} for now).
  std::vector<int64_t> devices = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<double> mesh_alpa_beta = {1, 1, 1};

  // [4, 2] -> [8, 1] => some resharding cost
  double cost = xla::spmd::ReshardingCostMixedMeshShape(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {1, 2}, {1, -1}, 8,
      mesh_alpa_beta, mesh_alpa_beta);
  EXPECT_GT(cost, 0);
  EXPECT_LT(cost, xla::spmd::kInfinityCost - 1);

  // [8, 1] -> [4, 2] => some resharding cost
  cost = xla::spmd::ReshardingCostMixedMeshShape(
      xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {1, -1}, {1, 2}, 8,
      mesh_alpa_beta, mesh_alpa_beta);
  EXPECT_GT(cost, 0);
  EXPECT_LT(cost, xla::spmd::kInfinityCost - 1);

  // [8, 1] -> [4, 1] => technically it should have small resharding cost, we
  // will return 0 for now.
  EXPECT_EQ(xla::spmd::ReshardingCostMixedMeshShape(
                xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {1, -1}, {1, -1},
                8, mesh_alpa_beta, mesh_alpa_beta),
            0);

  // [4, 1] -> [8, 1] => 0 cost
  EXPECT_EQ(xla::spmd::ReshardingCostMixedMeshShape(
                xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {1, -1}, {1, -1},
                8, mesh_alpa_beta, mesh_alpa_beta),
            0);

  // [2, 4] -> [8, 1] => not supported
  EXPECT_GE(xla::spmd::ReshardingCostMixedMeshShape(
                xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {2, 1}, {1, -1},
                8, mesh_alpa_beta, mesh_alpa_beta),
            xla::spmd::kInfinityCost - 1);

  // [8, 1] -> [2, 4] => not supported
  EXPECT_GE(xla::spmd::ReshardingCostMixedMeshShape(
                xla::ShapeUtil::MakeShape(xla::F32, {16, 32}), {1, -1}, {2, 1},
                8, mesh_alpa_beta, mesh_alpa_beta),
            xla::spmd::kInfinityCost - 1);
}

TEST(UtilTest, DecomposeMeshShapesTest) {
  // 2D mesh.
  EXPECT_THAT(xla::spmd::DecomposeMeshShapes({1, 8, 4}),
              ElementsAre(ElementsAre(1, 8, 1), ElementsAre(1, 8, 4)));
  EXPECT_THAT(xla::spmd::DecomposeMeshShapes({1, 4, 8}),
              ElementsAre(ElementsAre(1, 1, 8), ElementsAre(1, 4, 8)));
  EXPECT_THAT(xla::spmd::DecomposeMeshShapes({8, 1, 4}),
              ElementsAre(ElementsAre(8, 1, 1), ElementsAre(8, 1, 4)));
  EXPECT_THAT(xla::spmd::DecomposeMeshShapes({8, 4, 1}),
              ElementsAre(ElementsAre(8, 1, 1), ElementsAre(8, 4, 1)));
  EXPECT_THAT(xla::spmd::DecomposeMeshShapes({4, 1, 8}),
              ElementsAre(ElementsAre(1, 1, 8), ElementsAre(4, 1, 8)));
  EXPECT_THAT(xla::spmd::DecomposeMeshShapes({4, 8, 1}),
              ElementsAre(ElementsAre(1, 8, 1), ElementsAre(4, 8, 1)));

  // 3D mesh.
  EXPECT_THAT(xla::spmd::DecomposeMeshShapes({2, 4, 8}),
              ElementsAre(ElementsAre(1, 1, 8), ElementsAre(1, 4, 8),
                          ElementsAre(2, 4, 8)));
  EXPECT_THAT(xla::spmd::DecomposeMeshShapes({2, 8, 4}),
              ElementsAre(ElementsAre(1, 8, 1), ElementsAre(1, 8, 4),
                          ElementsAre(2, 8, 4)));
  EXPECT_THAT(xla::spmd::DecomposeMeshShapes({4, 2, 8}),
              ElementsAre(ElementsAre(1, 1, 8), ElementsAre(4, 1, 8),
                          ElementsAre(4, 2, 8)));
  EXPECT_THAT(xla::spmd::DecomposeMeshShapes({4, 8, 2}),
              ElementsAre(ElementsAre(1, 8, 1), ElementsAre(4, 8, 1),
                          ElementsAre(4, 8, 2)));
  EXPECT_THAT(xla::spmd::DecomposeMeshShapes({8, 2, 4}),
              ElementsAre(ElementsAre(8, 1, 1), ElementsAre(8, 1, 4),
                          ElementsAre(8, 2, 4)));
  EXPECT_THAT(xla::spmd::DecomposeMeshShapes({8, 4, 2}),
              ElementsAre(ElementsAre(8, 1, 1), ElementsAre(8, 4, 1),
                          ElementsAre(8, 4, 2)));
}

// TODO(pratikf): Need a few more unit tests for this function
TEST(UtilTest, ComputeIntermediateShapeTest) {
  xla::Array<int64_t> device_mesh({{0, 1}, {2, 3}, {4, 5}, {6, 7}});
  xla::HloSharding src_sharding =
      xla::HloSharding::Tile(xla::TileAssignment({4, 2}));
  xla::HloSharding dst_sharding =
      xla::HloSharding::Tile(xla::TileAssignment({1, 8}));
  xla::Shape shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {256, 256});
  auto intermediate_shape = xla::spmd::ComputeIntermediateShape(
      src_sharding, dst_sharding, shape, device_mesh);
  LOG(INFO) << intermediate_shape.ToString();
  EXPECT_EQ(intermediate_shape,
            xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {256, 4, 64}));
}

class CreateDifferentMeshShapesToTryTest : public xla::HloTestBase {
 protected:
  void TestCreateDifferentMeshShapesToTry(int64_t num_devices,
                                          int64_t num_mesh_dims,
                                          bool symmetrical_mesh,
                                          int64_t num_mesh_shapes_to_expect) {
    auto shapes = xla::spmd::CreateDifferentMeshShapesToTry(
        num_devices, num_mesh_dims, symmetrical_mesh);

    absl::flat_hash_set<std::vector<int64_t>> shapes_set(shapes.begin(),
                                                         shapes.end());
    EXPECT_EQ(shapes_set.size(), num_mesh_shapes_to_expect);
    for (auto mesh_shape : shapes) {
      EXPECT_EQ(mesh_shape.size(), num_mesh_dims);
      EXPECT_LE(xla::spmd::VectorGreaterThanOneElementCount(mesh_shape), 2);
      EXPECT_EQ(absl::c_accumulate(mesh_shape, 1,
                                   [](int64_t a, int64_t b) { return a * b; }),
                num_devices);
    }
  }
};

TEST_F(CreateDifferentMeshShapesToTryTest, Test1to3DMeshShapes) {
  // 1D Mesh
  TestCreateDifferentMeshShapesToTry(64, 1, /* symmetrical_mesh */ true, 1);
  TestCreateDifferentMeshShapesToTry(64, 1, /* symmetrical_mesh */ false, 1);

  // 2D Mesh
  TestCreateDifferentMeshShapesToTry(64, 2, /* symmetrical_mesh */ true, 4);
  TestCreateDifferentMeshShapesToTry(64, 2, /* symmetrical_mesh */ false, 7);

  // 3D Mesh
  TestCreateDifferentMeshShapesToTry(64, 3, /* symmetrical_mesh */ true, 4);
  TestCreateDifferentMeshShapesToTry(64, 3, /* symmetrical_mesh */ false, 18);
}

class UtilHloTest : public xla::HloTestBase {
 protected:
  const char* const hlo_string_alias_tuple_input_ = R"(
HloModule module, input_output_alias={ {0}: (0, {0}, may-alias), {1}: (0, {1}, may-alias), {2}: (0, {2}, may-alias), {3}: (0, {3}, may-alias)}

ENTRY %entry {
  arg_tuple.1 = (u32[], f32[32]{0}, f32[32]{0}, f32[32000]{0}) parameter(0)
  get-tuple-element.3 = f32[32000]{0} get-tuple-element(arg_tuple.1), index=3
  get-tuple-element.0 = u32[] get-tuple-element(arg_tuple.1), index=0
  get-tuple-element.2 = f32[32]{0} get-tuple-element(arg_tuple.1), index=2
  get-tuple-element.1 = f32[32]{0} get-tuple-element(arg_tuple.1), index=1

  ROOT tuple.61 = (u32[], f32[32]{0}, f32[32]{0}, f32[32000]{0}) tuple(get-tuple-element.0, get-tuple-element.1, get-tuple-element.2, get-tuple-element.3)
}
)";
  const char* const hlo_string_alias_not_tuple_input_ = R"(
HloModule module, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias), {2}: (2, {}, may-alias), {3}: (3, {}, may-alias)}

ENTRY %entry {
  param.0 = u32[] parameter(0)
  param.1 = f32[32]{0} parameter(1)
  param.2 = f32[32]{0} parameter(2)
  param.3 = f32[32000]{0} parameter(3)
  ROOT tuple.61 = (u32[], f32[32]{0}, f32[32]{0}, f32[32000]{0}) tuple(param.0, param.1, param.2, param.3)
}
)";
  xla::spmd::StrategyMap CreateDummyStrategyMap(xla::HloModule* module) {
    xla::spmd::StrategyMap map;
    xla::spmd::NodeIdx node_idx = 0;
    for (const xla::HloComputation* computation : module->computations()) {
      for (const xla::HloInstruction* instruction :
           computation->instructions()) {
        auto testing_vector = std::make_unique<xla::spmd::StrategyVector>();
        testing_vector->childs.reserve(
            instruction->shape().tuple_shapes_size());
        if (instruction->shape().IsTuple()) {
          testing_vector->is_tuple = true;
          for (size_t i = 0; i < instruction->shape().tuple_shapes_size();
               i++) {
            auto testing_child_vector =
                std::make_unique<xla::spmd::StrategyVector>();
            testing_child_vector->node_idx = node_idx++;
            testing_child_vector->is_tuple = false;
            testing_vector->childs.push_back(std::move(testing_child_vector));
          }
        } else {
          testing_vector->is_tuple = false;
          testing_vector->node_idx = node_idx++;
        }
        map[instruction] = std::move(testing_vector);
      }
    }
    return map;
  }

  const xla::HloInstruction* GetOnlyUserOf(const xla::HloInstruction* inst) {
    auto users = inst->users();
    EXPECT_EQ(users.size(), 1);
    auto user = users[0];
    EXPECT_NE(user, nullptr);
    return user;
  }

  void CheckOnlyUserOf(const xla::HloInstruction* inst,
                       const xla::HloInstruction* expected_user) {
    auto users = inst->users();
    ASSERT_EQ(users.size(), 1);
    auto user = users[0];
    ASSERT_NE(user, nullptr);
    EXPECT_EQ(user, expected_user);
  }
};

TEST_F(UtilHloTest, BuildAliasMap) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_string_alias_not_tuple_input_));
  xla::spmd::AliasMap alias_map = xla::spmd::BuildAliasMap(module.get());
  CHECK_EQ(alias_map.size(), 4);
}

TEST_F(UtilHloTest, BuildAliasSet) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           hlo_string_alias_not_tuple_input_));
  xla::spmd::StrategyMap map = CreateDummyStrategyMap(module.get());
  xla::spmd::AliasSet alias_set = xla::spmd::BuildAliasSet(module.get(), map);
  EXPECT_EQ(alias_set.size(), 4);
  EXPECT_THAT(alias_set, testing::UnorderedElementsAre(
                             testing::Pair(0, 4), testing::Pair(1, 5),
                             testing::Pair(2, 6), testing::Pair(3, 7)));
}

TEST_F(UtilHloTest, BuildAliasMapTupleParameter) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo_string_alias_tuple_input_));
  xla::spmd::AliasMap alias_map = xla::spmd::BuildAliasMap(module.get());
  CHECK_EQ(alias_map.size(), 4);
  for (auto pair : alias_map) {
    CHECK_EQ(pair.first, pair.second);
  }
}

TEST_F(UtilHloTest, BuildAliasSetTupleParameter) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, ParseAndReturnVerifiedModule(hlo_string_alias_tuple_input_));
  xla::spmd::StrategyMap map = CreateDummyStrategyMap(module.get());
  xla::spmd::AliasSet alias_set = xla::spmd::BuildAliasSet(module.get(), map);
  CHECK_EQ(alias_set.size(), 4);
  EXPECT_THAT(alias_set, testing::UnorderedElementsAre(
                             testing::Pair(0, 8), testing::Pair(1, 9),
                             testing::Pair(2, 10), testing::Pair(3, 11)));
}

TEST_F(UtilHloTest, AdjustShardingsWithPartialMeshShape) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %entry {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[4,2,1]0,1,2,3,4,5,6,7}
  %param1 = f32[64,32]{0,1} parameter(1), sharding={replicated}
  %dot = f32[4,256,32]{2,1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[64,32]{0,1} %param1), lhs_contracting_dims={2}, rhs_contracting_dims={0}, sharding={devices=[4,2,1]0,1,2,3,4,5,6,7}
  ROOT %copy = f32[4,256,32]{2,1,0} copy(%dot)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(xla::spmd::AdjustShardingsWithPartialMeshShape(
      GetInstructionVector(module.get()), {1, 4, 2}, 8));
  EXPECT_TRUE(xla::spmd::AdjustShardingsWithPartialMeshShape(
      GetInstructionVector(module.get()), {1, 4, 1}, 4));
  auto* param0 = FindInstruction(module.get(), "param0");
  EXPECT_THAT(param0, op::Sharding("{devices=[4,1,1]0,1,2,3}"));
  auto* dot = FindInstruction(module.get(), "dot");
  EXPECT_THAT(dot, op::Sharding("{devices=[4,1,1]0,1,2,3}"));
}

TEST_F(UtilHloTest, AdjustShardingsWithPartialMeshShapeSymmetrical) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %entry {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[2,2,1]0,1,2,3}
  %param1 = f32[64,32]{0,1} parameter(1), sharding={replicated}
  %dot = f32[4,256,32]{2,1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[64,32]{0,1} %param1), lhs_contracting_dims={2}, rhs_contracting_dims={0}, sharding={devices=[2,2,1]0,1,2,3}
  ROOT %copy = f32[4,256,32]{2,1,0} copy(%dot)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(xla::spmd::AdjustShardingsWithPartialMeshShape(
      GetInstructionVector(module.get()), {1, 2, 2}, 4));
  EXPECT_TRUE(xla::spmd::AdjustShardingsWithPartialMeshShape(
      GetInstructionVector(module.get()), {1, 2, 1}, 2));
  auto* param0 = FindInstruction(module.get(), "param0");
  EXPECT_THAT(param0, op::Sharding("{devices=[2,1,1]0,1}"));
  auto* dot = FindInstruction(module.get(), "dot");
  EXPECT_THAT(dot, op::Sharding("{devices=[2,1,1]0,1}"));
}

TEST_F(UtilHloTest, AdjustShardingsWithPartialMeshShapePartialSharding) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %entry {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[4,1,1,2]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %param1 = f32[64,32]{0,1} parameter(1), sharding={devices=[2,1,4]0,1,2,3,4,5,6,7 last_tile_dim_replicate}
  %dot = f32[4,256,32]{2,1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[64,32]{0,1} %param1), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  ROOT %copy = f32[4,256,32]{2,1,0} copy(%dot)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(xla::spmd::AdjustShardingsWithPartialMeshShape(
      GetInstructionVector(module.get()), {1, 4, 2}, 8));
  EXPECT_TRUE(xla::spmd::AdjustShardingsWithPartialMeshShape(
      GetInstructionVector(module.get()), {1, 4, 1}, 4));
  auto* param0 = FindInstruction(module.get(), "param0");
  EXPECT_THAT(param0, op::Sharding("{devices=[4,1,1]0,1,2,3}"));
  auto* param1 = FindInstruction(module.get(), "param1");
  EXPECT_THAT(param1, op::Sharding("{replicated}"));
}

TEST_F(UtilHloTest,
       AdjustShardingsWithPartialMeshShapePartialShardingSymmetrical) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %entry {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[2,1,1,2]0,1,2,3 last_tile_dim_replicate}
  %param1 = f32[64,32]{0,1} parameter(1), sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
  %dot = f32[4,256,32]{2,1,0} dot(f32[4,256,64]{2,1,0} %param0, f32[64,32]{0,1} %param1), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  ROOT %copy = f32[4,256,32]{2,1,0} copy(%dot)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(xla::spmd::AdjustShardingsWithPartialMeshShape(
      GetInstructionVector(module.get()), {1, 2, 2}, 4));
  EXPECT_TRUE(xla::spmd::AdjustShardingsWithPartialMeshShape(
      GetInstructionVector(module.get()), {1, 2, 1}, 2));
  auto* param0 = FindInstruction(module.get(), "param0");
  EXPECT_THAT(param0, op::Sharding("{replicated}"));
  auto* param1 = FindInstruction(module.get(), "param1");
  EXPECT_THAT(param1, op::Sharding("{replicated}"));
}

TEST_F(UtilHloTest, AdjustShardingsWithPartialMeshShapeTuple) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %entry {
  %param0 = f32[4,256,64]{2,1,0} parameter(0), sharding={devices=[4,2,1]0,1,2,3,4,5,6,7}
  %param1 = f32[64,32]{0,1} parameter(1), sharding={devices=[4,2]0,1,2,3,4,5,6,7}
  ROOT %tuple = (f32[4,256,64]{2,1,0}, f32[64,32]{0,1}) tuple(param0, param1), sharding={{devices=[4,2,1]0,1,2,3,4,5,6,7},{devices=[4,2]0,1,2,3,4,5,6,7}}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_FALSE(xla::spmd::AdjustShardingsWithPartialMeshShape(
      GetInstructionVector(module.get()), {1, 4, 2}, 8));
  EXPECT_TRUE(xla::spmd::AdjustShardingsWithPartialMeshShape(
      GetInstructionVector(module.get()), {1, 4, 1}, 4));
  auto* param0 = FindInstruction(module.get(), "param0");
  EXPECT_THAT(param0, op::Sharding("{devices=[4,1,1]0,1,2,3}"));
  auto* param1 = FindInstruction(module.get(), "param1");
  EXPECT_THAT(param1, op::Sharding("{devices=[4,1]0,1,2,3}"));
  auto* tuple = FindInstruction(module.get(), "tuple");
  EXPECT_THAT(
      tuple, op::Sharding("{{devices=[4,1,1]0,1,2,3},{devices=[4,1]0,1,2,3}}"));
}

TEST_F(UtilHloTest,
       FixMixedMeshShapeReshardingGetTupleElementChangedShardingDims) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %entry {
  %param0 = (f32[256,256]{0,1}, f32[256,256]{1,0}) parameter(0), sharding={{devices=[4,2]0,1,2,3,4,5,6,7}, {devices=[4,2]0,1,2,3,4,5,6,7}}
  %gte = f32[256,256]{1,0} get-tuple-element(param0), index=1, sharding={devices=[1,8]0,1,2,3,4,5,6,7}
  ROOT %cosine = f32[256,256]{1,0} cosine(%gte)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  xla::Array<int64_t> device_mesh({{0, 1}, {2, 3}, {4, 5}, {6, 7}});
  auto gte_inst = FindInstruction(module.get(), "gte");
  auto cosine_inst = FindInstruction(module.get(), "cosine");
  absl::flat_hash_map<std::string, std::vector<xla::HloSharding>>
      preserve_shardings;
  xla::spmd::FixMixedMeshShapeReshardingGetTupleElement(
      gte_inst, gte_inst->sharding(), device_mesh, &preserve_shardings);

  auto src_inter = GetOnlyUserOf(gte_inst);
  auto dst_inter = GetOnlyUserOf(src_inter);
  auto final_reshape = GetOnlyUserOf(dst_inter);
  CheckOnlyUserOf(final_reshape, cosine_inst);

  EXPECT_THAT(gte_inst, op::Sharding("{devices=[4,2]0,1,2,3,4,5,6,7}"));
  EXPECT_THAT(src_inter, op::Sharding("{devices=[4,2,1]0,1,2,3,4,5,6,7}"));
  EXPECT_THAT(dst_inter, op::Sharding("{devices=[1,4,2]0,1,2,3,4,5,6,7}"));
  EXPECT_THAT(final_reshape, op::Sharding("{devices=[1,8]0,1,2,3,4,5,6,7}"));
}

TEST_F(UtilHloTest,
       FixMixedMeshShapeReshardingGetTupleElementSameShardingDims) {
  const char* const hlo_string = R"(
HloModule module
ENTRY %entry {
  %param0 = (f32[256,256]{0,1}, f32[256,256]{1,0}) parameter(0), sharding={{devices=[4,4]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, {devices=[4,4]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}
  %gte = f32[256,256]{1,0} get-tuple-element(param0), index=1, sharding={devices=[2,8]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
  ROOT %cosine = f32[256,256]{1,0} cosine(%gte)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  xla::Array<int64_t> device_mesh(
      {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}, {12, 13}, {14, 15}});
  auto gte_inst = FindInstruction(module.get(), "gte");
  auto cosine_inst = FindInstruction(module.get(), "cosine");

  absl::flat_hash_map<std::string, std::vector<xla::HloSharding>>
      preserve_shardings;
  xla::spmd::FixMixedMeshShapeReshardingGetTupleElement(
      gte_inst, gte_inst->sharding(), device_mesh, &preserve_shardings);

  auto reshape_inst = GetOnlyUserOf(gte_inst);
  CheckOnlyUserOf(reshape_inst, cosine_inst);

  EXPECT_THAT(
      gte_inst,
      op::Sharding("{devices=[4,4]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"));
  EXPECT_THAT(
      reshape_inst,
      op::Sharding("{devices=[2,8]0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"));
}

TEST_F(UtilHloTest, ComputeInstructionExecutionCountsWhile) {
  const char* const hlo_string = R"(
HloModule module

%cond_inner {
  %vars.cond = (u32[], token[], token[]) parameter(0)
  %count.cond = u32[] get-tuple-element(%vars.cond), index=0
  %limit = u32[] constant(2)
  ROOT %lt = pred[] compare(%count.cond, %limit), direction=LT
}

%body_inner {
  %param = (u32[], token[], token[]) parameter(0)
  %count = u32[] get-tuple-element(%param), index=0
  %token1 = token[] get-tuple-element(%param), index=1
  %token2 = token[] get-tuple-element(%param), index=2
  %token3 = token[] after-all(u32[] %count, token[] %token1, token[] %token2)
  ROOT %tuple = (u32[], token[], token[]) tuple(%count, %token1, %token3)
}

%cond_outer {
  %vars.cond = (u32[], token[], token[]) parameter(0)
  %count.cond = u32[] get-tuple-element(%vars.cond), index=0
  %limit = u32[] constant(2)
  ROOT %lt = pred[] compare(%count.cond, %limit), direction=LT
}

%body_outer {
  %param = (u32[], token[], token[]) parameter(0)
  %count = u32[] get-tuple-element(%param), index=0
  %token1 = token[] get-tuple-element(%param), index=1
  %token2 = token[] get-tuple-element(%param), index=2
  %token3 = token[] after-all(u32[] %count, token[] %token1, token[] %token2)

  %inner_init = (u32[], token[], token[]) tuple(%count, %token1, %token3)
  ROOT %while = (u32[], token[], token[]) while(%inner_init), body=%body_inner, condition=%cond_inner
}

ENTRY %entry {
  %token1 = token[] after-all()
  %token2 = token[] after-all()
  %zero = u32[] constant(0)
  %init = (u32[], token[], token[]) tuple(%zero, %token1, %token2)
  %while = (u32[], token[], token[]) while(%init), body=%body_outer, condition=%cond_outer
  ROOT %result = token[] get-tuple-element(%while), index=2
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  int64_t loop_iteration_count_estimate = 100;
  auto execution_counts = xla::spmd::ComputeInstructionExecutionCounts(
      module.get(), loop_iteration_count_estimate);

  auto entry = FindComputation(module.get(), "entry");
  auto outer_while_body = FindComputation(module.get(), "body_outer");
  auto outer_while_cond = FindComputation(module.get(), "cond_outer");
  auto inner_while_body = FindComputation(module.get(), "body_inner");
  auto inner_while_cond = FindComputation(module.get(), "cond_inner");

  auto verify_execution_counts = [&](const xla::HloComputation* computation,
                                     int64_t execution_count) {
    for (auto instruction : computation->instructions()) {
      CHECK(execution_counts.count(instruction))
          << "No execution count for instruction " << instruction->ToString();
      CHECK_EQ(execution_counts.at(instruction), execution_count)
          << instruction->ToString();
    }
  };

  verify_execution_counts(entry, 1);
  verify_execution_counts(outer_while_cond, loop_iteration_count_estimate);
  verify_execution_counts(outer_while_body, loop_iteration_count_estimate);
  verify_execution_counts(inner_while_cond, loop_iteration_count_estimate *
                                                loop_iteration_count_estimate);
  verify_execution_counts(inner_while_body, loop_iteration_count_estimate *
                                                loop_iteration_count_estimate);
}

TEST_F(UtilHloTest, ComputeInstructionExecutionCountsConditional) {
  const char* const hlo_string = R"(
HloModule conditional

%Negate (x: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  ROOT %negate = f32[] negate(f32[] %x)
}

%Identity (y: f32[]) -> f32[] {
  %y = f32[] parameter(0)
  ROOT %copy = f32[] copy(f32[] %y)
}

%Floor (z: f32[]) -> f32[] {
  %z = f32[] parameter(0)
  ROOT %floor = f32[] floor(f32[] %z)
}

ENTRY %Parameters1.v4 () -> f32[] {
  %constant = s32[] constant(1)
  %constant.1 = f32[] constant(56)
  %constant.2 = f32[] constant(12)
  %constant.3 = f32[] constant(13)
  ROOT %conditional = f32[] conditional(s32[] %constant, f32[] %constant.1, f32[] %constant.2, f32[] %constant.3), branch_computations={%Negate, %Identity, %Floor}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  int64_t loop_iteration_count_estimate = 100;
  auto execution_counts = xla::spmd::ComputeInstructionExecutionCounts(
      module.get(), loop_iteration_count_estimate);

  auto negate_body = FindComputation(module.get(), "Negate");
  auto identity_body = FindComputation(module.get(), "Identity");
  auto floor_body = FindComputation(module.get(), "Floor");
  auto entry_body = FindComputation(module.get(), "Parameters1.v4");

  auto verify_execution_counts = [&](const xla::HloComputation* computation,
                                     int64_t execution_count) {
    for (auto instruction : computation->instructions()) {
      CHECK(execution_counts.count(instruction))
          << "No execution count for instruction " << instruction->ToString();
      CHECK_EQ(execution_counts.at(instruction), execution_count)
          << instruction->ToString();
    }
  };

  verify_execution_counts(entry_body, 1);
  verify_execution_counts(negate_body, 1);
  verify_execution_counts(identity_body, 1);
  verify_execution_counts(floor_body, 1);
}

// Partially inspired by memory_space_assignment_test.cc (NegateChain)
TEST_F(UtilHloTest, TestBuildCrosscutMap) {
  xla::HloComputation::Builder builder(TestName());
  const xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {2, 3});
  xla::HloInstruction* p0 = builder.AddInstruction(
      xla::HloInstruction::CreateParameter(0, shape, "p0"));
  xla::HloInstruction* p1 = builder.AddInstruction(
      xla::HloInstruction::CreateParameter(1, shape, "p1"));
  xla::HloInstruction* negate0 = builder.AddInstruction(
      xla::HloInstruction::CreateUnary(shape, xla::HloOpcode::kNegate, p0));
  xla::HloInstruction* negate1 =
      builder.AddInstruction(xla::HloInstruction::CreateUnary(
          shape, xla::HloOpcode::kNegate, negate0));
  xla::HloInstruction* negate2 =
      builder.AddInstruction(xla::HloInstruction::CreateUnary(
          shape, xla::HloOpcode::kNegate, negate1));
  xla::HloInstruction* negate3 =
      builder.AddInstruction(xla::HloInstruction::CreateUnary(
          shape, xla::HloOpcode::kNegate, negate2));
  xla::HloInstruction* add =
      builder.AddInstruction(xla::HloInstruction::CreateBinary(
          shape, xla::HloOpcode::kAdd, negate3, p1));
  auto module = CreateNewVerifiedModule();
  xla::HloComputation* computation =
      module->AddEntryComputation(builder.Build());
  xla::HloSchedule schedule(module.get());
  schedule.set_sequence(computation,
                        {p0, p1, negate0, negate1, negate2, negate3, add});
  TF_CHECK_OK(module->set_schedule(schedule));
  auto alias_analysis = xla::HloAliasAnalysis::Run(module.get()).value();
  std::unique_ptr<xla::HloLiveRange> hlo_live_range =
      xla::HloLiveRange::Run(module->schedule(), *alias_analysis,
                             module->entry_computation())
          .value();
  const xla::HloSharding sharding = xla::HloSharding::Manual();
  xla::spmd::ShardingStrategy default_strategy = {"default_strategy", sharding};
  xla::spmd::ShardingStrategy final_strategy = {"final_strategy", sharding};
  xla::spmd::StrategyVector p0_vector = {false, /*node_idx*/ 0, /*instr_id*/ 0};
  xla::spmd::StrategyVector p1_vector = {false, /*node_idx*/ 1, /*instr_id*/ 1};
  xla::spmd::StrategyVector negate0_vector = {
      false, /*node_idx*/ 2, /*instr_id*/ 2, {}, nullptr, {default_strategy}};
  xla::spmd::StrategyVector negate1_vector = {
      false, /*node_idx*/ 3, /*instr_id*/ 3, {}, nullptr, {default_strategy}};
  xla::spmd::StrategyVector negate2_vector = {
      false, /*node_idx*/ 4, /*instr_id*/ 4, {}, nullptr, {default_strategy}};
  // This last negation uses a different sharding strategy (in order to test
  // the crosscut key equality logic).
  xla::spmd::StrategyVector negate3_vector = {
      false, /*node_idx*/ 5, /*instr_id*/ 5, {}, nullptr, {final_strategy}};
  xla::spmd::StrategyVector add_vector = {false, /*node_idx*/ 6,
                                          /*instr_id*/ 6};
  const xla::spmd::LeafStrategies leaf_strategies = {
      &p0_vector,      &p1_vector,      &negate0_vector, &negate1_vector,
      &negate2_vector, &negate3_vector, &add_vector};

  const xla::spmd::CrosscutMap crosscut_map =
      BuildCrosscutMap(*hlo_live_range, leaf_strategies);

  const xla::spmd::CrosscutKey expected_crosscut_key = {
      xla::HloOpcode::kNegate, shape, {default_strategy}};
  const xla::spmd::CrosscutSet expected_crosscut_set = {
      {/*start*/ 2, /*node_idx*/ 2},
      {/*start*/ 3, /*node_idx*/ 3},
      {/*start*/ 4, /*node_idx*/ 4}};
  const xla::spmd::CrosscutMap expected_crosscut_map = {
      {expected_crosscut_key, expected_crosscut_set}};
  EXPECT_EQ(crosscut_map, expected_crosscut_map);
}

}  // namespace

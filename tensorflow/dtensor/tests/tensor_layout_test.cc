/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/cc/tensor_layout.h"

#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "tensorflow/core/platform/test.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;

// Simple implementation of a proto matcher comparing string representations.
// Only works as ShapeProto's textual representation is deterministic.
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const tensorflow::protobuf::Message& expected)
      : expected_(expected.SerializeAsString()) {}

  template <typename Message>
  bool MatchAndExplain(const Message& p,
                       ::testing::MatchResultListener*) const {
    return p.SerializeAsString() == expected_;
  }

  void DescribeTo(::std::ostream* os) const { *os << expected_; }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_;
  }

 private:
  const std::string expected_;
};

inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const tensorflow::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

class LayoutTest : public ::testing::Test {
 protected:
  Layout BatchLayout() {
    return Layout::FromString("sharding_specs:x,batch, mesh:|x=4,batch=8|*TPU")
        .ValueOrDie();
  }
};

TEST_F(LayoutTest, FromStringEmptyMesh) {
  Mesh mesh = Mesh::Empty();
  std::string mesh_str = mesh.ToString();
  EXPECT_EQ(mesh_str, Mesh::kEmptyMeshString);
}

TEST_F(LayoutTest, FromStringEmptyLayout) {
  Layout layout = Layout::Empty();
  std::string layout_str = layout.ToString();
  EXPECT_THAT(
      layout.ToProto(),
      EqualsProto(Layout::FromString(layout_str).ValueOrDie().ToProto()));
}

TEST_F(LayoutTest, LayoutToFromString) {
  Layout layout = BatchLayout();
  std::string layout_str = layout.ToString();
  EXPECT_THAT(
      layout.ToProto(),
      EqualsProto(Layout::FromString(layout_str).ValueOrDie().ToProto()));
}

TEST_F(LayoutTest, LayoutToFromStringNotSharded) {
  std::string layout_str = "sharding_specs:x," + string(Layout::kUnshardedDim) +
                           ", mesh:|x=1|0|0|/job:localhost/task:0/device:CPU:0";
  EXPECT_EQ(layout_str, Layout::FromString(layout_str)->ToString());
}

TEST_F(LayoutTest, LayoutToFromStringAny) {
  std::string layout_str =
      "sharding_specs:any, mesh:|x=1|0|0|/job:localhost/task:0/device:CPU:0";
  EXPECT_EQ(layout_str, Layout::FromString(layout_str)->ToString());
}

TEST_F(LayoutTest, AutoGenerateLayout) {
  std::string layout_str = "sharding_specs:x, mesh:|x=2,y=2|*CPU";
  std::string exp_layout_str =
      "sharding_specs:x, "
      "mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/"
      "job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/"
      "job:localhost/task:0/device:CPU:3";
  EXPECT_EQ(exp_layout_str, Layout::FromString(layout_str)->ToString());
}

TEST_F(LayoutTest, MeshToFromString) {
  Mesh mesh = BatchLayout().mesh();
  std::string mesh_str = mesh.ToString();
  EXPECT_THAT(mesh.ToProto(),
              EqualsProto(Mesh::FromString(mesh_str).ValueOrDie().ToProto()));
}

TEST_F(LayoutTest, GetType) {
  Mesh mesh = BatchLayout().mesh();
  EXPECT_TRUE(mesh.is_tpu_mesh());
}

TEST_F(LayoutTest, OnTPUMesh) {
  Layout layout = BatchLayout();
  EXPECT_TRUE(layout.mesh().is_tpu_mesh());
}

TEST_F(LayoutTest, NumShardsAsVector) {
  std::vector<int32> shards = {4, 8};
  EXPECT_EQ(BatchLayout().num_shards(), shards);
}

TEST_F(LayoutTest, IsReplicated) {
  EXPECT_FALSE(BatchLayout().IsFullyReplicated());
}

TEST_F(LayoutTest, LayoutDimLocations) {
  Layout layout = BatchLayout();
  absl::InlinedVector<int64, 4> offset = {1, 2};
  EXPECT_EQ(layout.device_location(10).ValueOrDie(), offset);
  offset = {2, 2};
  EXPECT_EQ(layout.device_location(18).ValueOrDie(), offset);
  offset = {3, 7};
  EXPECT_EQ(layout.device_location(31).ValueOrDie(), offset);

  EXPECT_FALSE(layout.device_location(32).ok());
  EXPECT_FALSE(layout.device_location(-1).ok());
}

TEST_F(LayoutTest, ScalarLayout) {
  Layout layout =
      Layout::FromString("sharding_specs:scalar, mesh:|x=4,y=4|*TPU")
          .ValueOrDie();
  EXPECT_EQ(layout.num_devices(), 16);
  EXPECT_TRUE(layout.mesh().is_tpu_mesh());
  EXPECT_EQ(layout.ToProto().mesh_config().mesh_dimensions(0).size(), 4);
  EXPECT_EQ(layout.rank(), 0);
}

TEST_F(LayoutTest, ParseSimpleTpuMesh) {
  Layout layout =
      Layout::FromString("sharding_specs:x, mesh:|x=4,y=4|*TPU").ValueOrDie();
  EXPECT_EQ(layout.num_devices(), 16);
  EXPECT_TRUE(layout.mesh().is_tpu_mesh());
  EXPECT_EQ(layout.ToProto().mesh_config().mesh_dimensions(0).size(), 4);
}

TEST_F(LayoutTest, ParseSimpleCpuMesh) {
  auto layout =
      Layout::FromString("sharding_specs:x,unsharded, mesh:|x=4,y=4|*CPU")
          .ValueOrDie();
  EXPECT_EQ(layout.num_devices(), 16);
  EXPECT_FALSE(layout.mesh().is_tpu_mesh());

  EXPECT_EQ(layout.ToProto().mesh_config().mesh_dimensions(0).size(), 4);
}

TEST_F(LayoutTest, ParseFailsOnRepeatedShardingSpec) {
  StatusOr<Layout> maybe_layout =
      Layout::FromString("sharding_specs:x,x, mesh:|x=1,y=2|*CPU");
  EXPECT_FALSE(maybe_layout.ok());
}

TEST_F(LayoutTest, ParseFailsOnInvalidScalarShardingSpec) {
  StatusOr<Layout> maybe_layout =
      Layout::FromString("sharding_specs:x,scalar, mesh:|x=1,y=2|*CPU");
  EXPECT_FALSE(maybe_layout.ok());
}

TEST_F(LayoutTest, ParseFailsOnShardingSpecOverNonExistentMeshDim) {
  StatusOr<Layout> maybe_layout =
      Layout::FromString("sharding_specs:x,z, mesh:|x=1,y=2|*CPU");
  EXPECT_FALSE(maybe_layout.ok());
}

TEST_F(LayoutTest, ParseFailsOnBadDeviceString) {
  auto layout =
      Layout::FromString("sharding_specs:x,unsharded, d:TPU mesh:x=4,y=4");
  EXPECT_FALSE(layout.ok()) << layout.status();
}

TEST_F(LayoutTest, ParseReplicatedLayout) {
  auto layout = Layout::FromString(
                    "sharding_specs:unsharded,unsharded, mesh:|x=4,y=4|*CPU")
                    .ValueOrDie();
  EXPECT_EQ(layout.num_devices(), 16);
  EXPECT_FALSE(layout.mesh().is_tpu_mesh());
  EXPECT_TRUE(layout.IsFullyReplicated());
  EXPECT_EQ(layout.ToProto().mesh_config().mesh_dimensions(0).size(), 4);
}

TEST_F(LayoutTest, SingleHostFullyReplicatedReducedMesh) {
  Layout replicated_layout =
      Layout::FromString(
          "sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU")
          .ValueOrDie();
  Mesh reduced_mesh = replicated_layout.ReducedMesh();
  EXPECT_EQ(reduced_mesh.size(), 1);
  EXPECT_THAT(reduced_mesh.hosts(), SizeIs(1));
}

TEST_F(LayoutTest, SingleHostFullShardedReducedMesh) {
  Layout layout = BatchLayout();
  Mesh original_mesh = layout.mesh();
  Mesh reduced_mesh = layout.ReducedMesh();
  EXPECT_EQ(original_mesh.ToString(), reduced_mesh.ToString());
  EXPECT_EQ(reduced_mesh.size(), 32);
  EXPECT_THAT(reduced_mesh.hosts(), SizeIs(1));
}

TEST_F(LayoutTest, MultiHostReplicatedReducedMesh) {
  StatusOr<Layout> layout = Layout::FromString(
      "sharding_specs:unsharded,unsharded, "
      "mesh:|x=4,y=2|0,1,2,3,4,5,6,7|4,5,6,7|"
      "/job:localhost/task:1/device:CPU:0,/job:localhost/task:1/device:CPU:1,"
      "/job:localhost/task:1/device:CPU:2,/job:localhost/task:1/device:CPU:3");

  Mesh reduced_mesh = layout->ReducedMesh();
  EXPECT_EQ(reduced_mesh.size(), 1);
  EXPECT_THAT(reduced_mesh.global_device_ids(), ElementsAre(0));
  EXPECT_THAT(reduced_mesh.local_device_ids(), IsEmpty());
  EXPECT_THAT(reduced_mesh.local_devices(), IsEmpty());
  EXPECT_THAT(reduced_mesh.hosts(), IsEmpty());
}

TEST_F(LayoutTest, MultiHostPartiallyShardedReducedMesh) {
  StatusOr<Layout> layout = Layout::FromString(
      "sharding_specs:x,unsharded, "
      "mesh:|x=4,y=2|0,1,2,3,4,5,6,7|4,5,6,7|"
      "/job:localhost/task:1/device:CPU:0,/job:localhost/task:1/device:CPU:1,"
      "/job:localhost/task:1/device:CPU:2,/job:localhost/task:1/device:CPU:3");

  Mesh reduced_mesh = layout->ReducedMesh();
  EXPECT_EQ(reduced_mesh.size(), 4);
  EXPECT_THAT(reduced_mesh.global_device_ids(), ElementsAre(0, 2, 4, 6));
  EXPECT_THAT(reduced_mesh.local_device_ids(), ElementsAre(4, 6));
  EXPECT_THAT(reduced_mesh.local_devices(),
              ElementsAre("/job:localhost/task:1/device:CPU:0",
                          "/job:localhost/task:1/device:CPU:2"));
  EXPECT_THAT(reduced_mesh.hosts(), SizeIs(1));
}

TEST_F(LayoutTest, MultiHostFullyShardedReducedMesh) {
  StatusOr<Layout> layout = Layout::FromString(
      "sharding_specs:x,y, "
      "mesh:|x=4,y=2|0,1,2,3,4,5,6,7|4,5,6,7|"
      "/job:localhost/task:1/device:CPU:0,/job:localhost/task:1/device:CPU:1,"
      "/job:localhost/task:1/device:CPU:2,/job:localhost/task:1/device:CPU:3");

  Mesh reduced_mesh = layout->ReducedMesh();
  EXPECT_EQ(reduced_mesh.size(), 8);
  EXPECT_THAT(reduced_mesh.global_device_ids(),
              ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
  EXPECT_THAT(reduced_mesh.local_device_ids(), ElementsAre(4, 5, 6, 7));
  EXPECT_THAT(reduced_mesh.local_devices(),
              ElementsAre("/job:localhost/task:1/device:CPU:0",
                          "/job:localhost/task:1/device:CPU:1",
                          "/job:localhost/task:1/device:CPU:2",
                          "/job:localhost/task:1/device:CPU:3"));
  EXPECT_EQ(reduced_mesh.hosts().size(), 1);
}

// TODO(luispazos) Decide if we want this to be the case.
TEST_F(LayoutTest, FlippedShardedMultiHostMeshes) {
  StatusOr<Layout> multi_host_layout_1 = Layout::FromString(
      "sharding_specs:x,y, "
      "mesh:|x=4,y=2|0,1,2,3,4,5,6,7|4,5,6,7|"
      "/job:localhost/task:1/device:CPU:0,/job:localhost/task:1/device:CPU:1,"
      "/job:localhost/task:1/device:CPU:2,/job:localhost/task:1/device:CPU:3");
  StatusOr<Layout> multi_host_layout_2 = Layout::FromString(
      "sharding_specs:x,y, "
      "mesh:|x=4,y=2|0,1,2,3,4,5,6,7|6,7,4,5|"
      "/job:localhost/task:1/device:CPU:2,/job:localhost/task:1/device:CPU:3,"
      "/job:localhost/task:1/device:CPU:0,/job:localhost/task:1/device:CPU:1");

  Mesh reduced_mesh_1 = multi_host_layout_1->ReducedMesh();
  Mesh reduced_mesh_2 = multi_host_layout_2->ReducedMesh();
  EXPECT_FALSE(reduced_mesh_1 == reduced_mesh_2);
}

TEST_F(LayoutTest, ShardEqualityOneDim) {
  ShardVector shard_vec1;
  Shard shard1{1};
  shard_vec1.shards.push_back(shard1);
  shard_vec1.num_shards_per_dim.push_back(1);

  ShardVector shard_vec2;
  Shard shard2{2};
  Shard shard3{3};
  shard_vec2.shards.push_back(shard1);
  shard_vec2.shards.push_back(shard2);
  shard_vec2.shards.push_back(shard3);
  shard_vec2.num_shards_per_dim.push_back(3);

  EXPECT_EQ(shard_vec1, shard_vec2);
}

TEST_F(LayoutTest, ShardEqualityOneDimOffset) {
  ShardVector shard_vec1;
  Shard shard1{3};
  shard_vec1.shards.push_back(shard1);
  shard_vec1.num_shards_per_dim.push_back(3);

  ShardVector shard_vec2;
  Shard shard2{7};
  Shard shard3{8};
  Shard shard4{9};
  shard_vec2.shards.push_back(shard2);
  shard_vec2.shards.push_back(shard3);
  shard_vec2.shards.push_back(shard4);
  shard_vec2.num_shards_per_dim.push_back(9);

  EXPECT_EQ(shard_vec1, shard_vec2);
}

TEST_F(LayoutTest, ShardEqualityTwoDims) {
  auto GenFullVector = [](std::vector<int> num_shards_per_dim) -> ShardVector {
    ShardVector shard_vec;
    shard_vec.num_shards_per_dim = num_shards_per_dim;
    for (int i = 1; i <= num_shards_per_dim[0]; ++i)
      for (int j = 1; j <= num_shards_per_dim[1]; ++j) {
        Shard shard{i, j};
        shard_vec.shards.push_back(shard);
      }
    return shard_vec;
  };
  std::vector<int> num_shards_per_dim_1{2, 4};
  ShardVector shard_vec1 = GenFullVector(num_shards_per_dim_1);

  std::vector<int> num_shards_per_dim_2{3, 3};
  ShardVector shard_vec2 = GenFullVector(num_shards_per_dim_2);
  EXPECT_EQ(shard_vec1, shard_vec2);
}

TEST_F(LayoutTest, Shards) {
  Layout layout =
      Layout::FromString("sharding_specs:x,y, mesh:|x=2,y=3|*CPU").ValueOrDie();
  ShardVector shard_vec = layout.GetShardVector();

  std::string expected_shard_vec_str =
      "shards:[(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)] num_shards_per_dim:(2,3)";
  EXPECT_EQ(shard_vec.ToString(), expected_shard_vec_str);
}

TEST_F(LayoutTest, ShardsInverted) {
  Layout layout =
      Layout::FromString("sharding_specs:y,x, mesh:|x=2,y=3|*CPU").ValueOrDie();
  ShardVector shards = layout.GetShardVector();
  std::string expected_shards =
      "shards:[(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)] num_shards_per_dim:(3,2)";
  EXPECT_EQ(shards.ToString(), expected_shards);
}

TEST_F(LayoutTest, HostShardMap) {
  Layout layout =
      Layout::FromString("sharding_specs:x,y, mesh:TPU|x=2,y=2|*TPU")
          .ValueOrDie();
  std::string host_name = layout.mesh().hosts()[0];
  auto host_map = layout.HostShardMap();

  std::string expected_shards =
      "shards:[(1,1),(1,2),(2,1),(2,2)] num_shards_per_dim:(2,2)";
  EXPECT_EQ(host_map.find(host_name)->second.ToString(), expected_shards);
}

TEST_F(LayoutTest, MultiHostMultiDeviceShards) {
  std::string host1 = "/job:localhost/task:0";
  std::string host2 = "/job:localhost/task:1";
  std::string device1 = "/device:TPU:0";
  std::string device2 = "/device:TPU:1";
  Layout layout =
      Layout::FromString(
          "sharding_specs:x,unsharded, mesh:TPU|x=4,y=1|0,1,2,3|0,1,2,3|" +
          host1 + device1 + "," + host1 + device2 + "," + host2 + device1 +
          "," + host2 + device2)
          .ValueOrDie();
  std::string expected_shard_vec =
      "shards:[(1,1),(2,1),(3,1),(4,1)] num_shards_per_dim:(4,1)";
  EXPECT_EQ(layout.GetShardVector().ToString(), expected_shard_vec);

  std::map<std::string, ShardVector> host_shard_map = layout.HostShardMap();

  std::string expected_shards_host1 =
      "shards:[(1,1),(2,1)] num_shards_per_dim:(4,1)";
  ShardVector host1_shard_vec = host_shard_map.find(host1)->second;
  EXPECT_EQ(host1_shard_vec.ToString(), expected_shards_host1);

  std::string expected_shards_host2 =
      "shards:[(3,1),(4,1)] num_shards_per_dim:(4,1)";
  ShardVector host2_shard_vec = host_shard_map.find(host2)->second;
  EXPECT_EQ(host2_shard_vec.ToString(), expected_shards_host2);
}

TEST_F(LayoutTest, MultiHostCommXYSharded) {
  std::string host_0 = "/job:localhost/task:0/";
  std::string host_1 = "/job:localhost/task:1/";

  StatusOr<Layout> send_layout =
      Layout::FromString("sharding_specs:y,x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|" +
                         host_0 + "device:CPU:0," + host_0 + "device:CPU:1," +
                         host_1 + "device:CPU:0," + host_1 + "device:CPU:1");
  StatusOr<Layout> recv_layout =
      Layout::FromString("sharding_specs:x,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|" +
                         host_0 + "device:TPU:0," + host_0 + "device:TPU:1," +
                         host_1 + "device:TPU:0," + host_1 + "device:TPU:1");

  std::vector<std::string> send_hosts = send_layout->ReducedMesh().hosts();
  std::vector<std::string> recv_hosts = recv_layout->ReducedMesh().hosts();
  EXPECT_TRUE(send_hosts == recv_hosts);
}

TEST_F(LayoutTest, MultiHostCommXSharded) {
  std::vector<std::string> hosts{"/job:localhost/task:0",
                                 "/job:localhost/task:1"};

  StatusOr<Layout> send_layout = Layout::FromString(
      "sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|" + hosts[0] +
      "/device:CPU:0," + hosts[0] + "/device:CPU:1," + hosts[1] +
      "/device:CPU:0," + hosts[1] + "/device:CPU:1");
  StatusOr<Layout> recv_layout = Layout::FromString(
      "sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|" + hosts[0] +
      "/device:TPU:0," + hosts[0] + "/device:TPU:1," + hosts[1] +
      "/device:TPU:0," + hosts[1] + "/device:TPU:1");

  std::vector<std::string> send_hosts = send_layout->ReducedMesh().hosts();
  std::vector<std::string> recv_hosts = recv_layout->ReducedMesh().hosts();
  EXPECT_TRUE(send_hosts == recv_hosts);

  std::map<std::string, ShardVector> send_host_shard_map =
      send_layout->HostShardMap();
  std::map<std::string, ShardVector> recv_host_shard_map =
      recv_layout->HostShardMap();

  // Check shards match in each host.
  for (const std::string& host : hosts) {
    ShardVector shard_vec_in_send_host = send_host_shard_map.find(host)->second;
    ShardVector shard_vec_in_recv_host = recv_host_shard_map.find(host)->second;
    EXPECT_EQ(shard_vec_in_send_host, shard_vec_in_recv_host);
  }
}

TEST_F(LayoutTest, Transposed2DLayout) {
  auto layout =
      Layout::FromString("sharding_specs:x,y, mesh:|x=2,y=2|*CPU").ValueOrDie();
  auto expected_layout =
      Layout::FromString("sharding_specs:y,x, mesh:|x=2,y=2|*CPU").ValueOrDie();
  EXPECT_EQ(Layout::Transposed2D(layout).ValueOrDie(), expected_layout);
}

TEST_F(LayoutTest, Transposed2DLayoutWithBatch) {
  auto layout = Layout::FromString(
                    "sharding_specs:b1,b2,x,y, mesh:|x=2,y=2,b1=2,b2=2|*CPU")
                    .ValueOrDie();
  auto expected_layout =
      Layout::FromString(
          "sharding_specs:b1,b2,y,x, mesh:|x=2,y=2,b1=2,b2=2|*CPU")
          .ValueOrDie();
  EXPECT_EQ(Layout::Transposed2D(layout).ValueOrDie(), expected_layout);
}

TEST_F(LayoutTest, MeshDimensionIndex) {
  auto layout =
      Layout::FromString("sharding_specs:x,y, mesh:|x=2,y=2|*CPU").ValueOrDie();
  EXPECT_EQ(layout.mesh().idx_for_dim("x").ValueOrDie(), 0);
  EXPECT_EQ(layout.mesh().idx_for_dim("y").ValueOrDie(), 1);
}

TEST_F(LayoutTest, TruncateBeginning) {
  auto layout = Layout::FromString("sharding_specs:x,y, mesh:CPU|x=2,y=2|*CPU")
                    .ValueOrDie();
  auto expected_layout =
      Layout::FromString("sharding_specs:x, mesh:CPU|x=2,y=2|*CPU")
          .ValueOrDie();
  EXPECT_EQ(layout.Truncate(/*split_point=*/1), expected_layout);
}

TEST_F(LayoutTest, TruncateEnd) {
  auto layout = Layout::FromString("sharding_specs:x,y, mesh:CPU|x=2,y=2|*CPU")
                    .ValueOrDie();
  auto expected_layout =
      Layout::FromString("sharding_specs:y, mesh:CPU|x=2,y=2|*CPU")
          .ValueOrDie();
  EXPECT_EQ(layout.Truncate(/*split_point=*/1, /*end=*/true), expected_layout);
}

TEST_F(LayoutTest, Concatenate) {
  auto layout_1 = Layout::FromString("sharding_specs:x, mesh:CPU|x=2,y=2|*CPU")
                      .ValueOrDie();
  auto layout_2 = Layout::FromString("sharding_specs:y, mesh:CPU|x=2,y=2|*CPU")
                      .ValueOrDie();
  auto expected_layout =
      Layout::FromString("sharding_specs:x,y, mesh:CPU|x=2,y=2|*CPU")
          .ValueOrDie();
  EXPECT_EQ(ConcatenateLayouts(layout_1, layout_2).ValueOrDie(),
            expected_layout);
}

TEST_F(LayoutTest, ConcatenateDifferentMesh) {
  auto layout_1 =
      Layout::FromString("sharding_specs:x, mesh:CPU|x=2|*CPU").ValueOrDie();
  auto layout_2 =
      Layout::FromString("sharding_specs:y, mesh:CPU|y=2|*CPU").ValueOrDie();
  auto layout = ConcatenateLayouts(layout_1, layout_2);
  EXPECT_FALSE(layout.ok()) << layout.status();
}

TEST_F(LayoutTest, ConcatenateSameDimension) {
  auto layout_1 = Layout::FromString("sharding_specs:x, mesh:CPU|x=2,y=2|*CPU")
                      .ValueOrDie();
  auto layout_2 = Layout::FromString("sharding_specs:x, mesh:CPU|x=2,y=2|*CPU")
                      .ValueOrDie();
  auto layout = ConcatenateLayouts(layout_1, layout_2);
  EXPECT_FALSE(layout.ok()) << layout.status();
}

TEST_F(LayoutTest, EmptyMeshDeviceType) {
  auto mesh = Mesh::Empty();
  EXPECT_EQ(mesh.device_type(), std::string());
}

TEST_F(LayoutTest, ConvertMeshDeviceType) {
  Mesh mesh = Mesh::FromString("mesh:|x=2,batch=1|*TPU").ValueOrDie();
  Mesh cpu_mesh = mesh.ToDeviceType("CPU").ValueOrDie();
  EXPECT_TRUE(cpu_mesh.is_cpu_mesh());

  std::string expected_task_name = "/job:localhost/replica:0/task:0/";
  Mesh expected_mesh =
      Mesh::FromString("mesh:|x=2,batch=1|0,1|0,1|" + expected_task_name +
                       "device:CPU:0," + expected_task_name + "device:CPU:1")
          .ValueOrDie();
  EXPECT_EQ(cpu_mesh, expected_mesh);
}

TEST_F(LayoutTest, EquivalentLayout) {
  Layout fully_sharded =
      Layout::FromString("sharding_specs:x,y, mesh:|x=2,y=1|*TPU").ValueOrDie();
  Layout x_sharded =
      Layout::FromString("sharding_specs:x,unsharded, mesh:|x=2,y=1|*TPU")
          .ValueOrDie();
  Layout y_sharded =
      Layout::FromString("sharding_specs:unsharded,y, mesh:|x=2,y=1|*TPU")
          .ValueOrDie();

  EXPECT_TRUE(fully_sharded.IsEquivalent(x_sharded));
  EXPECT_TRUE(x_sharded.IsEquivalent(fully_sharded));
  EXPECT_FALSE(fully_sharded.IsEquivalent(y_sharded));
  EXPECT_FALSE(y_sharded.IsEquivalent(fully_sharded));
}

}  // namespace
}  // namespace dtensor
}  // namespace tensorflow

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

#include "tensorflow/dtensor/cc/xla_spmd/layout_to_xla_sharding.h"

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

TEST(LayoutToXLAShardingTest, ReplicatedLayout1D) {
  std::string layout_str =
      "sharding_specs:unsharded, "
      "mesh:|x=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/"
      "task:0/device:CPU:1";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::REPLICATED, op_sharding.type());
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), IsEmpty());
  EXPECT_THAT(op_sharding.tile_assignment_devices(), IsEmpty());
}

TEST(LayoutToXLAShardingTest, ReplicatedLayout2D) {
  std::string layout_str =
      "sharding_specs:unsharded,unsharded "
      "mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/"
      "job:localhost/"
      "task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/"
      "task:0/device:CPU:3";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::REPLICATED, op_sharding.type());
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), IsEmpty());
  EXPECT_THAT(op_sharding.tile_assignment_devices(), IsEmpty());
}

TEST(LayoutToXLAShardingTest, ReplicatedLayout3D) {
  std::string layout_str =
      "sharding_specs:unsharded,unsharded,unsharded, "
      "mesh:|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/"
      "device:CPU:0,/"
      "job:localhost/"
      "task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/"
      "task:0/device:CPU:3,/job:localhost/task:0/device:CPU:4,/job:localhost/"
      "task:0/device:CPU:5,/job:localhost/task:0/device:CPU:6,/job:localhost/"
      "task:0/device:CPU:7";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::REPLICATED, op_sharding.type());
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), IsEmpty());
  EXPECT_THAT(op_sharding.tile_assignment_devices(), IsEmpty());
}

TEST(LayoutToXLAShardingTest, FullyShardedLayout1D) {
  std::string layout_str =
      "sharding_specs:x, "
      "mesh:|x=3|0,1,2|0,1,2|/job:localhost/task:0/device:CPU:0,/job:localhost/"
      "task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::OTHER, op_sharding.type());
  EXPECT_FALSE(op_sharding.replicate_on_last_tile_dim());
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), ElementsAre(3));
  EXPECT_THAT(op_sharding.tile_assignment_devices(), ElementsAre(0, 1, 2));
}

TEST(LayoutToXLAShardingTest, FullyShardedLayout2D) {
  std::string layout_str =
      "sharding_specs:x,y, "
      "mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/"
      "job:localhost/"
      "task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/"
      "task:0/device:CPU:3";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::OTHER, op_sharding.type());
  EXPECT_FALSE(op_sharding.replicate_on_last_tile_dim());
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), ElementsAre(2, 2));
  EXPECT_THAT(op_sharding.tile_assignment_devices(), ElementsAre(0, 1, 2, 3));
}

TEST(LayoutToXLAShardingTest, FullyShardedPermutedLayout2D) {
  std::string layout_str =
      "sharding_specs:y,x, "
      "mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/"
      "job:localhost/"
      "task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/"
      "task:0/device:CPU:3";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::OTHER, op_sharding.type());
  EXPECT_FALSE(op_sharding.replicate_on_last_tile_dim());
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), ElementsAre(2, 2));

  // Devices should now be ordered 'y' axis first.
  EXPECT_THAT(op_sharding.tile_assignment_devices(), ElementsAre(0, 2, 1, 3));
}

TEST(LayoutToXLAShardingTest, FullyShardedLayout3D) {
  std::string layout_str =
      "sharding_specs:x,y,z, "
      "mesh:|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/"
      "device:CPU:0,/"
      "job:localhost/"
      "task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/"
      "task:0/device:CPU:3,/job:localhost/task:0/device:CPU:4,/job:localhost/"
      "task:0/device:CPU:5,/job:localhost/task:0/device:CPU:6,/job:localhost/"
      "task:0/device:CPU:7";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::OTHER, op_sharding.type());
  EXPECT_FALSE(op_sharding.replicate_on_last_tile_dim());
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), ElementsAre(2, 2, 2));

  // Devices should now be ordered 'y' axis first.
  EXPECT_THAT(op_sharding.tile_assignment_devices(),
              ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
}

TEST(LayoutToXLAShardingTest, FullyShardedPermutedLayout3D_1) {
  std::string layout_str =
      "sharding_specs:z,x,y, "
      "mesh:|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/"
      "device:CPU:0,/"
      "job:localhost/"
      "task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/"
      "task:0/device:CPU:3,/job:localhost/task:0/device:CPU:4,/job:localhost/"
      "task:0/device:CPU:5,/job:localhost/task:0/device:CPU:6,/job:localhost/"
      "task:0/device:CPU:7";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::OTHER, op_sharding.type());
  EXPECT_FALSE(op_sharding.replicate_on_last_tile_dim());
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), ElementsAre(2, 2, 2));

  // Devices are permuted in z axis first and then x and y. It helps to manually
  // draw this to confirm it.
  EXPECT_THAT(op_sharding.tile_assignment_devices(),
              ElementsAre(0, 2, 4, 6, 1, 3, 5, 7));
}

TEST(LayoutToXLAShardingTest, FullyShardedPermutedLayout3D_2) {
  std::string layout_str =
      "sharding_specs:z,y,x, "
      "mesh:|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/"
      "device:CPU:0,/"
      "job:localhost/"
      "task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/"
      "task:0/device:CPU:3,/job:localhost/task:0/device:CPU:4,/job:localhost/"
      "task:0/device:CPU:5,/job:localhost/task:0/device:CPU:6,/job:localhost/"
      "task:0/device:CPU:7";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::OTHER, op_sharding.type());
  EXPECT_FALSE(op_sharding.replicate_on_last_tile_dim());
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), ElementsAre(2, 2, 2));

  // Devices are permuted in reverse order, it helps to draw this out manually
  // to understand this is correct.
  EXPECT_THAT(op_sharding.tile_assignment_devices(),
              ElementsAre(0, 4, 2, 6, 1, 5, 3, 7));
}

TEST(LayoutToXLAShardingTest, PartiallyShardedLayout2D) {
  std::string layout_str =
      "sharding_specs:x,unsharded, "
      "mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/"
      "job:localhost/"
      "task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/"
      "task:0/device:CPU:3";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::OTHER, op_sharding.type());
  EXPECT_TRUE(op_sharding.replicate_on_last_tile_dim());
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), ElementsAre(2, 1, 2));

  EXPECT_THAT(op_sharding.tile_assignment_devices(), ElementsAre(0, 1, 2, 3));
}

TEST(LayoutToXLAShardingTest, PartiallyShardedPermutedLayout2D) {
  std::string layout_str =
      "sharding_specs:y,unsharded, "
      "mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/"
      "job:localhost/"
      "task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/"
      "task:0/device:CPU:3";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::OTHER, op_sharding.type());
  EXPECT_TRUE(op_sharding.replicate_on_last_tile_dim());
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), ElementsAre(2, 1, 2));

  // Permuted on the Y dimension.
  EXPECT_THAT(op_sharding.tile_assignment_devices(), ElementsAre(0, 2, 1, 3));
}

TEST(LayoutToXLAShardingTest, PartiallyShardedLayout3D_1) {
  std::string layout_str =
      "sharding_specs:x,y,unsharded, "
      "mesh:|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/"
      "device:CPU:0,/"
      "job:localhost/"
      "task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/"
      "task:0/device:CPU:3,/job:localhost/task:0/device:CPU:4,/job:localhost/"
      "task:0/device:CPU:5,/job:localhost/task:0/device:CPU:6,/job:localhost/"
      "task:0/device:CPU:7";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::OTHER, op_sharding.type());
  EXPECT_TRUE(op_sharding.replicate_on_last_tile_dim());
  // Last dim is two since every replication group is size 2.
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(),
              ElementsAre(2, 2, 1, 2));

  EXPECT_THAT(op_sharding.tile_assignment_devices(),
              ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
}

TEST(LayoutToXLAShardingTest, PartiallyShardedLayout3D_2) {
  std::string layout_str =
      "sharding_specs:x,unsharded,unsharded, "
      "mesh:|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/"
      "device:CPU:0,/"
      "job:localhost/"
      "task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/"
      "task:0/device:CPU:3,/job:localhost/task:0/device:CPU:4,/job:localhost/"
      "task:0/device:CPU:5,/job:localhost/task:0/device:CPU:6,/job:localhost/"
      "task:0/device:CPU:7";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::OTHER, op_sharding.type());
  EXPECT_TRUE(op_sharding.replicate_on_last_tile_dim());
  // Last dim is four since every replication group is size 4.
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(),
              ElementsAre(2, 1, 1, 4));

  EXPECT_THAT(op_sharding.tile_assignment_devices(),
              ElementsAre(0, 1, 2, 3, 4, 5, 6, 7));
}

TEST(LayoutToXLAShardingTest, PartiallyShardedPermutedLayout3D_1) {
  std::string layout_str =
      "sharding_specs:z,y,unsharded, "
      "mesh:|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/"
      "device:CPU:0,/"
      "job:localhost/"
      "task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/"
      "task:0/device:CPU:3,/job:localhost/task:0/device:CPU:4,/job:localhost/"
      "task:0/device:CPU:5,/job:localhost/task:0/device:CPU:6,/job:localhost/"
      "task:0/device:CPU:7";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::OTHER, op_sharding.type());
  EXPECT_TRUE(op_sharding.replicate_on_last_tile_dim());
  // Last dim is two since every replication group is size 2.
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(),
              ElementsAre(2, 2, 1, 2));

  // Same permutation as 'z', 'y', 'x'.
  EXPECT_THAT(op_sharding.tile_assignment_devices(),
              ElementsAre(0, 4, 2, 6, 1, 5, 3, 7));
}

TEST(LayoutToXLAShardingTest, PartiallyShardedPermutedLayout3D_2) {
  std::string layout_str =
      "sharding_specs:y,unsharded,z, "
      "mesh:|x=2,y=2,z=2|0,1,2,3,4,5,6,7|0,1,2,3,4,5,6,7|/job:localhost/task:0/"
      "device:CPU:0,/"
      "job:localhost/"
      "task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/"
      "task:0/device:CPU:3,/job:localhost/task:0/device:CPU:4,/job:localhost/"
      "task:0/device:CPU:5,/job:localhost/task:0/device:CPU:6,/job:localhost/"
      "task:0/device:CPU:7";

  ::xla::OpSharding op_sharding =
      ConvertLayoutToXlaOpSharding(Layout::FromString(layout_str).value())
          .value();

  EXPECT_EQ(::xla::OpSharding::OTHER, op_sharding.type());
  EXPECT_TRUE(op_sharding.replicate_on_last_tile_dim());
  // Last dim is two since every replication group is size 2.
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(),
              ElementsAre(2, 1, 2, 2));

  // Same permutation as 'y', 'z'.
  EXPECT_THAT(op_sharding.tile_assignment_devices(),
              ElementsAre(0, 4, 1, 5, 2, 6, 3, 7));
}

}  // namespace
}  // namespace dtensor
}  // namespace tensorflow

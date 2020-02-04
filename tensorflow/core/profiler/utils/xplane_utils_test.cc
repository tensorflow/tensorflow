/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/xplane_utils.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

XEvent CreateEvent(int64 offset_ps, int64 duration_ps) {
  XEvent event;
  event.set_offset_ps(offset_ps);
  event.set_duration_ps(duration_ps);
  return event;
}

// Tests IsNested.
TEST(XPlaneUtilsTest, IsNestedTest) {
  XEvent event = CreateEvent(100, 100);
  XEvent parent = CreateEvent(50, 200);
  EXPECT_TRUE(IsNested(event, parent));
  // Returns false if there is no overlap.
  XEvent not_parent = CreateEvent(30, 50);
  EXPECT_FALSE(IsNested(event, not_parent));
  // Returns false if they overlap only partially.
  not_parent = CreateEvent(50, 100);
  EXPECT_FALSE(IsNested(event, not_parent));
}

TEST(XPlaneUtilsTest, RemovePlaneWithName) {
  XSpace space;
  RemovePlaneWithName(&space, "non-exist");
  EXPECT_EQ(space.planes_size(), 0);

  space.add_planes()->set_name("p1");
  space.add_planes()->set_name("p2");
  space.add_planes()->set_name("p3");
  RemovePlaneWithName(&space, "non-exist");
  EXPECT_EQ(space.planes_size(), 3);
  RemovePlaneWithName(&space, "p2");
  EXPECT_EQ(space.planes_size(), 2);
  RemovePlaneWithName(&space, "p1");
  EXPECT_EQ(space.planes_size(), 1);
  RemovePlaneWithName(&space, "p1");
  EXPECT_EQ(space.planes_size(), 1);
  RemovePlaneWithName(&space, "p3");
  EXPECT_EQ(space.planes_size(), 0);
}

TEST(XPlaneUtilsTest, RemoveEmptyPlanes) {
  XSpace space;
  RemoveEmptyPlanes(&space);
  EXPECT_EQ(space.planes_size(), 0);

  auto* plane1 = space.add_planes();
  plane1->set_name("p1");
  plane1->add_lines()->set_name("p1l1");
  plane1->add_lines()->set_name("p1l2");

  auto* plane2 = space.add_planes();
  plane2->set_name("p2");

  auto* plane3 = space.add_planes();
  plane3->set_name("p3");
  plane3->add_lines()->set_name("p3l1");

  auto* plane4 = space.add_planes();
  plane4->set_name("p4");

  RemoveEmptyPlanes(&space);
  ASSERT_EQ(space.planes_size(), 2);
  EXPECT_EQ(space.planes(0).name(), "p1");
  EXPECT_EQ(space.planes(1).name(), "p3");
}

TEST(XPlaneUtilsTest, RemoveEmptyLines) {
  XPlane plane;
  RemoveEmptyLines(&plane);
  EXPECT_EQ(plane.lines_size(), 0);

  auto* line1 = plane.add_lines();
  line1->set_name("l1");
  line1->add_events();
  line1->add_events();

  auto* line2 = plane.add_lines();
  line2->set_name("l2");

  auto* line3 = plane.add_lines();
  line3->set_name("l3");
  line3->add_events();

  auto* line4 = plane.add_lines();
  line4->set_name("l4");

  RemoveEmptyLines(&plane);
  ASSERT_EQ(plane.lines_size(), 2);
  EXPECT_EQ(plane.lines(0).name(), "l1");
  EXPECT_EQ(plane.lines(1).name(), "l3");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow

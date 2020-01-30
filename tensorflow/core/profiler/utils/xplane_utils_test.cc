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

}  // namespace
}  // namespace profiler
}  // namespace tensorflow

/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================h==================================*/

#include "xla/hlo/ir/stack_frames.h"

#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

TEST(StackFramesTest, Empty) {
  StackFrames dag;
  EXPECT_TRUE(dag.empty());
  EXPECT_EQ(dag.GetStackFrame(StackFrameId{1}).file_name, "");
}

TEST(StackFramesTest, AddAndGet) {
  StackFrames dag;
  HloStackFrame frame = {"file1.py", "func1", 10, 5, 0, 0, StackFrameId{}};
  StackFrameId id = dag.AddStackFrame(frame);

  EXPECT_FALSE(dag.empty());
  HloStackFrame got = dag.GetStackFrame(id);
  EXPECT_EQ(got.file_name, "file1.py");
  EXPECT_EQ(got.function_name, "func1");
  EXPECT_EQ(got.line, 10);
  EXPECT_EQ(got.column, 5);
  EXPECT_EQ(got.parent_frame_id, StackFrameId{});
}

TEST(StackFramesTest, Nested) {
  StackFrames dag;
  HloStackFrame frame1 = {"file1.py", "func1", 10, 0, 0, 0, StackFrameId{}};
  StackFrameId id1 = dag.AddStackFrame(frame1);

  HloStackFrame frame2 = {"file2.py", "func2", 20, 0, 0, 0, id1};
  StackFrameId id2 = dag.AddStackFrame(frame2);

  HloStackFrame got2 = dag.GetStackFrame(id2);
  EXPECT_EQ(got2.file_name, "file2.py");
  EXPECT_EQ(got2.parent_frame_id, id1);

  HloStackFrame got1 = dag.GetStackFrame(got2.parent_frame_id);
  EXPECT_EQ(got1.file_name, "file1.py");
  EXPECT_EQ(got1.parent_frame_id, StackFrameId{});
}

TEST(StackFramesTest, AddStackFrameDeduplication) {
  StackFrames dag;
  HloStackFrame frame1 = {"file1.py", "func1", 10, 20, 0, 0, StackFrameId{}};
  HloStackFrame frame2 = {"file1.py", "func1", 10, 20, 0, 0, StackFrameId{}};

  StackFrameId id1 = dag.AddStackFrame(frame1);
  StackFrameId id2 = dag.AddStackFrame(frame2);

  EXPECT_EQ(id1, id2);
  EXPECT_EQ(dag.proto().stack_frames_size(), 1);
  EXPECT_EQ(dag.proto().file_locations_size(), 1);
  EXPECT_EQ(dag.proto().file_names_size(), 1);
  EXPECT_EQ(dag.proto().function_names_size(), 1);
}

TEST(StackFramesTest, AddStackFrameDeepDeduplication) {
  StackFrames dag;
  HloStackFrame parent = {"file1.py", "func1", 10, 20, 0, 0, StackFrameId{}};
  StackFrameId parent_id = dag.AddStackFrame(parent);

  HloStackFrame child1 = {"file2.py", "func2", 30, 40, 0, 0, parent_id};
  HloStackFrame child2 = {"file2.py", "func2", 30, 40, 0, 0, parent_id};

  StackFrameId id1 = dag.AddStackFrame(child1);
  StackFrameId id2 = dag.AddStackFrame(child2);

  EXPECT_EQ(id1, id2);
  EXPECT_EQ(dag.proto().stack_frames_size(), 2);
}

TEST(StackFramesTest, IsPrefix) {
  StackFrames dag;
  HloStackFrame frameA = {"fileA.py", "funcA", 10, 0, 0, 0, StackFrameId{}};
  StackFrameId idA = dag.AddStackFrame(frameA);

  HloStackFrame frameB = {"fileB.py", "funcB", 20, 0, 0, 0, idA};
  StackFrameId idB = dag.AddStackFrame(frameB);

  HloStackFrame frameC = {"fileC.py", "funcC", 30, 0, 0, 0, idB};
  StackFrameId idC = dag.AddStackFrame(frameC);

  EXPECT_TRUE(dag.IsPrefix(StackFrameId{}, idC));
  EXPECT_TRUE(dag.IsPrefix(idA, idC));
  EXPECT_TRUE(dag.IsPrefix(idB, idC));
  EXPECT_TRUE(dag.IsPrefix(idC, idC));

  EXPECT_FALSE(dag.IsPrefix(idC, idB));
  EXPECT_FALSE(dag.IsPrefix(idB, idA));

  HloStackFrame frameD = {"fileD.py", "funcD", 40, 0, 0, 0, idA};
  StackFrameId idD = dag.AddStackFrame(frameD);
  EXPECT_FALSE(dag.IsPrefix(idB, idD));
  EXPECT_TRUE(dag.IsPrefix(idA, idD));
}

TEST(StackFramesTest, Concatenate) {
  StackFrames dag;
  HloStackFrame frameA = {"fileA.py", "funcA", 10, 0, 0, 0, StackFrameId{}};
  StackFrameId idA = dag.AddStackFrame(frameA);
  HloStackFrame frameB = {"fileB.py", "funcB", 20, 0, 0, 0, idA};
  StackFrameId idAB = dag.AddStackFrame(frameB);

  HloStackFrame frameX = {"fileX.py", "funcX", 100, 0, 0, 0, StackFrameId{}};
  StackFrameId idX = dag.AddStackFrame(frameX);
  HloStackFrame frameY = {"fileY.py", "funcY", 200, 0, 0, 0, idX};
  StackFrameId idXY = dag.AddStackFrame(frameY);

  // Concatenate [A, B] and [X, Y] should give [A, B, X, Y]
  StackFrameId idABXY = dag.Concatenate(idAB, idXY);

  HloStackFrame gotY = dag.GetStackFrame(idABXY);
  EXPECT_EQ(gotY.file_name, "fileY.py");

  HloStackFrame gotX = dag.GetStackFrame(gotY.parent_frame_id);
  EXPECT_EQ(gotX.file_name, "fileX.py");
  EXPECT_EQ(gotX.parent_frame_id, idAB);

  HloStackFrame gotB = dag.GetStackFrame(gotX.parent_frame_id);
  EXPECT_EQ(gotB.file_name, "fileB.py");
  EXPECT_EQ(gotB.parent_frame_id, idA);

  // Concatenate with 0
  EXPECT_EQ(dag.Concatenate(idAB, StackFrameId{}), idAB);
  EXPECT_EQ(dag.Concatenate(StackFrameId{}, idXY), idXY);
}

TEST(StackFramesTest, ConcatenateDeduplication) {
  StackFrames dag;
  HloStackFrame frameA = {"fileA.py", "funcA", 10, 0, 0, 0, StackFrameId{}};
  StackFrameId idA = dag.AddStackFrame(frameA);

  HloStackFrame frameX = {"fileX.py", "funcX", 100, 0, 0, 0, StackFrameId{}};
  StackFrameId idX = dag.AddStackFrame(frameX);

  StackFrameId idAX1 = dag.Concatenate(idA, idX);
  StackFrameId idAX2 = dag.Concatenate(idA, idX);
  EXPECT_EQ(idAX1, idAX2);
}

TEST(StackFramesTest, FromProto) {
  StackFrameIndexProto proto;
  proto.add_file_names("file1.py");
  proto.add_function_names("func1");
  auto* loc = proto.add_file_locations();
  loc->set_file_name_id(1);
  loc->set_function_name_id(1);
  loc->set_line(10);
  loc->set_column(5);
  auto* frame = proto.add_stack_frames();
  frame->set_file_location_id(1);
  frame->set_parent_frame_id(0);

  StackFrames dag = StackFrames::FromProto(proto).value();
  EXPECT_FALSE(dag.empty());
  HloStackFrame got = dag.GetStackFrame(StackFrameId{1});
  EXPECT_EQ(got.file_name, "file1.py");
  EXPECT_EQ(got.function_name, "func1");
  EXPECT_EQ(got.line, 10);
  EXPECT_EQ(got.column, 5);
  EXPECT_EQ(got.parent_frame_id, StackFrameId{});

  // Verify that FromProto errors on duplicates.
  proto.add_file_names("file1.py");  // Duplicate!
  EXPECT_FALSE(StackFrames::FromProto(proto).ok());
}

}  // namespace
}  // namespace xla

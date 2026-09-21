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
==============================================================================*/

#include "xla/hlo/translate/mhlo_to_hlo/stack_frame_index_builder.h"

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace mlir {
namespace {

using ::tsl::proto_testing::EqualsProto;

Location MakeFrameLoc(MLIRContext* ctx, const std::string& name,
                      const std::string& file, int line) {
  return NameLoc::get(StringAttr::get(ctx, name),
                      FileLineColLoc::get(StringAttr::get(ctx, file), line, 0));
}

std::vector<std::string> GetFrameNames(const xla::StackFrameIndexProto& proto,
                                       int frame_id) {
  std::vector<std::string> names;
  while (frame_id != StackFrameIndexBuilder::kInvalidIndex) {
    const auto& frame = proto.stack_frames(frame_id - 1);
    const auto& loc = proto.file_locations(frame.file_location_id() - 1);
    names.push_back(proto.function_names(loc.function_name_id() - 1));
    frame_id = frame.parent_frame_id();
  }
  return names;
}

TEST(StackFrameIndexBuilderTest, CallSiteLocConcatenation) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);
  Location b = MakeFrameLoc(&ctx, "B", "b.py", 2);
  Location c = MakeFrameLoc(&ctx, "C", "c.py", 3);
  Location d = MakeFrameLoc(&ctx, "D", "d.py", 4);

  Location ab = CallSiteLoc::get(a, b);
  Location cd = CallSiteLoc::get(c, d);
  Location abcd = CallSiteLoc::get(ab, cd);

  StackFrameIndexBuilder builder;
  auto result = builder.AddCallStackAndGetFirstFrameId(abcd);
  auto proto = builder.Build();

  std::vector<std::string> names = GetFrameNames(proto, result.last_frame_id);
  std::vector<std::string> expected = {"A", "B", "C", "D"};
  EXPECT_EQ(names, expected);
}

TEST(StackFrameIndexBuilderTest, DeeplyNestedCallSiteLoc) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);
  Location b = MakeFrameLoc(&ctx, "B", "b.py", 2);
  Location c = MakeFrameLoc(&ctx, "C", "c.py", 3);
  Location d = MakeFrameLoc(&ctx, "D", "d.py", 4);
  Location e = MakeFrameLoc(&ctx, "E", "e.py", 5);
  Location f = MakeFrameLoc(&ctx, "F", "f.py", 6);

  Location ab = CallSiteLoc::get(a, b);
  Location abc = CallSiteLoc::get(ab, c);
  Location ef = CallSiteLoc::get(e, f);
  Location def = CallSiteLoc::get(d, ef);
  Location abcdef = CallSiteLoc::get(abc, def);

  StackFrameIndexBuilder builder;
  auto result = builder.AddCallStackAndGetFirstFrameId(abcdef);
  auto proto = builder.Build();

  std::vector<std::string> names = GetFrameNames(proto, result.last_frame_id);
  std::vector<std::string> expected = {"A", "B", "C", "D", "E", "F"};
  EXPECT_EQ(names, expected);
}

TEST(StackFrameIndexBuilderTest, LinearChain) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);
  Location b = MakeFrameLoc(&ctx, "B", "b.py", 2);
  Location c = MakeFrameLoc(&ctx, "C", "c.py", 3);
  Location d = MakeFrameLoc(&ctx, "D", "d.py", 4);

  Location cd = CallSiteLoc::get(c, d);
  Location bcd = CallSiteLoc::get(b, cd);
  Location abcd = CallSiteLoc::get(a, bcd);

  StackFrameIndexBuilder builder;
  auto result = builder.AddCallStackAndGetFirstFrameId(abcd);
  auto proto = builder.Build();

  std::vector<std::string> names = GetFrameNames(proto, result.last_frame_id);
  std::vector<std::string> expected = {"A", "B", "C", "D"};
  EXPECT_EQ(names, expected);
}

// Ops that share a location, and locations that are callers of an indexed
// chain, resolve to the frames the first walk of the chain created.
TEST(StackFrameIndexBuilderTest, SharedRootLocation) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);
  Location b = MakeFrameLoc(&ctx, "B", "b.py", 2);
  Location c = MakeFrameLoc(&ctx, "C", "c.py", 3);
  Location d = MakeFrameLoc(&ctx, "D", "d.py", 4);

  Location cd = CallSiteLoc::get(c, d);
  Location bcd = CallSiteLoc::get(b, cd);
  Location abcd = CallSiteLoc::get(a, bcd);

  StackFrameIndexBuilder builder;
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(abcd).last_frame_id, 4);
  StackFrameIndexBuilder::AddStackFrameResult repeated =
      builder.AddCallStackAndGetFirstFrameId(abcd);
  EXPECT_EQ(repeated.last_frame_id, 4);
  EXPECT_EQ(repeated.last_frame_file, "a.py");
  EXPECT_EQ(repeated.last_frame_line, 1);
  EXPECT_EQ(repeated.last_frame_column, 0);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(cd).last_frame_id, 2);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(bcd).last_frame_id, 3);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(d).last_frame_id, 1);

  constexpr absl::string_view kExpected = R"pb(
    file_names: [ "d.py", "c.py", "b.py", "a.py" ]
    function_names: [ "D", "C", "B", "A" ]
    file_locations { file_name_id: 1 function_name_id: 1 line: 4 end_line: 4 }
    file_locations { file_name_id: 2 function_name_id: 2 line: 3 end_line: 3 }
    file_locations { file_name_id: 3 function_name_id: 3 line: 2 end_line: 2 }
    file_locations { file_name_id: 4 function_name_id: 4 line: 1 end_line: 1 }
    stack_frames { file_location_id: 1 parent_frame_id: 0 }
    stack_frames { file_location_id: 2 parent_frame_id: 1 }
    stack_frames { file_location_id: 3 parent_frame_id: 2 }
    stack_frames { file_location_id: 4 parent_frame_id: 3 }
  )pb";
  EXPECT_THAT(builder.Build(), EqualsProto(kExpected));
}

// A chain that shares its callers with an indexed chain only adds the frames
// above the shared part, in the order a walk of the whole chain would use.
TEST(StackFrameIndexBuilderTest, SharedCallerChain) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);
  Location b = MakeFrameLoc(&ctx, "B", "b.py", 2);
  Location c = MakeFrameLoc(&ctx, "C", "c.py", 3);
  Location d = MakeFrameLoc(&ctx, "D", "d.py", 4);
  Location e = MakeFrameLoc(&ctx, "E", "e.py", 5);
  Location f = MakeFrameLoc(&ctx, "F", "f.py", 6);

  Location cd = CallSiteLoc::get(c, d);
  Location bcd = CallSiteLoc::get(b, cd);
  Location abcd = CallSiteLoc::get(a, bcd);
  Location ebcd = CallSiteLoc::get(e, bcd);
  Location fcd = CallSiteLoc::get(f, cd);

  StackFrameIndexBuilder builder;
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(abcd).last_frame_id, 4);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(ebcd).last_frame_id, 5);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(fcd).last_frame_id, 6);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(ebcd).last_frame_id, 5);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(abcd).last_frame_id, 4);

  constexpr absl::string_view kExpected = R"pb(
    file_names: [ "d.py", "c.py", "b.py", "a.py", "e.py", "f.py" ]
    function_names: [ "D", "C", "B", "A", "E", "F" ]
    file_locations { file_name_id: 1 function_name_id: 1 line: 4 end_line: 4 }
    file_locations { file_name_id: 2 function_name_id: 2 line: 3 end_line: 3 }
    file_locations { file_name_id: 3 function_name_id: 3 line: 2 end_line: 2 }
    file_locations { file_name_id: 4 function_name_id: 4 line: 1 end_line: 1 }
    file_locations { file_name_id: 5 function_name_id: 5 line: 5 end_line: 5 }
    file_locations { file_name_id: 6 function_name_id: 6 line: 6 end_line: 6 }
    stack_frames { file_location_id: 1 parent_frame_id: 0 }
    stack_frames { file_location_id: 2 parent_frame_id: 1 }
    stack_frames { file_location_id: 3 parent_frame_id: 2 }
    stack_frames { file_location_id: 4 parent_frame_id: 3 }
    stack_frames { file_location_id: 5 parent_frame_id: 3 }
    stack_frames { file_location_id: 6 parent_frame_id: 2 }
  )pb";
  EXPECT_THAT(builder.Build(), EqualsProto(kExpected));
}

// A nested callee above a shared caller is added outermost first.
TEST(StackFrameIndexBuilderTest, NestedCalleeAboveSharedCaller) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);
  Location b = MakeFrameLoc(&ctx, "B", "b.py", 2);
  Location c = MakeFrameLoc(&ctx, "C", "c.py", 3);
  Location d = MakeFrameLoc(&ctx, "D", "d.py", 4);
  Location x = MakeFrameLoc(&ctx, "X", "x.py", 7);
  Location y = MakeFrameLoc(&ctx, "Y", "y.py", 8);

  Location cd = CallSiteLoc::get(c, d);
  Location bcd = CallSiteLoc::get(b, cd);
  Location abcd = CallSiteLoc::get(a, bcd);
  Location xy = CallSiteLoc::get(x, y);
  Location xybcd = CallSiteLoc::get(xy, bcd);

  StackFrameIndexBuilder builder;
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(abcd).last_frame_id, 4);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(xybcd).last_frame_id, 6);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(xybcd).last_frame_id, 6);

  constexpr absl::string_view kExpected = R"pb(
    file_names: [ "d.py", "c.py", "b.py", "a.py", "y.py", "x.py" ]
    function_names: [ "D", "C", "B", "A", "Y", "X" ]
    file_locations { file_name_id: 1 function_name_id: 1 line: 4 end_line: 4 }
    file_locations { file_name_id: 2 function_name_id: 2 line: 3 end_line: 3 }
    file_locations { file_name_id: 3 function_name_id: 3 line: 2 end_line: 2 }
    file_locations { file_name_id: 4 function_name_id: 4 line: 1 end_line: 1 }
    file_locations { file_name_id: 5 function_name_id: 5 line: 8 end_line: 8 }
    file_locations { file_name_id: 6 function_name_id: 6 line: 7 end_line: 7 }
    stack_frames { file_location_id: 1 parent_frame_id: 0 }
    stack_frames { file_location_id: 2 parent_frame_id: 1 }
    stack_frames { file_location_id: 3 parent_frame_id: 2 }
    stack_frames { file_location_id: 4 parent_frame_id: 3 }
    stack_frames { file_location_id: 5 parent_frame_id: 3 }
    stack_frames { file_location_id: 6 parent_frame_id: 5 }
  )pb";
  EXPECT_THAT(builder.Build(), EqualsProto(kExpected));
}

// One file location per (file, function, line, column); one frame per (file
// location, parent frame).
TEST(StackFrameIndexBuilderTest, SameFrameUnderDifferentCallers) {
  MLIRContext ctx;
  Location a = MakeFrameLoc(&ctx, "A", "a.py", 1);
  Location b = MakeFrameLoc(&ctx, "B", "b.py", 2);
  Location c = MakeFrameLoc(&ctx, "C", "c.py", 3);
  Location a_in_z = MakeFrameLoc(&ctx, "A", "z.py", 1);

  Location ab = CallSiteLoc::get(a, b);
  Location ac = CallSiteLoc::get(a, c);
  Location a_in_z_b = CallSiteLoc::get(a_in_z, b);

  StackFrameIndexBuilder builder;
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(ab).last_frame_id, 2);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(ac).last_frame_id, 4);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(a).last_frame_id, 5);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(a_in_z_b).last_frame_id, 6);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(ab).last_frame_id, 2);

  constexpr absl::string_view kExpected = R"pb(
    file_names: [ "b.py", "a.py", "c.py", "z.py" ]
    function_names: [ "B", "A", "C" ]
    file_locations { file_name_id: 1 function_name_id: 1 line: 2 end_line: 2 }
    file_locations { file_name_id: 2 function_name_id: 2 line: 1 end_line: 1 }
    file_locations { file_name_id: 3 function_name_id: 3 line: 3 end_line: 3 }
    file_locations { file_name_id: 4 function_name_id: 2 line: 1 end_line: 1 }
    stack_frames { file_location_id: 1 parent_frame_id: 0 }
    stack_frames { file_location_id: 2 parent_frame_id: 1 }
    stack_frames { file_location_id: 3 parent_frame_id: 0 }
    stack_frames { file_location_id: 2 parent_frame_id: 3 }
    stack_frames { file_location_id: 2 parent_frame_id: 0 }
    stack_frames { file_location_id: 4 parent_frame_id: 1 }
  )pb";
  EXPECT_THAT(builder.Build(), EqualsProto(kExpected));
}

// A location without frames is remembered as kInvalidIndex, and a repeat
// returns it without reading any frame.
TEST(StackFrameIndexBuilderTest, RootWithoutFrames) {
  MLIRContext ctx;
  Location op_name = NameLoc::get(StringAttr::get(&ctx, "op"));

  StackFrameIndexBuilder builder;
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(op_name).last_frame_id,
            StackFrameIndexBuilder::kInvalidIndex);
  EXPECT_EQ(builder.AddCallStackAndGetFirstFrameId(op_name).last_frame_id,
            StackFrameIndexBuilder::kInvalidIndex);
  EXPECT_THAT(builder.Build(), EqualsProto(""));
}

}  // namespace
}  // namespace mlir

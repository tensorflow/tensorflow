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

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/test.h"

namespace mlir {
namespace {

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

}  // namespace
}  // namespace mlir

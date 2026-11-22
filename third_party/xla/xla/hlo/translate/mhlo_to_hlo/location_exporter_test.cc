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

#include "xla/hlo/translate/mhlo_to_hlo/location_exporter.h"

#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/tsl/platform/test.h"

namespace mlir {
namespace mhlo {
namespace {

Location MakeNameLoc(MLIRContext* ctx, const std::string& name) {
  return NameLoc::get(StringAttr::get(ctx, name));
}

Location MakeFileLoc(MLIRContext* ctx) {
  return FileLineColLoc::get(StringAttr::get(ctx, "a.py"), 1, 1);
}

Location MakeFrameLoc(MLIRContext* ctx, const std::string& name) {
  return NameLoc::get(StringAttr::get(ctx, name), MakeFileLoc(ctx));
}

TEST(LocationExporterTest, CombinesFusedAndCallSiteNames) {
  MLIRContext ctx;
  Location a = MakeNameLoc(&ctx, "a");
  Location b = MakeNameLoc(&ctx, "b");
  Location c = MakeNameLoc(&ctx, "c");
  Location d = MakeNameLoc(&ctx, "d");
  Location file = MakeFileLoc(&ctx);
  Location frame = MakeFrameLoc(&ctx, "frame");
  // A CallSiteLoc without any op name.
  Location unnamed = CallSiteLoc::get(/*callee=*/file, /*caller=*/frame);

  EXPECT_EQ(GetDebugNameFromLocation(FusedLoc::get(&ctx, {a, b})), "a;b");
  EXPECT_EQ(GetDebugNameFromLocation(MakeNameLoc(&ctx, "name@function")),
            "name");
  EXPECT_EQ(
      GetDebugNameFromLocation(NameLoc::get(StringAttr::get(&ctx, "type:"), a)),
      "a");

  // The caller names come first. A source stack frame is an op name on the
  // callee side only.
  EXPECT_EQ(GetDebugNameFromLocation(CallSiteLoc::get(a, c)), "c/a");
  EXPECT_EQ(GetDebugNameFromLocation(CallSiteLoc::get(frame, c)), "c/frame");
  EXPECT_EQ(GetDebugNameFromLocation(CallSiteLoc::get(a, frame)), "a");
  EXPECT_EQ(
      GetDebugNameFromLocation(CallSiteLoc::get(a, CallSiteLoc::get(frame, c))),
      "c/a");
  EXPECT_EQ(GetDebugNameFromLocation(CallSiteLoc::get(
                FusedLoc::get(&ctx, {a, b}), FusedLoc::get(&ctx, {c, d}))),
            "c;d/a;b");

  // Locations without an op name do not add separators.
  EXPECT_EQ(GetDebugNameFromLocation(
                FusedLoc::get(&ctx, {CallSiteLoc::get(file, c), b})),
            "c;b");
  EXPECT_EQ(GetDebugNameFromLocation(
                FusedLoc::get(&ctx, {a, CallSiteLoc::get(b, frame)})),
            "a;b");
  EXPECT_EQ(GetDebugNameFromLocation(
                CallSiteLoc::get(FusedLoc::get(&ctx, {file, a}), c)),
            "c/a");
  EXPECT_EQ(GetDebugNameFromLocation(
                CallSiteLoc::get(FusedLoc::get(&ctx, {unnamed, a}), c)),
            "c/a");
  EXPECT_EQ(
      GetDebugNameFromLocation(CallSiteLoc::get(CallSiteLoc::get(a, file), c)),
      "c/a");
}

TEST(LocationExporterTest, OpNameSizeIsBounded) {
  MLIRContext ctx;
  // Locations are uniqued, so every CallSiteLoc below shares its callee and its
  // caller: 11 nested locations that expand to 2^11 copies of a 1000 byte name.
  const std::string name(1000, 'a');
  Location loc = MakeNameLoc(&ctx, name);
  for (int i = 0; i < 11; ++i) {
    loc = CallSiteLoc::get(loc, loc);
  }

  // Only whole names are kept: 4 of them fit in 4096 bytes.
  EXPECT_EQ(GetDebugNameFromLocation(loc),
            absl::StrJoin(std::vector<std::string>(4, name), "/"));
}

TEST(LocationExporterTest, DeepCallSiteChainIsTruncated) {
  MLIRContext ctx;
  // An op inlined through 10000 nested calls that are all named "s".
  Location loc = MakeNameLoc(&ctx, "op");
  for (int i = 0; i < 10000; ++i) {
    loc = CallSiteLoc::get(loc, MakeNameLoc(&ctx, "s"));
  }

  // Each call takes two of the 4096 visited locations (the CallSiteLoc and its
  // caller), so the names of the 2048 outermost calls are kept.
  EXPECT_EQ(GetDebugNameFromLocation(loc),
            absl::StrJoin(std::vector<std::string>(2048, "s"), "/"));
}

}  // namespace
}  // namespace mhlo
}  // namespace mlir

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

#include "xla/hlo/translate/hlo_to_mhlo/stack_location_utils.h"

#include <optional>
#include <ostream>
#include <string>

#include "absl/strings/string_view.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/platform/test.h"

namespace mlir {

// Lets gtest print locations instead of their raw pointer bytes.
void PrintTo(Location location, std::ostream* os) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  location.print(stream);
  *os << text;
}

namespace hlo {
namespace {

constexpr xla::StackFrameId kRoot{};

xla::StackFrameId AddFrame(xla::HloModule& module, absl::string_view file_name,
                           absl::string_view function_name, int line,
                           int column, xla::StackFrameId parent) {
  xla::HloStackFrame frame;
  frame.file_name = file_name;
  frame.function_name = function_name;
  frame.line = line;
  frame.column = column;
  frame.parent_frame_id = parent;
  return module.mutable_stack_frames().AddStackFrame(frame);
}

// The location of a single frame, as the importer has always built it.
Location ExpectedFrameLoc(Builder& builder, absl::string_view file_name,
                          absl::string_view function_name, int line,
                          int column) {
  return NameLoc::get(
      builder.getStringAttr(function_name),
      FileLineColLoc::get(builder.getStringAttr(file_name), line, column));
}

class StackLocationUtilsTest : public ::testing::Test {
 protected:
  // The memo is sized from the module's frame table, so it is created on
  // first use, after the test added its frames.
  StackFrameLocationCache& Cache() {
    if (!cache_.has_value()) {
      cache_.emplace(module_);
    }
    return *cache_;
  }

  Location FrameLocation(int frame_id) {
    return GetLocationFromFrameIndex(frame_id, builder_, &module_, &Cache());
  }

  // The memoized location of `id` for positive checks; a missing entry reads
  // as unknown. Absence is asserted through Cache().Lookup directly.
  Location Cached(xla::StackFrameId id) {
    if (LocationAttr cached = Cache().Lookup(id)) {
      return cached;
    }
    return UnknownLoc::get(&context_);
  }

  MLIRContext context_;
  Builder builder_{&context_};
  xla::HloModule module_{"module", xla::HloModuleConfig()};
  std::optional<StackFrameLocationCache> cache_;
};

TEST_F(StackLocationUtilsTest, ChainsSharingFramesGetIdenticalLocations) {
  // Two chains of depth three that share the two outermost frames.
  xla::StackFrameId a = AddFrame(module_, "a.py", "A", 1, 2, kRoot);
  xla::StackFrameId b = AddFrame(module_, "b.py", "B", 3, 4, a);
  xla::StackFrameId c = AddFrame(module_, "c.py", "C", 5, 6, b);
  xla::StackFrameId d = AddFrame(module_, "d.py", "D", 7, 8, b);

  Location loc_a = ExpectedFrameLoc(builder_, "a.py", "A", 1, 2);
  Location loc_b = ExpectedFrameLoc(builder_, "b.py", "B", 3, 4);
  Location loc_c = ExpectedFrameLoc(builder_, "c.py", "C", 5, 6);
  Location loc_d = ExpectedFrameLoc(builder_, "d.py", "D", 7, 8);
  Location expected_b = CallSiteLoc::get(loc_b, loc_a);
  Location expected_c = CallSiteLoc::get(loc_c, {loc_b, loc_a});
  Location expected_d = CallSiteLoc::get(loc_d, {loc_b, loc_a});

  // The first query memoizes every frame of its chain and nothing else.
  Location chain_c = FrameLocation(c.value);
  EXPECT_EQ(Cached(a), loc_a);
  EXPECT_EQ(Cached(b), expected_b);
  EXPECT_EQ(Cached(c), expected_c);
  EXPECT_FALSE(Cache().Lookup(d));

  // The second chain ends in the memoized b and gets memoized as well.
  Location chain_d = FrameLocation(d.value);
  EXPECT_EQ(Cached(d), expected_d);

  // Locations are uniqued, so equality is pointer identity.
  EXPECT_EQ(chain_c, expected_c);
  EXPECT_EQ(chain_d, expected_d);
  EXPECT_EQ(llvm::cast<CallSiteLoc>(chain_d).getCallee(), loc_d);
  EXPECT_EQ(llvm::cast<CallSiteLoc>(chain_d).getCaller(), expected_b);

  // Queries answered from the memo at every depth.
  EXPECT_EQ(FrameLocation(a.value), loc_a);
  EXPECT_EQ(FrameLocation(b.value), expected_b);
  EXPECT_EQ(FrameLocation(c.value), expected_c);
  EXPECT_EQ(FrameLocation(d.value), expected_d);

  // The uncached path agrees.
  for (xla::StackFrameId id : {a, b, c, d}) {
    EXPECT_EQ(GetLocationFromFrameIndex(id.value, builder_, &module_),
              FrameLocation(id.value));
  }
}

TEST_F(StackLocationUtilsTest, MemoizedParentChainIsReused) {
  xla::StackFrameId a = AddFrame(module_, "a.py", "A", 1, 2, kRoot);
  xla::StackFrameId b = AddFrame(module_, "b.py", "B", 3, 4, a);
  xla::StackFrameId d = AddFrame(module_, "d.py", "D", 7, 8, b);

  // Whatever the memo holds for b is used as the chain above d.
  Location sentinel = NameLoc::get(builder_.getStringAttr("sentinel"));
  Cache().Insert(b, sentinel);

  Location loc_d = ExpectedFrameLoc(builder_, "d.py", "D", 7, 8);
  Location expected_d = CallSiteLoc::get(loc_d, sentinel);
  EXPECT_EQ(FrameLocation(d.value), expected_d);
  EXPECT_EQ(Cached(d), expected_d);
  // The walk stopped at b, so a was never visited.
  EXPECT_FALSE(Cache().Lookup(a));
}

TEST_F(StackLocationUtilsTest, EmptyFrameEndsTheChain) {
  // a <- empty <- b <- c: the frames above the empty frame are not part of
  // the chains of b and c, even once a is memoized, and the empty frame is
  // never memoized.
  xla::StackFrameId a = AddFrame(module_, "a.py", "A", 1, 2, kRoot);
  xla::StackFrameId empty = AddFrame(module_, "", "", 0, 0, a);
  xla::StackFrameId b = AddFrame(module_, "b.py", "B", 3, 4, empty);
  xla::StackFrameId c = AddFrame(module_, "c.py", "C", 5, 6, b);

  Location unknown = UnknownLoc::get(&context_);
  Location loc_a = ExpectedFrameLoc(builder_, "a.py", "A", 1, 2);
  Location loc_b = ExpectedFrameLoc(builder_, "b.py", "B", 3, 4);
  Location loc_c = ExpectedFrameLoc(builder_, "c.py", "C", 5, 6);
  Location expected_c = CallSiteLoc::get(loc_c, loc_b);

  EXPECT_EQ(FrameLocation(a.value), loc_a);
  EXPECT_EQ(Cached(a), loc_a);
  EXPECT_EQ(FrameLocation(empty.value), unknown);
  EXPECT_FALSE(Cache().Lookup(empty));
  EXPECT_EQ(FrameLocation(c.value), expected_c);
  EXPECT_EQ(FrameLocation(b.value), loc_b);
  EXPECT_FALSE(Cache().Lookup(empty));
  EXPECT_EQ(FrameLocation(empty.value), unknown);
  EXPECT_EQ(FrameLocation(c.value), expected_c);
}

TEST_F(StackLocationUtilsTest, MissingFramesGiveUnknownLocation) {
  Location unknown = UnknownLoc::get(&context_);
  xla::StackFrameId a = AddFrame(module_, "a.py", "A", 1, 2, kRoot);
  // A frame whose parent id is past the table.
  xla::StackFrameId past_table{a.value + 5};
  xla::StackFrameId orphan = AddFrame(module_, "o.py", "O", 9, 1, past_table);

  EXPECT_EQ(FrameLocation(0), unknown);
  EXPECT_EQ(FrameLocation(past_table.value), unknown);
  EXPECT_EQ(FrameLocation(a.value),
            ExpectedFrameLoc(builder_, "a.py", "A", 1, 2));
  EXPECT_EQ(FrameLocation(orphan.value),
            ExpectedFrameLoc(builder_, "o.py", "O", 9, 1));

  // Missing frames are never memoized, present ones are.
  EXPECT_FALSE(Cache().Lookup(xla::StackFrameId{0}));
  EXPECT_FALSE(Cache().Lookup(past_table));
  EXPECT_TRUE(Cache().Lookup(a));
  EXPECT_TRUE(Cache().Lookup(orphan));
}

}  // namespace
}  // namespace hlo
}  // namespace mlir

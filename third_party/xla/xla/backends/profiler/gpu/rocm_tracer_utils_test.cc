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

#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"

#include <cstdint>
#include <string>

#include <gtest/gtest.h>

namespace xla {
namespace profiler {
namespace {

// rocm_tracer_utils has no ROCm dependency, so these tests run without a GPU.

TEST(AnnotationMapTest, HonoursConfiguredSize) {
  AnnotationMap map;
  map.Reset(2);
  map.Add(1, "first");
  map.Add(2, "second");
  map.Add(3, "third");

  EXPECT_EQ(map.LookUp(1), "first");
  EXPECT_EQ(map.LookUp(2), "second");
  // Over capacity: silently not retained. Add() is on the callback hot path
  // and deliberately does not log per-drop.
  EXPECT_TRUE(map.LookUp(3).empty());
}

TEST(AnnotationMapTest, FullStringSetAlsoStopsCorrelatingKnownStrings) {
  // Once the string set is full, later events are not correlated even when
  // their annotation is already stored. CUPTI's AnnotationMap does the same.
  AnnotationMap map;
  map.Reset(1);
  for (uint32_t i = 0; i < 100; ++i) {
    map.Add(i, "same_annotation");
  }
  EXPECT_EQ(map.LookUp(0), "same_annotation");
  EXPECT_TRUE(map.LookUp(99).empty());
}

TEST(AnnotationMapTest, ResetRaisesTheCap) {
  AnnotationMap map;
  map.Reset(1);
  map.Add(1, "first");
  map.Add(2, "second");
  ASSERT_TRUE(map.LookUp(2).empty()) << "precondition: cap of 1 is in force";

  map.Reset(3);
  map.Add(1, "first");
  map.Add(2, "second");
  map.Add(3, "third");

  EXPECT_EQ(map.LookUp(1), "first");
  EXPECT_EQ(map.LookUp(2), "second");
  EXPECT_EQ(map.LookUp(3), "third");
}

TEST(AnnotationMapTest, ResetToZeroRetainsNothing) {
  // 0 means "retain no annotation strings", not "unlimited".
  AnnotationMap map;
  map.Reset(8);
  map.Reset(0);
  map.Add(1, "first");

  EXPECT_TRUE(map.LookUp(1).empty());
}

TEST(AnnotationMapTest, ResetClearsEntriesAndLowersTheCap) {
  // What RocmTracer::Enable does at the start of each session: entries from
  // the old session are dropped and the new, here lower, cap applies.
  AnnotationMap map;
  map.Reset(8);
  map.Add(1, "old_entry");
  ASSERT_EQ(map.LookUp(1), "old_entry");

  map.Reset(2);
  EXPECT_TRUE(map.LookUp(1).empty()) << "Reset must clear old entries";

  map.Add(2, "new_a");
  map.Add(3, "new_b");
  map.Add(4, "past_cap");  // cap=2, so this is dropped
  EXPECT_EQ(map.LookUp(2), "new_a");
  EXPECT_EQ(map.LookUp(3), "new_b");
  EXPECT_TRUE(map.LookUp(4).empty());
}

TEST(AnnotationMapTest, ScopeRangeIdsStopAtCap) {
  AnnotationMap map;
  map.Reset(2);
  const int64_t ids[] = {10, 11};
  map.Add(1, "a", {}, ids);
  map.Add(2, "b", {}, ids);  // fills the string set
  map.Add(3, "c", {}, ids);  // string set full: nothing recorded
  EXPECT_EQ(map.LookUpScopeRangeId(1), 11);
  EXPECT_EQ(map.LookUpScopeRangeId(2), 11)
      << "the event that fills the last slot lost its scope range id";
  EXPECT_EQ(map.LookUpScopeRangeId(3), 0);

  map.Reset(2);
  EXPECT_EQ(map.LookUpScopeRangeId(1), 0);
  EXPECT_TRUE(map.TakeScopeRangeIdTree().empty());
}

// The cap is checked once per event, so an event's annotation, ROCTX range and
// scope range id are stored together or not at all.
TEST(AnnotationMapTest, EventIsStoredWholeOrNotAtAll) {
  AnnotationMap map;
  map.Reset(3);
  const int64_t ids[] = {10, 11};
  map.Add(1, "a", "r1", ids);  // takes 2 of 3 slots
  map.Add(2, "b", "r2", ids);  // needs 2 slots, 1 left: dropped whole
  map.Add(3, "c", {}, ids);    // needs 1 slot: stored whole

  EXPECT_TRUE(map.LookUp(2).empty());
  EXPECT_TRUE(map.LookUpRoctxRange(2).empty());
  EXPECT_EQ(map.LookUpScopeRangeId(2), 0);

  EXPECT_EQ(map.LookUp(3), "c");
  EXPECT_EQ(map.LookUpScopeRangeId(3), 11);
}

TEST(AnnotationMapTest, EmptyAnnotationIsIgnored) {
  // Empty annotations must not consume capacity; the ROCTX path can produce
  // them and they would otherwise crowd out real ones.
  AnnotationMap map;
  map.Reset(1);
  map.Add(1, "");
  map.Add(2, "real");

  EXPECT_TRUE(map.LookUp(1).empty());
  EXPECT_EQ(map.LookUp(2), "real");
}

TEST(AnnotationMapTest, StoresRoctxRange) {
  AnnotationMap map;
  map.Reset(1024);
  map.Add(99, "my_annotation", "my_roctx_label", {});
  EXPECT_EQ(map.LookUp(99), "my_annotation");
  EXPECT_EQ(map.LookUpRoctxRange(99), "my_roctx_label");

  EXPECT_EQ(map.LookUpRoctxRange(100), "");

  map.Reset(1024);
  EXPECT_EQ(map.LookUpRoctxRange(99), "");
}

TEST(AnnotationMapTest, RoctxRangeEmptyWhenNotProvided) {
  AnnotationMap map;
  map.Reset(1024);
  map.Add(42, "some_op", {}, {});
  EXPECT_EQ(map.LookUp(42), "some_op");
  EXPECT_EQ(map.LookUpRoctxRange(42), "");
}

// Add() stores the roctx_range even when the annotation is empty, so that
// standalone ROCTX annotations (no XLA AnnotationStack text) still produce
// kNVTXRange on kernel events.
TEST(AnnotationMapTest, StoresRoctxRangeWhenAnnotationEmpty) {
  AnnotationMap map;
  map.Reset(1024);
  map.Add(77, /*annotation=*/"", "roctx_only_label", {});
  EXPECT_EQ(map.LookUp(77), "")
      << "correlation_map should not have an entry when annotation is empty";
  EXPECT_EQ(map.LookUpRoctxRange(77), "roctx_only_label")
      << "roctx_range_map must store the label even with no annotation";
}

}  // namespace
}  // namespace profiler
}  // namespace xla

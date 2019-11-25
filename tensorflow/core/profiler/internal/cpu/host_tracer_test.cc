/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace profiler {
namespace cpu {

std::unique_ptr<ProfilerInterface> CreateHostTracer(
    const ProfilerOptions& options);

namespace {

using ::testing::UnorderedElementsAre;

NodeExecStats MakeNodeStats(const string& name, int32 thread_id,
                            const string& label = "") {
  NodeExecStats ns;
  ns.set_node_name(name);
  ns.set_thread_id(thread_id);
  if (!label.empty()) {
    ns.set_timeline_label(label);
  }
  return ns;
}

class NodeStatsMatcher {
 public:
  explicit NodeStatsMatcher(const NodeExecStats& expected)
      : expected_(expected) {}

  bool MatchAndExplain(const NodeExecStats& p,
                       ::testing::MatchResultListener* /* listener */) const {
    return p.node_name() == expected_.node_name() &&
           p.thread_id() == expected_.thread_id() &&
           p.timeline_label() == expected_.timeline_label();
  }

  void DescribeTo(::std::ostream* os) const { *os << expected_.DebugString(); }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_.DebugString();
  }

 private:
  const NodeExecStats expected_;
};

inline ::testing::PolymorphicMatcher<NodeStatsMatcher> EqualsNodeStats(
    const NodeExecStats& expected) {
  return ::testing::MakePolymorphicMatcher(NodeStatsMatcher(expected));
}

TEST(HostTracerTest, CollectsTraceMeEventsAsRunMetadata) {
  int32 thread_id = Env::Default()->GetCurrentThreadId();

  auto tracer = CreateHostTracer(ProfilerOptions());

  TF_ASSERT_OK(tracer->Start());
  { TraceMe traceme("hello"); }
  { TraceMe traceme("world"); }
  { TraceMe traceme("contains#inside"); }
  { TraceMe traceme("good#key1=value1#"); }
  { TraceMe traceme("morning#key1=value1,key2=value2#"); }
  { TraceMe traceme("incomplete#key1=value1,key2#"); }
  TF_ASSERT_OK(tracer->Stop());

  RunMetadata run_metadata;
  TF_ASSERT_OK(tracer->CollectData(&run_metadata));

  EXPECT_EQ(run_metadata.step_stats().dev_stats_size(), 1);
  EXPECT_EQ(run_metadata.step_stats().dev_stats(0).node_stats_size(), 6);
  EXPECT_THAT(
      run_metadata.step_stats().dev_stats(0).node_stats(),
      UnorderedElementsAre(
          EqualsNodeStats(MakeNodeStats("hello", thread_id)),
          EqualsNodeStats(MakeNodeStats("world", thread_id)),
          EqualsNodeStats(MakeNodeStats("contains#inside", thread_id)),
          EqualsNodeStats(MakeNodeStats("good", thread_id, "key1=value1")),
          EqualsNodeStats(
              MakeNodeStats("morning", thread_id, "key1=value1,key2=value2")),
          EqualsNodeStats(
              MakeNodeStats("incomplete", thread_id, "key1=value1,key2"))));
}

TEST(HostTracerTest, CollectsTraceMeEventsAsXSpace) {
  int32 thread_id = Env::Default()->GetCurrentThreadId();
  string thread_name;
  ASSERT_TRUE(Env::Default()->GetCurrentThreadName(&thread_name));

  auto tracer = CreateHostTracer(ProfilerOptions());

  TF_ASSERT_OK(tracer->Start());
  { TraceMe traceme("hello"); }
  { TraceMe traceme("world"); }
  { TraceMe traceme("contains#inside"); }
  { TraceMe traceme("good#key1=value1#"); }
  { TraceMe traceme("morning#key1=value1,key2=value2#"); }
  { TraceMe traceme("incomplete#key1=value1,key2#"); }
  TF_ASSERT_OK(tracer->Stop());

  XSpace space;
  TF_ASSERT_OK(tracer->CollectData(&space));

  ASSERT_EQ(space.planes_size(), 1);
  const auto& plane = space.planes(0);
  EXPECT_EQ(plane.name(), "Host Threads");
  ASSERT_EQ(plane.lines_size(), 1);
  ASSERT_EQ(plane.event_metadata_size(), 6);
  ASSERT_EQ(plane.stat_metadata_size(), 2);
  const auto& event_metadata = plane.event_metadata();
  const auto& stat_metadata = plane.stat_metadata();
  const auto& line = plane.lines(0);
  EXPECT_EQ(line.id(), thread_id);
  EXPECT_EQ(line.name(), thread_name);
  ASSERT_EQ(line.events_size(), 6);
  const auto& events = line.events();
  EXPECT_EQ(events[0].metadata_id(), 1);
  EXPECT_EQ(event_metadata.at(1).name(), "hello");
  ASSERT_EQ(events[0].stats_size(), 0);
  EXPECT_EQ(events[1].metadata_id(), 2);
  EXPECT_EQ(event_metadata.at(2).name(), "world");
  ASSERT_EQ(events[1].stats_size(), 0);
  EXPECT_EQ(events[2].metadata_id(), 3);
  EXPECT_EQ(event_metadata.at(3).name(), "contains#inside");
  ASSERT_EQ(events[2].stats_size(), 0);
  EXPECT_EQ(events[3].metadata_id(), 4);
  EXPECT_EQ(event_metadata.at(4).name(), "good");
  ASSERT_EQ(events[3].stats_size(), 1);
  EXPECT_EQ(events[3].stats(0).metadata_id(), 1);
  EXPECT_EQ(stat_metadata.at(1).name(), "key1");
  EXPECT_EQ(events[3].stats(0).str_value(), "value1");
  EXPECT_EQ(events[4].metadata_id(), 5);
  EXPECT_EQ(event_metadata.at(5).name(), "morning");
  ASSERT_EQ(events[4].stats_size(), 2);
  EXPECT_EQ(events[4].stats(0).metadata_id(), 1);
  EXPECT_EQ(events[4].stats(0).str_value(), "value1");
  EXPECT_EQ(events[4].stats(1).metadata_id(), 2);
  EXPECT_EQ(stat_metadata.at(2).name(), "key2");
  EXPECT_EQ(events[4].stats(1).str_value(), "value2");
  EXPECT_EQ(events[5].metadata_id(), 6);
  EXPECT_EQ(event_metadata.at(6).name(), "incomplete");
  ASSERT_EQ(events[5].stats_size(), 1);
  EXPECT_EQ(events[5].stats(0).metadata_id(), 1);
  EXPECT_EQ(events[5].stats(0).str_value(), "value1");
}

}  // namespace
}  // namespace cpu
}  // namespace profiler
}  // namespace tensorflow

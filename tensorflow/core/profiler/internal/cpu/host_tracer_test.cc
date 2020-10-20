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
#include <memory>
#include <ostream>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace profiler {

std::unique_ptr<ProfilerInterface> CreateHostTracer(
    const ProfileOptions& options);

namespace {

using ::testing::UnorderedElementsAre;

NodeExecStats MakeNodeStats(absl::string_view name, uint32 thread_id,
                            absl::string_view label = "") {
  NodeExecStats ns;
  ns.set_node_name(std::string(name));
  ns.set_thread_id(thread_id);
  if (!label.empty()) {
    ns.set_timeline_label(std::string(label));
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
  uint32 thread_id = Env::Default()->GetCurrentThreadId();

  auto tracer = CreateHostTracer(ProfilerSession::DefaultOptions());

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
  uint32 thread_id;
  std::string thread_name = "MyThreadName";
  XSpace space;

  // We start a thread with a known and controled name. As of the time of
  // writing, not all platforms (example: Windows) allow reading through the
  // system to the current thread name/description. By starting a thread with a
  // name, we control this behavior entirely within the TensorFlow subsystems.
  std::unique_ptr<Thread> traced_thread(
      Env::Default()->StartThread(ThreadOptions(), thread_name, [&] {
        // Some implementations add additional information to the thread name.
        // Recapture this information.
        ASSERT_TRUE(Env::Default()->GetCurrentThreadName(&thread_name));
        thread_id = Env::Default()->GetCurrentThreadId();

        auto tracer = CreateHostTracer(ProfilerSession::DefaultOptions());

        TF_ASSERT_OK(tracer->Start());
        { TraceMe traceme("hello"); }
        { TraceMe traceme("world"); }
        { TraceMe traceme("contains#inside"); }
        { TraceMe traceme("good#key1=value1#"); }
        { TraceMe traceme("morning#key1=value1,key2=value2#"); }
        { TraceMe traceme("incomplete#key1=value1,key2#"); }
        // Special cases for tf.data
        { TraceMe traceme("Iterator::XXX::YYY::ParallelMap"); }
        TF_ASSERT_OK(tracer->Stop());

        TF_ASSERT_OK(tracer->CollectData(&space));
      }));
  traced_thread.reset();      // Join thread, waiting for completion.
  ASSERT_NO_FATAL_FAILURE();  // Test for failure in child thread.

  ASSERT_EQ(space.planes_size(), 1);
  const auto& plane = space.planes(0);
  XPlaneVisitor xplane(&plane);
  ASSERT_EQ(plane.name(), kHostThreadsPlaneName);
  ASSERT_EQ(plane.lines_size(), 1);
  ASSERT_EQ(plane.event_metadata_size(), 7);
  ASSERT_EQ(plane.stat_metadata_size(), 4);
  const auto& line = plane.lines(0);
  EXPECT_EQ(line.id(), thread_id);
  EXPECT_EQ(line.name(), thread_name);
  ASSERT_EQ(line.events_size(), 7);
  const auto& events = line.events();

  XEventVisitor e0(&xplane, &line, &events[0]);
  EXPECT_EQ(e0.Name(), "hello");
  ASSERT_EQ(events[0].stats_size(), 0);

  XEventVisitor e1(&xplane, &line, &events[1]);
  EXPECT_EQ(e1.Name(), "world");
  ASSERT_EQ(events[1].stats_size(), 0);

  XEventVisitor e2(&xplane, &line, &events[2]);
  EXPECT_EQ(e2.Name(), "contains#inside");
  ASSERT_EQ(events[2].stats_size(), 0);

  XEventVisitor e3(&xplane, &line, &events[3]);
  EXPECT_EQ(e3.Name(), "good");
  ASSERT_EQ(events[3].stats_size(), 1);
  {
    absl::optional<std::string> value;
    e3.ForEachStat([&](const XStatVisitor& stat) {
      if (stat.Name() == "key1") value = stat.ToString();
    });
    ASSERT_TRUE(value);           // The stat key is present.
    EXPECT_EQ(*value, "value1");  // The stat value is expected.
  }

  XEventVisitor e4(&xplane, &line, &events[4]);
  EXPECT_EQ(e4.Name(), "morning");
  ASSERT_EQ(events[4].stats_size(), 2);
  {
    absl::optional<std::string> value1, value2;
    e4.ForEachStat([&](const XStatVisitor& stat) {
      if (stat.Name() == "key1") {
        value1 = stat.ToString();
      } else if (stat.Name() == "key2") {
        value2 = stat.ToString();
      }
    });
    ASSERT_TRUE(value1 && value2);  // The stat keys are presents.
    EXPECT_EQ(*value1, "value1");   // The stat value1 is expected.
    EXPECT_EQ(*value2, "value2");   // The stat value2 is expected.
  }

  XEventVisitor e5(&xplane, &line, &events[5]);
  EXPECT_EQ(e5.Name(), "incomplete");
  ASSERT_EQ(events[5].stats_size(), 1);
  {
    absl::optional<std::string> value1, value2;
    e5.ForEachStat([&](const XStatVisitor& stat) {
      if (stat.Name() == "key1") {
        value1 = stat.ToString();
      } else if (stat.Name() == "key2") {
        value2 = stat.ToString();
      }
    });
    ASSERT_TRUE(value1 && !value2);  // One of the stat key is present.
    EXPECT_EQ(*value1, "value1");    // The stat value is expected.
  }

  // Dataset Ops will trim intermediate namespace.
  XEventVisitor e6(&xplane, &line, &events[6]);
  EXPECT_EQ(e6.Name(), "Iterator::XXX::YYY::ParallelMap");

  EXPECT_EQ(e6.DisplayName(), "Iterator::ParallelMap");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow

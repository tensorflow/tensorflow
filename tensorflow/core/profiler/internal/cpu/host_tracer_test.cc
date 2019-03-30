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
#include "tensorflow/core/profiler/internal/cpu/host_tracer.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/optional.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace profiler {
namespace cpu {
namespace {

using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

NodeExecStats MakeNodeStats(const string& name, uint64 thread_id,
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

TEST(HostTracerTest, CollectsTraceMeEvents) {
  uint32 thread_id = Env::Default()->GetCurrentThreadId();

  auto tracer = HostTracer::Create(/*host_trace_level=*/1);

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

void ValidateResult(const RunMetadata& run_metadata, const string& trace_name) {
  uint32 thread_id = Env::Default()->GetCurrentThreadId();

  EXPECT_THAT(
      run_metadata.step_stats().dev_stats(0).node_stats(),
      ElementsAre(EqualsNodeStats(MakeNodeStats(trace_name, thread_id))));
}

TEST(HostTracerTest, CollectsTraceMeEventsBetweenTracing) {
  auto tracer = HostTracer::Create(/*host_trace_level=*/1);
  RunMetadata run_metadata;
  RunMetadata run_metadata2;

  TF_ASSERT_OK(tracer->Start());
  { TraceMe traceme("hello"); }
  TF_ASSERT_OK(tracer->CollectData(&run_metadata));
  { TraceMe traceme("world"); }
  TF_ASSERT_OK(tracer->CollectData(&run_metadata2));
  TF_ASSERT_OK(tracer->Stop());

  ValidateResult(run_metadata, "hello");
  ValidateResult(run_metadata2, "world");
}

}  // namespace
}  // namespace cpu
}  // namespace profiler
}  // namespace tensorflow

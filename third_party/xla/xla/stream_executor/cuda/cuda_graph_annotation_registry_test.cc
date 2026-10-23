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

#include "xla/stream_executor/cuda/cuda_graph_annotation_registry.h"

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace stream_executor::gpu {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Optional;
using Registry = CudaGraphAnnotationRegistry;
using TrackMode = CudaGraphAnnotationRegistry::TrackMode;

constexpr uint32_t kGraph = 10;
constexpr uint32_t kOtherGraph = 20;
constexpr uint64_t kNode = 1001;
constexpr uint64_t kNode2 = 1002;
constexpr uint64_t kOtherNode = 2001;

// Returns a copy of the node's annotation, if the registry has one.
std::optional<std::string> Lookup(
    const Registry& registry, uint32_t graph_id, uint64_t node_id,
    Registry::KeySpace space = Registry::KeySpace::kTemplateGraph) {
  std::optional<std::string> annotation;
  registry.VisitNodeAnnotation(
      graph_id, node_id,
      [&annotation](absl::string_view a) { annotation.emplace(a); }, space);
  return annotation;
}

// Tracks a newly created graph and returns its token.
Registry::Token Track(Registry& registry, uint32_t graph_id) {
  return registry.TrackGraph(graph_id, TrackMode::kNewGraph).token;
}

TEST(CudaGraphAnnotationRegistryTest, GetOrCreateReturnsExistingToken) {
  Registry registry;
  EXPECT_TRUE(registry.empty());
  EXPECT_FALSE(registry.IsGraphTracked(kGraph));

  Registry::TrackResult first =
      registry.TrackGraph(kGraph, TrackMode::kGetOrCreate);
  EXPECT_NE(first.token, Registry::kInvalidToken);
  EXPECT_TRUE(first.newly_tracked);
  EXPECT_TRUE(registry.IsGraphTracked(kGraph));
  EXPECT_FALSE(registry.empty());

  Registry::TrackResult again =
      registry.TrackGraph(kGraph, TrackMode::kGetOrCreate);
  EXPECT_EQ(again.token, first.token);
  EXPECT_FALSE(again.newly_tracked);

  Registry::TrackResult other =
      registry.TrackGraph(kOtherGraph, TrackMode::kNewGraph);
  EXPECT_NE(other.token, Registry::kInvalidToken);
  EXPECT_NE(other.token, first.token);
  EXPECT_TRUE(other.newly_tracked);
  EXPECT_EQ(registry.GetStats().live_graphs, 2);
}

TEST(CudaGraphAnnotationRegistryTest, GetNodeAnnotationsForTesting) {
  Registry registry;
  Registry::Token token = Track(registry, kGraph);
  Registry::Token other_token = Track(registry, kOtherGraph);
  // Registers the annotations across two graphs in reverse order, so the result
  // is sorted only if GetNodeAnnotationsForTesting sorts it.
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, token, kNode + i,
                                                "op" + std::to_string(7 - i)));
    ASSERT_TRUE(registry.RegisterNodeAnnotation(kOtherGraph, other_token,
                                                kOtherNode + i,
                                                "op" + std::to_string(3 - i)));
  }
  Registry::Token exec_token =
      registry
          .TrackGraph(kGraph, TrackMode::kNewGraph,
                      Registry::KeySpace::kPre134ExecGraph)
          .token;
  ASSERT_TRUE(registry.RegisterNodeAnnotation(
      kGraph, exec_token, kNode, "exec", Registry::KeySpace::kPre134ExecGraph));

  EXPECT_THAT(
      registry.GetNodeAnnotationsForTesting(Registry::KeySpace::kTemplateGraph),
      ElementsAre("op0", "op1", "op2", "op3", "op4", "op5", "op6", "op7"));
  EXPECT_THAT(registry.GetNodeAnnotationsForTesting(
                  Registry::KeySpace::kPre134ExecGraph),
              ElementsAre("exec"));
}

TEST(CudaGraphAnnotationRegistryTest, InsertIfAbsent) {
  Registry registry;
  Registry::Token token = Track(registry, kGraph);

  EXPECT_FALSE(registry.RegisterNodeAnnotation(kGraph, token, kNode, ""));
  EXPECT_TRUE(registry.RegisterNodeAnnotation(kGraph, token, kNode, "first"));
  EXPECT_FALSE(registry.RegisterNodeAnnotation(kGraph, token, kNode, "second"));
  EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("first")));

  // Writes need a live entry and its token.
  EXPECT_FALSE(registry.RegisterNodeAnnotation(kGraph, token + 1, kNode2, "x"));
  EXPECT_FALSE(registry.RegisterNodeAnnotation(kGraph, Registry::kInvalidToken,
                                               kNode2, "x"));
  EXPECT_FALSE(
      registry.RegisterNodeAnnotation(kOtherGraph, token, kOtherNode, "x"));
  EXPECT_EQ(Lookup(registry, kGraph, kNode2), std::nullopt);
  EXPECT_FALSE(registry.IsGraphTracked(kOtherGraph));
  EXPECT_EQ(registry.GetStats().live_graphs, 1);
}

TEST(CudaGraphAnnotationRegistryTest, TokenMismatchNoOp) {
  Registry registry;
  Registry::Token token = Track(registry, kGraph);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, token, kNode, "a"));

  registry.UnregisterGraph(kGraph, token + 1);
  EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("a")));

  // An untracked ID matches no token either.
  registry.UnregisterGraph(kOtherGraph, token);
  EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("a")));

  registry.UnregisterGraph(kGraph, token);
  EXPECT_EQ(Lookup(registry, kGraph, kNode), std::nullopt);
  EXPECT_TRUE(registry.empty());

  // The old token can't bring the entry back or unregister it twice.
  EXPECT_FALSE(registry.RegisterNodeAnnotation(kGraph, token, kNode, "a"));
  registry.UnregisterGraph(kGraph, token);
  EXPECT_TRUE(registry.empty());
}

TEST(CudaGraphAnnotationRegistryTest, NewGraphRetiresStale) {
  Registry registry;
  Registry::Token t1 = Track(registry, kGraph);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, t1, kNode, "old"));

  // A new graph got the ID before the old graph's cleanup ran.
  Registry::TrackResult second =
      registry.TrackGraph(kGraph, TrackMode::kNewGraph);
  Registry::Token t2 = second.token;
  EXPECT_TRUE(second.newly_tracked);
  EXPECT_NE(t2, t1);
  EXPECT_EQ(Lookup(registry, kGraph, kNode), std::nullopt);
  EXPECT_FALSE(registry.RegisterNodeAnnotation(kGraph, t1, kNode, "old"));
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, t2, kNode, "new"));

  // The old graph's late cleanup must leave the new graph's entry alone.
  registry.UnregisterGraph(kGraph, t1);
  EXPECT_TRUE(registry.IsGraphTracked(kGraph));
  EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("new")));
  EXPECT_EQ(registry.GetStats().live_graphs, 1);
}

TEST(CudaGraphAnnotationRegistryTest, NewGraphParksStaleEntryWhileHeld) {
  Registry registry;
  Registry::HarvestHold hold = registry.AcquireHarvestHold();
  Registry::Token t1 = Track(registry, kGraph);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, t1, kNode, "old"));
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, t1, kNode2, "old2"));

  // The retired entry takes the park path, so the old graph's events resolve.
  Registry::Token t2 = Track(registry, kGraph);
  EXPECT_EQ(registry.GetStats().parked_graphs, 1);
  EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("old")));

  // Parking the newer entry replaces the older parked one.
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, t2, kNode, "new"));
  registry.UnregisterGraph(kGraph, t2);
  Registry::Stats stats = registry.GetStats();
  EXPECT_EQ(stats.parked_graphs, 1);
  EXPECT_EQ(stats.parked_bytes, 3);
  EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("new")));
  EXPECT_EQ(Lookup(registry, kGraph, kNode2), std::nullopt);
}

TEST(CudaGraphAnnotationRegistryTest, ParkThenPurge) {
  Registry registry;
  Registry::Token token = Track(registry, kGraph);
  ASSERT_TRUE(
      registry.RegisterNodeAnnotation(kGraph, token, kNode, "annotation"));
  {
    Registry::HarvestHold hold = registry.AcquireHarvestHold();
    registry.UnregisterGraph(kGraph, token);
    Registry::Stats stats = registry.GetStats();
    EXPECT_EQ(stats.live_graphs, 0);
    EXPECT_EQ(stats.parked_graphs, 1);
    EXPECT_EQ(stats.parked_bytes, 10);
    EXPECT_EQ(stats.active_holds, 1);
    EXPECT_FALSE(registry.IsGraphTracked(kGraph));
    EXPECT_FALSE(registry.empty());
    // The destroyed graph's events still resolve until the harvest ends.
    EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("annotation")));
    // But the graph is dead: its token no longer admits writes.
    EXPECT_FALSE(registry.RegisterNodeAnnotation(kGraph, token, kNode2, "x"));
  }
  Registry::Stats stats = registry.GetStats();
  EXPECT_EQ(stats.parked_graphs, 0);
  EXPECT_EQ(stats.parked_bytes, 0);
  EXPECT_EQ(stats.active_holds, 0);
  EXPECT_EQ(Lookup(registry, kGraph, kNode), std::nullopt);
  EXPECT_TRUE(registry.empty());
}

TEST(CudaGraphAnnotationRegistryTest, OverlappingHolds) {
  Registry registry;
  Registry::Token token = Track(registry, kGraph);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, token, kNode, "a"));
  std::optional<Registry::HarvestHold> first(registry.AcquireHarvestHold());
  Registry::HarvestHold second = registry.AcquireHarvestHold();
  EXPECT_EQ(registry.GetStats().active_holds, 2);
  registry.UnregisterGraph(kGraph, token);

  // Moving a hold transfers it without releasing it.
  std::optional<Registry::HarvestHold> moved(std::move(second));
  EXPECT_EQ(registry.GetStats().active_holds, 2);

  // Releasing one of two holds keeps the parked entry.
  first.reset();
  EXPECT_EQ(registry.GetStats().active_holds, 1);
  EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("a")));

  // Releasing the last hold purges the parked entry.
  moved.reset();
  Registry::Stats stats = registry.GetStats();
  EXPECT_EQ(stats.active_holds, 0);
  EXPECT_EQ(stats.parked_graphs, 0);
  EXPECT_EQ(Lookup(registry, kGraph, kNode), std::nullopt);
}

TEST(CudaGraphAnnotationRegistryTest, EmptyEntryNotParked) {
  Registry registry;
  Registry::HarvestHold hold = registry.AcquireHarvestHold();
  Registry::Token token = Track(registry, kGraph);
  registry.UnregisterGraph(kGraph, token);

  EXPECT_EQ(registry.GetStats().parked_graphs, 0);
  EXPECT_TRUE(registry.empty());
}

TEST(CudaGraphAnnotationRegistryTest, ParkCapDrops) {
  // Room for "12345", but not for "123456" as well.
  Registry registry(/*max_parked_bytes=*/10);
  Registry::Token t1 = Track(registry, kGraph);
  Registry::Token t2 = Track(registry, kOtherGraph);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, t1, kNode, "12345"));
  ASSERT_TRUE(
      registry.RegisterNodeAnnotation(kOtherGraph, t2, kOtherNode, "123456"));

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  // Only the last release warns, and only about the drops.
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                       HasSubstr("dropped the annotations of 1 graph")))
      .Times(1);
  log.StartCapturingLogs();
  {
    Registry::HarvestHold hold = registry.AcquireHarvestHold();
    registry.UnregisterGraph(kGraph, t1);
    registry.UnregisterGraph(kOtherGraph, t2);
    Registry::Stats stats = registry.GetStats();
    EXPECT_EQ(stats.parked_graphs, 1);
    EXPECT_EQ(stats.parked_bytes, 5);
    EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("12345")));
    EXPECT_EQ(Lookup(registry, kOtherGraph, kOtherNode), std::nullopt);
  }
  EXPECT_TRUE(registry.empty());
}

TEST(CudaGraphAnnotationRegistryTest,
     ParkReplacementOverCapKeepsOldParkedEntry) {
  // Room for "1234" (4 bytes), not for "123456" (6 bytes).
  Registry registry(/*max_parked_bytes=*/5);
  Registry::HarvestHold hold = registry.AcquireHarvestHold();

  Registry::Token t1 = Track(registry, kGraph);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, t1, kNode, "1234"));
  registry.UnregisterGraph(kGraph, t1);
  EXPECT_EQ(registry.GetStats().parked_graphs, 1);
  EXPECT_EQ(registry.GetStats().parked_bytes, 4);

  // A new graph with the same ID has a 6-byte annotation that exceeds the cap
  // even after subtracting the old 4-byte entry. Retiring it must drop the new
  // entry and preserve the old parked entry.
  Registry::Token t2 = Track(registry, kGraph);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, t2, kNode, "123456"));
  registry.UnregisterGraph(kGraph, t2);

  Registry::Stats stats = registry.GetStats();
  EXPECT_EQ(stats.parked_graphs, 1);
  EXPECT_EQ(stats.parked_bytes, 4);
  EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("1234")));
}

TEST(CudaGraphAnnotationRegistryTest, ParkReplacementUnderCapReplacesOldEntry) {
  // Room for both the old 6-byte entry and the new 7-byte entry.
  Registry registry(/*max_parked_bytes=*/20);
  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  // The replaced parked entry counts as one dropped graph.
  EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _,
                       HasSubstr("dropped the annotations of 1 graph")))
      .Times(1);
  log.StartCapturingLogs();
  {
    Registry::HarvestHold hold = registry.AcquireHarvestHold();

    Registry::Token t1 = Track(registry, kGraph);
    ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, t1, kNode, "1234"));
    ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, t1, kNode2, "56"));
    registry.UnregisterGraph(kGraph, t1);
    Registry::Stats stats = registry.GetStats();
    EXPECT_EQ(stats.parked_graphs, 1);
    EXPECT_EQ(stats.parked_nodes, 2);
    EXPECT_EQ(stats.parked_bytes, 6);

    // A new graph with the same ID is retired too. Its entry fits under the
    // cap, so it replaces the older parked entry.
    Registry::Token t2 = Track(registry, kGraph);
    ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, t2, kNode, "abcdefg"));
    registry.UnregisterGraph(kGraph, t2);

    stats = registry.GetStats();
    EXPECT_EQ(stats.live_graphs, 0);
    EXPECT_EQ(stats.parked_graphs, 1);
    EXPECT_EQ(stats.parked_nodes, 1);
    EXPECT_EQ(stats.parked_bytes, 7);
    EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("abcdefg")));
    EXPECT_EQ(Lookup(registry, kGraph, kNode2), std::nullopt);
  }
  EXPECT_TRUE(registry.empty());
}

TEST(CudaGraphAnnotationRegistryTest,
     TemplateAndPre134ExecKeySpacesAreDisjoint) {
  constexpr uint32_t kSameGraphId = 7;
  Registry registry;

  // Track template graph 7 and register an annotation.
  Registry::TrackResult tmpl = registry.TrackGraph(
      kSameGraphId, TrackMode::kNewGraph, Registry::KeySpace::kTemplateGraph);
  ASSERT_TRUE(tmpl.newly_tracked);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(
      kSameGraphId, tmpl.token, kNode, "template_op",
      Registry::KeySpace::kTemplateGraph));

  // Track pre-13.4 exec graph 7 with the same integer ID and kNewGraph; it must
  // NOT retire or overwrite the live template graph 7.
  Registry::TrackResult exec = registry.TrackGraph(
      kSameGraphId, TrackMode::kNewGraph, Registry::KeySpace::kPre134ExecGraph);
  ASSERT_TRUE(exec.newly_tracked);
  EXPECT_TRUE(registry.IsGraphTracked(kSameGraphId,
                                      Registry::KeySpace::kTemplateGraph));
  EXPECT_TRUE(registry.IsGraphTracked(kSameGraphId,
                                      Registry::KeySpace::kPre134ExecGraph));
  EXPECT_EQ(registry.GetStats().live_graphs, 2);

  // Copy from template graph 7 into exec graph 7, plus a direct exec node.
  EXPECT_TRUE(registry.CopyNodeAnnotation(
      kSameGraphId, exec.token, kOtherNode, kSameGraphId, kNode, "fallback",
      Registry::KeySpace::kPre134ExecGraph));
  EXPECT_TRUE(registry.RegisterNodeAnnotation(
      kSameGraphId, exec.token, kNode, "exec_only_op",
      Registry::KeySpace::kPre134ExecGraph));

  // Each key space sees only its own entries.
  EXPECT_THAT(
      Lookup(registry, kSameGraphId, kNode, Registry::KeySpace::kTemplateGraph),
      Optional(Eq("template_op")));
  EXPECT_EQ(Lookup(registry, kSameGraphId, kOtherNode,
                   Registry::KeySpace::kTemplateGraph),
            std::nullopt);
  EXPECT_THAT(Lookup(registry, kSameGraphId, kNode,
                     Registry::KeySpace::kPre134ExecGraph),
              Optional(Eq("exec_only_op")));
  EXPECT_THAT(Lookup(registry, kSameGraphId, kOtherNode,
                     Registry::KeySpace::kPre134ExecGraph),
              Optional(Eq("template_op")));

  // Unregistering exec graph 7 leaves template graph 7 live and intact.
  registry.UnregisterGraph(kSameGraphId, exec.token,
                           Registry::KeySpace::kPre134ExecGraph);
  EXPECT_TRUE(registry.IsGraphTracked(kSameGraphId,
                                      Registry::KeySpace::kTemplateGraph));
  EXPECT_FALSE(registry.IsGraphTracked(kSameGraphId,
                                       Registry::KeySpace::kPre134ExecGraph));
  EXPECT_THAT(
      Lookup(registry, kSameGraphId, kNode, Registry::KeySpace::kTemplateGraph),
      Optional(Eq("template_op")));
  EXPECT_EQ(Lookup(registry, kSameGraphId, kNode,
                   Registry::KeySpace::kPre134ExecGraph),
            std::nullopt);
}

TEST(CudaGraphAnnotationRegistryTest, LiveBeforeParked) {
  Registry registry;
  Registry::HarvestHold hold = registry.AcquireHarvestHold();
  Registry::Token t1 = Track(registry, kGraph);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, t1, kNode, "parked"));
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, t1, kNode2, "parked2"));
  registry.UnregisterGraph(kGraph, t1);

  // A newer graph with the same ID. Parked entries don't count as tracked.
  Registry::TrackResult t2 =
      registry.TrackGraph(kGraph, TrackMode::kGetOrCreate);
  EXPECT_TRUE(t2.newly_tracked);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, t2.token, kNode, "live"));
  EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("live")));
  EXPECT_THAT(Lookup(registry, kGraph, kNode2), Optional(Eq("parked2")));
}

TEST(CudaGraphAnnotationRegistryTest, CopyFromLive) {
  Registry registry;
  Registry::Token source = Track(registry, kOtherGraph);
  Registry::Token clone = Track(registry, kGraph);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kOtherGraph, source, kOtherNode,
                                              "original"));

  EXPECT_TRUE(registry.CopyNodeAnnotation(kGraph, clone, kNode, kOtherGraph,
                                          kOtherNode, "fallback"));
  EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("original")));
}

TEST(CudaGraphAnnotationRegistryTest, CopyFromParked) {
  Registry registry;
  Registry::HarvestHold hold = registry.AcquireHarvestHold();
  Registry::Token source = Track(registry, kOtherGraph);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kOtherGraph, source, kOtherNode,
                                              "original"));
  registry.UnregisterGraph(kOtherGraph, source);
  Registry::Token clone = Track(registry, kGraph);

  EXPECT_TRUE(registry.CopyNodeAnnotation(kGraph, clone, kNode, kOtherGraph,
                                          kOtherNode, "fallback"));
  EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("original")));
}

TEST(CudaGraphAnnotationRegistryTest, CopyFallback) {
  Registry registry;
  Registry::Token clone = Track(registry, kGraph);

  EXPECT_FALSE(registry.CopyNodeAnnotation(kGraph, clone, kNode, kOtherGraph,
                                           kOtherNode, /*fallback=*/""));
  EXPECT_EQ(Lookup(registry, kGraph, kNode), std::nullopt);
  EXPECT_TRUE(registry.CopyNodeAnnotation(kGraph, clone, kNode, kOtherGraph,
                                          kOtherNode, "fallback"));
  EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("fallback")));
  // Copies need the live entry's token too.
  EXPECT_FALSE(registry.CopyNodeAnnotation(kGraph, clone + 1, kNode2,
                                           kOtherGraph, kOtherNode, "x"));
  EXPECT_EQ(Lookup(registry, kGraph, kNode2), std::nullopt);
}

TEST(CudaGraphAnnotationRegistryTest, CopyNoOverwrite) {
  Registry registry;
  Registry::Token source = Track(registry, kOtherGraph);
  Registry::Token clone = Track(registry, kGraph);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kOtherGraph, source, kOtherNode,
                                              "original"));
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, clone, kNode, "precise"));

  EXPECT_FALSE(registry.CopyNodeAnnotation(kGraph, clone, kNode, kOtherGraph,
                                           kOtherNode, "fallback"));
  EXPECT_THAT(Lookup(registry, kGraph, kNode), Optional(Eq("precise")));

  // The source may be a node of the destination graph itself.
  EXPECT_TRUE(
      registry.CopyNodeAnnotation(kGraph, clone, kNode2, kGraph, kNode, ""));
  EXPECT_THAT(Lookup(registry, kGraph, kNode2), Optional(Eq("precise")));
}

TEST(CudaGraphAnnotationRegistryTest, VisitorNotCalledOnMiss) {
  Registry registry;
  Registry::Token token = Track(registry, kGraph);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, token, kNode, "a"));
  int calls = 0;
  auto visitor = [&calls](absl::string_view) { ++calls; };

  EXPECT_FALSE(registry.VisitNodeAnnotation(kGraph, kNode2, visitor));
  EXPECT_FALSE(registry.VisitNodeAnnotation(kOtherGraph, kNode, visitor));
  EXPECT_EQ(calls, 0);
  EXPECT_TRUE(registry.VisitNodeAnnotation(kGraph, kNode, visitor));
  EXPECT_EQ(calls, 1);
}

TEST(CudaGraphAnnotationRegistryTest, ClearKeepsHolds) {
  Registry registry;
  Registry::HarvestHold hold = registry.AcquireHarvestHold();
  Registry::Token live = Track(registry, kGraph);
  Registry::Token parked = Track(registry, kOtherGraph);
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, live, kNode, "a"));
  ASSERT_TRUE(
      registry.RegisterNodeAnnotation(kOtherGraph, parked, kOtherNode, "b"));
  registry.UnregisterGraph(kOtherGraph, parked);

  registry.ClearForTesting();
  EXPECT_TRUE(registry.empty());
  Registry::Stats stats = registry.GetStats();
  EXPECT_EQ(stats.live_graphs, 0);
  EXPECT_EQ(stats.parked_graphs, 0);
  EXPECT_EQ(stats.parked_bytes, 0);
  EXPECT_EQ(stats.active_holds, 1);

  // Tokens keep increasing, so a cleared graph's late cleanup stays stale.
  Registry::Token token = Track(registry, kGraph);
  EXPECT_GT(token, parked);
  registry.UnregisterGraph(kGraph, live);
  EXPECT_TRUE(registry.IsGraphTracked(kGraph));

  // The hold survived, so unregistered entries are still parked.
  ASSERT_TRUE(registry.RegisterNodeAnnotation(kGraph, token, kNode, "c"));
  registry.UnregisterGraph(kGraph, token);
  EXPECT_EQ(registry.GetStats().parked_graphs, 1);
}

// Writers track, register, copy and unregister graphs while visitors read and
// another thread keeps taking overlapping holds. Run it under TSAN.
TEST(CudaGraphAnnotationRegistryTest, Concurrency) {
  constexpr int kNumWriters = 8;
  constexpr int kGraphsPerWriter = 200;
  constexpr int kNodesPerGraph = 4;
  constexpr int kNumGraphs = kNumWriters * kGraphsPerWriter;
  // Each writer owns a disjoint range of graph IDs, starting at 1.
  auto annotation = [](uint32_t graph_id, int node) {
    return absl::StrCat("graph", graph_id, "_node", node);
  };
  Registry registry;
  std::atomic<int> writers_running = kNumWriters;
  std::atomic<int> visits_hit = 0;
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "registry_concurrency",
                                 kNumWriters + 3);
    for (int w = 0; w < kNumWriters; ++w) {
      pool.Schedule([&, w] {
        for (int g = 0; g < kGraphsPerWriter; ++g) {
          const uint32_t graph_id = w * kGraphsPerWriter + g + 1;
          Registry::Token token = Track(registry, graph_id);
          Registry::TrackResult again =
              registry.TrackGraph(graph_id, TrackMode::kGetOrCreate);
          EXPECT_EQ(again.token, token);
          EXPECT_FALSE(again.newly_tracked);
          for (int n = 0; n + 1 < kNodesPerGraph; ++n) {
            EXPECT_TRUE(registry.RegisterNodeAnnotation(
                graph_id, token, static_cast<uint64_t>(n),
                annotation(graph_id, n)));
          }
          // The last node copies node 0 of the same graph.
          EXPECT_TRUE(registry.CopyNodeAnnotation(
              graph_id, token, /*node_id=*/3, graph_id,
              /*source_node_id=*/0, "fallback"));
          // Unregister every other graph, the second time with a stale token.
          if (g % 2 == 0) {
            registry.UnregisterGraph(graph_id, token);
            registry.UnregisterGraph(graph_id, token);
          }
        }
        writers_running.fetch_sub(1, std::memory_order_release);
      });
    }
    for (int v = 0; v < 2; ++v) {
      pool.Schedule([&] {
        while (writers_running.load(std::memory_order_acquire) > 0) {
          for (uint32_t graph_id = 1; graph_id <= kNumGraphs; graph_id += 7) {
            constexpr uint64_t kNodeId = 3;
            if (registry.VisitNodeAnnotation(
                    graph_id, kNodeId, [&](absl::string_view a) {
                      EXPECT_EQ(a, annotation(graph_id, 0));
                    })) {
              visits_hit.fetch_add(1, std::memory_order_relaxed);
            }
          }
        }
      });
    }
    pool.Schedule([&] {
      while (writers_running.load(std::memory_order_acquire) > 0) {
        Registry::HarvestHold outer = registry.AcquireHarvestHold();
        {
          Registry::HarvestHold inner = registry.AcquireHarvestHold();
          absl::SleepFor(absl::Microseconds(50));
        }
        EXPECT_GE(registry.GetStats().active_holds, 1);
      }
    });
  }  // Joins all threads.

  Registry::Stats stats = registry.GetStats();
  EXPECT_EQ(stats.active_holds, 0);
  EXPECT_EQ(stats.live_graphs, kNumGraphs / 2);
  // The last release purged everything that was parked.
  EXPECT_EQ(stats.parked_graphs, 0);
  EXPECT_EQ(stats.parked_bytes, 0);
  EXPECT_FALSE(registry.empty());
  for (uint32_t graph_id = 1; graph_id <= kNumGraphs; ++graph_id) {
    const bool unregistered = (graph_id - 1) % kGraphsPerWriter % 2 == 0;
    EXPECT_EQ(registry.IsGraphTracked(graph_id), !unregistered);
    if (unregistered) {
      EXPECT_EQ(Lookup(registry, graph_id, 0), std::nullopt);
    } else {
      EXPECT_THAT(Lookup(registry, graph_id, 0),
                  Optional(Eq(annotation(graph_id, 0))));
    }
  }
}

TEST(CudaGraphAnnotationRegistryTest, StatsReportLiveAndParkedSizes) {
  Registry& registry = Registry::Default();
  registry.ClearForTesting();

  auto token1 = Track(registry, 1);
  registry.RegisterNodeAnnotation(1, token1, 10, "node_a");
  registry.RegisterNodeAnnotation(1, token1, 11, "node_b");

  auto token2 = Track(registry, 2);
  registry.RegisterNodeAnnotation(2, token2, 20, "node_c");

  {
    Registry::Stats stats = registry.GetStats();
    EXPECT_EQ(stats.live_graphs, 2);
    EXPECT_EQ(stats.live_nodes, 3);
    EXPECT_EQ(stats.live_bytes, 18);
  }

  Registry::HarvestHold hold = registry.AcquireHarvestHold();
  registry.UnregisterGraph(1, token1);

  {
    Registry::Stats stats = registry.GetStats();
    EXPECT_EQ(stats.live_graphs, 1);
    EXPECT_EQ(stats.live_nodes, 1);
    EXPECT_EQ(stats.live_bytes, 6);
    EXPECT_EQ(stats.parked_graphs, 1);
    EXPECT_EQ(stats.parked_nodes, 2);
    EXPECT_EQ(stats.parked_bytes, 12);

    std::string str = absl::StrCat(stats);
    EXPECT_THAT(str, ::testing::HasSubstr("live_bytes=6"));
    EXPECT_THAT(str, ::testing::HasSubstr("parked_nodes=2"));
  }
}

TEST(SelectTelemetryStrategyTest, EvaluatesRulesCorrectly) {
  using TelemetryStrategy = CudaGraphAnnotationRegistry::TelemetryStrategy;

  struct TestCase {
    unsigned compiled_major;
    unsigned compiled_minor;
    unsigned drv_major;
    unsigned drv_minor;
    unsigned runtime_major;
    unsigned runtime_minor;
    TelemetryStrategy expected;
  } test_cases[] = {
      // zero versions
      {0, 0, 13, 5, 13, 5, TelemetryStrategy::kDisabled},
      {13, 4, 0, 0, 13, 5, TelemetryStrategy::kDisabled},
      {13, 4, 13, 5, 0, 0, TelemetryStrategy::kDisabled},

      // compiled 13.2 / 13.3 / 13.4
      // driver 13.1 / 13.2 / 13.3 / 13.4 / 13.5
      // runtime lower than the driver, and the reverse

      // compiled 13.2, effective < 13.2 (disabled)
      {13, 2, 13, 5, 13, 1, TelemetryStrategy::kDisabled},
      {13, 2, 13, 1, 13, 5, TelemetryStrategy::kDisabled},

      // compiled < 13.2 (disabled)
      {13, 1, 13, 5, 13, 5, TelemetryStrategy::kDisabled},

      // compiled 13.2, effective 13.2
      {13, 2, 13, 2, 13, 2, TelemetryStrategy::kPre134ExecGraph},
      // compiled 13.2, effective 13.5
      {13, 2, 13, 5, 13, 5, TelemetryStrategy::kPre134ExecGraph},
      // compiled 13.3, effective 13.5
      {13, 3, 13, 5, 13, 5, TelemetryStrategy::kPre134ExecGraph},

      // compiled 13.4, effective 13.3
      {13, 4, 13, 5, 13, 3, TelemetryStrategy::kPre134ExecGraph},
      {13, 4, 13, 3, 13, 5, TelemetryStrategy::kPre134ExecGraph},
      // compiled 13.4, effective 13.4
      {13, 4, 13, 4, 13, 4, TelemetryStrategy::kCuda134SourceGraph},
      // compiled 13.4, effective 13.5
      {13, 4, 13, 5, 13, 5, TelemetryStrategy::kCuda134SourceGraph},
  };

  for (const auto& test_case : test_cases) {
    SemanticVersion compiled_toolkit{test_case.compiled_major,
                                     test_case.compiled_minor, 0};
    SemanticVersion driver{test_case.drv_major, test_case.drv_minor, 0};
    SemanticVersion runtime{test_case.runtime_major, test_case.runtime_minor,
                            0};

    EXPECT_EQ(CudaGraphAnnotationRegistry::SelectTelemetryStrategy(
                  compiled_toolkit, driver, runtime),
              test_case.expected)
        << "compiled_toolkit: " << compiled_toolkit << " driver: " << driver
        << " runtime: " << runtime;
  }
}
}  // namespace
}  // namespace stream_executor::gpu

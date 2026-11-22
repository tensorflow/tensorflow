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

#include "xla/backends/cpu/runtime/rng_state_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/call_thunk.h"
#include "xla/backends/cpu/runtime/rng_state_resource.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_executor.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/execution_graph.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

using NodeEdge = ExecutionGraph::NodeEdge;

// Exposes a thunk's buffer and resource uses to `ExecutionGraph`, the same way
// `ThunkExecutor` builds its graph.
class ThunkUses : public ExecutionGraph::Operation {
 public:
  explicit ThunkUses(const Thunk& thunk)
      : buffer_uses_(thunk.buffer_uses()),
        resource_uses_(thunk.resource_uses()) {}

  absl::string_view name() const final { return "thunk"; }
  absl::Span<const BufferUse> BufferUses() const final { return buffer_uses_; }
  absl::Span<const ResourceUse> ResourceUses() const final {
    return resource_uses_;
  }

 private:
  Thunk::BufferUses buffer_uses_;
  Thunk::ResourceUses resource_uses_;
};

absl::StatusOr<ExecutionGraph> CreateGraph(const ThunkSequence& sequence) {
  std::vector<ThunkUses> ops;
  ops.reserve(sequence.size());
  for (const std::unique_ptr<Thunk>& thunk : sequence) {
    ops.emplace_back(*thunk);
  }
  return ExecutionGraph::Create<ThunkUses>(ops);
}

// Builds `[rng(delta=128), call { rng(delta=256) }]` with disjoint state
// buffers, so only resource uses can order the two RNG thunks.
absl::StatusOr<ThunkSequence> CreateTopLevelAndNestedRng(
    const BufferAllocation& alloc0, const BufferAllocation& alloc1,
    std::shared_ptr<RngStateResource> top_state,
    std::shared_ptr<RngStateResource> nested_state) {
  ThunkSequence sequence;
  ABSL_ASSIGN_OR_RETURN(sequence.emplace_back(),
                   RngGetAndUpdateStateThunk::Create(
                       {"rng0"}, CreateBufferAllocationSlice(alloc0),
                       /*delta=*/128, std::move(top_state)));

  ThunkSequence called;
  ABSL_ASSIGN_OR_RETURN(called.emplace_back(),
                   RngGetAndUpdateStateThunk::Create(
                       {"rng1"}, CreateBufferAllocationSlice(alloc1),
                       /*delta=*/256, std::move(nested_state)));
  ABSL_ASSIGN_OR_RETURN(sequence.emplace_back(),
                   CallThunk::Create({"call"}, std::move(called)));
  return sequence;
}

TEST(RngGetAndUpdateStateThunkTest, SharedResourceOrdersThunkNestedInCall) {
  Literal state0 = LiteralUtil::CreateR1<uint64_t>({0, 0});
  Literal state1 = LiteralUtil::CreateR1<uint64_t>({0, 0});
  auto [alloc0, alloc1] = CreateBufferAllocation(state0, state1);

  auto rng_state = std::make_shared<RngStateResource>();
  ASSERT_OK_AND_ASSIGN(
      ThunkSequence sequence,
      CreateTopLevelAndNestedRng(alloc0, alloc1, rng_state, rng_state));

  EXPECT_THAT(sequence[1]->resource_uses(),
              ElementsAre(ResourceUse::Write(rng_state)));

  ASSERT_OK_AND_ASSIGN(ExecutionGraph graph, CreateGraph(sequence));
  EXPECT_THAT(graph.out_edges(0),
              ElementsAre(NodeEdge{NodeEdge::Kind::kExecution, 1}));
  EXPECT_THAT(graph.in_edges(1),
              ElementsAre(NodeEdge{NodeEdge::Kind::kExecution, 0}));

  // Both thunks advance one counter, so the two outputs differ by the delta of
  // whichever thunk ran first.
  ASSERT_OK_AND_ASSIGN(ThunkExecutor executor,
                       ThunkExecutor::Create(std::move(sequence)));
  BufferAllocations allocations = CreateBufferAllocations(state0, state1);
  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  auto event = executor.Execute(params);
  tsl::BlockUntilReady(event);
  ASSERT_TRUE(event.IsConcrete());
  EXPECT_EQ(state1.data<uint64_t>()[0], state0.data<uint64_t>()[0] + 128);
}

TEST(RngGetAndUpdateStateThunkTest, DistinctResourcesAreNotOrdered) {
  Literal state0 = LiteralUtil::CreateR1<uint64_t>({0, 0});
  Literal state1 = LiteralUtil::CreateR1<uint64_t>({0, 0});
  auto [alloc0, alloc1] = CreateBufferAllocation(state0, state1);

  ASSERT_OK_AND_ASSIGN(ThunkSequence sequence,
                       CreateTopLevelAndNestedRng(
                           alloc0, alloc1, std::make_shared<RngStateResource>(),
                           std::make_shared<RngStateResource>()));

  ASSERT_OK_AND_ASSIGN(ExecutionGraph graph, CreateGraph(sequence));
  EXPECT_THAT(graph.out_edges(0), IsEmpty());
  EXPECT_THAT(graph.in_edges(1), IsEmpty());
  EXPECT_THAT(graph.source(), UnorderedElementsAre(0, 1));
}

}  // namespace
}  // namespace xla::cpu

/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/runtime/execution_graph.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

using NodeEdge = ExecutionGraph::NodeEdge;

// Give aliases to the edge kinds to make tests more readable.
static constexpr auto kExecution = NodeEdge::Kind::kExecution;
static constexpr auto kScheduling = NodeEdge::Kind::kScheduling;

// A test-only operation for verifying execution graph implementation.
class Operation : public ExecutionGraph::Operation {
 public:
  explicit Operation(std::vector<BufferUse> buffers,
                     std::vector<ResourceUse> resources = {})
      : buffers_(std::move(buffers)), resources_(std::move(resources)) {}

  absl::Span<const BufferUse> BufferUses() const final { return buffers_; }

  absl::Span<const ResourceUse> ResourceUses() const final {
    return resources_;
  }

 private:
  std::vector<BufferUse> buffers_;
  std::vector<ResourceUse> resources_;
};

TEST(ExecutionGraphTest, EdgePriority) {
  // Scheduling edge has weaker ordering guarantee than an execution edge.
  EXPECT_LE(kScheduling, kExecution);
}

TEST(ExecutionGraphTest, DependencyOrdering) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);

  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/40);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/40, /*size=*/40);
  BufferAllocation::Slice slice2(&alloc, /*offset=*/20, /*size=*/40);

  std::vector<Operation> operations;
  operations.push_back(
      Operation({BufferUse::Read(slice0), BufferUse::Write(slice0)}));
  operations.push_back(
      Operation({BufferUse::Read(slice1), BufferUse::Write(slice1)}));
  operations.push_back(
      Operation({BufferUse::Read(slice2), BufferUse::Write(slice2)}));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionGraph execution_graph,
                          ExecutionGraph::Create<Operation>(operations));

  EXPECT_FALSE(execution_graph.is_sequential());
  EXPECT_THAT(execution_graph.source(), ElementsAre(0, 1));
  EXPECT_THAT(execution_graph.sink(), ElementsAre(2));

  EXPECT_THAT(execution_graph.out_edges(0),
              ElementsAre(NodeEdge{kExecution, 2}));
  EXPECT_THAT(execution_graph.out_edges(1),
              ElementsAre(NodeEdge{kExecution, 2}));
  EXPECT_THAT(execution_graph.in_edges(2),
              ElementsAre(NodeEdge{kExecution, 0}, NodeEdge{kExecution, 1}));

  EXPECT_EQ(execution_graph.priority(0), 1);
  EXPECT_EQ(execution_graph.priority(1), 1);
  EXPECT_EQ(execution_graph.priority(2), 0);
}

TEST(ExecutionGraphTest, SequentialOrdering) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/40);

  std::vector<Operation> operations;
  operations.push_back(
      Operation({BufferUse::Read(slice), BufferUse::Write(slice)}));
  operations.push_back(
      Operation({BufferUse::Read(slice), BufferUse::Write(slice)}));
  operations.push_back(
      Operation({BufferUse::Read(slice), BufferUse::Write(slice)}));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionGraph execution_graph,
                          ExecutionGraph::Create<Operation>(operations));

  EXPECT_TRUE(execution_graph.is_sequential());
  EXPECT_THAT(execution_graph.source(), ElementsAre(0));
  EXPECT_THAT(execution_graph.sink(), ElementsAre(2));

  EXPECT_THAT(execution_graph.out_edges(0),
              ElementsAre(NodeEdge{kExecution, 1}));
  EXPECT_THAT(execution_graph.out_edges(1),
              ElementsAre(NodeEdge{kExecution, 2}));
  EXPECT_THAT(execution_graph.in_edges(1),
              ElementsAre(NodeEdge{kExecution, 0}));
  EXPECT_THAT(execution_graph.in_edges(2),
              ElementsAre(NodeEdge{kExecution, 1}));

  EXPECT_EQ(execution_graph.priority(0), 2);
  EXPECT_EQ(execution_graph.priority(1), 1);
  EXPECT_EQ(execution_graph.priority(2), 0);
}

TEST(ExecutionGraphTest, TokenResourceOrdering) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);

  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/40);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/40, /*size=*/40);

  auto resource = Resource::Create(Resource::Kind::kToken);

  std::vector<Operation> operations;
  operations.push_back(
      Operation({BufferUse::Read(slice0), BufferUse::Write(slice0)},
                {ResourceUse::Write(resource)}));
  operations.push_back(
      Operation({BufferUse::Read(slice1), BufferUse::Write(slice1)},
                {ResourceUse::Write(resource)}));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionGraph execution_graph,
                          ExecutionGraph::Create<Operation>(operations));

  EXPECT_TRUE(execution_graph.is_sequential());
  EXPECT_THAT(execution_graph.source(), ElementsAre(0));
  EXPECT_THAT(execution_graph.sink(), ElementsAre(1));

  EXPECT_THAT(execution_graph.out_edges(0),
              ElementsAre(NodeEdge{kExecution, 1}));
  EXPECT_THAT(execution_graph.in_edges(1),
              ElementsAre(NodeEdge{kExecution, 0}));

  EXPECT_EQ(execution_graph.priority(0), 1);
  EXPECT_EQ(execution_graph.priority(1), 0);
}

TEST(ExecutionGraphTest, CollectivesResourceOrdering) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);

  BufferAllocation::Slice slice0(&alloc, /*offset=*/0, /*size=*/40);
  BufferAllocation::Slice slice1(&alloc, /*offset=*/40, /*size=*/40);

  auto resource = Resource::Create(Resource::Kind::kCollectiveCommunicator);

  std::vector<Operation> operations;
  operations.push_back(
      Operation({BufferUse::Read(slice0), BufferUse::Write(slice0)},
                {ResourceUse::Write(resource)}));
  operations.push_back(
      Operation({BufferUse::Read(slice1), BufferUse::Write(slice1)},
                {ResourceUse::Write(resource)}));
  operations.push_back(
      Operation({BufferUse::Read(slice1), BufferUse::Write(slice1)},
                {ResourceUse::Write(resource)}));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionGraph execution_graph,
                          ExecutionGraph::Create<Operation>(operations));

  EXPECT_TRUE(execution_graph.is_sequential());
  EXPECT_THAT(execution_graph.source(), ElementsAre(0));
  EXPECT_THAT(execution_graph.sink(), ElementsAre(2));

  EXPECT_THAT(execution_graph.out_edges(0),
              ElementsAre(NodeEdge{kScheduling, 1}));

  EXPECT_THAT(execution_graph.in_edges(1),
              ElementsAre(NodeEdge{kScheduling, 0}));

  // We have buffer conflicts, and a resource conflict, so in this case we
  // must add an execution edge as it provides stronger ordering guarantee
  EXPECT_THAT(execution_graph.out_edges(1),
              ElementsAre(NodeEdge{kExecution, 2}));
  EXPECT_THAT(execution_graph.in_edges(2),
              ElementsAre(NodeEdge{kExecution, 1}));

  EXPECT_EQ(execution_graph.priority(0), 2);
  EXPECT_EQ(execution_graph.priority(1), 1);
  EXPECT_EQ(execution_graph.priority(2), 0);
}

TEST(ExecutionGraphTest, TransitiveReduction) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/40);

  std::vector<Operation> operations;
  operations.push_back(
      Operation({BufferUse::Read(slice), BufferUse::Write(slice)}));
  operations.push_back(
      Operation({BufferUse::Read(slice), BufferUse::Write(slice)}));
  operations.push_back(
      Operation({BufferUse::Read(slice), BufferUse::Write(slice)}));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionGraph execution_graph,
                          ExecutionGraph::Create<Operation>(operations));

  EXPECT_THAT(execution_graph.source(), ElementsAre(0));
  EXPECT_THAT(execution_graph.sink(), ElementsAre(2));

  EXPECT_THAT(execution_graph.out_edges(0),
              ElementsAre(NodeEdge{kExecution, 1}));
  EXPECT_THAT(execution_graph.in_edges(1),
              ElementsAre(NodeEdge{kExecution, 0}));
  EXPECT_THAT(execution_graph.out_edges(1),
              ElementsAre(NodeEdge{kExecution, 2}));
  EXPECT_THAT(execution_graph.in_edges(2),
              ElementsAre(NodeEdge{kExecution, 1}));

  EXPECT_EQ(execution_graph.priority(0), 2);
  EXPECT_EQ(execution_graph.priority(1), 1);
  EXPECT_EQ(execution_graph.priority(2), 0);
}

TEST(ExecutionGraphTest, TransitiveReductionKeepsExecutionEdge) {
  BufferAllocation alloc(/*index=*/0, /*size=*/80, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, /*offset=*/0, /*size=*/40);

  auto resource = Resource::Create(Resource::Kind::kCollectiveCommunicator);

  std::vector<Operation> operations;

  // All three operations connected with scheduling edges, but because execution
  // edge provides stronger ordering guarantee, we must keep an 0-2 execution
  // edge, or we might get a data race.
  operations.push_back(
      Operation({BufferUse::Write(slice)}, {ResourceUse::Write(resource)}));
  operations.push_back(
      Operation(/*buffers=*/{}, {ResourceUse::Write(resource)}));
  operations.push_back(
      Operation({BufferUse::Write(slice)}, {ResourceUse::Write(resource)}));

  TF_ASSERT_OK_AND_ASSIGN(ExecutionGraph execution_graph,
                          ExecutionGraph::Create<Operation>(operations));

  EXPECT_THAT(execution_graph.source(), ElementsAre(0));
  EXPECT_THAT(execution_graph.sink(), ElementsAre(2));

  EXPECT_THAT(execution_graph.out_edges(0),
              ElementsAre(NodeEdge{kScheduling, 1}, NodeEdge{kExecution, 2}));

  EXPECT_THAT(execution_graph.in_edges(1),
              ElementsAre(NodeEdge{kScheduling, 0}));
  EXPECT_THAT(execution_graph.out_edges(1),
              ElementsAre(NodeEdge{kScheduling, 2}));

  EXPECT_THAT(execution_graph.in_edges(2),
              ElementsAre(NodeEdge{kExecution, 0}, NodeEdge{kScheduling, 1}));

  EXPECT_EQ(execution_graph.priority(0), 2);
  EXPECT_EQ(execution_graph.priority(1), 1);
  EXPECT_EQ(execution_graph.priority(2), 0);
}

}  // namespace
}  // namespace xla

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

#include "xla/backends/gpu/runtime/command_executor.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/execution_graph.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {
namespace {

// A minimal Command for testing: configurable buffer uses, no-op Record.
class FakeCmd : public Command {
 public:
  explicit FakeCmd(Command::BufferUses uses = {})
      : Command(CommandType::kUnknownCmd), uses_(std::move(uses)) {}

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams&, const RecordParams&, RecordAction,
      se::CommandBuffer*) override {
    return nullptr;
  }

  BufferUses buffer_uses() const override { return uses_; }

 private:
  BufferUses uses_;
};

// Convenience aliases for synchronization modes.
constexpr auto kSerialize = CommandExecutor::SynchronizationMode::kSerialize;
constexpr auto kConcurrent = CommandExecutor::SynchronizationMode::kConcurrent;
constexpr auto kLHS = CommandExecutor::SynchronizationMode::kLHS;

TEST(CommandExecutorTest, DuplicateAllocsCollapsedToOne) {
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  Shape shape = ShapeUtil::MakeShape(U8, {100});
  auto slice = BufferAllocation::Slice(&alloc0, 0, 100);

  CommandSequence cmds;
  cmds.Emplace<FakeCmd>(Command::BufferUses{BufferUse::Read(slice, shape)});
  cmds.Emplace<FakeCmd>(Command::BufferUses{BufferUse::Write(slice, shape)});
  TF_ASSERT_OK_AND_ASSIGN(auto executor,
                          CommandExecutor::Create(std::move(cmds), kSerialize));

  // Both commands reference the same allocation index — should appear once.
  EXPECT_EQ(executor.allocs_indices().size(), 1);
  EXPECT_EQ(executor.allocs_indices()[0], 0);
}

//===----------------------------------------------------------------------===//
// extra_resources design tests
//===----------------------------------------------------------------------===//

TEST(CommandExecutorTest, CreateWithExtraResourcesConcurrent) {
  auto shared_resource = Resource::Create(Resource::kToken);

  CommandSequence cmds;
  cmds.Emplace<FakeCmd>();
  cmds.Emplace<FakeCmd>();

  // cmd0 writes shared_resource, cmd1 reads it — adds a resource dependency.
  std::vector<Command::ResourceUses> extra = {
      {ResourceUse::Write(shared_resource)},
      {ResourceUse::Read(shared_resource)},
  };
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      CommandExecutor::Create(std::move(cmds), kConcurrent, std::move(extra)));
  EXPECT_EQ(executor.size(), 2);
}

TEST(CommandExecutorTest, CreateWithExtraResourcesLHS) {
  auto shared_resource = Resource::Create(Resource::kToken);

  CommandSequence cmds;
  cmds.Emplace<FakeCmd>();
  cmds.Emplace<FakeCmd>();

  std::vector<Command::ResourceUses> extra = {
      {ResourceUse::Write(shared_resource)},
      {ResourceUse::Read(shared_resource)},
  };
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      CommandExecutor::Create(std::move(cmds), kLHS, std::move(extra)));
  EXPECT_EQ(executor.size(), 2);
}

//===----------------------------------------------------------------------===//
// RenderExecutionGraph failure cases (do not require a registered renderer)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// RenderExecutionGraph tests with a fake registered renderer.
//
// The FakeRenderer is registered once (SetUpTestSuite) and captures the
// operations passed to GenerateGraphAsString so tests can inspect them.
//===----------------------------------------------------------------------===//

// Fake renderer that records per-operation metadata for assertions.
class FakeRenderer : public ExecutionGraph::Renderer {
 public:
  struct CapturedOp {
    std::string name;
    size_t resource_use_count;
    size_t buffer_use_count;
  };

  std::string GenerateGraphAsString(
      absl::Span<const ExecutionGraph::Operation* const> operations) override {
    captured_.clear();
    for (const auto* op : operations) {
      captured_.push_back({std::string(op->name()), op->ResourceUses().size(),
                           op->BufferUses().size()});
    }
    return "fake_graph";
  }

  absl::StatusOr<std::string> PublishGraph(
      absl::string_view graph_as_string) override {
    return std::string(graph_as_string);
  }

  const std::vector<CapturedOp>& captured() const { return captured_; }

 private:
  std::vector<CapturedOp> captured_;
};

class CommandExecutorRendererTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Register a shared FakeRenderer once for this test suite.
    // We keep a raw pointer so tests can inspect captured state.
    fake_renderer_ = new FakeRenderer();
    ExecutionGraph::RegisterRenderer(
        std::unique_ptr<ExecutionGraph::Renderer>(fake_renderer_));
  }

  static FakeRenderer* fake_renderer_;
};

FakeRenderer* CommandExecutorRendererTest::fake_renderer_ = nullptr;

// Basic smoke test: render succeeds and returns the fake URL.
TEST_F(CommandExecutorRendererTest, RenderSucceeds) {
  CommandSequence cmds;
  cmds.Emplace<FakeCmd>();
  cmds.Emplace<FakeCmd>();
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor, CommandExecutor::Create(std::move(cmds), kConcurrent));

  TF_ASSERT_OK_AND_ASSIGN(auto url, executor.RenderExecutionGraph());
  EXPECT_EQ(url, "fake_graph");
  EXPECT_EQ(fake_renderer_->captured().size(), 2);
}

// Without extra_resources, each CommandOperation has exactly one resource use:
// the Write(cmd->token()) that is always added by the CommandOperation ctor.
TEST_F(CommandExecutorRendererTest, RenderWithoutExtraResourcesHasOneResource) {
  CommandSequence cmds;
  cmds.Emplace<FakeCmd>();
  cmds.Emplace<FakeCmd>();
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor, CommandExecutor::Create(std::move(cmds), kConcurrent));

  TF_ASSERT_OK(executor.RenderExecutionGraph().status());
  ASSERT_EQ(fake_renderer_->captured().size(), 2);
  EXPECT_EQ(fake_renderer_->captured()[0].resource_use_count, 1);
  EXPECT_EQ(fake_renderer_->captured()[1].resource_use_count, 1);
}

// With extra_resources, RenderExecutionGraph() must use extra_resources_
// (not empty {}) when calling CreateCommandOperations. Each CommandOperation
// should show Write(token) + the extra resource = 2 total.
TEST_F(CommandExecutorRendererTest, RenderUsesStoredExtraResources) {
  auto shared_resource = Resource::Create(Resource::kToken);

  CommandSequence cmds;
  cmds.Emplace<FakeCmd>();
  cmds.Emplace<FakeCmd>();

  // cmd0 writes shared_resource; cmd1 reads it.
  std::vector<Command::ResourceUses> extra = {
      {ResourceUse::Write(shared_resource)},
      {ResourceUse::Read(shared_resource)},
  };
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      CommandExecutor::Create(std::move(cmds), kConcurrent, std::move(extra)));

  TF_ASSERT_OK_AND_ASSIGN(auto url, executor.RenderExecutionGraph());
  EXPECT_EQ(url, "fake_graph");

  ASSERT_EQ(fake_renderer_->captured().size(), 2);
  // cmd0: Write(cmd0->token()) + Write(shared_resource) = 2
  EXPECT_EQ(fake_renderer_->captured()[0].resource_use_count, 2);
  // cmd1: Write(cmd1->token()) + Read(shared_resource) = 2
  EXPECT_EQ(fake_renderer_->captured()[1].resource_use_count, 2);
}

// Same verification for kLHS mode — extra_resources must be passed through.
TEST_F(CommandExecutorRendererTest, RenderUsesStoredExtraResourcesLHSMode) {
  auto shared_resource = Resource::Create(Resource::kToken);

  CommandSequence cmds;
  cmds.Emplace<FakeCmd>();
  cmds.Emplace<FakeCmd>();

  std::vector<Command::ResourceUses> extra = {
      {ResourceUse::Write(shared_resource)},
      {ResourceUse::Read(shared_resource)},
  };
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      CommandExecutor::Create(std::move(cmds), kLHS, std::move(extra)));

  TF_ASSERT_OK(executor.RenderExecutionGraph().status());
  ASSERT_EQ(fake_renderer_->captured().size(), 2);
  // cmd0: Write(cmd0->token()) + Write(shared_resource) = 2.
  EXPECT_EQ(fake_renderer_->captured()[0].resource_use_count, 2);
  // cmd1: Write(cmd1->token()) + Read(shared_resource) [extra]
  //       + Read(cmd0->token()) [lhs_extras: sequential dependency] = 3.
  EXPECT_EQ(fake_renderer_->captured()[1].resource_use_count, 3);
}

// Calling RenderExecutionGraph twice produces consistent results.
TEST_F(CommandExecutorRendererTest, RenderIsIdempotent) {
  auto shared_resource = Resource::Create(Resource::kToken);

  CommandSequence cmds;
  cmds.Emplace<FakeCmd>();
  cmds.Emplace<FakeCmd>();

  std::vector<Command::ResourceUses> extra = {
      {ResourceUse::Write(shared_resource)},
      {ResourceUse::Read(shared_resource)},
  };
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      CommandExecutor::Create(std::move(cmds), kConcurrent, std::move(extra)));

  TF_ASSERT_OK_AND_ASSIGN(auto url1, executor.RenderExecutionGraph());
  auto captured1 = fake_renderer_->captured();

  TF_ASSERT_OK_AND_ASSIGN(auto url2, executor.RenderExecutionGraph());
  auto captured2 = fake_renderer_->captured();

  EXPECT_EQ(url1, url2);
  ASSERT_EQ(captured1.size(), captured2.size());
  for (size_t i = 0; i < captured1.size(); ++i) {
    EXPECT_EQ(captured1[i].resource_use_count, captured2[i].resource_use_count);
  }
}

}  // namespace
}  // namespace xla::gpu

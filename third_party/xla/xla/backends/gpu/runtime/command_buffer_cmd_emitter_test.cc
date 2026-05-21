/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"

#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/runtime/execution_graph.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;

MATCHER_P(HasEdgeTo, node_id, "") { return arg.id == node_id; }

class FakeKernelThunk : public KernelThunk {
 public:
  FakeKernelThunk(Thunk::ThunkInfo thunk_info, BufferAllocation::Slice slice)
      : KernelThunk(thunk_info,
                    /*kernel_name=*/"kernel",
                    /*kernel_arguments=*/BuildKernelArguments(slice),
                    /*launch_dimensions=*/LaunchDimensions(),
                    /*cluster_dim=*/se::ClusterDim(),
                    /*shmem_bytes=*/0,
                    /*tma_metadata=*/se::gpu::TmaMetadata(),
                    /*zeroed_output_buffer_indices=*/std::vector<int64_t>{}) {}

 private:
  static emitters::KernelArguments BuildKernelArguments(
      BufferAllocation::Slice slice) {
    emitters::KernelArgument arg(ShapeUtil::MakeShape(F32, {256}), slice);
    arg.set_written(true);
    return emitters::KernelArguments(
        std::vector<emitters::KernelArgument>{arg});
  }
};

class CommandBufferCmdEmitterTest : public testing::Test {
 public:
  Thunk::ThunkInfo NextThunkInfo(absl::string_view profile_annotation) {
    Thunk::ThunkInfo thunk_info;
    thunk_info.thunk_id = thunk_id_generator_.GetNextThunkId();
    thunk_info.profile_annotation = profile_annotation;
    return thunk_info;
  }

  std::map<std::string, ExecutionGraph::NodeId> NamesToNodeIds(
      CommandExecutor& commands) {
    std::map<std::string, ExecutionGraph::NodeId> name_to_node_id;
    ExecutionGraph::NodeId node_id = 0;
    CHECK_OK(commands.Walk([&name_to_node_id, &node_id](Command* command) {
      name_to_node_id[std::string(command->profile_annotation())] = node_id++;
      return absl::OkStatus();
    }));
    return name_to_node_id;
  }

 private:
  ThunkIdGenerator thunk_id_generator_;
};

TEST_F(CommandBufferCmdEmitterTest, ConcurrentAndSequentialExecutionGraphs) {
  BufferAllocation allocation(/*index=*/0, /*size=*/2048, /*color=*/0);

  ThunkSequence thunks;
  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("a"), slice_a));
  BufferAllocation::Slice slice_b(&allocation, /*offset=*/1024, /*size=*/2048);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("b"), slice_b));

  ConvertToCommandsOptions options;
  {
    // Synchronization mode kConcurrent creates a concurrent execution graph.
    options.synchronization_mode =
        CommandExecutor::SynchronizationMode::kConcurrent;
    TF_ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                            ConvertToCommands(thunks, options));
    EXPECT_FALSE(commands.execution_graph()->is_sequential());
  }

  {
    // Synchronization mode kConcurrentRegions creates a sequential execution
    // graph when all thunks are in different concurrent regions.
    thunks[0]->set_concurrent_region_id(42);
    thunks[1]->set_concurrent_region_id(43);
    options.synchronization_mode =
        CommandExecutor::SynchronizationMode::kConcurrentRegions;
    TF_ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                            ConvertToCommands(thunks, options));
    EXPECT_TRUE(commands.execution_graph()->is_sequential());
  }

  {
    // Synchronization mode kConcurrentRegions creates a concurrent execution
    // graph when thunks are in the same regions.
    thunks[0]->set_concurrent_region_id(44);
    thunks[1]->set_concurrent_region_id(44);
    options.synchronization_mode =
        CommandExecutor::SynchronizationMode::kConcurrentRegions;
    TF_ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                            ConvertToCommands(thunks, options));
    EXPECT_FALSE(commands.execution_graph()->is_sequential());
  }
}

TEST_F(CommandBufferCmdEmitterTest,
       ConcurrentRegionsLimitesConcurrencyToTwoChains) {
  BufferAllocation allocation(/*index=*/0, /*size=*/3 * 1024, /*color=*/0);
  ThunkSequence thunks;

  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("a"), slice_a));

  BufferAllocation::Slice slice_b(&allocation, /*offset=*/2 * 1024,
                                  /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("b"), slice_b));

  BufferAllocation::Slice slice_c(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("c"), slice_c));

  BufferAllocation::Slice slice_d(&allocation, /*offset=*/1024, /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("d"), slice_d));

  ConvertToCommandsOptions options;
  // In kConcurrent mode, there are 3 source nodes (a, b, d) and 3 sink nodes
  // (c, b, d).
  {
    options.synchronization_mode =
        CommandExecutor::SynchronizationMode::kConcurrent;
    TF_ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                            ConvertToCommands(thunks, options));
    EXPECT_FALSE(commands.execution_graph()->is_sequential());
    EXPECT_EQ(commands.execution_graph()->nodes_defs().size(), 4);
    EXPECT_EQ(commands.execution_graph()->source().size(), 3);
    EXPECT_EQ(commands.execution_graph()->sink().size(), 3);
  }

  // In kConcurrentRegions mode and artificial dependencies between b and d is
  // inserted, hence there are 2 source nodes (a, b) and 2 sink nodes (c, d).
  {
    options.synchronization_mode =
        CommandExecutor::SynchronizationMode::kConcurrentRegions;
    for (auto& thunk : thunks) {
      thunk->set_concurrent_region_id(44);
    }
    TF_ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                            ConvertToCommands(thunks, options));
    EXPECT_FALSE(commands.execution_graph()->is_sequential());
    EXPECT_EQ(commands.execution_graph()->nodes_defs().size(), 4);
    EXPECT_EQ(commands.execution_graph()->source().size(), 2);
    EXPECT_EQ(commands.execution_graph()->sink().size(), 2);
    auto node_id = NamesToNodeIds(commands);

    EXPECT_EQ(node_id["a"], 0);
    EXPECT_EQ(node_id["b"], 1);
    EXPECT_EQ(node_id["c"], 2);
    EXPECT_EQ(node_id["d"], 3);

    EXPECT_EQ(
        commands.execution_graph()->nodes_defs()[node_id["a"]].out_edges[0].id,
        node_id["c"]);
    EXPECT_EQ(
        commands.execution_graph()->nodes_defs()[node_id["b"]].out_edges[0].id,
        node_id["d"]);
  }
}

TEST_F(CommandBufferCmdEmitterTest, ConcurrentRegionsExecutesOutOfOrder) {
  BufferAllocation allocation(/*index=*/0, /*size=*/2 * 1024, /*color=*/0);
  ThunkSequence thunks;

  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("a"), slice_a));

  BufferAllocation::Slice slice_b(&allocation, /*offset=*/0,
                                  /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("b"), slice_b));

  BufferAllocation::Slice slice_c(&allocation, /*offset=*/1024, /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("c"), slice_c));

  BufferAllocation::Slice slice_d(&allocation, /*offset=*/1024, /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("d"), slice_d));

  ConvertToCommandsOptions options;
  options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrentRegions;
  for (auto& thunk : thunks) {
    thunk->set_concurrent_region_id(44);
  }
  TF_ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                          ConvertToCommands(thunks, options));
  EXPECT_FALSE(commands.execution_graph()->is_sequential());
  EXPECT_EQ(commands.execution_graph()->nodes_defs().size(), 4);
  EXPECT_EQ(commands.execution_graph()->source().size(), 2);
  EXPECT_EQ(commands.execution_graph()->sink().size(), 2);
  auto node_id = NamesToNodeIds(commands);

  // In the ThunkSequence the order is a, b, c, d. In the execution graph (a, c)
  // and (b, d) are executed concurrently. Hence, node c has a lower id than
  // node b.
  EXPECT_EQ(node_id["a"], 0);
  EXPECT_EQ(node_id["b"], 2);
  EXPECT_EQ(node_id["c"], 1);
  EXPECT_EQ(node_id["d"], 3);

  EXPECT_EQ(
      commands.execution_graph()->nodes_defs()[node_id["a"]].out_edges[0].id,
      node_id["b"]);
  EXPECT_EQ(
      commands.execution_graph()->nodes_defs()[node_id["c"]].out_edges[0].id,
      node_id["d"]);
}

TEST_F(CommandBufferCmdEmitterTest,
       ConcurrentRegionsAddsDependenciesBetweenRegions) {
  BufferAllocation allocation(/*index=*/0, /*size=*/5 * 1024, /*color=*/0);
  ThunkSequence thunks;

  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("a"), slice_a));
  thunks.back()->set_concurrent_region_id(42);

  BufferAllocation::Slice slice_b(&allocation, /*offset=*/1024, /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("b"), slice_b));
  thunks.back()->set_concurrent_region_id(42);

  BufferAllocation::Slice slice_c(&allocation, /*offset=*/2 * 1024,
                                  /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("c"), slice_c));
  thunks.back()->set_concurrent_region_id(43);

  BufferAllocation::Slice slice_d(&allocation, /*offset=*/3 * 1024,
                                  /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("d"), slice_d));
  thunks.back()->set_concurrent_region_id(43);

  BufferAllocation::Slice slice_e(&allocation, /*offset=*/4 * 1024,
                                  /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("e"), slice_e));
  thunks.back()->set_concurrent_region_id(44);

  ConvertToCommandsOptions options;
  options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrentRegions;
  TF_ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                          ConvertToCommands(thunks, options));
  // The execution graph contains dependencies between nodes in concurrent
  // regions: (a, b), (c, d) and (e).
  EXPECT_FALSE(commands.execution_graph()->is_sequential());
  auto node_id = NamesToNodeIds(commands);
  EXPECT_THAT(
      commands.execution_graph()->nodes_defs()[node_id["a"]].out_edges,
      UnorderedElementsAre(HasEdgeTo(node_id["c"]), HasEdgeTo(node_id["d"])));
  EXPECT_THAT(
      commands.execution_graph()->nodes_defs()[node_id["b"]].out_edges,
      UnorderedElementsAre(HasEdgeTo(node_id["c"]), HasEdgeTo(node_id["d"])));
  EXPECT_THAT(commands.execution_graph()->nodes_defs()[node_id["c"]].out_edges,
              UnorderedElementsAre(HasEdgeTo(node_id["e"])));
  EXPECT_THAT(commands.execution_graph()->nodes_defs()[node_id["d"]].out_edges,
              UnorderedElementsAre(HasEdgeTo(node_id["e"])));
}

TEST_F(CommandBufferCmdEmitterTest,
       ConcurrentRegionsAddsDependenciesBetweenRegionsDifferenLaneLength) {
  BufferAllocation allocation(/*index=*/0, /*size=*/5 * 1024, /*color=*/0);
  ThunkSequence thunks;

  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("a"), slice_a));
  thunks.back()->set_concurrent_region_id(42);

  BufferAllocation::Slice slice_b(&allocation, /*offset=*/1024, /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("b"), slice_b));
  thunks.back()->set_concurrent_region_id(42);

  BufferAllocation::Slice slice_c(&allocation, /*offset=*/2 * 1024,
                                  /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("c"), slice_c));
  thunks.back()->set_concurrent_region_id(42);

  BufferAllocation::Slice slice_d(&allocation, /*offset=*/3 * 1024,
                                  /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("d"), slice_d));
  thunks.back()->set_concurrent_region_id(43);

  BufferAllocation::Slice slice_e(&allocation, /*offset=*/4 * 1024,
                                  /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("e"), slice_e));
  thunks.back()->set_concurrent_region_id(43);

  ConvertToCommandsOptions options;
  options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrentRegions;
  TF_ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                          ConvertToCommands(thunks, options));
  // The execution graph contains dependencies between nodes in concurrent
  // regions: (a, b, c), (d, e).
  EXPECT_FALSE(commands.execution_graph()->is_sequential());
  auto node_id = NamesToNodeIds(commands);
  EXPECT_THAT(
      commands.execution_graph()->nodes_defs()[node_id["b"]].out_edges,
      UnorderedElementsAre(HasEdgeTo(node_id["d"]), HasEdgeTo(node_id["e"])));
  EXPECT_THAT(
      commands.execution_graph()->nodes_defs()[node_id["c"]].out_edges,
      UnorderedElementsAre(HasEdgeTo(node_id["d"]), HasEdgeTo(node_id["e"])));
}

TEST_F(CommandBufferCmdEmitterTest, ConcurrentRegionsScheduleHasLaneAffinity) {
  BufferAllocation allocation(/*index=*/0, /*size=*/3 * 1024, /*color=*/0);
  ThunkSequence thunks;

  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("a"), slice_a));

  BufferAllocation::Slice slice_b(&allocation, /*offset=*/1024, /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("b"), slice_b));

  BufferAllocation::Slice slice_c(&allocation, /*offset=*/2 * 1024,
                                  /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("c"), slice_c));

  BufferAllocation::Slice slice_d(&allocation, /*offset=*/3 * 1024,
                                  /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("d"), slice_d));

  BufferAllocation::Slice slice_e(&allocation, /*offset=*/0, /*size=*/1024);
  thunks.push_back(
      std::make_unique<FakeKernelThunk>(NextThunkInfo("e"), slice_e));
  for (auto& thunk : thunks) {
    thunk->set_concurrent_region_id(44);
  }

  ConvertToCommandsOptions options;
  options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrentRegions;
  TF_ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                          ConvertToCommands(thunks, options));
  auto node_id = NamesToNodeIds(commands);
  EXPECT_EQ(node_id["a"], 0);
  EXPECT_EQ(node_id["b"], 1);
  // Nodes c and d are ready from the start. However, node e is scheduled ahead
  // of them, because it has a data dependency on a.
  EXPECT_EQ(node_id["e"], 2);

  // Nade e depends on a only and should be scheduled on the same lane as a.
  EXPECT_EQ(
      commands.execution_graph()->nodes_defs()[node_id["a"]].out_edges.size(),
      1);
  EXPECT_EQ(
      commands.execution_graph()->nodes_defs()[node_id["a"]].out_edges[0].id,
      node_id["e"]);
}

TEST_F(CommandBufferCmdEmitterTest, ConvertsConditionalThunkToCommand) {
  BufferAllocation branch_index_alloc(/*index=*/0, /*size=*/sizeof(int32_t),
                                      /*color=*/0);
  BufferAllocation data_alloc(/*index=*/1, /*size=*/2 * 1024, /*color=*/0);

  BufferAllocation::Slice branch_index_slice(&branch_index_alloc, /*offset=*/0,
                                             /*size=*/sizeof(int32_t));
  BufferAllocation::Slice branch0_slice(&data_alloc, /*offset=*/0,
                                        /*size=*/1024);
  BufferAllocation::Slice branch1_slice(&data_alloc, /*offset=*/1024,
                                        /*size=*/1024);

  ThunkSequence branch0;
  branch0.push_back(std::make_unique<FakeKernelThunk>(NextThunkInfo("branch0"),
                                                      branch0_slice));

  ThunkSequence branch1;
  branch1.push_back(std::make_unique<FakeKernelThunk>(NextThunkInfo("branch1"),
                                                      branch1_slice));

  std::vector<ThunkSequence> branches;
  branches.push_back(std::move(branch0));
  branches.push_back(std::move(branch1));

  auto conditional = std::make_unique<ConditionalThunk>(
      NextThunkInfo("conditional"),
      ShapedSlice{branch_index_slice, ShapeUtil::MakeShape(S32, {})},
      std::move(branches));
  ConditionalThunk* conditional_ptr = conditional.get();

  ThunkSequence thunks;
  thunks.push_back(std::move(conditional));

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, ConvertToCommandsOptions()));

  EXPECT_EQ(conditional_ptr->command_type(), CommandType::kCaseCmd);

  std::vector<std::string> command_names;
  CHECK_OK(commands.Walk([&](Command* command) {
    command_names.push_back(std::string(command->profile_annotation()));
    return absl::OkStatus();
  }));
  EXPECT_THAT(command_names, ElementsAre("conditional", "branch0", "branch1"));
}

TEST_F(CommandBufferCmdEmitterTest, ConvertsConditionalThunkRepeatedly) {
  BufferAllocation branch_index_alloc(/*index=*/0, /*size=*/sizeof(int32_t),
                                      /*color=*/0);
  BufferAllocation data_alloc(/*index=*/1, /*size=*/2 * 1024, /*color=*/0);

  BufferAllocation::Slice branch_index_slice(&branch_index_alloc, /*offset=*/0,
                                             /*size=*/sizeof(int32_t));
  BufferAllocation::Slice branch0_slice(&data_alloc, /*offset=*/0,
                                        /*size=*/1024);
  BufferAllocation::Slice branch1_slice(&data_alloc, /*offset=*/1024,
                                        /*size=*/1024);

  ThunkSequence branch0;
  branch0.push_back(std::make_unique<FakeKernelThunk>(NextThunkInfo("branch0"),
                                                      branch0_slice));

  ThunkSequence branch1;
  branch1.push_back(std::make_unique<FakeKernelThunk>(NextThunkInfo("branch1"),
                                                      branch1_slice));

  std::vector<ThunkSequence> branches;
  branches.push_back(std::move(branch0));
  branches.push_back(std::move(branch1));

  ThunkSequence thunks;
  thunks.push_back(std::make_unique<ConditionalThunk>(
      NextThunkInfo("conditional"),
      ShapedSlice{branch_index_slice, ShapeUtil::MakeShape(S32, {})},
      std::move(branches)));

  auto collect_command_names = [](CommandExecutor& commands) {
    std::vector<std::string> command_names;
    CHECK_OK(commands.Walk([&](Command* command) {
      command_names.push_back(std::string(command->profile_annotation()));
      return absl::OkStatus();
    }));
    return command_names;
  };

  ASSERT_OK_AND_ASSIGN(CommandExecutor first_commands,
                       ConvertToCommands(thunks, ConvertToCommandsOptions()));
  EXPECT_THAT(collect_command_names(first_commands),
              ElementsAre("conditional", "branch0", "branch1"));

  ConvertToCommandsOptions concurrent_options;
  concurrent_options.synchronization_mode =
      CommandExecutor::SynchronizationMode::kConcurrent;
  ASSERT_OK_AND_ASSIGN(CommandExecutor second_commands,
                       ConvertToCommands(thunks, concurrent_options));
  ASSERT_TRUE(second_commands.execution_graph().has_value());
  EXPECT_THAT(collect_command_names(second_commands),
              ElementsAre("conditional", "branch0", "branch1"));
}

TEST_F(CommandBufferCmdEmitterTest,
       ConvertsBoolConditionalBranchesInCaseOrder) {
  BufferAllocation branch_index_alloc(/*index=*/0, /*size=*/sizeof(bool),
                                      /*color=*/0);
  BufferAllocation data_alloc(/*index=*/1, /*size=*/2 * 1024, /*color=*/0);

  BufferAllocation::Slice branch_index_slice(&branch_index_alloc, /*offset=*/0,
                                             /*size=*/sizeof(bool));
  BufferAllocation::Slice false_slice(&data_alloc, /*offset=*/0, /*size=*/1024);
  BufferAllocation::Slice true_slice(&data_alloc, /*offset=*/1024,
                                     /*size=*/1024);

  ThunkSequence false_branch;
  false_branch.push_back(std::make_unique<FakeKernelThunk>(
      NextThunkInfo("false_branch"), false_slice));

  ThunkSequence true_branch;
  true_branch.push_back(std::make_unique<FakeKernelThunk>(
      NextThunkInfo("true_branch"), true_slice));

  std::vector<ThunkSequence> branches;
  branches.push_back(std::move(false_branch));
  branches.push_back(std::move(true_branch));

  ThunkSequence thunks;
  thunks.push_back(std::make_unique<ConditionalThunk>(
      NextThunkInfo("conditional"),
      ShapedSlice{branch_index_slice, ShapeUtil::MakeShape(PRED, {})},
      std::move(branches)));

  ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                       ConvertToCommands(thunks, ConvertToCommandsOptions()));

  std::vector<std::string> command_names;
  CHECK_OK(commands.Walk([&](Command* command) {
    command_names.push_back(std::string(command->profile_annotation()));
    return absl::OkStatus();
  }));
  EXPECT_THAT(command_names,
              ElementsAre("conditional", "true_branch", "false_branch"));
}

}  // namespace xla::gpu

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

// Tests for the cordoned-off pre-CUDA-13.4 (CUDA 13.2-13.3) executable graph
// telemetry projection in `cuda_command_buffer_telemetry_pre_13_4`.

#include "xla/stream_executor/cuda/cuda_command_buffer_telemetry_pre_13_4.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_command_buffer.h"
#include "xla/stream_executor/cuda/cuda_command_buffer_telemetry_test_util.h"
#include "xla/stream_executor/cuda/cuda_graph_annotation_registry.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/scoped_command_buffer_annotation.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

class CudaCommandBufferTestPeer {
 public:
  static CUgraph graph(CudaCommandBuffer* cmd) { return cmd->graph_; }
  static CUgraphExec graph_exec(CudaCommandBuffer* cmd) {
    return cmd->graph_exec_;
  }
};

namespace {

using ::testing::Eq;
using ::testing::Optional;
using Command = CommandBuffer::Command;
using KeySpace = CudaGraphAnnotationRegistry::KeySpace;
using Scope = ScopedCommandBufferAnnotation;
using TelemetryStrategy = CudaGraphAnnotationRegistry::TelemetryStrategy;
using TrackMode = CudaGraphAnnotationRegistry::TrackMode;

uint64_t ExecNodeId(uint32_t exec_id, uint32_t flat_local_id) {
  return (uint64_t{exec_id} << 32) | flat_local_id;
}

std::optional<std::string> LookupExec(uint32_t exec_id,
                                      uint32_t flat_local_id) {
  return Lookup(NodeKey{exec_id, ExecNodeId(exec_id, flat_local_id)},
                KeySpace::kPre134ExecGraph);
}

TEST_F(CudaCommandBufferTelemetryTest, FinalizeProjectsExecGraphOnPre134) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> primary, NewPrimary());
  auto module_scope = Scope::Root(/*enabled=*/true, "XlaModule:#module=test#");
  {
    Scope op_scope("fusion.1:#tf_op=add#");
    ASSERT_OK_AND_ASSIGN(const Command* cmd, Launch(*primary));
    (void)cmd;
  }
  ASSERT_OK(primary->Finalize());

  auto* cuda_cmd = absl::down_cast<CudaCommandBuffer*>(primary.get());
  const std::optional<uint32_t> maybe_exec_id =
      command_buffer_telemetry::GetGraphExecId(
          CudaCommandBufferTestPeer::graph_exec(cuda_cmd));
  ASSERT_TRUE(maybe_exec_id.has_value());
  const uint32_t exec_id = *maybe_exec_id;

  const DeviceDescription& desc = executor_->GetDeviceDescription();
  if (CudaGraphAnnotationRegistry::SelectTelemetryStrategy(
          desc.compile_time_toolkit_version(), desc.driver_version(),
          desc.runtime_version()) == TelemetryStrategy::kPre134ExecGraph) {
    EXPECT_THAT(LookupExec(exec_id, 0),
                Optional(Eq("XlaModule:#module=test#::fusion.1:#tf_op=add#")));
  } else {
    EXPECT_EQ(LookupExec(exec_id, 0), std::nullopt);
  }
}

TEST_F(CudaCommandBufferTelemetryTest,
       RegisterPre134ExecGraphNodesProjectsHierarchyAndCleansUpOnDestroy) {
  DeviceAddress<bool> pred = executor_->AllocateArray<bool>(1);
  ASSERT_FALSE(pred.is_null());
  absl::Cleanup free_pred = [&] { executor_->Deallocate(&pred); };

  uint32_t exec_id = 0;
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> primary, NewPrimary());
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> inner_clone,
                         NewNested());
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> nested_clone,
                         NewNested());

    auto module_scope =
        Scope::Root(/*enabled=*/true, "XlaModule:#module=test#");

    const Command* kernel_cmd = nullptr;
    {
      Scope op_scope("fusion.1:#tf_op=add#");
      ASSERT_OK_AND_ASSIGN(kernel_cmd, Launch(*primary));
    }
    const Command* memset_cmd = nullptr;
    {
      Scope op_scope("memset.1:#tf_op=zero#");
      ASSERT_OK_AND_ASSIGN(
          memset_cmd, primary->CreateMemset(&a_, uint32_t{0}, 4, {kernel_cmd}));
    }
    const Command* memcpy_cmd = nullptr;
    {
      Scope op_scope("copy.1:#tf_op=copy#");
      ASSERT_OK_AND_ASSIGN(
          memcpy_cmd,
          primary->CreateMemcpyD2D(&b_, a_, 4 * sizeof(int32_t), {memset_cmd}));
    }
    const Command* inner_k = nullptr;
    {
      Scope op_scope("inner_child:#tf_op=conv#");
      ASSERT_OK_AND_ASSIGN(inner_k, Launch(*inner_clone));
    }
    const Command* cloned_child_cmd = nullptr;
    {
      Scope op_scope("traced_child:#tf_op=gemm#");
      ASSERT_OK_AND_ASSIGN(const Command* child_k, Launch(*nested_clone));
      ASSERT_OK_AND_ASSIGN(
          const Command* grand_child_cmd,
          nested_clone->CreateChildCommand(*inner_clone, {child_k}));
      (void)grand_child_cmd;
      ASSERT_OK_AND_ASSIGN(cloned_child_cmd, primary->CreateChildCommand(
                                                 *nested_clone, {memcpy_cmd}));
    }
    {
      Scope op_scope("while_op:#tf_op=while#");
      auto create_cond = [](CommandBuffer*,
                            absl::Span<const Command* const> deps)
          -> absl::StatusOr<std::vector<const Command*>> {
        return std::vector<const Command*>(deps.begin(), deps.end());
      };
      auto create_body = [&](CommandBuffer* body,
                             absl::Span<const Command* const> deps)
          -> absl::StatusOr<std::vector<const Command*>> {
        ABSL_ASSIGN_OR_RETURN(const Command* body_kernel, Launch(*body, deps));
        return std::vector<const Command*>{body_kernel};
      };
      ASSERT_OK(
          primary
              ->CreateWhile(pred, create_cond, create_body, {cloned_child_cmd})
              .status());
    }

    ASSERT_OK(primary->Finalize());

    auto* cuda_cmd = absl::down_cast<CudaCommandBuffer*>(primary.get());
    const std::optional<uint32_t> maybe_exec_id =
        command_buffer_telemetry::GetGraphExecId(
            CudaCommandBufferTestPeer::graph_exec(cuda_cmd));
    ASSERT_TRUE(maybe_exec_id.has_value());
    exec_id = *maybe_exec_id;

    // Also track a template graph with the same 32-bit integer ID `exec_id` in
    // KeySpace::kTemplateGraph to verify RegisterPre134ExecGraphNodes does not
    // clobber template entries when `cuGraphExecGetId` collides with
    // `cuGraphGetId`. Skipped if one of this test's own graphs already has that
    // ID: `kNewGraph` would retire its live entry.
    const bool force_collision =
        !registry().IsGraphTracked(exec_id, KeySpace::kTemplateGraph);
    if (force_collision) {
      const CudaGraphAnnotationRegistry::TrackResult colliding_tmpl =
          registry().TrackGraph(exec_id, TrackMode::kNewGraph,
                                KeySpace::kTemplateGraph);
      ASSERT_TRUE(registry().RegisterNodeAnnotation(
          exec_id, colliding_tmpl.token, /*node_id=*/42,
          "colliding_template_op", KeySpace::kTemplateGraph));
    }

    // Invoke the pre-13.4 projection directly so it is exercised even when
    // the test runs on a CUDA 13.4+ driver.
    command_buffer_telemetry::RegisterPre134ExecGraphNodes(
        CudaCommandBufferTestPeer::graph(cuda_cmd),
        CudaCommandBufferTestPeer::graph_exec(cuda_cmd));

    // The colliding template entry must remain live and untouched.
    if (force_collision) {
      EXPECT_THAT(Lookup(NodeKey{exec_id, 42}, KeySpace::kTemplateGraph),
                  Optional(Eq("colliding_template_op")));
    }

    // 0-based two-pass flat local IDs:
    //   Root graph [0..5]:
    //     0: kernel_cmd
    //     1: memset_cmd
    //     2: memcpy_cmd
    //     3: cloned_child_cmd (container node -> unannotated)
    //     4: CreateWhile init-condition kernel
    //     5: CreateWhile conditional node (container node -> unannotated)
    //   Subgraphs in DFS pre-order [6..10]:
    //     6: embedded child kernel inside cloned_child_cmd
    //     7: grand_child_cmd inside cloned_child_cmd (container -> unannotated)
    //     8: grand-embedded kernel inside grand_child_cmd (copies inner_child!)
    //     9: CreateWhile body kernel
    //     10: CreateWhile body-condition kernel
    EXPECT_THAT(LookupExec(exec_id, 0),
                Optional(Eq("XlaModule:#module=test#::fusion.1:#tf_op=add#")));
    EXPECT_THAT(LookupExec(exec_id, 1),
                Optional(Eq("XlaModule:#module=test#::memset.1:#tf_op=zero#")));
    EXPECT_THAT(LookupExec(exec_id, 2),
                Optional(Eq("XlaModule:#module=test#::copy.1:#tf_op=copy#")));
    EXPECT_EQ(LookupExec(exec_id, 3), std::nullopt);
    EXPECT_THAT(
        LookupExec(exec_id, 4),
        Optional(Eq("XlaModule:#module=test#::while_op:#tf_op=while#")));
    EXPECT_EQ(LookupExec(exec_id, 5), std::nullopt);
    EXPECT_THAT(
        LookupExec(exec_id, 6),
        Optional(Eq("XlaModule:#module=test#::traced_child:#tf_op=gemm#")));
    EXPECT_EQ(LookupExec(exec_id, 7), std::nullopt);
    EXPECT_THAT(
        LookupExec(exec_id, 8),
        Optional(Eq("XlaModule:#module=test#::inner_child:#tf_op=conv#")));
    EXPECT_THAT(
        LookupExec(exec_id, 9),
        Optional(Eq("XlaModule:#module=test#::while_op:#tf_op=while#")));
    EXPECT_THAT(
        LookupExec(exec_id, 10),
        Optional(Eq("XlaModule:#module=test#::while_op:#tf_op=while#")));
  }

  // `~CudaCommandBuffer()` destroys `CUgraphExec` before `CUgraph`, releasing
  // the last reference to the user object attached to `root_graph` and removing
  // the `exec_id` entry from `KeySpace::kPre134ExecGraph` while leaving the
  // colliding `KeySpace::kTemplateGraph` entry intact.
  EXPECT_TRUE(PollUntil([&] { return !LookupExec(exec_id, 0).has_value(); }));
  EXPECT_THAT(Lookup(NodeKey{exec_id, 42}, KeySpace::kTemplateGraph),
              Optional(Eq("colliding_template_op")));
}

}  // namespace
}  // namespace stream_executor::gpu

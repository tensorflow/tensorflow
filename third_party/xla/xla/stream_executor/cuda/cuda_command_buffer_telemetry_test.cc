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

// Tests of the CUDA 13.2+ command buffer telemetry producer: which graph nodes
// `CudaCommandBuffer` registers in `CudaGraphAnnotationRegistry`, with
// which annotations, and when the entries go away.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_command_buffer_telemetry_test_util.h"
#include "xla/stream_executor/cuda/cuda_graph_annotation_registry.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/scoped_command_buffer_annotation.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"

namespace stream_executor::gpu {
namespace {

using ::testing::Eq;
using ::testing::Optional;
using Command = CommandBuffer::Command;
using Scope = ScopedCommandBufferAnnotation;

TEST_F(CudaCommandBufferTelemetryTest,
       CudaGraphAnnotationRegistryPopulatedAndCleanedUp) {
  NodeKey kernel_key;
  NodeKey embedded_key;
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> cmd_buffer,
                         NewPrimary());
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> nested_clone,
                         NewNested());

    auto module_scope =
        Scope::Root(/*enabled=*/true, "XlaModule:#module=test#");

    const Command* kernel_cmd = nullptr;
    {
      Scope op_scope("fusion.1:#tf_op=add#");
      ASSERT_OK_AND_ASSIGN(kernel_cmd, Launch(*cmd_buffer));
    }
    const Command* memset_cmd = nullptr;
    {
      Scope op_scope("memset.1:#tf_op=zero#");
      ASSERT_OK_AND_ASSIGN(memset_cmd, cmd_buffer->CreateMemset(
                                           &a_, uint32_t{0}, 4, {kernel_cmd}));
    }
    const Command* memcpy_cmd = nullptr;
    {
      Scope op_scope("copy.1:#tf_op=copy#");
      ASSERT_OK_AND_ASSIGN(
          memcpy_cmd, cmd_buffer->CreateMemcpyD2D(&b_, a_, 4 * sizeof(int32_t),
                                                  {memset_cmd}));
    }
    const Command* child_k = nullptr;
    const Command* cloned_child_cmd = nullptr;
    {
      Scope op_scope("traced_child:#tf_op=gemm#");
      ASSERT_OK_AND_ASSIGN(child_k, Launch(*nested_clone));
      ASSERT_OK_AND_ASSIGN(cloned_child_cmd, cmd_buffer->CreateChildCommand(
                                                 *nested_clone, {memcpy_cmd}));
    }

    ASSERT_OK_AND_ASSIGN(kernel_key, KeyOf(kernel_cmd));
    ASSERT_OK_AND_ASSIGN(NodeKey memset_key, KeyOf(memset_cmd));
    ASSERT_OK_AND_ASSIGN(NodeKey memcpy_key, KeyOf(memcpy_cmd));
    EXPECT_EQ(memset_key.graph_id, kernel_key.graph_id);
    EXPECT_EQ(memcpy_key.graph_id, kernel_key.graph_id);
    ASSERT_OK_AND_ASSIGN(embedded_key, EmbeddedKey(cloned_child_cmd, child_k));

    EXPECT_THAT(Lookup(kernel_key),
                Optional(Eq("XlaModule:#module=test#::fusion.1:#tf_op=add#")));
    EXPECT_THAT(Lookup(memset_key),
                Optional(Eq("XlaModule:#module=test#::memset.1:#tf_op=zero#")));
    EXPECT_THAT(Lookup(memcpy_key),
                Optional(Eq("XlaModule:#module=test#::copy.1:#tf_op=copy#")));
    EXPECT_THAT(
        Lookup(embedded_key),
        Optional(Eq("XlaModule:#module=test#::traced_child:#tf_op=gemm#")));

    ASSERT_OK(cmd_buffer->Finalize());

    // In-place leaf-node updates on the hot path must preserve the original
    // creation-time annotation without re-registering.
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> nested_clone_updated,
                         NewNested());
    ASSERT_OK(cmd_buffer->Update());
    {
      Scope op_scope("fusion.1_updated:#tf_op=add#");
      ASSERT_OK(cmd_buffer->UpdateLaunch(
          kernel_cmd, ThreadDim(4, 1, 1), BlockDim(1, 1, 1), std::nullopt,
          *kernel_, *PackKernelArgs(0, a_, a_, b_)));
    }
    NodeKey updated_key;
    {
      Scope op_scope("traced_child_updated:#tf_op=gemm#");
      ASSERT_OK_AND_ASSIGN(const Command* child_k_updated,
                           Launch(*nested_clone_updated));
      ASSERT_OK_AND_ASSIGN(updated_key, KeyOf(child_k_updated));
      ASSERT_OK(cmd_buffer->UpdateChildCommand(cloned_child_cmd,
                                               *nested_clone_updated));
    }
    ASSERT_OK(cmd_buffer->Finalize());

    EXPECT_THAT(Lookup(kernel_key),
                Optional(Eq("XlaModule:#module=test#::fusion.1:#tf_op=add#")));
    // After the update, CUPTI reports the child node's work with the IDs of
    // the nested graph it was updated from, which were registered when its
    // node was created. The graph embedded at creation keeps its annotation.
    EXPECT_THAT(
        Lookup(updated_key),
        Optional(
            Eq("XlaModule:#module=test#::traced_child_updated:#tf_op=gemm#")));
    EXPECT_THAT(
        Lookup(embedded_key),
        Optional(Eq("XlaModule:#module=test#::traced_child:#tf_op=gemm#")));
  }

  // Destroying the command buffers unregisters every graph.
  EXPECT_TRUE(registry().empty());
  EXPECT_EQ(Lookup(kernel_key), std::nullopt);
  EXPECT_EQ(Lookup(embedded_key), std::nullopt);
}

TEST_F(CudaCommandBufferTelemetryTest, FlagOffDoesNoWork) {
  // With telemetry off `CommandBufferThunk` opens a disabled root, so pushes do
  // nothing, nothing is recorded, and graphs must not even be tracked.
  auto module_scope = Scope::Root(/*enabled=*/false, "XlaModule:#module=test#");
  ASSERT_FALSE(Scope::IsActive());
  auto trace_memset = [&] {
    return TraceCommandBufferFactory::Create(
        executor_, stream_.get(),
        [&](Stream* stream) {
          return stream->Memset32(&a_, 42, 4 * sizeof(int32_t));
        },
        CommandBuffer::Mode::kNested);
  };

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> primary, NewPrimary());
  ASSERT_OK_AND_ASSIGN(const Command* kernel, Launch(*primary));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> traced, trace_memset());
  ASSERT_OK_AND_ASSIGN(const Command* child,
                       primary->CreateChildCommand(*traced, {kernel}));
  ASSERT_OK(primary->Finalize());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> retraced, trace_memset());
  ASSERT_OK(UpdateChild(*primary, child, *retraced));

  EXPECT_TRUE(registry().empty());
}

TEST_F(CudaCommandBufferTelemetryTest, ClonedChildCopiesFromOriginal) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> nested, NewNested());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> primary, NewPrimary());
  // Recorded with recording off, so this node of the original has no entry.
  ASSERT_OK_AND_ASSIGN(const Command* k0, Launch(*nested));
  const Command* k1 = nullptr;
  const Command* k2 = nullptr;
  const Command* child = nullptr;
  {
    ScopedOpRecording recording("a");
    ASSERT_OK_AND_ASSIGN(k1, Launch(*nested, {k0}));
  }
  {
    ScopedOpRecording recording("b");
    ASSERT_OK_AND_ASSIGN(k2, Launch(*nested, {k1}));
  }
  {
    ScopedOpRecording recording("parent");
    ASSERT_OK_AND_ASSIGN(child, primary->CreateChildCommand(*nested, {}));
  }

  ASSERT_OK_AND_ASSIGN(NodeKey k0_key, KeyOf(k0));
  ASSERT_OK_AND_ASSIGN(NodeKey e0_key, EmbeddedKey(child, k0));
  ASSERT_OK_AND_ASSIGN(NodeKey e1_key, EmbeddedKey(child, k1));
  ASSERT_OK_AND_ASSIGN(NodeKey e2_key, EmbeddedKey(child, k2));
  // Each node of the clone gets its counterpart's annotation, which is more
  // precise than the parent's, and the parent's only if there is none.
  EXPECT_THAT(Lookup(e1_key), Optional(Eq("m::a")));
  EXPECT_THAT(Lookup(e2_key), Optional(Eq("m::b")));
  EXPECT_THAT(Lookup(e0_key), Optional(Eq("m::parent")));
  EXPECT_EQ(Lookup(k0_key), std::nullopt);

  ASSERT_OK(primary->Finalize());
  primary.reset();
  nested.reset();
  EXPECT_TRUE(registry().empty());
}

TEST_F(CudaCommandBufferTelemetryTest, UpdateKeepsPreciseNestedAnnotations) {
  // [kernel, child graph node embedding [kernel]]: graphs of this shape can
  // replace each other in `cuGraphExecChildGraphNodeSetParams`.
  struct Nested {
    std::unique_ptr<CommandBuffer> inner;
    std::unique_ptr<CommandBuffer> outer;
    const Command* inner_kernel = nullptr;
    const Command* outer_kernel = nullptr;
    const Command* child = nullptr;
  };
  auto record_nested = [&](const char* kernel_op,
                           const char* child_op) -> absl::StatusOr<Nested> {
    Nested nested;
    ABSL_ASSIGN_OR_RETURN(nested.inner, NewNested());
    ABSL_ASSIGN_OR_RETURN(nested.inner_kernel, Launch(*nested.inner));
    ABSL_ASSIGN_OR_RETURN(nested.outer, NewNested());
    {
      ScopedOpRecording recording(kernel_op);
      ABSL_ASSIGN_OR_RETURN(nested.outer_kernel, Launch(*nested.outer));
    }
    ScopedOpRecording recording(child_op);
    ABSL_ASSIGN_OR_RETURN(nested.child, nested.outer->CreateChildCommand(
                                       *nested.inner, {nested.outer_kernel}));
    return nested;
  };

  ASSERT_OK_AND_ASSIGN(Nested original, record_nested(nullptr, nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> primary, NewPrimary());
  ASSERT_OK_AND_ASSIGN(const Command* child,
                       primary->CreateChildCommand(*original.outer, {}));
  ASSERT_OK(primary->Finalize());

  // A tracked graph was registered as its nodes were created, so the update
  // retains its entries without walking or overwriting them.
  ASSERT_OK_AND_ASSIGN(Nested tracked, record_nested("outer", "inner"));
  {
    ScopedOpRecording recording("update");
    ASSERT_OK(UpdateChild(*primary, child, *tracked.outer));
  }
  ASSERT_OK_AND_ASSIGN(NodeKey kernel_key, KeyOf(tracked.outer_kernel));
  ASSERT_OK_AND_ASSIGN(NodeKey embedded_key,
                       EmbeddedKey(tracked.child, tracked.inner_kernel));
  tracked.outer.reset();
  tracked.inner.reset();
  EXPECT_THAT(Lookup(kernel_key), Optional(Eq("m::outer")));
  EXPECT_THAT(Lookup(embedded_key), Optional(Eq("m::inner")));

  primary.reset();
  EXPECT_TRUE(registry().empty());
}

TEST_F(CudaCommandBufferTelemetryTest, UpdateSharesEntriesOfRecordedNested) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> original, NewNested());
  ASSERT_OK(Launch(*original).status());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> primary, NewPrimary());
  ASSERT_OK_AND_ASSIGN(const Command* child,
                       primary->CreateChildCommand(*original, {}));
  ASSERT_OK(primary->Finalize());
  // Recorded with recording off, so nothing so far is in the registry.
  ASSERT_TRUE(registry().empty());

  // Records a nested graph for op `op`, updates `child` to it, and destroys it
  // right away, as commands that record a temporary nested command buffer per
  // Record do.
  auto update_to_temporary = [&](const char* op) -> absl::StatusOr<NodeKey> {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<CommandBuffer> nested, NewNested());
    const Command* kernel = nullptr;
    {
      ScopedOpRecording recording(op);
      ABSL_ASSIGN_OR_RETURN(kernel, Launch(*nested));
    }
    ScopedOpRecording recording("update");
    ABSL_RETURN_IF_ERROR(UpdateChild(*primary, child, *nested));
    return KeyOf(kernel);
  };

  // The child node retains the entry of the nested command buffer after the
  // nested command buffer is destroyed.
  ASSERT_OK_AND_ASSIGN(NodeKey first, update_to_temporary("first"));
  EXPECT_THAT(Lookup(first), Optional(Eq("m::first")));
  EXPECT_EQ(registry().GetStats().live_graphs, size_t{1});

  // Each update releases the graph the node ran before: however many updates
  // there are, the registry only has the entry of the graph the node runs.
  NodeKey prev = first;
  for (int i = 0; i < 16; ++i) {
    ASSERT_OK_AND_ASSIGN(NodeKey next, update_to_temporary("next"));
    EXPECT_EQ(Lookup(prev), std::nullopt);
    EXPECT_THAT(Lookup(next), Optional(Eq("m::next")));
    EXPECT_EQ(registry().GetStats().live_graphs, size_t{1});
    prev = next;
  }

  // Under a harvest hold, the replaced graph is parked until the hold ends.
  {
    CudaGraphAnnotationRegistry::HarvestHold hold =
        registry().AcquireHarvestHold();
    ASSERT_OK_AND_ASSIGN(NodeKey held, update_to_temporary("held"));
    EXPECT_FALSE(registry().IsGraphTracked(prev.graph_id));
    EXPECT_THAT(Lookup(prev), Optional(Eq("m::next")));
    EXPECT_THAT(Lookup(held), Optional(Eq("m::held")));
  }
  EXPECT_EQ(Lookup(prev), std::nullopt);

  primary.reset();
  EXPECT_TRUE(registry().empty());
}

TEST_F(CudaCommandBufferTelemetryTest, UpdateToCachedNestedKeepsItsEntries) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> original, NewNested());
  ASSERT_OK(Launch(*original).status());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> primary, NewPrimary());
  ASSERT_OK_AND_ASSIGN(const Command* child,
                       primary->CreateChildCommand(*original, {}));
  ASSERT_OK(primary->Finalize());

  // Two nested command buffers that stay alive between updates, as in a trace
  // cache with room for two.
  struct Cached {
    std::unique_ptr<CommandBuffer> nested;
    NodeKey key;
  };
  auto record = [&](const char* op) -> absl::StatusOr<Cached> {
    Cached cached;
    ABSL_ASSIGN_OR_RETURN(cached.nested, NewNested());
    ScopedOpRecording recording(op);
    ABSL_ASSIGN_OR_RETURN(const Command* kernel, Launch(*cached.nested));
    ABSL_ASSIGN_OR_RETURN(cached.key, KeyOf(kernel));
    return cached;
  };
  ASSERT_OK_AND_ASSIGN(Cached a, record("a"));
  ASSERT_OK_AND_ASSIGN(Cached b, record("b"));
  auto update_to = [&](const Cached& cached) {
    ScopedOpRecording recording("update");
    return UpdateChild(*primary, child, *cached.nested);
  };

  // Switching between them releases nothing that a cached command buffer still
  // holds, and installing one again keeps its precise annotation.
  for (int i = 0; i < 2; ++i) {
    ASSERT_OK(update_to(a));
    ASSERT_OK(update_to(b));
    EXPECT_THAT(Lookup(a.key), Optional(Eq("m::a")));
    EXPECT_THAT(Lookup(b.key), Optional(Eq("m::b")));
  }

  // Evicting the installed command buffer keeps its entry for the node.
  ASSERT_OK(update_to(a));
  a.nested.reset();
  EXPECT_THAT(Lookup(a.key), Optional(Eq("m::a")));
  // Until the node switches away from it.
  ASSERT_OK(update_to(b));
  EXPECT_EQ(Lookup(a.key), std::nullopt);
  EXPECT_THAT(Lookup(b.key), Optional(Eq("m::b")));

  // Evicting a command buffer that no node runs releases its entries.
  ASSERT_OK_AND_ASSIGN(Cached c, record("c"));
  ASSERT_THAT(Lookup(c.key), Optional(Eq("m::c")));
  c.nested.reset();
  EXPECT_EQ(Lookup(c.key), std::nullopt);

  primary.reset();
  EXPECT_THAT(Lookup(b.key), Optional(Eq("m::b")));
  b.nested.reset();
  EXPECT_TRUE(registry().empty());
}

TEST_F(CudaCommandBufferTelemetryTest,
       MovedChildWithSubgraphsLivesAsLongAsPrimary) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> inner, NewNested());
  const Command* inner_kernel = nullptr;
  {
    ScopedOpRecording recording("inner");
    ASSERT_OK_AND_ASSIGN(inner_kernel, Launch(*inner));
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> primary, NewPrimary());
  NodeKey moved_key;
  NodeKey embedded_key;
  {
    ScopedOpRecording recording("moved");
    // The moved child has both its own kernel and an embedded clone of `inner`,
    // whose entries the primary takes over
    // (`CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE`).
    ASSERT_OK(primary
                  ->CreateChildCommand(
                      [&](CommandBuffer* nested) -> absl::Status {
                        ABSL_ASSIGN_OR_RETURN(const Command* kernel,
                                         Launch(*nested));
                        ABSL_ASSIGN_OR_RETURN(moved_key, KeyOf(kernel));
                        ABSL_ASSIGN_OR_RETURN(
                            const Command* child,
                            nested->CreateChildCommand(*inner, {kernel}));
                        ABSL_ASSIGN_OR_RETURN(embedded_key,
                                         EmbeddedKey(child, inner_kernel));
                        return absl::OkStatus();
                      },
                      {})
                  .status());
  }
  ASSERT_OK(primary->Finalize());
  EXPECT_THAT(Lookup(moved_key), Optional(Eq("m::moved")));
  EXPECT_THAT(Lookup(embedded_key), Optional(Eq("m::inner")));

  // Both the moved graph and its embedded clone live in the primary's graph.
  inner.reset();
  EXPECT_THAT(Lookup(moved_key), Optional(Eq("m::moved")));
  EXPECT_THAT(Lookup(embedded_key), Optional(Eq("m::inner")));
  primary.reset();
  EXPECT_TRUE(registry().empty());
}

TEST_F(CudaCommandBufferTelemetryTest,
       NestedUnregistersItsGraphsOnDestruction) {
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> nested, NewNested());
    ScopedOpRecording recording("nested");
    ASSERT_OK_AND_ASSIGN(const Command* kernel, Launch(*nested));
    ASSERT_OK_AND_ASSIGN(NodeKey key, KeyOf(kernel));
    EXPECT_THAT(Lookup(key), Optional(Eq("m::nested")));
  }
  EXPECT_TRUE(registry().empty());

  // A command buffer built by stream capture, as `TracedCommandBuffer` does.
  {
    ScopedOpRecording recording("traced");
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> traced,
                         TraceCommandBufferFactory::Create(
                             executor_, stream_.get(),
                             [&](Stream* stream) {
                               return stream->Memset32(&a_, 42,
                                                       4 * sizeof(int32_t));
                             },
                             CommandBuffer::Mode::kNested));
    EXPECT_FALSE(registry().empty());
  }
  EXPECT_TRUE(registry().empty());
}

TEST_F(CudaCommandBufferTelemetryTest, WhileBodyRegisteredAndCleanedUp) {
  DeviceAddress<bool> pred = executor_->AllocateArray<bool>(1);
  absl::Cleanup free_pred = [&] { executor_->Deallocate(&pred); };
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> primary, NewPrimary());
  NodeKey root_key;
  NodeKey body_key;
  {
    ScopedOpRecording recording("while");
    ASSERT_OK_AND_ASSIGN(const Command* kernel, Launch(*primary));
    ASSERT_OK_AND_ASSIGN(root_key, KeyOf(kernel));
    // Records nothing: the while command itself adds the kernels that set its
    // condition, to the primary and to the body.
    auto create_cond = [](CommandBuffer*, absl::Span<const Command* const> deps)
        -> absl::StatusOr<std::vector<const Command*>> {
      return std::vector<const Command*>(deps.begin(), deps.end());
    };
    auto create_body = [&](CommandBuffer* body,
                           absl::Span<const Command* const> deps)
        -> absl::StatusOr<std::vector<const Command*>> {
      Scope op_scope("body");
      ABSL_ASSIGN_OR_RETURN(const Command* body_kernel, Launch(*body, deps));
      ABSL_ASSIGN_OR_RETURN(body_key, KeyOf(body_kernel));
      return std::vector<const Command*>{body_kernel};
    };
    ASSERT_OK(primary->CreateWhile(pred, create_cond, create_body, {kernel})
                  .status());
  }
  // The body is a graph of its own, whose entry the root command buffer owns.
  EXPECT_NE(body_key.graph_id, root_key.graph_id);
  EXPECT_THAT(Lookup(body_key), Optional(Eq("m::while::body")));
  ASSERT_OK(primary->Finalize());

  primary.reset();
  EXPECT_TRUE(registry().empty());
  EXPECT_EQ(Lookup(body_key), std::nullopt);
}

}  // namespace
}  // namespace stream_executor::gpu

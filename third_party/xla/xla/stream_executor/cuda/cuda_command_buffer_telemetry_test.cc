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

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
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

  // Once the command buffers are destroyed, their cleanups unregister every
  // graph. CUDA runs them asynchronously.
  EXPECT_TRUE(PollUntil([] { return registry().empty(); }));
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
  EXPECT_TRUE(PollUntil([] { return registry().empty(); }));
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
  // Records the outer kernel for op `kernel_op` and the child graph node for
  // op `child_op`, each with recording off if its op is null.
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

  // An untracked graph gets the update's annotation, but only for nodes that
  // have none: its tracked child graph is left alone.
  ASSERT_OK_AND_ASSIGN(Nested untracked, record_nested(nullptr, "inner"));
  {
    ScopedOpRecording recording("update");
    ASSERT_OK(UpdateChild(*primary, child, *untracked.outer));
  }
  ASSERT_OK_AND_ASSIGN(NodeKey kernel_key, KeyOf(untracked.outer_kernel));
  ASSERT_OK_AND_ASSIGN(NodeKey embedded_key,
                       EmbeddedKey(untracked.child, untracked.inner_kernel));
  EXPECT_THAT(Lookup(kernel_key), Optional(Eq("m::update")));
  EXPECT_THAT(Lookup(embedded_key), Optional(Eq("m::inner")));

  // A tracked graph was registered as its nodes were created, so the update
  // doesn't walk it, and its unregistered child graph stays unregistered.
  ASSERT_OK_AND_ASSIGN(Nested tracked, record_nested("outer", nullptr));
  {
    ScopedOpRecording recording("update");
    ASSERT_OK(UpdateChild(*primary, child, *tracked.outer));
  }
  ASSERT_OK_AND_ASSIGN(kernel_key, KeyOf(tracked.outer_kernel));
  ASSERT_OK_AND_ASSIGN(embedded_key,
                       EmbeddedKey(tracked.child, tracked.inner_kernel));
  EXPECT_THAT(Lookup(kernel_key), Optional(Eq("m::outer")));
  EXPECT_EQ(Lookup(embedded_key), std::nullopt);
}

TEST_F(CudaCommandBufferTelemetryTest, UpdateWithTemporaryNestedGraph) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> original, NewNested());
  ASSERT_OK(Launch(*original).status());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> primary, NewPrimary());
  ASSERT_OK_AND_ASSIGN(const Command* child,
                       primary->CreateChildCommand(*original, {}));
  ASSERT_OK(primary->Finalize());

  // Updates `child` to a new nested graph and destroys that graph right away,
  // as host execution does for every Record. The graph is recorded with
  // recording off, so the update tracks it. Returns the key of its kernel.
  auto update_to_temporary = [&](const char* op) -> absl::StatusOr<NodeKey> {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<CommandBuffer> nested, NewNested());
    ABSL_ASSIGN_OR_RETURN(const Command* kernel, Launch(*nested));
    ScopedOpRecording recording(op);
    ABSL_RETURN_IF_ERROR(UpdateChild(*primary, child, *nested));
    return KeyOf(kernel);
  };

  ASSERT_OK_AND_ASSIGN(NodeKey first, update_to_temporary("first"));
  // The update attached the cleanup before `SetParams` cloned the graph's
  // references into the executable graph, so the entry outlives the graph.
  // Wait for the cleanup of a graph destroyed later, by which time the
  // cleanup would most likely have run otherwise.
  NodeKey control;
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> nested, NewNested());
    ScopedOpRecording recording("control");
    ASSERT_OK_AND_ASSIGN(const Command* kernel, Launch(*nested));
    ASSERT_OK_AND_ASSIGN(control, KeyOf(kernel));
    ASSERT_TRUE(registry().IsGraphTracked(control.graph_id));
  }
  ASSERT_TRUE(
      PollUntil([&] { return !registry().IsGraphTracked(control.graph_id); }));
  EXPECT_THAT(Lookup(first), Optional(Eq("m::first")));

  // The next update replaces the executable graph's references, the last ones
  // to the first graph's cleanup.
  ASSERT_OK_AND_ASSIGN(NodeKey second, update_to_temporary("second"));
  EXPECT_TRUE(PollUntil([&] { return !Lookup(first).has_value(); }));
  EXPECT_THAT(Lookup(second), Optional(Eq("m::second")));

  // While a harvest hold is active, the entry is parked instead: it is no
  // longer tracked, but still resolves until the hold is released.
  {
    CudaGraphAnnotationRegistry::HarvestHold hold =
        registry().AcquireHarvestHold();
    ASSERT_OK_AND_ASSIGN(NodeKey third, update_to_temporary("third"));
    EXPECT_TRUE(
        PollUntil([&] { return !registry().IsGraphTracked(second.graph_id); }));
    EXPECT_THAT(Lookup(second), Optional(Eq("m::second")));
    EXPECT_THAT(Lookup(third), Optional(Eq("m::third")));
  }
  EXPECT_EQ(Lookup(second), std::nullopt);

  primary.reset();
  EXPECT_TRUE(PollUntil([] { return registry().empty(); }));
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
  // The body is a graph of its own, whose cleanup the root graph holds.
  EXPECT_NE(body_key.graph_id, root_key.graph_id);
  EXPECT_THAT(Lookup(body_key), Optional(Eq("m::while::body")));
  ASSERT_OK(primary->Finalize());

  primary.reset();
  EXPECT_TRUE(PollUntil([] { return registry().empty(); }));
  EXPECT_EQ(Lookup(body_key), std::nullopt);
}

}  // namespace
}  // namespace stream_executor::gpu

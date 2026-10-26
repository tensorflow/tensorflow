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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_TELEMETRY_TEST_UTIL_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_TELEMETRY_TEST_UTIL_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_graph_annotation_registry.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/scoped_command_buffer_annotation.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// Identifies a graph node the way CUPTI does in `sourceGraphId` and
// `sourceGraphNodeId`, which is also the key of its registry entry.
struct NodeKey {
  uint32_t graph_id = 0;
  uint64_t node_id = 0;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const NodeKey& key) {
    absl::Format(&sink, "(%d, %#x)", key.graph_id, key.node_id);
  }
};

// Returns the key of the graph node that `command`, a `GpuCommand`, created.
absl::StatusOr<NodeKey> KeyOf(const CommandBuffer::Command* command);

// Returns the key of the counterpart of node `original` in the graph embedded
// by the child graph node that `child` created from `original`'s graph.
absl::StatusOr<NodeKey> EmbeddedKey(const CommandBuffer::Command* child,
                                    const CommandBuffer::Command* original);

// Returns a copy of the annotation of `key` in `space` in the default
// registry, or `std::nullopt` if there is none.
std::optional<std::string> Lookup(
    const NodeKey& key,
    CudaGraphAnnotationRegistry::KeySpace space =
        CudaGraphAnnotationRegistry::KeySpace::kTemplateGraph);

// Calls `done` every 10 ms until it returns true, but for at most 10 s, and
// returns whether it did. Needed to observe cleanups: CUDA runs user-object
// destructors asynchronously, on an internal thread.
bool PollUntil(absl::FunctionRef<bool()> done);

// Records the HLO annotation "m::<op>" for the nodes created during its
// lifetime, as XLA does while it records op `op` of module "m": "m" is an
// enabled root that turns recording on. Does nothing if `op` is null.
class ScopedOpRecording {
 public:
  // `Root()` returns a non-movable scope, so it initializes `root_` directly
  // (C++17 guaranteed copy elision) instead of being emplaced. With a null
  // `op` the root is disabled, so the empty `op_push_` does nothing either.
  explicit ScopedOpRecording(const char* op)
      : root_(ScopedCommandBufferAnnotation::Root(/*enabled=*/op != nullptr,
                                                  "m")),
        op_push_(op != nullptr ? op : "") {}

  ScopedOpRecording(const ScopedOpRecording&) = delete;
  ScopedOpRecording& operator=(const ScopedOpRecording&) = delete;

 private:
  ScopedCommandBufferAnnotation root_;
  ScopedCommandBufferAnnotation op_push_;
};

// Fixture for command buffer telemetry tests: skips tests unless both the CUDA
// headers and the driver are 13.2+, and starts each test with an empty default
// registry.
class CudaCommandBufferTelemetryTest : public ::testing::Test {
 protected:
  void SetUp() override;
  void TearDown() override;

  static CudaGraphAnnotationRegistry& registry() {
    return CudaGraphAnnotationRegistry::Default();
  }

  absl::StatusOr<std::unique_ptr<CommandBuffer>> NewPrimary();
  absl::StatusOr<std::unique_ptr<CommandBuffer>> NewNested();

  // Adds a launch of a small kernel to `command_buffer`. All launches have the
  // same parameters, so graphs of them can replace each other in updates.
  absl::StatusOr<const CommandBuffer::Command*> Launch(
      CommandBuffer& command_buffer,
      absl::Span<const CommandBuffer::Command* const> dependencies = {});

  // Updates child command `child` of the finalized `primary` to `nested`, and
  // finalizes `primary` again.
  static absl::Status UpdateChild(CommandBuffer& primary,
                                  const CommandBuffer::Command* child,
                                  const CommandBuffer& nested);

  StreamExecutor* executor_ = nullptr;
  std::unique_ptr<Stream> stream_;
  std::unique_ptr<Kernel> kernel_;
  DeviceAddress<int32_t> a_;
  DeviceAddress<int32_t> b_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_TELEMETRY_TEST_UTIL_H_

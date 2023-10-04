/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/stream_executor/gpu/gpu_command_buffer.h"

#include <atomic>
#include <cstdint>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_kernel.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace stream_executor::gpu {

//===----------------------------------------------------------------------===//
// GpuCommandBuffer resource usage tracking
//===----------------------------------------------------------------------===//

static std::atomic<int64_t> allocated_execs(0);
static std::atomic<int64_t> alive_execs(0);

static int64_t NotifyExecCreated() {
  alive_execs.fetch_add(1, std::memory_order_relaxed);
  return allocated_execs.fetch_add(1, std::memory_order_relaxed);
}

static int64_t NotifyExecDestroyed() {
  DCHECK_GE(alive_execs.load(std::memory_order_relaxed), 1);
  return alive_execs.fetch_sub(1, std::memory_order_relaxed) - 1;
}

/*static*/ int64_t GpuCommandBuffer::AllocatedExecs() {
  return allocated_execs.load(std::memory_order_relaxed);
}

/*static*/ int64_t GpuCommandBuffer::AliveExecs() {
  return alive_execs.load(std::memory_order_relaxed);
}

//===----------------------------------------------------------------------===//
// GpuCommandBuffer implementation
//===----------------------------------------------------------------------===//

GpuCommandBuffer::GpuCommandBuffer(CommandBuffer::Mode mode,
                                   GpuExecutor* parent, GpuGraphHandle graph)
    : mode_(mode), parent_(parent), graph_(graph), exec_(nullptr) {}

GpuCommandBuffer::~GpuCommandBuffer() {
  if (exec_ != nullptr) {
    VLOG(5) << "Destroy GPU command buffer executable graph "
            << "(remaining alive executable graphs: " << NotifyExecDestroyed()
            << ")";
    auto st = GpuDriver::DestroyGraphExec(exec_);
    CHECK(st.ok()) << "Failed to destroy GPU graph exec: " << st.message();
  }
  if (graph_ != nullptr) {
    auto st = GpuDriver::DestroyGraph(graph_);
    CHECK(st.ok()) << "Failed to destroy GPU graph: " << st.message();
  }
}

static GpuDevicePtr AsDevicePtr(const DeviceMemoryBase& mem) {
  return reinterpret_cast<GpuDevicePtr>(const_cast<void*>(mem.opaque()));
}

tsl::Status GpuCommandBuffer::Trace(
    Stream* stream, absl::AnyInvocable<tsl::Status()> function) {
  // TODO(ezhulenev): Check that graph is empty, because we should not be mixing
  // graph tracing with explicit graph construction.
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  VLOG(5) << "Trace into GPU command buffer graph " << graph_
          << " on a stream: " << stream->DebugStreamPointers();

  auto gpu_stream = AsGpuStreamValue(stream);

  // Switch stream into the capture mode.
  uint64_t start_nanos = tsl::Env::Default()->NowNanos();
  TF_RETURN_IF_ERROR(GpuDriver::StreamBeginCapture(
      gpu_stream, GpuDriver::StreamCaptureMode::kThreadLocal));

  auto traced = function();

  // Always stop capturing the stream before checking `traced` result.
  TF_RETURN_IF_ERROR(GpuDriver::StreamEndCapture(gpu_stream, &graph_));
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();

  if (!traced.ok())
    return absl::InternalError(
        absl::StrCat("Failed to capture gpu graph: ", traced.message()));

  VLOG(5) << "Traced into the GPU command buffer graph " << graph_ << " (took "
          << (end_nanos - start_nanos) / 1000 << " μs)";

  return tsl::OkStatus();
}

tsl::Status GpuCommandBuffer::CheckNotFinalized() {
  if (finalized_)
    return absl::InternalError(
        "Command buffer can't be updated after it was finalized");
  return tsl::OkStatus();
}

tsl::Status GpuCommandBuffer::Launch(const ThreadDim& threads,
                                     const BlockDim& blocks,
                                     const KernelBase& kernel,
                                     const KernelArgsArrayBase& args) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  const GpuKernel* gpu_kernel = AsGpuKernel(&kernel);
  GpuFunctionHandle gpu_func = gpu_kernel->AsGpuFunctionHandle();

  void** kernel_params = const_cast<void**>(args.argument_addresses().data());

  GpuGraphNodeHandle node;
  TF_RETURN_IF_ERROR(GpuDriver::GraphAddKernelNode(
      &node, graph_, {}, kernel.name(), gpu_func, blocks.x, blocks.y, blocks.z,
      threads.x, threads.y, threads.z, args.number_of_shared_bytes(),
      kernel_params, /*extra=*/nullptr));

  return tsl::OkStatus();
}

tsl::Status GpuCommandBuffer::MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                                   const DeviceMemoryBase& src,
                                                   uint64_t size) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  GpuGraphNodeHandle node;
  TF_RETURN_IF_ERROR(GpuDriver::GraphAddMemcpyD2DNode(
      parent_->gpu_context(), &node, graph_, {}, AsDevicePtr(*dst),
      AsDevicePtr(src), size));

  return tsl::OkStatus();
}

tsl::Status GpuCommandBuffer::Finalize() {
  TF_RETURN_IF_ERROR(CheckNotFinalized());

  if (mode_ == CommandBuffer::Mode::kPrimary) {
    GpuDriver::GraphInstantiateFlags flags;

    uint64_t start_nanos = tsl::Env::Default()->NowNanos();
    TF_RETURN_IF_ERROR(GpuDriver::GraphInstantiate(&exec_, graph_, flags));
    uint64_t end_nanos = tsl::Env::Default()->NowNanos();

    VLOG(5) << "Instantiated executable graph #" << NotifyExecCreated()
            << " in " << (end_nanos - start_nanos) / 1000
            << " μs (alive executable graphs: " << AliveExecs() << ")";

  } else {
    VLOG(5) << "Finalize nested command buffer without instantiating "
               "executable graph";
  }

  finalized_ = true;
  return tsl::OkStatus();
}

}  // namespace stream_executor::gpu

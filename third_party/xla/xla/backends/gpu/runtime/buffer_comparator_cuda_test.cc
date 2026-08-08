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

#include <cstdint>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/backends/gpu/runtime/buffer_comparator.h"
#include "xla/backends/gpu/runtime/buffer_comparator_test.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/cuda/cuda_stream.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

struct GraphDeleter {
  void operator()(CUgraph graph) const { (void)cuGraphDestroy(graph); }
};

}  // namespace

// Verifies that using BufferComparator::CompareEqual on a helper thread does
// not invalidate an active stream capture session on the main thread.
//
// When CUDA stream capture is active, context-wide synchronization operations
// (like cuCtxSynchronize) are prohibited. Since BufferComparator uses a
// host-pinned allocator for output result buffers, its deallocation does not
// trigger context synchronization and thus leaves the stream capture session
// intact. (If it were to use a VMM/device allocator, the context-wide sync
// inside the allocator destructor would fail and invalidate the capture).
TEST_F(BufferComparatorTest,
       CompareEqualDoesNotInvalidateConcurrentStreamCapture) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       stream_exec_->CreateStream());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> capture_stream,
                       stream_exec_->CreateStream());

  std::vector<float> lhs_data = {1.0f, 2.0f, 3.0f};
  std::vector<float> rhs_data = {1.0f, 2.0f, 3.0f};

  se::DeviceAddressHandle lhs(
      stream_exec_, stream_exec_->AllocateArray<float>(lhs_data.size()));
  se::DeviceAddressHandle rhs(
      stream_exec_, stream_exec_->AllocateArray<float>(rhs_data.size()));

  ASSERT_OK(
      stream->Memcpy(lhs.address_ptr(), lhs_data.data(), lhs.address().size()));
  ASSERT_OK(
      stream->Memcpy(rhs.address_ptr(), rhs_data.data(), rhs.address().size()));
  ASSERT_OK(stream->BlockHostUntilDone());

  se::gpu::CudaStream* cuda_capture_stream =
      absl::down_cast<se::gpu::CudaStream*>(capture_stream.get());

  CUgraph graph;
  ASSERT_OK(stream_executor::cuda::ToStatus(cuGraphCreate(&graph, 0)));
  std::unique_ptr<CUgraph_st, GraphDeleter> cleanup_graph(graph);

  ASSERT_OK_AND_ASSIGN(
      stream_executor::gpu::CudaStream::CaptureHandle capture_handle,
      cuda_capture_stream->BeginCapture(graph, nullptr, nullptr, 0,
                                        CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));

  BufferComparator comparator(
      ShapeUtil::MakeShape(F32, {static_cast<int64_t>(lhs_data.size())}),
      kDefaultTolerance);

  absl::StatusOr<bool> result;
  // Spawning a helper thread here simulates this concurrent autotuning behavior
  // and verifies that the deallocation synchronization from the autotuning
  // thread does not context-wide invalidate the active thread-local capture
  // session on the main thread.
  std::thread t([&] {
    result =
        comparator.CompareEqual(stream.get(), lhs.address(), rhs.address());
  });
  t.join();

  // CompareEqual should succeed because BlockHostUntilDone() on the helper
  // thread is not blocked by the main thread's thread-local capture.
  EXPECT_OK(result);

  // End capture.
  EXPECT_OK(capture_handle.EndCapture());
}

}  // namespace gpu
}  // namespace xla

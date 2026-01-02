/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_GPU_TFRT_TFRT_GPU_ASYNC_HOST_TO_DEVICE_TRANSFER_MANAGER_H_
#define XLA_PJRT_GPU_TFRT_TFRT_GPU_ASYNC_HOST_TO_DEVICE_TRANSFER_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/executable_run_options.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_device.h"
#include "xla/pjrt/gpu/tfrt/tracked_gpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Checks that the input buffers passed in by the user have the correct size
// on device for the compiled program.

class TfrtGpuAsyncHostToDeviceTransferManager final
    : public PjRtClient::AsyncHostToDeviceTransferManager {
 public:
  static absl::StatusOr<
      std::unique_ptr<TfrtGpuAsyncHostToDeviceTransferManager>>
  Create(absl::Span<const PjRtClient::ShapeSpec> shape_specs,
         std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
         TfrtGpuDevice* device, TfrtGpuClient* client,
         PjRtMemorySpace* memory_space);
  TfrtGpuAsyncHostToDeviceTransferManager(
      absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers,
      absl::InlinedVector<tsl::AsyncValueRef<GpuDeviceMemory>, 4> buffer_ptrs,
      absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events,
      absl::InlinedVector<Shape, 4> device_shapes, TfrtGpuDevice* device);

  ~TfrtGpuAsyncHostToDeviceTransferManager() override;

  size_t buffer_count() const override { return buffer_sizes_.size(); };

  size_t buffer_size(int buffer_index) const override {
    DCHECK_LT(buffer_index, buffer_sizes_.size());
    return buffer_sizes_[buffer_index];
  }

  PjRtDevice* device() const override { return device_; }

  std::unique_ptr<PjRtBuffer> RetrieveBuffer(int buffer_index) override {
    absl::MutexLock l(mu_);
    DCHECK_LT(buffer_index, buffers_.size());
    return std::move(buffers_[buffer_index]);
  };

  absl::Status TransferLiteralToBuffer(
      int buffer_index, const LiteralSlice& literal,
      absl::AnyInvocable<void() &&> on_done) override;
  absl::Status TransferRawDataToBuffer(
      int buffer_index, absl::string_view data,
      absl::AnyInvocable<void() &&> on_done) override {
    return TransferRawDataToSubBuffer(buffer_index, data.data(),
                                      /*offset=*/0, data.size(),
                                      /*is_last_transfer=*/true,
                                      std::move(on_done));
  }

  absl::Status TransferRawDataToSubBuffer(
      int buffer_index, const void* data, int64_t offset, int64_t transfer_size,
      bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) override;
  void SetBufferError(int buffer_index, absl::Status error) override;
  void AddTransferMetadata(const TransferMetadata& meta) override {}

 private:
  static absl::InlinedVector<size_t, 4> GetBufferSizes(
      absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4>& buffers);

  void CleanUp(int buffer_index, absl::AnyInvocable<void() &&> on_done);

  absl::Mutex mu_;
  // The newly created buffers, which will be returned to the caller via
  // Retrieve.
  absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers_
      ABSL_GUARDED_BY(mu_);

  absl::InlinedVector<tsl::AsyncValueRef<GpuDeviceMemory>, 4> buffer_ptrs_
      ABSL_GUARDED_BY(mu_);
  // Cached versions of the sizes of all the buffers, so we can return them
  // without acquiring mu_.
  const absl::InlinedVector<size_t, 4> buffer_sizes_;
  // True if the last transfer for a buffer has been initiated. Used to
  // prevent a client initiating another transfer after the last transfer has
  // already been initiated.
  absl::InlinedVector<bool, 4> last_transfer_started_ ABSL_GUARDED_BY(mu_);
  // The buffer definition events on all the buffers, unblocked once the
  // corresponding buffer transfer has completed.
  absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events_
      ABSL_GUARDED_BY(mu_);
  // Device shapes for all buffers with either compact or custom layout.
  const absl::InlinedVector<Shape, 4> device_shapes_;
  // Count of transfers that have been started but have not yet called
  // cleanup. Used to block in the destructor to avoid dangling pointers in
  // cleanup.
  absl::InlinedVector<size_t, 4> transfers_in_flight_ ABSL_GUARDED_BY(mu_);

  TfrtGpuDevice* const device_;  // not owned.
  TfrtGpuClient* const client_;  // not owned.
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_TFRT_TFRT_GPU_ASYNC_HOST_TO_DEVICE_TRANSFER_MANAGER_H_

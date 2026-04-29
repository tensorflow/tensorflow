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

#ifndef XLA_PJRT_GPU_TFRT_TFRT_GPU_DEVICE_H_
#define XLA_PJRT_GPU_TFRT_TFRT_GPU_DEVICE_H_

#include <cstdint>
#include <memory>
#include <random>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/executable_run_options.h"
#include "xla/literal.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_buffer.h"
#include "xla/pjrt/gpu/tfrt/thread_checker.h"
#include "xla/pjrt/gpu/tfrt/tracked_gpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_device_description.h"
#include "xla/pjrt/scoped_async_tracking_event.h"
#include "xla/pjrt/semaphore.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/transfer_manager.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

class TfrtGpuClient;

class TfrtGpuDevice final : public PjRtDevice {
 public:
  struct Options {
    int id;
    int32_t process_index;
    int32_t process_index_in_partition;
    int partition_index;
    LocalDeviceId local_device_id;
    LocalChipId local_hardware_id;
    se::StreamExecutor* executor;
    int max_inflight_computations;
    std::string platform_version;
    std::string compute_capability;
    std::string device_vendor;
    int core_count;
  };

  explicit TfrtGpuDevice(Options&& options);

  ~TfrtGpuDevice() override;

  void SetClient(TfrtGpuClient* client);

  const PjRtStreamExecutorDeviceDescription& description() const override {
    return description_;
  }

  PjRtClient* client() const override;

  bool IsAddressable() const override { return executor_ != nullptr; }

  int id() const override { return id_; }

  LocalDeviceId local_device_id() const override { return local_device_id_; }

  // Used as `device_ordinal`.
  LocalChipId local_hardware_id() const override { return local_hardware_id_; }

  absl::Status TransferToInfeed(const LiteralSlice& literal) override;

  absl::Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

  // Returns the semaphore to control the max inflight computations.
  Semaphore& max_inflight_computations_semaphore() {
    return max_inflight_computations_semaphore_;
  }

  void AttachMemorySpace(PjRtMemorySpace* memory_space,
                         bool is_default = false);

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind_id(int id) const;

  absl::StatusOr<PjRtMemorySpace*> memory_space_by_kind(
      absl::string_view kind) const override;

  absl::StatusOr<PjRtMemorySpace*> default_memory_space() const override;

  std::unique_ptr<ScopedAsyncTrackingEvent> CreateAsyncTrackingEvent(
      absl::string_view description) const override {
    return nullptr;
  }

  absl::StatusOr<tsl::AllocatorStats> GetAllocatorStats() const override;

  // Returns a fresh, PRNG-generated random seed for an XLA computation.
  int GetNewPrngSeed();

  se::Stream* stream() const {
    TfrtGpuThreadChecker::AssertCudaCallAllowedOnThisThread();
    return stream_.get();
  }

  se::Stream* d2h_stream() const {
    TfrtGpuThreadChecker::AssertCudaCallAllowedOnThisThread();
    return d2h_stream_.get();
  }

  se::StreamExecutor* executor() const {
    TfrtGpuThreadChecker::AssertCudaCallAllowedOnThisThread();
    return executor_;
  }

  tsl::AsyncValueRef<GpuEvent> SetLastCollectiveLaunchEvent(
      tsl::AsyncValueRef<GpuEvent> event);

 private:
  friend class TfrtGpuClient;
  friend class TfrtGpuExecutable;
  friend class TfrtGpuBuffer;

  absl::StatusOr<TransferManager*> GetTransferManager();

  int id_;
  TfrtGpuClient* client_ = nullptr;
  const LocalDeviceId local_device_id_;
  const LocalChipId local_hardware_id_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<se::Stream> d2h_stream_;
  absl::InlinedVector<PjRtMemorySpace*, 1> memory_spaces_;
  absl::flat_hash_map<int, PjRtMemorySpace*> memory_spaces_by_kind_id_;

  absl::Mutex mu_;
  std::random_device prng_seed_device_ ABSL_GUARDED_BY(mu_);
  std::mt19937 prng_seed_generator_ ABSL_GUARDED_BY(mu_);
  std::uniform_int_distribution<> prng_seed_distribution_ ABSL_GUARDED_BY(mu_);
  // Launching collectives are prone to deadlock when we use fixed-sized
  // thread pools and stream pools, since ExecuteHelper will block until all
  // replicas reach the barrier. We ensure that
  // 1. Thread pool size is at least as large as device_count so one collective
  //    launch over all devices can succeed.
  // 2. Gang-schedule each collective by conservatively ensuring a total order
  //    of collectives and launching only one collective at a time to avoid
  //    having no active threads to make progress
  tsl::AsyncValueRef<GpuEvent> last_collective_launch_event_
      ABSL_GUARDED_BY(mu_);

  PjRtStreamExecutorDeviceDescription description_;
  PjRtMemorySpace* default_memory_space_ = nullptr;

  // Semaphore used to limit how many programs can be enqueued by the host
  // ahead of the device.
  xla::Semaphore max_inflight_computations_semaphore_;
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_TFRT_TFRT_GPU_DEVICE_H_

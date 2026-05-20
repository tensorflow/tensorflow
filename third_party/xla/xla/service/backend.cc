/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/backend.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/platform_util.h"
#include "xla/service/stream_pool.h"
#include "xla/service/transfer_manager.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/integrations/device_mem_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/bfc_allocator.h"
#include "xla/tsl/framework/device_id.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "tsl/platform/cpu_info.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla {
namespace se = ::stream_executor;

BackendOptions& BackendOptions::set_platform(se::Platform* platform) {
  platform_ = platform;
  return *this;
}

se::Platform* BackendOptions::platform() const { return platform_; }

BackendOptions& BackendOptions::set_intra_op_parallelism_threads(
    int num_threads) {
  intra_op_parallelism_threads_ = num_threads;
  return *this;
}

int BackendOptions::intra_op_parallelism_threads() const {
  return intra_op_parallelism_threads_;
}

BackendOptions& BackendOptions::set_allowed_devices(
    const std::optional<std::set<int>>& allowed_devices) {
  allowed_devices_ = allowed_devices;
  return *this;
}

const std::optional<std::set<int>>& BackendOptions::allowed_devices() const {
  return allowed_devices_;
}

// Define this in .cc file to avoid having to include eigen or forward declare
// these types in the header.
struct Backend::IntraOpThreadPool {
  explicit IntraOpThreadPool(const int num_threads)
      : pool(new tsl::thread::ThreadPool(tsl::Env::Default(), "XLAEigen",
                                         num_threads)),
        device(new Eigen::ThreadPoolDevice(pool->AsEigenThreadPool(),
                                           pool->NumThreads())) {}

  std::unique_ptr<tsl::thread::ThreadPool> pool;
  std::unique_ptr<Eigen::ThreadPoolDevice> device;
};

namespace {

// Thin tsl::Allocator wrapper around StreamExecutor for a given memory space.
// Used to plug non-default memory spaces (e.g. host pinned) into
// MultiDeviceAdapter via TfAllocatorAdapter.
class StreamExecutorAllocator : public tsl::Allocator {
 public:
  StreamExecutorAllocator(se::StreamExecutor* executor, int64_t memory_space)
      : executor_(executor), memory_space_(memory_space) {}

  std::string Name() override {
    return absl::StrCat("SE_", executor_->device_ordinal(), "_space_",
                        memory_space_);
  }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    auto result = executor_->AllocateArray<char>(num_bytes, memory_space_);
    return result.opaque();
  }

  void DeallocateRaw(void* ptr) override {
    se::DeviceAddressBase mem(ptr);
    executor_->Deallocate(&mem);
  }

 private:
  se::StreamExecutor* executor_;
  int64_t memory_space_;
};

// Creates a MultiDeviceAdapter with a BFC allocator for device memory (space 0)
// and a passthrough StreamExecutor allocator for host pinned memory (space 5).
std::pair<std::vector<std::unique_ptr<se::Stream>>,
          std::shared_ptr<se::MultiDeviceAdapter>>
CreateGpuAllocators(const se::Platform* platform,
                    absl::Span<se::StreamExecutor* const> stream_executors) {
  std::vector<se::MultiDeviceAdapter::AllocatorInfo> allocators;
  std::vector<std::unique_ptr<se::Stream>> streams;
  for (auto* executor : stream_executors) {
    auto stream = executor->CreateStream();
    CHECK_OK(stream) << "Failed to create stream for device "
                     << executor->device_ordinal();
    int32_t ordinal = executor->device_ordinal();
    se::Stream* stream_ptr = (*stream).get();
    streams.push_back(std::move(*stream));

    int64_t free_memory;
    int64_t total_memory;
    CHECK(executor->DeviceMemoryUsage(&free_memory, &total_memory))
        << "Failed to query available memory from device " << ordinal;

    // BFC allocator for default device memory space.
    auto sub = std::make_unique<se::DeviceMemAllocator>(
        executor, tsl::PlatformDeviceId(ordinal));
    tsl::BFCAllocator::Options opts;
    opts.allow_growth = true;
    auto bfc = std::make_shared<tsl::BFCAllocator>(
        std::move(sub), total_memory,
        absl::StrCat("XLA_backend_", ordinal, "_bfc"), opts);
    allocators.push_back(
        {std::move(bfc), stream_ptr, /*memory_space=*/0, ordinal});

    // Passthrough allocator for host memory space.
    auto host_alloc = std::make_shared<StreamExecutorAllocator>(
        executor, static_cast<int64_t>(se::MemorySpace::kHost));
    allocators.push_back(
        {std::move(host_alloc), stream_ptr,
         /*memory_space=*/static_cast<int64_t>(se::MemorySpace::kHost),
         ordinal});
  }

  return {std::move(streams), std::make_shared<se::MultiDeviceAdapter>(
                                  platform, std::move(allocators))};
}

}  // namespace

/* static */ absl::StatusOr<std::unique_ptr<Backend>> Backend::CreateBackend(
    const BackendOptions& options) {
  se::Platform* platform = options.platform();
  TF_ASSIGN_OR_RETURN(auto compiler, Compiler::GetForPlatform(platform->id()));
  TF_ASSIGN_OR_RETURN(
      auto stream_executors,
      PlatformUtil::GetStreamExecutors(platform, options.allowed_devices()));
  TF_ASSIGN_OR_RETURN(auto transfer_manager,
                      TransferManager::GetForPlatform(platform));
  TF_ASSIGN_OR_RETURN(auto computation_placer,
                      ComputationPlacer::GetForPlatform(platform->id()));
  std::unique_ptr<Backend> backend(new Backend(
      platform, std::move(compiler), stream_executors, transfer_manager,
      computation_placer, options.intra_op_parallelism_threads()));
  return std::move(backend);
}

/* static */ absl::StatusOr<std::unique_ptr<Backend>>
Backend::CreateDefaultBackend() {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetDefaultPlatform());
  BackendOptions backend_options;
  backend_options.set_platform(platform);
  return CreateBackend(backend_options);
}

absl::StatusOr<StreamPool::Ptr> Backend::BorrowStream(
    int device_ordinal, se::StreamPriority priority) {
  TF_ASSIGN_OR_RETURN(auto executor, stream_executor(device_ordinal));
  return BorrowStream(executor, priority);
}

absl::StatusOr<StreamPool::Ptr> Backend::BorrowStream(
    se::StreamExecutor* executor, se::StreamPriority priority) {
  absl::MutexLock l(mu_);
  if (!stream_pools_.contains(executor)) {
    stream_pools_.emplace(executor, std::make_unique<StreamPool>(executor));
  }
  return stream_pools_.at(executor)->BorrowStream(priority);
}

absl::StatusOr<std::vector<StreamPool::Ptr>> Backend::BorrowStreams(
    int device_ordinal, int num_streams, se::StreamPriority priority) {
  absl::MutexLock l(mu_);
  TF_ASSIGN_OR_RETURN(auto executor, stream_executor(device_ordinal));
  if (!stream_pools_.contains(executor)) {
    stream_pools_.emplace(executor, std::make_unique<StreamPool>(executor));
  }

  std::vector<StreamPool::Ptr> ptrs;
  for (int i = 0; i < num_streams; i++) {
    StreamPool::Ptr ptr = stream_pools_.at(executor)->BorrowStream(priority);
    ptrs.push_back(std::move(ptr));
  }
  return ptrs;
}

Backend::Backend(se::Platform* platform, std::unique_ptr<Compiler> compiler,
                 absl::Span<se::StreamExecutor* const> stream_executors,
                 TransferManager* transfer_manager,
                 ComputationPlacer* computation_placer,
                 int intra_op_parallelism_threads)
    : platform_(platform),
      compiler_(std::move(compiler)),
      transfer_manager_(transfer_manager),
      computation_placer_(computation_placer),
      stream_executors_(stream_executors.begin(), stream_executors.end()) {
  if (platform->id() == se::cuda::kCudaPlatformId ||
      platform->id() == se::rocm::kROCmPlatformId) {
    std::tie(allocator_streams_, memory_allocator_) =
        CreateGpuAllocators(platform, stream_executors_);
  } else {
    memory_allocator_ =
        std::make_shared<stream_executor::StreamExecutorAddressAllocator>(
            platform, stream_executors_);
  }

  CHECK(!stream_executors_.empty())
      << "Service found no devices for backend " << platform_->Name() << '.';

  if (platform->id() == se::host::kHostPlatformId) {
    const int num_threads = intra_op_parallelism_threads > 0
                                ? intra_op_parallelism_threads
                                : tsl::port::MaxParallelism();
    intra_op_thread_pool_ = std::make_unique<IntraOpThreadPool>(num_threads);
  }
}

Backend::~Backend() = default;

int Backend::default_device_ordinal() const {
  return default_stream_executor()->device_ordinal();
}

const Eigen::ThreadPoolDevice* Backend::eigen_intra_op_thread_pool_device()
    const {
  if (intra_op_thread_pool_ == nullptr) {
    return nullptr;
  }
  return intra_op_thread_pool_->device.get();
}

tsl::thread::ThreadPool* Backend::eigen_intra_op_thread_pool() const {
  if (intra_op_thread_pool_ == nullptr) {
    return nullptr;
  }
  return intra_op_thread_pool_->pool.get();
}

absl::StatusOr<se::StreamExecutor*> Backend::stream_executor(
    int device_ordinal) const {
  if (device_ordinal < 0 ||
      device_ordinal > stream_executors_.back()->device_ordinal()) {
    return InvalidArgument(
        "Invalid device ordinal value (%d). Valid range is [0, %d].",
        device_ordinal, stream_executors_.back()->device_ordinal());
  }
  for (auto* executor : stream_executors_) {
    if (executor->device_ordinal() == device_ordinal) {
      return executor;
    }
  }
  return InvalidArgument("device %s not supported by XLA service",
                         device_name(device_ordinal));
}

absl::StatusOr<bool> Backend::devices_equivalent(int device_ordinal_a,
                                                 int device_ordinal_b) const {
  // Use the name from device description to determine equivalence. This is a
  // bit crude but works for GPUs which is the important case where we compile
  // an executable for one GPU and want to know if it will run (well) on
  // another.
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor_a,
                      stream_executor(device_ordinal_a));
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor_b,
                      stream_executor(device_ordinal_b));
  return (executor_a->GetDeviceDescription().name() ==
          executor_b->GetDeviceDescription().name());
}

absl::Status Backend::ResetDevices() {
  return transfer_manager_->ResetDevices(stream_executors_);
}

}  // namespace xla

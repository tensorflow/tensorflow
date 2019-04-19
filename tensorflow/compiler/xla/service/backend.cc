/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/compiler/xla/service/backend.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

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
    const absl::optional<std::set<int>>& allowed_devices) {
  allowed_devices_ = allowed_devices;
  return *this;
}

const absl::optional<std::set<int>>& BackendOptions::allowed_devices() const {
  return allowed_devices_;
}

// Define this in .cc file to avoid having to include eigen or forward declare
// these types in the header.
struct Backend::IntraOpThreadPool {
  explicit IntraOpThreadPool(const int num_threads)
      : pool(new tensorflow::thread::ThreadPool(tensorflow::Env::Default(),
                                                "XLAEigen", num_threads)),
        device(new Eigen::ThreadPoolDevice(pool->AsEigenThreadPool(),
                                           pool->NumThreads())) {}

  std::unique_ptr<tensorflow::thread::ThreadPool> pool;
  std::unique_ptr<Eigen::ThreadPoolDevice> device;
};

/* static */ StatusOr<std::unique_ptr<Backend>> Backend::CreateBackend(
    const BackendOptions& options) {
  se::Platform* platform = options.platform();
  TF_ASSIGN_OR_RETURN(auto compiler, Compiler::GetForPlatform(platform));
  TF_ASSIGN_OR_RETURN(
      auto stream_executors,
      PlatformUtil::GetStreamExecutors(platform, options.allowed_devices()));
  TF_ASSIGN_OR_RETURN(auto transfer_manager,
                      TransferManager::GetForPlatform(platform));
  TF_ASSIGN_OR_RETURN(auto computation_placer,
                      ComputationPlacer::GetForPlatform(platform));
  std::unique_ptr<Backend> backend(
      new Backend(platform, compiler, stream_executors, transfer_manager,
                  computation_placer, options.intra_op_parallelism_threads()));
  return std::move(backend);
}

/* static */ StatusOr<std::unique_ptr<Backend>>
Backend::CreateDefaultBackend() {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetDefaultPlatform());
  BackendOptions backend_options;
  backend_options.set_platform(platform);
  return CreateBackend(backend_options);
}

StatusOr<StreamPool::Ptr> Backend::BorrowStream(int device_ordinal) {
  TF_ASSIGN_OR_RETURN(auto executor, stream_executor(device_ordinal));
  return BorrowStream(executor);
}

StatusOr<StreamPool::Ptr> Backend::BorrowStream(se::StreamExecutor* executor) {
  tensorflow::mutex_lock l(mu_);
  if (!stream_pools_.contains(executor)) {
    stream_pools_.emplace(executor, absl::make_unique<StreamPool>());
  }
  return stream_pools_.at(executor)->BorrowStream(executor);
}

Backend::Backend(se::Platform* platform, Compiler* compiler,
                 absl::Span<se::StreamExecutor* const> stream_executors,
                 TransferManager* transfer_manager,
                 ComputationPlacer* computation_placer,
                 int intra_op_parallelism_threads)
    : platform_(platform),
      compiler_(compiler),
      transfer_manager_(transfer_manager),
      computation_placer_(computation_placer) {
  // The given set of stream executors set may include invalid executors.
  for (se::StreamExecutor* exec : stream_executors) {
    if (exec != nullptr) {
      stream_executors_.push_back(exec);
    }
  }
  // Create a memory allocator for the valid stream executors.
  memory_allocator_ = absl::make_unique<StreamExecutorMemoryAllocator>(
      platform, stream_executors);
  CHECK(!stream_executors_.empty())
      << "Service found no devices for backend " << platform_->Name() << '.';

  if (platform->id() == se::host::kHostPlatformId) {
    const int num_threads = intra_op_parallelism_threads > 0
                                ? intra_op_parallelism_threads
                                : tensorflow::port::NumSchedulableCPUs();
    intra_op_thread_pool_.reset(new IntraOpThreadPool(num_threads));
  }
}

Backend::~Backend() {}

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

tensorflow::thread::ThreadPool* Backend::eigen_intra_op_thread_pool() const {
  if (intra_op_thread_pool_ == nullptr) {
    return nullptr;
  }
  return intra_op_thread_pool_->pool.get();
}

StatusOr<se::StreamExecutor*> Backend::stream_executor(
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

StatusOr<bool> Backend::devices_equivalent(int device_ordinal_a,
                                           int device_ordinal_b) {
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

Status Backend::ResetDevices() {
  return transfer_manager_->ResetDevices(stream_executors_);
}

}  // namespace xla

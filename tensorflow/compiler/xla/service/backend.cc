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

#include "tensorflow/compiler/xla/service/backend.h"

#include <algorithm>
#include <string>
#include <utility>

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/legacy_flags/backend_flags.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/eigen_thread_pool.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace se = ::perftools::gputools;

namespace xla {

// Define this in .cc file to avoid having to include eigen or forward declare
// these types in the header.
struct Backend::EigenThreadPoolWrapper {
  explicit EigenThreadPoolWrapper()
      : pool(new tensorflow::thread::ThreadPool(
            tensorflow::Env::Default(), "XLAEigen",
            tensorflow::port::NumSchedulableCPUs())),
        wrapper(new tensorflow::EigenThreadPoolWrapper(pool.get())),
        device(new Eigen::ThreadPoolDevice(wrapper.get(),
                                           wrapper->NumThreads())) {}

  std::unique_ptr<tensorflow::thread::ThreadPool> pool;
  std::unique_ptr<tensorflow::EigenThreadPoolWrapper> wrapper;
  std::unique_ptr<Eigen::ThreadPoolDevice> device;
};

/* static */ StatusOr<std::unique_ptr<Backend>> Backend::CreateBackend(
    perftools::gputools::Platform* platform, int64 replica_count) {
  if (replica_count == -1) {
    legacy_flags::BackendFlags* flags = legacy_flags::GetBackendFlags();
    replica_count = flags->xla_replicas;
  }
  TF_ASSIGN_OR_RETURN(auto compiler, Compiler::GetForPlatform(platform));
  TF_ASSIGN_OR_RETURN(auto stream_executors,
                      PlatformUtil::GetStreamExecutors(platform));
  TF_ASSIGN_OR_RETURN(auto transfer_manager,
                      TransferManager::GetForPlatform(platform));
  std::unique_ptr<Backend> backend(new Backend(
      replica_count, platform, compiler, stream_executors, transfer_manager));
  TF_RETURN_IF_ERROR(backend->PoolStreams(kInitialStreamsToPool,
                                          backend->default_stream_executor()));
  return std::move(backend);
}

/* static */ StatusOr<std::unique_ptr<Backend>>
Backend::CreateDefaultBackend() {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetDefaultPlatform());
  return CreateBackend(platform);
}

tensorflow::Status Backend::PoolStreams(int n, se::StreamExecutor* executor) {
  std::vector<std::unique_ptr<se::Stream>> primed;
  for (int i = 0; i < n; ++i) {
    TF_ASSIGN_OR_RETURN(auto stream, AcquireStream(executor));
    primed.emplace_back(std::move(stream));
  }
  for (int i = 0; i < n; ++i) {
    ReleaseStream(std::move(primed.back()));
    primed.pop_back();
  }
  return tensorflow::Status::OK();
}

StatusOr<std::unique_ptr<perftools::gputools::Stream>> Backend::AcquireStream(
    perftools::gputools::StreamExecutor* executor) {
  tensorflow::mutex_lock lock(mutex_);
  auto& cached_streams = cached_streams_[executor];
  if (!cached_streams.empty()) {
    auto result = std::move(cached_streams.back());
    cached_streams.pop_back();
    return std::move(result);
  }

  auto stream = MakeUnique<se::Stream>(executor);
  if (!stream->Init().ok()) {
    return InternalError("failed to initialize stream");
  }
  return std::move(stream);
}

void Backend::ReleaseStream(
    std::unique_ptr<perftools::gputools::Stream> stream) {
  tensorflow::mutex_lock lock(mutex_);
  auto& streams = cached_streams_[stream->parent()];
  streams.emplace_back(std::move(stream));
}

Backend::Backend(
    int64 replica_count, perftools::gputools::Platform* platform,
    Compiler* compiler,
    tensorflow::gtl::ArraySlice<se::StreamExecutor*> stream_executors,
    TransferManager* transfer_manager)
    : platform_(platform),
      compiler_(compiler),
      transfer_manager_(transfer_manager),
      replica_count_(replica_count) {
  // The given set of stream executors set may include invalid executors.
  for (se::StreamExecutor* exec : stream_executors) {
    if (exec != nullptr) {
      stream_executors_.push_back(exec);
    }
  }
  CHECK_GE(replica_count, 1) << "Must request at least 1 replica.";

  // Create a memory allocator for the valid stream executors.
  memory_allocator_ =
      MakeUnique<StreamExecutorMemoryAllocator>(platform, stream_executors);

  // First check that there are some non-null stream executors to avoid issuing
  // an error mentioning replicas in the common case of requesting just 1
  // replica, which means no replication.
  CHECK(!stream_executors_.empty())
      << "Service found no devices for backend " << platform_->Name() << '.';
  CHECK_GE(stream_executors_.size(), replica_count)
      << "Requested more replicas than there are devices for backend "
      << platform_->Name() << '.';

  if (platform->id() == se::host::kHostPlatformId) {
    inter_op_thread_pool_.reset(new tensorflow::thread::ThreadPool(
        tensorflow::Env::Default(), "xla_inter_op",
        tensorflow::port::NumSchedulableCPUs()));
    intra_op_thread_pool_wrapper_.reset(new EigenThreadPoolWrapper());
  }
}

Backend::~Backend() {}

int Backend::default_device_ordinal() const {
  return default_stream_executor()->device_ordinal();
}

StatusOr<std::vector<perftools::gputools::StreamExecutor*>> Backend::Replicas(
    int device_ordinal) const {
  if (stream_executors_[device_ordinal] == nullptr) {
    return InvalidArgument("device %s not supported by XLA service",
                           device_name(device_ordinal).c_str());
  }

  // Find replica_count_ stream executors starting from the given device
  // ordinal.
  std::vector<perftools::gputools::StreamExecutor*> replicas;
  for (se::StreamExecutor* exec : stream_executors_) {
    CHECK(exec != nullptr);
    if (exec->device_ordinal() >= device_ordinal) {
      replicas.push_back(exec);
      if (replicas.size() >= replica_count_) {
        return replicas;
      }
    }
  }

  return InvalidArgument(
      "Not enough devices for replicas for the device ordinal %d",
      device_ordinal);
}

std::vector<perftools::gputools::StreamExecutor*> Backend::Replicas() const {
  CHECK_GE(stream_executors_.size(), replica_count_);
  return Replicas(default_device_ordinal()).ValueOrDie();
}

tensorflow::thread::ThreadPool* Backend::inter_op_thread_pool() const {
  return inter_op_thread_pool_.get();
}

const Eigen::ThreadPoolDevice* Backend::eigen_intra_op_thread_pool_device()
    const {
  if (intra_op_thread_pool_wrapper_ == nullptr) return nullptr;
  return intra_op_thread_pool_wrapper_->device.get();
}

StatusOr<perftools::gputools::StreamExecutor*> Backend::stream_executor(
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
                         device_name(device_ordinal).c_str());
}

StatusOr<bool> Backend::devices_equivalent(int device_ordinal_a,
                                           int device_ordinal_b) {
  // Use the name from device description to determine equivalence. This is a
  // bit crude but works for GPUs which is the important case where we compile
  // an executable for one GPU and want to know if it will run (well) on
  // another.
  TF_ASSIGN_OR_RETURN(perftools::gputools::StreamExecutor * executor_a,
                      stream_executor(device_ordinal_a));
  TF_ASSIGN_OR_RETURN(perftools::gputools::StreamExecutor * executor_b,
                      stream_executor(device_ordinal_b));
  return (executor_a->GetDeviceDescription().name() ==
          executor_b->GetDeviceDescription().name());
}

Status Backend::ResetDevices() {
  return transfer_manager_->ResetDevices(stream_executors_);
}

}  // namespace xla

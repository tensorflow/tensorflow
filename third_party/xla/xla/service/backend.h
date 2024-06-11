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

#ifndef XLA_SERVICE_BACKEND_H_
#define XLA_SERVICE_BACKEND_H_

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/stream_pool.h"
#include "xla/service/transfer_manager.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"

namespace Eigen {
struct ThreadPoolDevice;
}

namespace xla {

// Options to configure the backend when it is created.
class BackendOptions {
 public:
  // Set the platform backing the backend, or nullptr for the default platform.
  BackendOptions& set_platform(se::Platform* platform);
  se::Platform* platform() const;

  // Sets the thread pool size for parallel execution of an individual operator.
  // The default value of -1 will result in initializing the thread pool with
  // the number of threads equal to the number of cores in the system.
  BackendOptions& set_intra_op_parallelism_threads(int num_threads);
  int intra_op_parallelism_threads() const;

  // Sets the allowed_devices for selectively constructing stream executors
  // on the platform.
  BackendOptions& set_allowed_devices(
      const std::optional<std::set<int>>& allowed_devices);
  const std::optional<std::set<int>>& allowed_devices() const;

 private:
  se::Platform* platform_ = nullptr;
  int intra_op_parallelism_threads_ = -1;
  std::optional<std::set<int>> allowed_devices_;
};

// Class which encapsulates an XLA backend. It includes everything necessary
// to compile and execute computations on a particular platform.
//
// It also offers a pooling API for creation/use of initialized streams:
//
//    StreamPool::Ptr stream = backend->BorrowStream().value();
class Backend {
 public:
  // Creates a new backend.
  static absl::StatusOr<std::unique_ptr<Backend>> CreateBackend(
      const BackendOptions& options);

  // Creates a backend for the default platform. The default platform is defined
  // in PlatformUtil.
  static absl::StatusOr<std::unique_ptr<Backend>> CreateDefaultBackend();

  ~Backend();

  // Accessors for the various objects.
  se::Platform* platform() const { return platform_; }
  Compiler* compiler() const { return compiler_; }
  se::DeviceMemoryAllocator* memory_allocator() const {
    return memory_allocator_.get();
  }
  std::shared_ptr<se::DeviceMemoryAllocator> shared_memory_allocator() const {
    return memory_allocator_;
  }
  TransferManager* transfer_manager() const { return transfer_manager_; }
  ComputationPlacer* computation_placer() const { return computation_placer_; }

  // Returns the number of devices of the platform type which are visible. Not
  // all of these devices may be usable by XLA.
  int device_count() const { return stream_executors_.size(); }

  // Returns the device ordinal number of the default device.
  int default_device_ordinal() const;

  // Returns stream executors of all supported devices for this backend. The
  // executors are ordered by the device ordinal.
  const std::vector<se::StreamExecutor*>& stream_executors() const {
    return stream_executors_;
  }

  // Returns the stream executor for the given device ordinal.
  absl::StatusOr<se::StreamExecutor*> stream_executor(int device_ordinal) const;

  // Returns the stream executor for the default device ordinal. This stream
  // executor can only be used when the number of computations is 1 (replication
  // can be > 1).
  se::StreamExecutor* default_stream_executor() const {
    CHECK(!stream_executors_.empty());
    return stream_executors_[0];
  }

  // Borrows a stream for use by the caller with a given priority, either by
  // grabbing it from an internal pool, or by constructing/initializating it,
  // and returns the result to the caller.
  absl::StatusOr<StreamPool::Ptr> BorrowStream(
      int device_ordinal,
      se::StreamPriority priority = se::StreamPriority::Default);
  absl::StatusOr<StreamPool::Ptr> BorrowStream(
      se::StreamExecutor* executor,
      se::StreamPriority priority = se::StreamPriority::Default);
  absl::StatusOr<std::vector<StreamPool::Ptr>> BorrowStreams(
      int device_ordinal, int num_streams,
      se::StreamPriority priority = se::StreamPriority::Default);

  // Returns a function to borrow streams with a given priority,
  // as `BorrowStreams` above does.
  // Purely for convenience, the caller could rather make this anonymous
  // function itself.
  std::function<absl::StatusOr<std::vector<StreamPool::Ptr>>(
      int, int, se::StreamPriority)>
  StreamBorrowerWithPriority() {
    return [this](int device_ordinal, int num_streams,
                  se::StreamPriority priority) {
      return BorrowStreams(device_ordinal, num_streams, priority);
    };
  }

  // Returns whether the given device ordinal of the backend is supported.
  bool device_ordinal_supported(int device_ordinal) const {
    return (device_ordinal >= 0 && device_ordinal < device_count() &&
            stream_executors_[device_ordinal] != nullptr);
  }

  // Return a string identifier for the given device, eg: "GPU:3".
  std::string device_name(int device_ordinal) const {
    return absl::StrCat(platform_->Name(), ":", device_ordinal);
  }

  // Returns true if the devices with the given ordinals are equivalent from
  // XLA's perspective. That is, an executable compiled for one device would
  // be equivalent to an executable compiled for the other.
  absl::StatusOr<bool> devices_equivalent(int device_ordinal_a,
                                          int device_ordinal_b);

  // For the host platform, returns the configured eigen threadpool device to be
  // used for scheduling work. For other platforms, returns NULL.
  const Eigen::ThreadPoolDevice* eigen_intra_op_thread_pool_device() const;
  tsl::thread::ThreadPool* eigen_intra_op_thread_pool() const;

  // Resets the devices associated with this backend.
  absl::Status ResetDevices();

 private:
  Backend(se::Platform* platform, Compiler* compiler,
          absl::Span<se::StreamExecutor* const> stream_executors,
          TransferManager* transfer_manager,
          ComputationPlacer* computation_placer,
          int intra_op_parallelism_threads);
  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;

  se::Platform* platform_;
  Compiler* compiler_;
  TransferManager* transfer_manager_;
  ComputationPlacer* computation_placer_;

  // Vector of stream executors. stream_executors_[0] is the default executor.
  std::vector<se::StreamExecutor*> stream_executors_;

  absl::Mutex mu_;

  // Mapping from stream executor to stream pools, used by `BorrowStream` above.
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<StreamPool>>
      stream_pools_ ABSL_GUARDED_BY(mu_);

  // The default memory allocator to use.
  // This must be a shared_ptr, as this is passed all the way down to the
  // cluster compilation. This allows asynchronous compilation to hold a
  // referecence until the compilation is finished.
  std::shared_ptr<se::StreamExecutorMemoryAllocator> memory_allocator_;

  // For the CPU backend, an Eigen threadpool device for use by Eigen code.
  struct IntraOpThreadPool;
  std::unique_ptr<IntraOpThreadPool> intra_op_thread_pool_;
};

}  // namespace xla

#endif  // XLA_SERVICE_BACKEND_H_

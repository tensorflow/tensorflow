/*
 * Copyright 2023 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PYTHON_IFRT_PROXY_SERVER_IFRT_BACKEND_H_
#define XLA_PYTHON_IFRT_PROXY_SERVER_IFRT_BACKEND_H_

#include <cstdint>
#include <functional>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/server/host_buffer.h"
#include "xla/python/ifrt_proxy/server/host_callback.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace ifrt {
namespace proxy {

// The abstract class `BackendInterface` defines the interface used by the IFRT
// service to interact with a variety of backend runtime system it can utilize.
class BackendInterface {
 public:
  virtual ~BackendInterface() = default;

  // Currently, responses (particularly those that carry buffer contents) can be
  // of non-trivial size. Once we figured out how best to move the data, we may
  // want to revise the shared_ptr below to the `IfrtResponse` proto itself.
  // Also, if and when we have a move-only Future in xla::ifrt, we may consider
  // changing it to std::unique_ptr.
  using Response = absl::StatusOr<std::shared_ptr<IfrtResponse>>;

  // Processes a given IFRT Request and returns a Future of an IfrtResponse.
  virtual Future<Response> Process(std::unique_ptr<IfrtRequest> request) = 0;
};

// IfrtBackend implements a backend that already has a linkable C++ client that
// conforms to the xla::ifrt API.
class IfrtBackend final : public BackendInterface {
 public:
  // Creates an returns an IfrtBackend that uses the given IFRT Client to
  // process the incoming proxy client requests. The `ifrt_client` param cannot
  // be a nullptr.
  static absl::StatusOr<std::unique_ptr<IfrtBackend>> Create(
      IfrtProxyVersion version, uint64_t session_id,
      std::unique_ptr<xla::ifrt::Client> ifrt_client,
      std::shared_ptr<HostBufferStore> host_buffer_store);

  ~IfrtBackend() override;

  // IFRT Proxy version negotiated between the client and the server.
  const IfrtProxyVersion& version() const { return version_; }

  Future<Response> Process(std::unique_ptr<IfrtRequest> request) override;

 private:
  // Generates unique handles for returning to the client. All object types
  // currently use this single "handle space".
  class HandleGenerator {
   public:
    uint64_t New();

    // Bulk allocates a given number of handles and saves them into the provided
    // Span.
    void BulkNew(absl::Span<uint64_t> handles);

   private:
    absl::Mutex mu_;
    uint64_t current_ ABSL_GUARDED_BY(mu_) = 1;
  };

  IfrtBackend(IfrtProxyVersion version, uint64_t session_id,
              std::unique_ptr<xla::ifrt::Client> ifrt_client,
              std::shared_ptr<HostBufferStore> host_buffer_store);

  // Executes the given function on the given thread pool and returns a future
  // that becomes ready when the function returns. If the thread pool is not
  // given, uses a default thread pool implementation that does not limit the
  // maximum number of threads.
  Future<Response> AsyncExecute(std::function<Response()> handle_fn,
                                tsl::thread::ThreadPool* thread_pool = nullptr);

  //////////////////////////////////////////////////////////////////////
  // Handlers for individual requests
  //

  Response HandleInit(std::unique_ptr<IfrtRequest> request);

  Future<Response> HandleCheckFutureRequest(
      std::unique_ptr<IfrtRequest> request);

  Response HandleMakeArrayFromHostBufferRequest(
      std::unique_ptr<IfrtRequest> request);
  Response HandleAssembleArrayFromSingleDeviceArraysRequest(
      std::unique_ptr<IfrtRequest> request);
  Future<Response> HandleCopyToHostBufferRequest(
      std::unique_ptr<IfrtRequest> request);
  Response HandleDisassembleIntoSingleDeviceArraysRequest(
      std::unique_ptr<IfrtRequest> request);
  Response HandleReshardRequest(std::unique_ptr<IfrtRequest> request);
  Response HandleFullyReplicatedShardRequest(
      std::unique_ptr<IfrtRequest> request);
  Future<Response> HandleCheckArrayReadyRequest(
      std::unique_ptr<IfrtRequest> request);
  Response HandleDeleteArrayRequest(std::unique_ptr<IfrtRequest> request);
  Response HandleIsArrayDeletedRequest(std::unique_ptr<IfrtRequest> request);
  Response HandleDestructArrayRequest(std::unique_ptr<IfrtRequest> request);

  Future<Response> HandleCompileRequest(std::unique_ptr<IfrtRequest> request);

  Future<Response> HandleLoadedExecutableMetadataRequest(
      std::unique_ptr<IfrtRequest> request);
  Response HandleLoadedExecutableExecuteRequest(
      std::unique_ptr<IfrtRequest> request);
  Response HandleLoadedExecutableDeleteRequest(
      std::unique_ptr<IfrtRequest> request);
  Response HandleLoadedExecutableIsDeletedRequest(
      std::unique_ptr<IfrtRequest> request);
  Response HandleLoadedExecutableDestructRequest(
      std::unique_ptr<IfrtRequest> request);

  Future<Response> HandleLoadedHostCallbackPollRequest(
      std::unique_ptr<IfrtRequest> request);
  Response HandleLoadedHostCallbackReturnRequest(
      std::unique_ptr<IfrtRequest> request);

  Response HandleGetDefaultDeviceAssignmentRequest(
      std::unique_ptr<IfrtRequest> request);

  //////////////////////////////////////////////////////////////////////
  // Convenient methods for object lookups
  //

  absl::StatusOr<std::shared_ptr<xla::ifrt::LoadedExecutable>>
  GetLoadedExecutable(uint64_t handle);

  absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> GetArray(uint64_t handle);
  absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> GetArrayLocked(
      uint64_t handle) ABSL_SHARED_LOCKS_REQUIRED(arrays_mutex_);

  HandleGenerator handle_generator_;

  // Must not change during the life of this object.
  const IfrtProxyVersion version_;
  const uint64_t session_id_;
  const std::unique_ptr<xla::ifrt::Client> client_;
  const std::shared_ptr<HostBufferStore> host_buffer_store_;

  absl::Mutex futures_mutex_;
  absl::flat_hash_map<uint64_t, Future<>> futures_
      ABSL_GUARDED_BY(futures_mutex_);

  absl::Mutex arrays_mutex_;
  absl::flat_hash_map<uint64_t, tsl::RCReference<xla::ifrt::Array>> arrays_
      ABSL_GUARDED_BY(arrays_mutex_);

  absl::Mutex executables_mutex_;
  absl::flat_hash_map<uint64_t, std::shared_ptr<xla::ifrt::LoadedExecutable>>
      executables_ ABSL_GUARDED_BY(executables_mutex_);

  absl::Mutex host_callback_queues_mutex_;
  absl::flat_hash_map<uint64_t, std::shared_ptr<RemoteLoadedHostCallbackQueue>>
      host_callback_queues_ ABSL_GUARDED_BY(host_callback_queues_mutex_);

  absl::Mutex host_callback_executions_mutex_;
  absl::flat_hash_map<uint64_t, RemoteLoadedHostCallbackQueue::ExecutionRequest>
      host_callback_executions_
          ABSL_GUARDED_BY(host_callback_executions_mutex_);

  absl::Mutex in_flight_count_mutex_;
  int64_t in_flight_count_ ABSL_GUARDED_BY(in_flight_count_mutex_) = 0;

  // Use a separate thread pool for compilation as XLA compilation often
  // requires a bigger stack.
  tsl::thread::ThreadPool compile_thread_pool_;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_SERVER_IFRT_BACKEND_H_

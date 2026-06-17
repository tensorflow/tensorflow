// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/client/grpc_host_buffer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/client_callback.h"
#include "grpcpp/support/sync_stream.h"
#include "xla/pjrt/distributed/util.h"
#include "xla/pjrt/semaphore.h"
#include "xla/python/ifrt_proxy/client/global_flags.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.grpc.pb.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/prof_util.h"
#include "xla/python/ifrt_proxy/common/transfer_util.h"
#include "xla/python/ifrt_proxy/common/versions.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/protobuf/status.pb.h"
#include "tsl/platform/unbounded_work_queue.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace ifrt {
namespace proxy {

static constexpr int64_t kChunkSize = 1024 * 1024;

static void SetDataFromStringView(GrpcHostBufferStoreRequest& req,
                                  absl::string_view data) {
#if defined(PLATFORM_GOOGLE)
  req.set_alias_data(data);
#else
  // TODO(b/325306748): Find a way to not do a memory-copy.
  req.set_data(std::string(data));
#endif
}

static std::shared_ptr<xla::Semaphore::ScopedReservation>
ScopedAcquireSemaphore(std::optional<xla::Semaphore>& semaphore) {
  if (!semaphore.has_value()) {
    return nullptr;
  }
  auto reservation = semaphore->ScopedAcquire(1);
  return std::make_shared<xla::Semaphore::ScopedReservation>(
      std::move(reservation));
}

GrpcClientHostBufferStore::GrpcClientHostBufferStore(
    std::shared_ptr<grpc::GrpcIfrtService::StubInterface> stub,
    IfrtProxyVersion version, uint64_t session_id)
    : stub_(std::move(stub)),
      version_(std::move(version)),
      session_id_(session_id),
      work_queue_(std::make_unique<tsl::UnboundedWorkQueue>(
          tsl::Env::Default(), "HostBufferStoreLookupsWorkQueue")) {
  if (GetGlobalClientFlags()->grpc_max_ongoing_host_buffer_stores > 0) {
    store_throttler_.emplace(
        GetGlobalClientFlags()->grpc_max_ongoing_host_buffer_stores);
  }
  if (GetGlobalClientFlags()->grpc_max_ongoing_host_buffer_lookups > 0) {
    lookup_throttler_.emplace(
        GetGlobalClientFlags()->grpc_max_ongoing_host_buffer_lookups);
  }
}

GrpcClientHostBufferStore::~GrpcClientHostBufferStore() {
  LOG(INFO) << "Waiting for destruction of HostBufferStoreLookupsWorkQueue...";
  work_queue_.reset();
  LOG(INFO) << "Destructed HostBufferStoreLookupsWorkQueue.";
}

tsl::Future<> GrpcClientHostBufferStore::StoreToDisk(uint64_t handle,
                                                     absl::string_view data) {
  auto [promise, future] = tsl::MakePromise<>();

  XFlowHelper flow("GrpcClientHostBufferStore::StoreToDisk");
  flow.InstantActivity<XFlowHelper::kSend>();

  work_queue_->Schedule([this, handle, promise = std::move(promise).ToShared(),
                         data, flow]() mutable -> void {
    auto span = flow.Span<XFlowHelper::kRecv>();

    GrpcHostBufferReadFromDiskRequest request;
    request.mutable_metadata()->set_session_id(session_id_);
    request.mutable_metadata()->set_handle(handle);
    request.mutable_metadata()->set_buffer_size(data.size());
    VLOG(3) << "GrpcClientHostBufferStore::StoreToDisk start "
            << request.ShortDebugString();

    auto file_path = LargeTransferFilePath(handle);
    if (file_path == std::nullopt) {
      promise->Set(absl::FailedPreconditionError(
          "IFRT proxy: cannot retrieve file path for StoreToDisk()."));
      return;
    }
    if (auto s = tsl::WriteStringToFile(tsl::Env::Default(), *file_path, data);
        !s.ok()) {
      LOG(WARNING) << "Failed to write to file " << *file_path << ": " << s;
      promise->Set(std::move(s));
      return;
    }

    ::grpc::ClientContext context;
    GrpcHostBufferReadFromDiskResponse response;
    promise->Set(xla::FromGrpcStatus(
        stub_->HostBufferReadFromDisk(&context, request, &response)));
    VLOG(3) << "GrpcClientHostBufferStore::StoreToDisk done "
            << request.ShortDebugString();
  });
  return std::move(future);
}

tsl::Future<> GrpcClientHostBufferStore::Store(uint64_t handle,
                                               absl::string_view data) {
  if (version_.protocol_version() >=
      protocol_version::kGrpcAllowLargeTransferOptimizationViaSharedDirectory) {
    int64_t threshold = GetGlobalClientFlags()
                            ->grpc_large_transfer_optimization_threshold_bytes;
    if (data.size() >= threshold) {
      return StoreToDisk(handle, data);
    }
  }

  auto [promise, future] = tsl::MakePromise<>();

  XFlowHelper flow("GrpcClientHostBufferStore::StoreAsync");
  flow.InstantActivity<XFlowHelper::kSend>();

  std::unique_ptr<std::string> buffered_data;

  auto reservation = ScopedAcquireSemaphore(store_throttler_);
  work_queue_->Schedule([this, reservation = std::move(reservation), handle,
                         promise = std::move(promise).ToShared(), data,
                         flow]() mutable -> void {
    auto span = flow.Span<XFlowHelper::kRecv>();
    GrpcHostBufferStoreMetadata metadata;
    metadata.set_session_id(session_id_);
    metadata.set_handle(handle);
    metadata.set_buffer_size(data.size());
    VLOG(3) << "GrpcClientHostBufferStore::Store start "
            << metadata.ShortDebugString();

    ::grpc::ClientContext context;
    context.AddMetadata("ifrt-proxy-grpc-host-buffer-store-metadata-bin",
                        metadata.SerializeAsString());

    GrpcHostBufferStoreResponse response;
    auto writer = stub_->HostBufferStore(&context, &response);

    {
      tsl::profiler::TraceMe trace_me_send_data([size = data.size()]() {
        return tsl::profiler::TraceMeEncode(
            "GrpcClientHostBufferStore::StoreAsync_Send", {{"size", size}});
      });
      for (int64_t offset = 0; offset < data.size(); offset += kChunkSize) {
        GrpcHostBufferStoreRequest request;
        SetDataFromStringView(request, data.substr(offset, kChunkSize));
        writer->Write(request);
      }

      if (!writer->WritesDone()) {
        absl::Status s = xla::FromGrpcStatus(writer->Finish());
        promise->Set(absl::InternalError(absl::StrCat(
            "Failed to write all host buffer chunks, Finish() returned: ",
            s.ToString())));
        return;
      }
    }

    VLOG(3) << "GrpcClientHostBufferStore::Store done "
            << metadata.ShortDebugString();
    promise->Set(xla::FromGrpcStatus(writer->Finish()));
  });
  return std::move(future);
}

tsl::Future<> GrpcClientHostBufferStore::Store(uint64_t handle,
                                               const absl::Cord& data) {
  // The current implementation synchronously sends host buffer chunks. We may
  // consider making it asynchronous if the caller can leverage such asynchrony.
  tsl::profiler::TraceMe traceme("GrpcClientHostBufferStore::StoreSync");

  GrpcHostBufferStoreMetadata metadata;
  metadata.set_session_id(session_id_);
  metadata.set_handle(handle);
  metadata.set_buffer_size(data.size());
  VLOG(3) << "GrpcClientHostBufferStore::Store start "
          << metadata.ShortDebugString();

  ::grpc::ClientContext context;
  context.AddMetadata("ifrt-proxy-grpc-host-buffer-store-metadata-bin",
                      metadata.SerializeAsString());

  auto reservation = ScopedAcquireSemaphore(store_throttler_);

  GrpcHostBufferStoreResponse response;
  auto writer = stub_->HostBufferStore(&context, &response);

  {
    tsl::profiler::TraceMe trace_me_send_data([size = data.size()]() {
      return tsl::profiler::TraceMeEncode(
          "GrpcClientHostBufferStore::StoreAsync_Send", {{"size", size}});
    });
    for (absl::string_view chunk : data.Chunks()) {
      for (int64_t offset = 0; offset < chunk.size(); offset += kChunkSize) {
        GrpcHostBufferStoreRequest request;
        SetDataFromStringView(request, chunk.substr(offset, kChunkSize));
        writer->Write(request);
      }
    }
    if (!writer->WritesDone()) {
      absl::Status s = xla::FromGrpcStatus(writer->Finish());
      return tsl::Future<>(absl::InternalError(absl::StrCat(
          "Failed to write all host buffer chunks, Finish() returned: ",
          s.ToString())));
    }
  }
  VLOG(3) << "GrpcClientHostBufferStore::Store done "
          << metadata.ShortDebugString();

  return tsl::Future<>(xla::FromGrpcStatus(writer->Finish()));
}

tsl::Future<absl::Cord> GrpcClientHostBufferStore::Lookup(uint64_t handle) {
  auto [promise, future] = tsl::MakePromise<absl::Cord>();

  XFlowHelper flow("GrpcClientHostBufferStore::Lookup");
  flow.InstantActivity<XFlowHelper::kSend>();

  auto reservation = ScopedAcquireSemaphore(lookup_throttler_);
  work_queue_->Schedule([this, reservation = std::move(reservation), handle,
                         promise = std::move(promise).ToShared(),
                         flow]() mutable -> void {
    auto span = flow.Span<XFlowHelper::kRecv>();
    GrpcHostBufferLookupRequest request;
    request.set_handle(handle);
    request.set_session_id(session_id_);
    VLOG(3) << "GrpcClientHostBufferStore::Lookup start "
            << request.ShortDebugString();

    ::grpc::ClientContext context;

    std::unique_ptr<::grpc::ClientReaderInterface<GrpcHostBufferLookupResponse>>
        stream = stub_->HostBufferLookup(&context, request);

    absl::Cord data;
    GrpcHostBufferLookupResponse response;
    while (stream->Read(&response)) {
      data.Append(response.data());
    }

    absl::Status status = xla::FromGrpcStatus(stream->Finish());
    if (status.ok()) {
      promise->Set(std::move(data));
    } else {
      promise->Set(status);
    }
    VLOG(3) << "GrpcClientHostBufferStore::Lookup done "
            << request.ShortDebugString();
  });

  return std::move(future);
}

tsl::Future<> GrpcClientHostBufferStore::Delete(uint64_t handle) {
  GrpcHostBufferDeleteRequest request;
  request.set_session_id(session_id_);
  request.set_handle(handle);
  VLOG(3) << "GrpcClientHostBufferStore::Delete start "
          << request.ShortDebugString();

  ::grpc::ClientContext context;
  GrpcHostBufferDeleteResponse response;
  auto result = xla::FromGrpcStatus(
      stub_->HostBufferDelete(&context, request, &response));
  VLOG(3) << "GrpcClientHostBufferStore::Delete done "
          << request.ShortDebugString();
  return tsl::Future<>(result);
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

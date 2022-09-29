/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_service.h"

#include <chrono>  // NOLINT

#include "grpcpp/support/byte_buffer.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/platform/coding.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_rpc_support.h"

namespace tensorflow {
namespace {
using ::tensorflow::tpu::CompilationCacheEntryRef;
using ::tensorflow::tpu::TpuCompilationCacheEntry;
using ::tensorflow::tpu::TpuCompilationCacheInterface;

static constexpr int kGetTpuProgramServingThreads = 32;
}  // namespace

TpuCompilationCacheService::TpuCompilationCacheService(
    ::grpc::ServerBuilder* server_builder, TpuCompilationCacheInterface* cache)
    : running_(true),
      cache_(cache),
      server_builder_(server_builder),
      cq_(server_builder_->AddCompletionQueue()),
      thread_pool_(absl::make_unique<thread::ThreadPool>(
          Env::Default(), "TpuCompilationCacheService",
          kGetTpuProgramServingThreads)) {
  cache_->Ref();
  server_builder_->RegisterService(&service_);
}

TpuCompilationCacheService::~TpuCompilationCacheService() {
  // This ordering is important. We must first shutdown our CQ and allow the
  // polling thread and dispatch pool to shutdown before releasing our cache
  // reference. The gRPC server must be Shutdown() by this point or we will
  // deadlock here.  The running_ boolean is necessary to avoid adding new
  // operations to the CQ after is has shutdown.
  running_ = false;
  cq_->Shutdown();
  polling_thread_.reset();
  thread_pool_.reset();
  cache_->Unref();
}

void TpuCompilationCacheService::Start() {
  server_ = server_builder_->BuildAndStart();
  ThreadOptions opts;
  polling_thread_.reset(Env::Default()->StartThread(
      opts, "TpuCompilationCachePoller", [this]() { HandleRPCsLoop(); }));
}

bool TpuCompilationCacheService::Shutdown(int timeout_sec) {
  if (server_ != nullptr) {
    std::chrono::system_clock::time_point timeout =
        std::chrono::system_clock::now() + std::chrono::seconds(timeout_sec);
    server_->Shutdown(std::chrono::system_clock::now() +
                      std::chrono::seconds(timeout_sec));
    if (std::chrono::system_clock::now() >= timeout) {
      return false;
    }
    return true;
  } else {
    return false;
  }
}

void TpuCompilationCacheService::SetMemoryQuota(size_t max_bytes) {
  ::grpc::ResourceQuota quota;
  quota.Resize(max_bytes);
  server_builder_->SetResourceQuota(quota);
}

// Fetch a cache result for the given request and serialize the result directly
// into a ByteBuffer.
void TpuCompilationCacheService::GetTpuProgram(GetTpuProgramCall* call) {
  std::unique_ptr<CompilationCacheEntryRef> entry;

  VLOG(1) << "GetTpuProgram: " << call->request.DebugString();
  Status s;
  switch (call->request.key_oneof_case()) {
    case tpu::GetTpuProgramRequest::kKey:
      s = cache_->Lookup(call->request.key(), &entry);
      break;

    case tpu::GetTpuProgramRequest::kUidAndIndex:
      s = cache_->Lookup(call->request.uid_and_index().uid(),
                         call->request.uid_and_index().proto_index(), &entry);
      break;

    default:
      s = errors::Internal("Bad GetTpuProgram RPC request oneof case ",
                           call->request.key_oneof_case());
      break;
  }
  if (!s.ok()) {
    return call->SendResponse(ToGrpcStatus(s));
  }

  s = entry->ToSubEntryRef(call->request.fetch_target());
  if (!s.ok()) {
    return call->SendResponse(::grpc::Status(
        ::grpc::StatusCode::INVALID_ARGUMENT,
        absl::StrCat(
            "Error getting the fetching target ",
            CompilationCacheFetchTarget_Name(call->request.fetch_target())),
        s.error_message()));
  }

  TpuCompilationCacheEntry cache_entry = entry->get();
  if (cache_entry.tpu_program_group() == nullptr) {
    // It's possible that the sharding/unsharding entry does not exist, but the
    // main entry must exist.
    CHECK_NE(call->request.fetch_target(),
             tpu::CompilationCacheFetchTarget::MAIN);
  }

  xla::StatusOr<std::vector<::grpc::Slice>> buffer_slices =
      tpu::SerializeCacheEntryToBufferSlices(cache_entry);

  if (!buffer_slices.ok()) {
    return call->SendResponse(ToGrpcStatus(buffer_slices.status()));
  }

  call->response =
      ::grpc::ByteBuffer{&buffer_slices.value()[0], buffer_slices->size()};
  return call->SendResponse(::grpc::Status());
}

void TpuCompilationCacheService::HandleGetTpuProgram(GetTpuProgramCall* call) {
  thread_pool_->Schedule([this, call]() { GetTpuProgram(call); });
  if (running_) {
    GetTpuProgramCall::EnqueueRequestForMethod(
        &service_, cq_.get(),
        static_cast<int>(ServiceType::MethodId::kGetTpuProgram),
        &TpuCompilationCacheService::HandleGetTpuProgram,
        /*supports_cancel=*/false);
  }
}

void TpuCompilationCacheService::HandleRPCsLoop() {
  void* tag;
  bool ok;

  for (int i = 0; i < 50; ++i) {
    GetTpuProgramCall::EnqueueRequestForMethod(
        &service_, cq_.get(),
        static_cast<int>(ServiceType::MethodId::kGetTpuProgram),
        &TpuCompilationCacheService::HandleGetTpuProgram,
        /*supports_cancel=*/false);
  }

  while (cq_->Next(&tag, &ok)) {
    VLOG(2) << "HandleRPCS: " << tag;
    UntypedCall<TpuCompilationCacheService>::Tag* callback_tag =
        static_cast<UntypedCall<TpuCompilationCacheService>::Tag*>(tag);
    callback_tag->OnCompleted(this, ok);
  }

  VLOG(2) << "Cache thread shutting down.";
}
}  // namespace tensorflow

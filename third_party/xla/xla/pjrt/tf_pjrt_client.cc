/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/pjrt/tf_pjrt_client.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/pjrt_client.h"

namespace xla {

TfPjRtBuffer::TfPjRtBuffer(TfPjRtClient* client,
                           std::unique_ptr<PjRtBuffer> wrapped)
    : client_(client), wrapped_(std::move(wrapped)) {
  client_->TrackBuffer(this);
}

TfPjRtBuffer::~TfPjRtBuffer() { client_->UntrackBuffer(this); }

PjRtClient* TfPjRtBuffer::client() const { return client_; }
PjRtClient* TfPjRtExecutable::client() const { return client_; }

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfPjRtBuffer::CopyToDevice(
    PjRtDevice* dst_device) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> result,
                      wrapped_->CopyToDevice(dst_device));
  return std::unique_ptr<PjRtBuffer>(
      std::make_unique<TfPjRtBuffer>(client_, std::move(result)));
}

TfPjRtExecutable::TfPjRtExecutable(
    TfPjRtClient* client, std::unique_ptr<PjRtLoadedExecutable> wrapped)
    : client_(client), wrapped_(std::move(wrapped)) {}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
TfPjRtExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<>>>& returned_futures) {
  std::vector<std::vector<PjRtBuffer*>> unwrapped_argument_handles;
  unwrapped_argument_handles.reserve(argument_handles.size());
  for (auto& handles : argument_handles) {
    unwrapped_argument_handles.emplace_back();
    auto& unwrapped_handles = unwrapped_argument_handles.back();
    unwrapped_handles.reserve(handles.size());
    for (PjRtBuffer* buffer : handles) {
      unwrapped_handles.push_back(
          tensorflow::down_cast<TfPjRtBuffer*>(buffer)->wrapped());
    }
  }
  TF_ASSIGN_OR_RETURN(auto out, wrapped_->Execute(unwrapped_argument_handles,
                                                  options, returned_futures));
  for (auto& buffer_list : out) {
    for (std::unique_ptr<PjRtBuffer>& buffer : buffer_list) {
      buffer = std::make_unique<TfPjRtBuffer>(client_, std::move(buffer));
    }
  }
  return out;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
TfPjRtExecutable::ExecuteSharded(absl::Span<PjRtBuffer* const> argument_handles,
                                 PjRtDevice* device,
                                 const ExecuteOptions& options,
                                 std::optional<PjRtFuture<>>& returned_future,
                                 bool fill_future) {
  std::vector<PjRtBuffer*> unwrapped_argument_handles;
  unwrapped_argument_handles.reserve(argument_handles.size());
  for (PjRtBuffer* buffer : argument_handles) {
    unwrapped_argument_handles.push_back(
        tensorflow::down_cast<TfPjRtBuffer*>(buffer)->wrapped());
  }
  TF_ASSIGN_OR_RETURN(auto out, wrapped_->ExecuteSharded(
                                    unwrapped_argument_handles, device, options,
                                    returned_future, fill_future));
  for (std::unique_ptr<PjRtBuffer>& buffer : out) {
    buffer = std::make_unique<TfPjRtBuffer>(client_, std::move(buffer));
  }
  return out;
}
absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
TfPjRtExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
  std::vector<PjRtBuffer*> unwrapped_argument_handles;
  unwrapped_argument_handles.reserve(argument_handles.size());
  for (PjRtBuffer* buffer : argument_handles) {
    unwrapped_argument_handles.push_back(
        tensorflow::down_cast<TfPjRtBuffer*>(buffer)->wrapped());
  }
  TF_ASSIGN_OR_RETURN(auto out, wrapped_->ExecutePortable(
                                    unwrapped_argument_handles, device, options,
                                    returned_future, fill_future));
  for (std::unique_ptr<PjRtBuffer>& buffer : out) {
    buffer = std::make_unique<TfPjRtBuffer>(client_, std::move(buffer));
  }
  return out;
}

TfPjRtClient::TfPjRtClient(std::unique_ptr<PjRtClient> wrapped)
    : wrapped_(std::move(wrapped)) {
  LOG(INFO) << "TfPjRtClient created.";
  int num_mutexes = wrapped_->addressable_device_count();
  alive_buffers_ = std::vector<DeviceBuffers>(num_mutexes);
  for (int i = 0; i < num_mutexes; ++i) {
    mutex_id_from_device_id_.insert(
        {wrapped_->addressable_devices()[i]->id(), i});
  }
}

TfPjRtClient::~TfPjRtClient() { LOG(INFO) << "TfPjRtClient destroyed."; }

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfPjRtClient::WrapBuffer(
    absl::StatusOr<std::unique_ptr<PjRtBuffer>> to_wrap) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> buffer, std::move(to_wrap));
  return std::unique_ptr<PjRtBuffer>(
      std::make_unique<TfPjRtBuffer>(this, std::move(buffer)));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfPjRtClient::WrapExecutable(
    absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> to_wrap) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtLoadedExecutable> executable,
                      std::move(to_wrap));
  return std::unique_ptr<PjRtLoadedExecutable>(
      std::make_unique<TfPjRtExecutable>(this, std::move(executable)));
}

static int GetMutexId(
    const TfPjRtBuffer* buffer,
    const absl::flat_hash_map<int, int>& mutex_id_from_device_id) {
  auto iters = mutex_id_from_device_id.find(buffer->wrapped()->device()->id());
  CHECK(iters != mutex_id_from_device_id.end())
      << "Mutex id not found for device id: "
      << buffer->wrapped()->device()->id();
  return iters->second;
}

void TfPjRtClient::TrackBuffer(TfPjRtBuffer* buffer) {
  int mutex_id = GetMutexId(buffer, mutex_id_from_device_id_);
  {
    absl::MutexLock lock(&alive_buffers_[mutex_id].mu);
    alive_buffers_[mutex_id].alive_buffers.insert(buffer);
  }
}

void TfPjRtClient::UntrackBuffer(const TfPjRtBuffer* buffer) {
  if (buffer->wrapped() == nullptr) {
    return;
  }
  int mutex_id = GetMutexId(buffer, mutex_id_from_device_id_);
  {
    absl::MutexLock lock(&alive_buffers_[mutex_id].mu);
    alive_buffers_[mutex_id].alive_buffers.erase(buffer);
  }
}

void TfPjRtClient::DestroyWrappedBuffersAndClient() {
  int num_mutexes = alive_buffers_.size();
  for (int i = 0; i < num_mutexes; ++i) {
    absl::MutexLock lock(&alive_buffers_[i].mu);
    for (auto* buffer : alive_buffers_[i].alive_buffers) {
      buffer->DestroyWrappedBuffer();
    }
  }
  wrapped_.reset(nullptr);
  LOG(INFO) << "TfPjRtClient::DestroyWrappedBuffersAndClient completed.";
}

std::unique_ptr<TfPjRtClient> TfPjRtClient::CreateTfPjRtClient(
    std::unique_ptr<PjRtClient> wrapped) {
  return std::make_unique<TfPjRtClient>(std::move(wrapped));
}

}  // namespace xla

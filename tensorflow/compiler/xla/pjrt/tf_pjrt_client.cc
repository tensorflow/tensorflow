/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/pjrt/tf_pjrt_client.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace xla {

TfPjRtBuffer::TfPjRtBuffer(TfPjRtClient* client,
                           std::unique_ptr<PjRtBuffer> wrapped)
    : client_(client), wrapped_(std::move(wrapped)) {
  client_->TrackBuffer(this);
}

TfPjRtBuffer::~TfPjRtBuffer() { client_->UntrackBuffer(this); }

PjRtClient* TfPjRtBuffer::client() const { return client_; }
PjRtClient* TfPjRtExecutable::client() const { return client_; }

StatusOr<std::unique_ptr<PjRtBuffer>> TfPjRtBuffer::CopyToDevice(
    PjRtDevice* dst_device) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> result,
                      wrapped_->CopyToDevice(dst_device));
  return std::unique_ptr<PjRtBuffer>(
      std::make_unique<TfPjRtBuffer>(client_, std::move(result)));
}

TfPjRtExecutable::TfPjRtExecutable(
    TfPjRtClient* client, std::unique_ptr<PjRtLoadedExecutable> wrapped)
    : client_(client), wrapped_(std::move(wrapped)) {}

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
TfPjRtExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
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

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
TfPjRtExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
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
StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
TfPjRtExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
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
}

TfPjRtClient::~TfPjRtClient() { LOG(INFO) << "TfPjRtClient destroyed."; }

StatusOr<std::unique_ptr<PjRtBuffer>> TfPjRtClient::WrapBuffer(
    StatusOr<std::unique_ptr<PjRtBuffer>> to_wrap) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> buffer, std::move(to_wrap));
  return std::unique_ptr<PjRtBuffer>(
      std::make_unique<TfPjRtBuffer>(this, std::move(buffer)));
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>> TfPjRtClient::WrapExecutable(
    StatusOr<std::unique_ptr<PjRtLoadedExecutable>> to_wrap) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtLoadedExecutable> executable,
                      std::move(to_wrap));
  return std::unique_ptr<PjRtLoadedExecutable>(
      std::make_unique<TfPjRtExecutable>(this, std::move(executable)));
}

void TfPjRtClient::TrackBuffer(TfPjRtBuffer* buffer) {
  mu_.Lock();
  alive_buffers_.insert(buffer);
  mu_.Unlock();
}

void TfPjRtClient::UntrackBuffer(const TfPjRtBuffer* buffer) {
  mu_.Lock();
  alive_buffers_.erase(buffer);
  mu_.Unlock();
}

void TfPjRtClient::DestroyWrappedBuffersAndClient() {
  mu_.Lock();
  for (auto* buffer : alive_buffers_) {
    buffer->DestroyWrappedBuffer();
  }
  mu_.Unlock();
  wrapped_.reset(nullptr);
  LOG(INFO) << "TfPjRtClient::DestroyWrappedBuffersAndClient completed.";
}

std::unique_ptr<TfPjRtClient> TfPjRtClient::CreateTfPjRtClient(
    std::unique_ptr<PjRtClient> wrapped) {
  return std::make_unique<TfPjRtClient>(std::move(wrapped));
}

}  // namespace xla

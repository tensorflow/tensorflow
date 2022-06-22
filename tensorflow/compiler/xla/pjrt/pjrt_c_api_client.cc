/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/pjrt_c_api_client.h"

#include <string>
#include <utility>

// TODO(skyewm): remove when everything goes through C API
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "tensorflow/core/tpu/pjrt_api.h"

namespace xla {

PjRtCApiClient::PjRtCApiClient(
    const PJRT_Api* c_api, PJRT_Client* c_client,
    std::vector<std::unique_ptr<PjRtCApiDevice>> devices)
    : c_api_(c_api), c_client_(c_client), owned_devices_(std::move(devices)) {
  wrapped_ = c_client_->client.get();

  for (int i = 0; i < owned_devices_.size(); ++i) {
    const std::unique_ptr<PjRtCApiDevice>& device = owned_devices_[i];
    wrapped_device_map_[wrapped_->devices()[i]] = device.get();
    device->SetClient(this);
    devices_.push_back(device.get());
    if (device->IsAddressable()) {
      int idx = device->local_hardware_id();
      if (idx >= addressable_devices_.size()) {
        addressable_devices_.resize(idx + 1);
      }
      CHECK(addressable_devices_[idx] == nullptr) << idx;
      addressable_devices_[idx] = device.get();
    }
  }
}

absl::string_view PjRtCApiClient::platform_name() const {
  PJRT_Client_PlatformName_Args args;
  args.client = c_client_;
  args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  args.priv = nullptr;
  PJRT_Error* error = c_api_->PJRT_Client_PlatformName(&args);
  // TODO(b/236710439): handle error
  CHECK(error == nullptr);

  absl::string_view platform_name(args.platform_name, args.platform_name_size);
  return platform_name;
}

PjRtCApiClient::~PjRtCApiClient() {
  PJRT_Client_Destroy_Args free_args;
  free_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  free_args.priv = nullptr;
  free_args.client = c_client_;
  PJRT_Error* error = c_api_->PJRT_Client_Destroy(&free_args);
  // TODO(skyewm): handle error
  CHECK(error == nullptr);
}

StatusOr<std::optional<std::string>> PjRtCApiClient::ExecutableFingerprint(
    const PjRtExecutable& executable) const {
  return wrapped_->ExecutableFingerprint(
      *PjRtCApiExecutable::GetWrapped(&executable));
}

StatusOr<std::string> PjRtCApiClient::SerializeExecutable(
    const PjRtExecutable& executable) const {
  return wrapped_->SerializeExecutable(
      *PjRtCApiExecutable::GetWrapped(&executable));
}

StatusOr<std::uintptr_t> PjRtCApiClient::UnsafeBufferPointer(
    PjRtBuffer* buffer) {
  return wrapped_->UnsafeBufferPointer(PjRtCApiBuffer::GetWrapped(buffer));
}

StatusOr<std::unique_ptr<PjRtExecutable>> PjRtCApiClient::WrapExecutable(
    StatusOr<std::unique_ptr<PjRtExecutable>> to_wrap) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                      std::move(to_wrap));
  return std::unique_ptr<PjRtExecutable>(
      std::make_unique<PjRtCApiExecutable>(this, std::move(executable)));
}

StatusOr<std::unique_ptr<PjRtBuffer>> PjRtCApiClient::WrapBuffer(
    StatusOr<std::unique_ptr<PjRtBuffer>> to_wrap) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> buffer, std::move(to_wrap));
  return std::unique_ptr<PjRtBuffer>(
      std::make_unique<PjRtCApiBuffer>(this, std::move(buffer)));
}

PjRtCApiExecutable::PjRtCApiExecutable(PjRtCApiClient* client,
                                       std::unique_ptr<PjRtExecutable> wrapped)
    : client_(client), wrapped_(std::move(wrapped)) {
  addressable_devices_.reserve(wrapped_->addressable_devices().size());
  for (PjRtDevice* device : wrapped_->addressable_devices()) {
    addressable_devices_.push_back(client_->GetCApiDevice(device));
  }
}

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
PjRtCApiExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
  std::vector<std::vector<PjRtBuffer*>> wrapped_args;
  for (const std::vector<PjRtBuffer*>& args : argument_handles) {
    wrapped_args.push_back(PjRtCApiBuffer::GetWrappedVector(args));
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> out,
      wrapped_->Execute(wrapped_args, options, returned_futures));

  for (auto& buffer_list : out) {
    for (std::unique_ptr<PjRtBuffer>& buffer : buffer_list) {
      buffer = std::make_unique<PjRtCApiBuffer>(client_, std::move(buffer));
    }
  }
  return out;
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtCApiExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
  std::vector<PjRtBuffer*> wrapped_args =
      PjRtCApiBuffer::GetWrappedVector(argument_handles);

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<PjRtBuffer>> out,
      wrapped_->ExecuteSharded(wrapped_args, PjRtCApiDevice::GetWrapped(device),
                               options, returned_future, fill_future));

  for (std::unique_ptr<PjRtBuffer>& buffer : out) {
    buffer = std::make_unique<PjRtCApiBuffer>(client_, std::move(buffer));
  }
  return out;
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtCApiExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
  std::vector<PjRtBuffer*> wrapped_args =
      PjRtCApiBuffer::GetWrappedVector(argument_handles);

  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<PjRtBuffer>> out,
                      wrapped_->ExecutePortable(
                          wrapped_args, PjRtCApiDevice::GetWrapped(device),
                          options, returned_future, fill_future));

  for (std::unique_ptr<PjRtBuffer>& buffer : out) {
    buffer = std::make_unique<PjRtCApiBuffer>(client_, std::move(buffer));
  }
  return out;
}

StatusOr<std::unique_ptr<PjRtClient>> GetCApiClient() {
  const PJRT_Api* c_api = tensorflow::tpu::PjrtApi();
  // TODO(skyewm): make status
  CHECK(c_api != nullptr);

  PJRT_Client_Create_Args init_args;
  init_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  init_args.priv = nullptr;
  PJRT_Error* error = c_api->PJRT_Client_Create(&init_args);
  // TODO(skyewm): handle error
  CHECK(error == nullptr);
  PJRT_Client* c_client = init_args.client;
  PjRtClient* wrapped = c_client->client.get();

  std::vector<std::unique_ptr<PjRtCApiDevice>> devices;
  devices.reserve(wrapped->devices().size());
  for (PjRtDevice* device : wrapped->devices()) {
    devices.emplace_back(std::make_unique<PjRtCApiDevice>(device));
  }
  return std::unique_ptr<PjRtClient>(
      std::make_unique<PjRtCApiClient>(c_api, c_client, std::move(devices)));
}

}  // namespace xla

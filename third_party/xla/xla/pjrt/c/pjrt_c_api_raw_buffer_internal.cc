/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/c/pjrt_c_api_raw_buffer_internal.h"

#include <utility>

#include "absl/status/status.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/tsl/concurrency/ref_count.h"

struct PJRT_RawBuffer {
  tsl::RCReference<xla::PjRtRawBuffer> buffer;
  PJRT_Client* client;
};

namespace pjrt {

PJRT_Error* PJRT_RawBuffer_CreateRawAliasOfBuffer(
    PJRT_RawBuffer_CreateRawAliasOfBuffer_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_RawBuffer_CreateRawAliasOfBuffer_Args",
      PJRT_RawBuffer_CreateRawAliasOfBuffer_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_ASSIGN_OR_RETURN(auto result, xla::PjRtRawBuffer::CreateRawAliasOfBuffer(
                                         args->buffer->buffer.get()));
  args->raw_buffer =
      new PJRT_RawBuffer{std::move(result), args->buffer->client};
  return nullptr;
}
PJRT_Error* PJRT_RawBuffer_Destroy(PJRT_RawBuffer_Destroy_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_RawBuffer_Destroy_Args", PJRT_RawBuffer_Destroy_Args_STRUCT_SIZE,
      args->struct_size));
  delete args->buffer;
  return nullptr;
}
PJRT_Error* PJRT_RawBuffer_GetOnDeviceSizeInBytes(
    PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args",
      PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args_STRUCT_SIZE,
      args->struct_size));
  args->on_device_size_in_bytes =
      args->buffer->buffer->GetOnDeviceSizeInBytes();
  return nullptr;
}
PJRT_Error* PJRT_RawBuffer_GetMemorySpace(
    PJRT_RawBuffer_GetMemorySpace_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_RawBuffer_GetMemorySpace_Args",
      PJRT_RawBuffer_GetMemorySpace_Args_STRUCT_SIZE, args->struct_size));
  args->memory_space = PJRT_Client_FindMemoryWrapper(
      args->buffer->buffer->memory_space(), args->buffer->client);
  if (args->memory_space == nullptr) {
    return new PJRT_Error{
        absl::UnimplementedError("Could find memory_space() for RawBuffer")};
  }
  return nullptr;
}

PJRT_Error* PJRT_RawBuffer_CopyRawHostToDevice(
    PJRT_RawBuffer_CopyRawHostToDevice_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_RawBuffer_CopyRawHostToDevice_Args",
      PJRT_RawBuffer_CopyRawHostToDevice_Args_STRUCT_SIZE, args->struct_size));
  auto result = args->buffer->buffer->CopyRawHostToDevice(
      args->src, args->offset, args->transfer_size);
  args->event = new PJRT_Event{std::move(result)};
  return nullptr;
}
PJRT_Error* PJRT_RawBuffer_CopyRawDeviceToHost(
    PJRT_RawBuffer_CopyRawDeviceToHost_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_RawBuffer_CopyRawDeviceToHost_Args",
      PJRT_RawBuffer_CopyRawDeviceToHost_Args_STRUCT_SIZE, args->struct_size));
  auto result = args->buffer->buffer->CopyRawDeviceToHost(
      args->dst, args->offset, args->transfer_size);
  args->event = new PJRT_Event{std::move(result)};
  return nullptr;
}

PJRT_RawBuffer_Extension CreateRawBufferExtension(PJRT_Extension_Base* next) {
  return {
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_RawBuffer_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer,
          /*next=*/next,
      },
      /*PJRT_RawBuffer_CreateRawAliasOfBuffer=*/
      pjrt::PJRT_RawBuffer_CreateRawAliasOfBuffer,
      /*PJRT_RawBuffer_Destroy=*/pjrt::PJRT_RawBuffer_Destroy,
      /*PJRT_RawBuffer_GetOnDeviceSizeInBytes=*/
      pjrt::PJRT_RawBuffer_GetOnDeviceSizeInBytes,
      /*PJRT_RawBuffer_GetMemorySpace=*/pjrt::PJRT_RawBuffer_GetMemorySpace,
      /*PJRT_RawBuffer_CopyRawHostToDevice=*/
      pjrt::PJRT_RawBuffer_CopyRawHostToDevice,
      /*PJRT_RawBuffer_CopyRawDeviceToHost=*/
      pjrt::PJRT_RawBuffer_CopyRawDeviceToHost,
  };
}

}  // namespace pjrt

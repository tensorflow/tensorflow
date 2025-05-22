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

#include "xla/pjrt/c/pjrt_c_api_raw_buffer_external.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/casts.h"

#define PJRT_RETURN_FUTURE_IF_ERROR(expr, c_api)                         \
  do {                                                                   \
    PJRT_Error* error = (expr);                                          \
    std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> _error(         \
        error, pjrt::MakeErrorDeleter(c_api));                           \
    absl::Status _status = pjrt::PjrtErrorToStatus(_error.get(), c_api); \
    if (!_status.ok()) {                                                 \
      return xla::PjRtFuture<>(_status);                                 \
    }                                                                    \
  } while (false)

namespace pjrt {

void PjRtCApiRawBuffer_Destroy(const PJRT_Api* c_api,
                               const PJRT_RawBuffer_Extension* extension,
                               PJRT_RawBuffer* buffer) {
  PJRT_RawBuffer_Destroy_Args args;
  args.struct_size = PJRT_RawBuffer_Destroy_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;
  pjrt::LogFatalIfPjrtError(extension->PJRT_RawBuffer_Destroy(&args), c_api);
}

PJRT_Memory* PjRtCApiRawBuffer_GetMemorySpace(
    const PJRT_Api* c_api, const PJRT_RawBuffer_Extension* extension,
    PJRT_RawBuffer* buffer) {
  PJRT_RawBuffer_GetMemorySpace_Args args;
  args.struct_size = PJRT_RawBuffer_GetMemorySpace_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;
  args.memory_space = nullptr;
  pjrt::LogFatalIfPjrtError(extension->PJRT_RawBuffer_GetMemorySpace(&args),
                            c_api);
  return args.memory_space;
}

size_t PjRtCApiRawBuffer_GetOnDeviceSizeInBytes(
    const PJRT_Api* c_api, const PJRT_RawBuffer_Extension* extension,
    PJRT_RawBuffer* buffer) {
  PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args args;
  args.struct_size = PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;
  pjrt::LogFatalIfPjrtError(
      extension->PJRT_RawBuffer_GetOnDeviceSizeInBytes(&args), c_api);
  return args.on_device_size_in_bytes;
}

xla::PjRtFuture<> PjRtCApiRawBuffer_CopyRawHostToDevice(
    const PJRT_Api* c_api, const PJRT_RawBuffer_Extension* extension,
    PJRT_RawBuffer* buffer, const void* src, int64_t offset,
    int64_t transfer_size) {
  PJRT_RawBuffer_CopyRawHostToDevice_Args args;
  args.struct_size = PJRT_RawBuffer_CopyRawHostToDevice_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;
  args.src = src;
  args.offset = offset;
  args.transfer_size = transfer_size;
  args.event = nullptr;
  PJRT_RETURN_FUTURE_IF_ERROR(
      extension->PJRT_RawBuffer_CopyRawHostToDevice(&args), c_api);
  return pjrt::ConvertCEventToCppFuture(args.event, c_api);
}

xla::PjRtFuture<> PjRtCApiRawBuffer_CopyRawDeviceToHost(
    const PJRT_Api* c_api, const PJRT_RawBuffer_Extension* extension,
    PJRT_RawBuffer* buffer, void* dst, int64_t offset, int64_t transfer_size) {
  PJRT_RawBuffer_CopyRawDeviceToHost_Args args;
  args.struct_size = PJRT_RawBuffer_CopyRawDeviceToHost_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;
  args.dst = dst;
  args.offset = offset;
  args.transfer_size = transfer_size;
  args.event = nullptr;
  PJRT_RETURN_FUTURE_IF_ERROR(
      extension->PJRT_RawBuffer_CopyRawDeviceToHost(&args), c_api);
  return pjrt::ConvertCEventToCppFuture(args.event, c_api);
}

absl::StatusOr<PJRT_RawBuffer*> PjRtCApiBuffer_CreateRawAliasOfBuffer(
    const PJRT_Api* c_api, const PJRT_RawBuffer_Extension* extension,
    PJRT_Buffer* buffer) {
  PJRT_RawBuffer_CreateRawAliasOfBuffer_Args args;
  args.struct_size = PJRT_RawBuffer_CreateRawAliasOfBuffer_Args_STRUCT_SIZE;
  args.extension_start = nullptr;
  args.buffer = buffer;
  args.raw_buffer = nullptr;
  RETURN_STATUS_IF_PJRT_ERROR(
      extension->PJRT_RawBuffer_CreateRawAliasOfBuffer(&args), c_api);
  return args.raw_buffer;
}

}  // namespace pjrt

namespace xla {

PjRtCApiRawBuffer::~PjRtCApiRawBuffer() {
  pjrt::PjRtCApiRawBuffer_Destroy(c_api_, c_extension_, c_buffer_);
}

PjRtMemorySpace* PjRtCApiRawBuffer::memory_space() const {
  return client_->GetCppMemory(
      pjrt::PjRtCApiRawBuffer_GetMemorySpace(c_api_, c_extension_, c_buffer_));
}

size_t PjRtCApiRawBuffer::GetOnDeviceSizeInBytes() const {
  return pjrt::PjRtCApiRawBuffer_GetOnDeviceSizeInBytes(c_api_, c_extension_,
                                                        c_buffer_);
}

PjRtFuture<> PjRtCApiRawBuffer::CopyRawHostToDevice(const void* src,
                                                    int64_t offset,
                                                    int64_t transfer_size) {
  return pjrt::PjRtCApiRawBuffer_CopyRawHostToDevice(
      c_api_, c_extension_, c_buffer_, src, offset, transfer_size);
}

PjRtFuture<> PjRtCApiRawBuffer::CopyRawDeviceToHost(void* dst, int64_t offset,
                                                    int64_t transfer_size) {
  return pjrt::PjRtCApiRawBuffer_CopyRawDeviceToHost(
      c_api_, c_extension_, c_buffer_, dst, offset, transfer_size);
}

static std::optional<absl::StatusOr<tsl::RCReference<PjRtRawBuffer>>>
PjRtCApiBuffer_CreateRawAliasOfBuffer_Factory(PjRtBuffer* buffer) {
  if (auto* c_api_buffer = dynamic_cast<xla::PjRtCApiBuffer*>(buffer)) {
    auto* c_api = c_api_buffer->pjrt_c_api();
    PJRT_RawBuffer_Extension* extension =
        pjrt::FindExtension<PJRT_RawBuffer_Extension>(
            c_api, PJRT_Extension_Type::PJRT_Extension_Type_RawBuffer);
    if (!extension) {
      return absl::UnimplementedError(
          "RawBuffer extension not implemented in this PJRT plugin.");
    }
    TF_ASSIGN_OR_RETURN(PJRT_RawBuffer * raw_buffer,
                        pjrt::PjRtCApiBuffer_CreateRawAliasOfBuffer(
                            c_api, extension, c_api_buffer->c_buffer()));
    return tsl::MakeRef<PjRtCApiRawBuffer>(
        raw_buffer,
        tensorflow::down_cast<PjRtCApiClient*>(c_api_buffer->client()), c_api,
        extension);
  }
  return std::nullopt;
}

REGISTER_PJRT_RAW_BUFFER_FACTORY(PjRtCApiBuffer_CreateRawAliasOfBuffer_Factory);

}  // namespace xla

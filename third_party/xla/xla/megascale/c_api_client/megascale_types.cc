/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/megascale/c_api_client/megascale_types.h"

#include <cstdint>
#include <memory>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_megascale_extension.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/plugin/plugin_names.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace megascale {
namespace c_api_client {

namespace {

absl::StatusOr<const PJRT_Megascale_Extension*> GetExtension(
    const PJRT_Api** c_api_out = nullptr) {
  TF_ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(kTpuPjrtName));
  if (c_api_out != nullptr) {
    *c_api_out = c_api;
  }
  PJRT_Megascale_Extension* extension =
      pjrt::FindExtension<PJRT_Megascale_Extension>(
          c_api, PJRT_Extension_Type_Megascale);
  if (extension == nullptr) {
    return absl::InternalError("Megascale extension is not available.");
  }
  return extension;
}

}  // namespace

absl::StatusOr<MultiSliceDeviceId> MultiSliceDeviceId::Create(
    int64_t megascale_id) {
  const PJRT_Api* c_api;
  TF_ASSIGN_OR_RETURN(const PJRT_Megascale_Extension* ext,
                      GetExtension(&c_api));
  PJRT_Megascale_MegascaleId_To_DeviceId_Args args;
  args.struct_size = PJRT_Megascale_MegascaleId_To_DeviceId_Args_STRUCT_SIZE;
  args.megascale_id = megascale_id;
  RETURN_STATUS_IF_PJRT_ERROR(ext->megascale_id_to_device_id(&args), c_api);
  return MultiSliceDeviceId(megascale_id, args.slice_id,
                            args.per_slice_device_id);
}

absl::StatusOr<MultiSliceDeviceId> MultiSliceDeviceId::Create(
    int32_t slice_id, int32_t per_slice_device_id) {
  const PJRT_Api* c_api;
  TF_ASSIGN_OR_RETURN(const PJRT_Megascale_Extension* ext,
                      GetExtension(&c_api));
  PJRT_Megascale_DeviceId_To_MegascaleId_Args args;
  args.struct_size = PJRT_Megascale_DeviceId_To_MegascaleId_Args_STRUCT_SIZE;
  args.slice_id = slice_id;
  args.per_slice_device_id = per_slice_device_id;
  RETURN_STATUS_IF_PJRT_ERROR(ext->device_id_to_megascale_id(&args), c_api);
  return MultiSliceDeviceId(args.megascale_id, slice_id, per_slice_device_id);
}

CApiPjRtClientContext::~CApiPjRtClientContext() {
  PJRT_Megascale_DeleteClientContext_Args args;
  args.struct_size = PJRT_Megascale_DeleteClientContext_Args_STRUCT_SIZE;
  args.client_context = client_context_;
  std::unique_ptr<PJRT_Error, pjrt::PJRT_ErrorDeleter> error(
      extension_->delete_client_context(&args), pjrt::MakeErrorDeleter(c_api_));
  if (error != nullptr) {
    LOG(ERROR) << "Failed to delete CApiPjRtClientContext: "
               << pjrt::PjrtErrorToStatus(error.get(), c_api_);
  }
}

absl::Status CApiPjRtClientContext::Initialize() {
  PJRT_Megascale_ClientContext_Initialize_Args args;
  args.struct_size = PJRT_Megascale_ClientContext_Initialize_Args_STRUCT_SIZE;
  args.client_context = client_context_;
  RETURN_STATUS_IF_PJRT_ERROR(extension_->client_context_initialize(&args),
                              c_api_);
  return absl::OkStatus();
}

absl::Status CApiPjRtClientContext::UnblockPendingWork(
    int32_t launch_id, absl::Duration expire_after) {
  PJRT_Megascale_ClientContext_UnblockPendingWork_Args args;
  args.struct_size =
      PJRT_Megascale_ClientContext_UnblockPendingWork_Args_STRUCT_SIZE;
  args.client_context = client_context_;
  args.launch_id = launch_id;
  args.expire_after_ms = absl::ToInt64Milliseconds(expire_after);
  RETURN_STATUS_IF_PJRT_ERROR(
      extension_->client_context_unblock_pending_work(&args), c_api_);
  return absl::OkStatus();
}

absl::StatusOr<int> CApiPjRtClientContext::megascale_port() {
  PJRT_Megascale_ClientContext_MegascalePort_Args args;
  args.struct_size =
      PJRT_Megascale_ClientContext_MegascalePort_Args_STRUCT_SIZE;
  args.client_context = client_context_;
  args.port = 0;
  RETURN_STATUS_IF_PJRT_ERROR(extension_->client_context_megascale_port(&args),
                              c_api_);
  return args.port;
}

}  // namespace c_api_client
}  // namespace megascale
}  // namespace xla

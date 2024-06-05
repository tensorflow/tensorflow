/* Copyright 2024 The OpenXLA Authors.

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

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/ffi_api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/service/custom_call_target_registry.h"

namespace pjrt {

static PJRT_Error* PJRT_FFI_UserData_Add(PJRT_FFI_UserData_Add_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_FFI_UserData_Add_Args", PJRT_FFI_UserData_Add_Args_STRUCT_SIZE,
      args->struct_size));

  if (args->context == nullptr) {
    return new PJRT_Error{absl::InvalidArgumentError(
        "PJRT FFI extension requires execute context to be not nullptr")};
  }

  xla::ffi::ExecutionContext::TypeId type_id(args->user_data.type_id);
  PJRT_RETURN_IF_ERROR(args->context->execute_context->ffi_context().Insert(
      type_id, args->user_data.data, args->user_data.deleter));
  return nullptr;
}

static PJRT_Error* PJRT_FFI_Register_Handler(
    PJRT_FFI_Register_Handler_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_FFI_Register_Handler_Args",
      PJRT_FFI_Register_Handler_Args_STRUCT_SIZE, args->struct_size));
  std::string target_name(args->target_name, args->target_name_size);
  std::string platform_name(args->platform_name, args->platform_name_size);
  switch (args->api_version) {
    case 0:
      xla::CustomCallTargetRegistry::Global()->Register(
          target_name, args->handler, platform_name);
      return nullptr;
    case 1:
      xla::ffi::Ffi::RegisterStaticHandler(
          xla::ffi::GetXlaFfiApi(), target_name, platform_name,
          reinterpret_cast<XLA_FFI_Handler*>(args->handler));
      return nullptr;
    default:
      return new PJRT_Error{absl::UnimplementedError(
          absl::StrFormat("API version %d not supported for PJRT GPU plugin. "
                          "Supported versions are 0 and 1.",
                          args->api_version))};
  }
}

PJRT_FFI_Extension CreateFfiExtension(PJRT_Extension_Base* next) {
  return {
      /*struct_size=*/PJRT_FFI_Extension_STRUCT_SIZE,
      /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_FFI,
      /*next=*/next,
      /*user_data_add=*/PJRT_FFI_UserData_Add,
      /*register_handler=*/PJRT_FFI_Register_Handler,
  };
}

}  // namespace pjrt

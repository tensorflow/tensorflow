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
#include "absl/strings/string_view.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/ffi/type_registry.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"

namespace pjrt {

static PJRT_Error* PJRT_FFI_Type_Register(PJRT_FFI_Type_Register_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_FFI_Type_Register_Args", PJRT_FFI_Type_Register_Args_STRUCT_SIZE,
      args->struct_size));

  absl::string_view type_name(args->type_name, args->type_name_size);
  xla::ffi::TypeRegistry::TypeId type_id(args->type_id);
  xla::ffi::TypeRegistry::TypeInfo type_info = {
      args->type_info->deleter,
  };

  if (type_id == xla::ffi::TypeRegistry::kUnknownTypeId) {
    // If type_id is unknown, we are registering a new type and XLA will assign
    // a unique type id to it.
    PJRT_ASSIGN_OR_RETURN(
        auto assigned_type_id,
        xla::ffi::TypeRegistry::AssignTypeId(type_name, type_info));
    args->type_id = assigned_type_id.value();

  } else {
    // If type_id is set, we are relying on the caller-provided unique type id.
    PJRT_RETURN_IF_ERROR(
        xla::ffi::TypeRegistry::RegisterTypeId(type_name, type_id, type_info));
  }

  return nullptr;
}

static PJRT_Error* PJRT_FFI_UserData_Add(PJRT_FFI_UserData_Add_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_FFI_UserData_Add_Args", PJRT_FFI_UserData_Add_Args_STRUCT_SIZE,
      args->struct_size));

  if (args->context == nullptr) {
    return new PJRT_Error{absl::InvalidArgumentError(
        "PJRT FFI extension requires execute context to be not nullptr")};
  }

  xla::ffi::TypeRegistry::TypeId type_id(args->user_data.type_id);
  PJRT_RETURN_IF_ERROR(args->context->execute_context->ffi_context().Insert(
      type_id, args->user_data.data));
  return nullptr;
}

static PJRT_Error* PJRT_FFI_Register_Handler(
    PJRT_FFI_Register_Handler_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_FFI_Register_Handler_Args",
      PJRT_FFI_Register_Handler_Args_STRUCT_SIZE, args->struct_size));
  std::string target_name(args->target_name, args->target_name_size);
  std::string platform_name(args->platform_name, args->platform_name_size);

  // Validate that handler is not null
  if (args->handler == nullptr) {
    return new PJRT_Error{
        absl::InvalidArgumentError("FFI handler cannot be null")};
  }

  // Only support typed FFI handlers
  xla::ffi::Ffi::RegisterStaticHandler(
      xla::ffi::GetXlaFfiApi(), target_name, platform_name,
      reinterpret_cast<XLA_FFI_Handler*>(args->handler),
      static_cast<XLA_FFI_Handler_TraitsBits>(args->traits));
  return nullptr;
}

PJRT_FFI_Extension CreateFfiExtension(PJRT_Extension_Base* next) {
  return {
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_FFI_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_FFI,
          /*next=*/next,
      },
      /*type_register=*/PJRT_FFI_Type_Register,
      /*user_data_add=*/PJRT_FFI_UserData_Add,
      /*register_handler=*/PJRT_FFI_Register_Handler,
  };
}

}  // namespace pjrt

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

#include "xla/pjrt/c/pjrt_c_api_triton_internal.h"

#include <cstring>

#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_triton_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/triton.h"

namespace pjrt {

PJRT_Error* PJRT_Triton_Compile(PJRT_Triton_Compile_Args* args) {
  static constexpr size_t PJRT_Triton_Compile_Args_STRUCT_SIZE_V1 =
      PJRT_STRUCT_SIZE(PJRT_Triton_Compile_Args, out_smem_bytes);

  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Triton_Compile_Args", PJRT_Triton_Compile_Args_STRUCT_SIZE_V1,
      args->struct_size));

  PJRT_ASSIGN_OR_RETURN(
      auto result, xla::triton::Compile(
                       absl::string_view(args->module, args->module_size),
                       absl::string_view(args->arch_name, args->arch_name_size),
                       args->num_warps, args->num_ctas, args->num_stages));

  bool is_v1_struct =
      args->struct_size == PJRT_Triton_Compile_Args_STRUCT_SIZE_V1;
  if (xla::triton::AsmText* ptr =
          std::get_if<xla::triton::AsmText>(&result.compiled_output)) {
    args->out_asm = new char[ptr->value.size()];
    std::memcpy(const_cast<void*>(static_cast<const void*>(args->out_asm)),
                ptr->value.data(), ptr->value.size());
    args->out_asm_size = ptr->value.size();
    if (!is_v1_struct) {
      args->out_path = nullptr;
      args->out_path_size = 0;
    }
  } else if (xla::triton::HsacoPath* ptr =
                 std::get_if<xla::triton::HsacoPath>(&result.compiled_output)) {
    if (is_v1_struct) {
      return new PJRT_Error{absl::InvalidArgumentError(
          "Triton compilation returned ROCm HsacoPath, but client is using V1 "
          "PJRT_Triton_Compile_Args struct version which only supports CUDA "
          "PTX AsmText output.")};
    } else {
      args->out_asm = nullptr;
      args->out_asm_size = 0;
      args->out_path = new char[ptr->value.size()];
      std::memcpy(const_cast<void*>(static_cast<const void*>(args->out_path)),
                  ptr->value.data(), ptr->value.size());
      args->out_path_size = ptr->value.size();
    }
  }
  args->out_smem_bytes = result.smem_bytes;
  return nullptr;
}

}  // namespace pjrt

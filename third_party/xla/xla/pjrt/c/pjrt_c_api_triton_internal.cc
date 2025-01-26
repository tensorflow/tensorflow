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
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Triton_Compile_Args", PJRT_Triton_Compile_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_ASSIGN_OR_RETURN(
      auto result, xla::triton::Compile(
                       absl::string_view(args->module, args->module_size),
                       absl::string_view(args->arch_name, args->arch_name_size),
                       args->num_warps, args->num_ctas, args->num_stages));

  auto* asm_copy = new char[result.asm_text.size()];
  std::memcpy(asm_copy, result.asm_text.data(), result.asm_text.size());
  args->out_asm = asm_copy;
  args->out_asm_size = result.asm_text.size();
  args->out_smem_bytes = result.smem_bytes;
  args->out_cluster_dim_x = result.cluster_dim_x;
  args->out_cluster_dim_y = result.cluster_dim_y;
  args->out_cluster_dim_z = result.cluster_dim_z;
  return nullptr;
}

}  // namespace pjrt

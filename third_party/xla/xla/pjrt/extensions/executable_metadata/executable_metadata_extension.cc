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

#include "xla/pjrt/extensions/executable_metadata/executable_metadata_extension.h"

#include "xla/pjrt/c/pjrt_c_api.h"

namespace pjrt {

void DestroySerializedMetadata(
    PJRT_ExecutableMetadata_DestroySerializedMetadata_Args* args) {
  delete[] args->metadata->serialized_metadata;
  delete args->metadata;
}

PJRT_ExecutableMetadata_Extension CreateExecutableMetadataExtension(
    PJRT_Extension_Base* next,
    PJRT_ExecutableMetadata_GetExecutableMetadata get_executable_metadata) {
  return PJRT_ExecutableMetadata_Extension{
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_ExecutableMetadata_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type_ExecutableMetadata,
          /*next=*/next,
      },
      /*get_executable_metadata=*/
      get_executable_metadata,
      /*Destroy_serialized_metadata=*/DestroySerializedMetadata};
}

}  // namespace pjrt

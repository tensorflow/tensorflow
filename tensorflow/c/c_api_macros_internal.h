/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_C_API_MACROS_INTERNAL_H_
#define TENSORFLOW_C_C_API_MACROS_INTERNAL_H_

#ifdef __cplusplus
#include "tensorflow/core/platform/status.h"

// Macro to verify that the field `struct_size` of STRUCT_OBJ is initialized.
// `struct_size` is used for struct member compatibility check between core TF
// and plug-ins with the same C API minor version. More info here:
// https://github.com/tensorflow/community/blob/master/rfcs/20200612-stream-executor-c-api/C_API_versioning_strategy.md
#define TF_VALIDATE_STRUCT_SIZE(STRUCT_NAME, STRUCT_OBJ, SIZE_VALUE_NAME) \
  do {                                                                    \
    if (STRUCT_OBJ.struct_size == 0) {                                    \
      return tensorflow::Status(absl::StatusCode::kFailedPrecondition,    \
                                "Expected initialized `" #STRUCT_NAME     \
                                "` structure with `struct_size` field "   \
                                "set to " #SIZE_VALUE_NAME                \
                                ". Found `struct_size` = 0.");            \
    }                                                                     \
  } while (0)

// Macro to verify that the field NAME of STRUCT_OBJ is not null.
#define TF_VALIDATE_NOT_NULL(STRUCT_NAME, STRUCT_OBJ, NAME)            \
  do {                                                                 \
    if (STRUCT_OBJ.NAME == 0) {                                        \
      return tensorflow::Status(absl::StatusCode::kFailedPrecondition, \
                                "'" #NAME "' field in " #STRUCT_NAME   \
                                " must be set.");                      \
    }                                                                  \
  } while (0)

#endif  // __cplusplus
#endif  // TENSORFLOW_C_C_API_MACROS_INTERNAL_H_

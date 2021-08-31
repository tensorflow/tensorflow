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
#ifndef TENSORFLOW_C_EXPERIMENTAL_PLUGGABLE_DEVICE_UTILS_PLUGGABLE_DEVICE_UTILS_H_
#define TENSORFLOW_C_EXPERIMENTAL_PLUGGABLE_DEVICE_UTILS_PLUGGABLE_DEVICE_UTILS_H_

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/platform/status.h"

namespace pluggable_device {

#define VALIDATE_STRUCT_SIZE(STRUCT_NAME, STRUCT_OBJ, SIZE_VALUE_NAME)    \
  do {                                                                    \
    if (STRUCT_OBJ.struct_size == 0) {                                    \
      return tensorflow::Status(tensorflow::error::FAILED_PRECONDITION,   \
                                "struct_size field in " #STRUCT_NAME      \
                                " must be set to " #SIZE_VALUE_NAME "."); \
    }                                                                     \
  } while (0)

#define VALIDATE_MEMBER(STRUCT_NAME, STRUCT_OBJ, NAME)                  \
  do {                                                                  \
    if (STRUCT_OBJ.NAME == 0) {                                         \
      return tensorflow::Status(tensorflow::error::FAILED_PRECONDITION, \
                                "'" #NAME "' field in " #STRUCT_NAME    \
                                " must be set.");                       \
    }                                                                   \
  } while (0)

tensorflow::Status ValidateDeviceType(tensorflow::StringPiece type);

struct TFStatusDeleter {
  void operator()(TF_Status* s) const { TF_DeleteStatus(s); }
};

}  // namespace pluggable_device
#endif  // TENSORFLOW_C_EXPERIMENTAL_PLUGGABLE_DEVICE_UTILS_PLUGGABLE_DEVICE_UTILS_H_

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/c_api_internal.h"

namespace tensorflow {

void Set_TF_Status_from_Status(TF_Status* tf_status, const Status& status) {
  auto code = status.code();
  auto message = status.error_message().c_str();

  if (code != tensorflow::error::OK) {
    TF_SetStatus(tf_status, (TF_Code)code, message);
  }
}

Status StatusFromTF_Status(const TF_Status* tf_status) {
  return tf_status->status;
}

}  // namespace tensorflow

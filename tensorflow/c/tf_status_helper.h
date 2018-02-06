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

#ifndef TENSORFLOW_C_TF_STATUS_HELPER_H
#define TENSORFLOW_C_TF_STATUS_HELPER_H

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Set the attribute of "tf_status" from the attributes of "status".
void Set_TF_Status_from_Status(TF_Status* tf_status, const Status& status);

// Returns a "status" from "tf_status".
Status StatusFromTF_Status(const TF_Status* tf_status);

}  // namespace tensorflow

#endif  // TENSORFLOW_C_TF_STATUS_HELPER_H

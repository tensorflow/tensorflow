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

#ifndef TENSORFLOW_C_TF_STATUS_HELPER_H_
#define TENSORFLOW_C_TF_STATUS_HELPER_H_

#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/platform/status.h"

namespace tsl {
// Set the attribute of "tf_status" from the attributes of "status".
void Set_TF_Status_from_Status(TF_Status* tf_status, const tsl::Status& status);

// Returns a "status" from "tf_status".
tensorflow::Status StatusFromTF_Status(const TF_Status* tf_status);
}  // namespace tsl

namespace tensorflow {
using tsl::Set_TF_Status_from_Status;  // NOLINT
using tsl::StatusFromTF_Status;        // NOLINT

namespace internal {
struct TF_StatusDeleter {
  void operator()(TF_Status* tf_status) const { TF_DeleteStatus(tf_status); }
};
}  // namespace internal

using TF_StatusPtr = std::unique_ptr<TF_Status, internal::TF_StatusDeleter>;

}  // namespace tensorflow

#endif  // TENSORFLOW_C_TF_STATUS_HELPER_H_

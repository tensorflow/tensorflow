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

#ifndef TENSORFLOW_TSL_C_TSL_STATUS_HELPER_H_
#define TENSORFLOW_TSL_C_TSL_STATUS_HELPER_H_

#include <memory>

#include "tensorflow/tsl/c/tsl_status.h"
#include "tensorflow/tsl/platform/status.h"

namespace tsl {

TSL_Code TSLCodeFromStatusCode(absl::StatusCode code);

absl::StatusCode StatusCodeFromTSLCode(TSL_Code code);

// Returns a "status" from "tsl_status".
Status StatusFromTSL_Status(const TSL_Status* tsl_status);

}  // namespace tsl

#endif  // TENSORFLOW_TSL_C_TSL_STATUS_HELPER_H_

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

#include <string.h>

#include "tensorflow/compiler/xla/service/custom_call_status.h"

// Call the API from a .c file to make sure it works with pure C code.

void CSetSuccess(XlaCustomCallStatus* status) {
  XlaCustomCallStatusSetSuccess(status);
}

void CSetFailure(XlaCustomCallStatus* status, const char* message,
                 size_t message_len) {
  XlaCustomCallStatusSetFailure(status, message, message_len);
}

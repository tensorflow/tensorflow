/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/custom_call_status_internal.h"

namespace xla {
// Internal functions

std::optional<absl::string_view> CustomCallStatusGetMessage(
    const XlaCustomCallStatus* status) {
  return status->message;
}

}  // namespace xla

void XlaCustomCallStatusSetSuccess(XlaCustomCallStatus* status) {
  status->message = std::nullopt;
}

void XlaCustomCallStatusSetFailure(XlaCustomCallStatus* status,
                                   const char* message, size_t message_len) {
  status->message = std::string(message, 0, message_len);
}

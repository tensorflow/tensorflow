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

#ifndef XLA_SERVICE_CUSTOM_CALL_STATUS_INTERNAL_H_
#define XLA_SERVICE_CUSTOM_CALL_STATUS_INTERNAL_H_

#include <optional>

#include "absl/strings/string_view.h"
#include "xla/service/custom_call_status.h"

struct XlaCustomCallStatus_ {
  // The message being present means "failure". Otherwise means "success".
  std::optional<std::string> message;
};

namespace xla {

// Get a view of the internal error message of the XlaCustomCallStatus. Only
// lives as long as the XlaCustomCallStatus. Returns an empty optional if the
// result was "success".
std::optional<absl::string_view> CustomCallStatusGetMessage(
    const XlaCustomCallStatus* status);

}  // namespace xla

#endif  // XLA_SERVICE_CUSTOM_CALL_INTERNAL_STATUS_H_

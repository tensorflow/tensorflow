/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/error_logging.h"

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace tsl::error_logging {

absl::Status Log(absl::string_view component, absl::string_view subcomponent,
                 absl::string_view error_msg) {
  // no-op, intentionally empty function
  return absl::OkStatus();
}

}  // namespace tsl::error_logging

/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/tsl/lib/gtl/value_or_die.h"

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/logging.h"

namespace tsl::gtl::internal_value_or_die {

ABSL_ATTRIBUTE_NORETURN void DieBecauseEmptyValue(const char* file, int line,
                                                  const absl::Status* status) {
  if (status == nullptr) {
    LOG(FATAL).AtLocation(file, line) << "ValueOrDie on empty value.";
  } else {
    LOG(FATAL).AtLocation(file, line) << "ValueOrDie on " << *status;
  }
}

}  // namespace tsl::gtl::internal_value_or_die

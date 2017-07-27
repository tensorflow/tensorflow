/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/statusor.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace internal_statusor {

void Helper::HandleInvalidStatusCtorArg(Status* status) {
  const char* kMessage =
      "An OK status is not a valid constructor argument to StatusOr<T>";
  LOG(ERROR) << kMessage;
  // Fall back to tensorflow::error::INTERNAL.
  *status = ::tensorflow::errors::Internal(kMessage);
}

void Helper::Crash(const Status& status) {
  LOG(FATAL) << "Attempting to fetch value instead of handling error "
             << status;
}

}  // namespace internal_statusor
}  // namespace xla

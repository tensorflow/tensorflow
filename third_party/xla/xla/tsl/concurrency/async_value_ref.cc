/* Copyright 2022 Google LLC. All Rights Reserved.

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

#include "xla/tsl/concurrency/async_value_ref.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/logging.h"

namespace tsl {

RCReference<IndirectAsyncValue> MakeIndirectAsyncValue() {
  return TakeRef(internal::AllocateAndConstruct<IndirectAsyncValue>());
}

RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(absl::Status status) {
  CHECK(!status.ok()) << "status must be an error";  // NOLINT
  return TakeRef(
      internal::AllocateAndConstruct<ErrorAsyncValue>(std::move(status)));
}

RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(absl::string_view message) {
  // Converting to `absl::string_view` because implicit conversion is not
  // supported in android builds.
  absl::string_view message_view(message.data(), message.size());
  return MakeErrorAsyncValueRef(absl::InternalError(message_view));
}

}  // namespace tsl

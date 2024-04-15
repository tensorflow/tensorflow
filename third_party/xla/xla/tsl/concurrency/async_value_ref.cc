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

#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace tsl {

RCReference<IndirectAsyncValue> MakeIndirectAsyncValue() {
  return TakeRef(internal::AllocateAndConstruct<IndirectAsyncValue>());
}

RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(absl::Status status) {
  auto* error_value =
      internal::AllocateAndConstruct<ErrorAsyncValue>(std::move(status));

  return TakeRef(error_value);
}

RCReference<ErrorAsyncValue> MakeErrorAsyncValueRef(std::string_view message) {
  return MakeErrorAsyncValueRef(absl::InternalError(message));
}

}  // namespace tsl

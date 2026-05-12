/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/fusion_debug_callback.h"

#include <functional>
#include <memory>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "xla/literal.h"

namespace xla {

namespace {
absl::Mutex g_mutex(absl::kConstInit);
auto& g_callback() ABSL_EXCLUSIVE_LOCKS_REQUIRED(g_mutex) {
  static absl::NoDestructor<FusionDebugCallbackFunction> value;
  return *value;
}
}  // namespace

// TODO(amirsaadat): Implement the callback id generation.
FusionDebugCallbackId RegisterFusionDebugCallback(
    FusionDebugCallbackFunction cb) {
  absl::MutexLock lock(g_mutex);
  g_callback() = std::move(cb);
  return FusionDebugCallbackId(0);
}

void UnregisterFusionDebugCallback(FusionDebugCallbackId callback_id) {
  absl::MutexLock lock(g_mutex);
  g_callback() = nullptr;
}

void TriggerFusionDebugCallback(
    const std::shared_ptr<const xla::Literal>& literal,
    const FusionDebugCallbackAttributes& attributes) {
  FusionDebugCallbackFunction callback;
  {
    absl::MutexLock lock(g_mutex);
    callback = g_callback();
  }
  if (callback != nullptr) {
    callback(literal, attributes);
  }
}

}  // namespace xla

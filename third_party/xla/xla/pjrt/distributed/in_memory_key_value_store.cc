/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/distributed/in_memory_key_value_store.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"

namespace xla {

absl::StatusOr<std::string> InMemoryKeyValueStore::Get(absl::string_view key,
                                                       absl::Duration timeout) {
  absl::Notification done;
  absl::StatusOr<std::string> result;
  auto call_opts =
      AsyncGet(key, [&done, &result](const absl::StatusOr<std::string>& res) {
        result = res;
        done.Notify();
      });

  if (!done.WaitForNotificationWithTimeout(timeout)) {
    if (call_opts != nullptr) {
      call_opts->StartCancel();
    }
    return absl::NotFoundError(
        absl::StrCat(key, " is not found in the kv store."));
  }
  return result;
}

absl::StatusOr<std::string> InMemoryKeyValueStore::TryGet(
    absl::string_view key) {
  std::optional<std::string> val = kv_store_.Get(key);
  if (!val.has_value()) {
    return absl::NotFoundError(
        absl::StrCat(key, " is not found in the kv store."));
  }
  return *val;
}

std::shared_ptr<tsl::CallOptions> InMemoryKeyValueStore::AsyncGet(
    absl::string_view key,
    tsl::CoordinationServiceAgent::StatusOrValueCallback done) {
  auto call_opts = std::make_shared<tsl::CallOptions>();
  auto promise_and_future = tsl::MakePromiseOnce<std::string>();
  tsl::PromiseOnce<std::string> promise = std::move(promise_and_future.first);
  tsl::Future<std::string> future = std::move(promise_and_future.second);

  future.OnReady([done = std::move(done)](
                     const absl::StatusOr<std::string>& res) { done(res); });

  call_opts->SetCancelCallback([promise]() mutable {
    promise.Set(absl::CancelledError("AsyncGet was cancelled."));
  });

  kv_store_.AddCallbackForKey(
      key, [promise](const absl::StatusOr<absl::string_view>& res) mutable {
        if (res.ok()) {
          promise.Set(std::string(*res));
        } else {
          promise.Set(res.status());
        }
      });

  return call_opts;
}

absl::Status InMemoryKeyValueStore::Set(absl::string_view key,
                                        absl::string_view value) {
  return kv_store_.Put(key, value, allow_overwrite_);
}

absl::Status InMemoryKeyValueStore::Delete(absl::string_view key) {
  kv_store_.Delete(key);
  return absl::OkStatus();
}

}  // namespace xla

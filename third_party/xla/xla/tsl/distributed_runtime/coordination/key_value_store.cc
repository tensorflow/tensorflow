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

#include "xla/tsl/distributed_runtime/coordination/key_value_store.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/internal/endian.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace tsl {

KeyValueStore::~KeyValueStore() {
  absl::MutexLock l(&mu_);

  absl::Status cancelled = absl::CancelledError("KeyValueStore destructed");
  for (auto& [key, callbacks] : callbacks_) {
    for (Callback& callback : callbacks) {
      callback(cancelled);
    }
  }
}

absl::Status KeyValueStore::Put(absl::string_view key, absl::string_view value,
                                bool allow_overwrite) {
  absl::MutexLock l(&mu_);

  if (allow_overwrite) {
    data_[key] = value;
    NotifyCallbacksForKey(key, value);
    return absl::OkStatus();
  }

  auto [it, inserted] = data_.try_emplace(key, value);
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrCat("key ", key, " already exists."));
  }
  NotifyCallbacksForKey(key, value);
  return absl::OkStatus();
}

std::optional<std::string> KeyValueStore::Get(absl::string_view key) {
  absl::MutexLock l(&mu_);
  auto it = data_.find(key);
  if (it == data_.end()) {
    return std::nullopt;
  }
  return it->second;
}

absl::StatusOr<std::string> KeyValueStore::IncrementBy(absl::string_view key,
                                                       int64_t increment) {
  absl::MutexLock l(mu_);

  auto [it, inserted] = data_.try_emplace(key, "");
  if (inserted) {
    std::string new_value(sizeof(int64_t), '\0');
    it->second = std::move(new_value);
  } else if (it->second.size() != sizeof(int64_t)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "key ", key, " has an unexpected value size: ", it->second.size(),
        " bytes, expected ", sizeof(int64_t), " bytes."));
  }
  int64_t new_value =
      static_cast<int64_t>(absl::big_endian::Load64(it->second.data())) +
      increment;

  absl::big_endian::Store64(it->second.data(), new_value);
  NotifyCallbacksForKey(key, it->second);
  return it->second;
}

std::vector<tensorflow::KeyValueEntry> KeyValueStore::GetPrefix(
    absl::string_view prefix) {
  absl::MutexLock l(&mu_);

  std::vector<tensorflow::KeyValueEntry> entries;
  for (auto it = data_.lower_bound(prefix); it != data_.end(); ++it) {
    const auto& [key, value] = *it;
    if (!absl::StartsWith(key, prefix)) {
      break;
    }
    tensorflow::KeyValueEntry entry;
    entry.set_key(key);
    entry.set_value(value);
    entries.push_back(std::move(entry));
  }
  return entries;
}

void KeyValueStore::Delete(absl::string_view key) {
  absl::MutexLock l(&mu_);
  data_.erase(key);
}

void KeyValueStore::DeletePrefix(absl::string_view prefix) {
  absl::MutexLock l(&mu_);

  auto begin = data_.lower_bound(prefix);
  auto it = begin;
  for (; it != data_.end(); ++it) {
    const auto& [key, value] = *it;
    if (!absl::StartsWith(key, prefix)) {
      break;
    }
  }
  data_.erase(begin, it);
}

void KeyValueStore::AddCallbackForKey(absl::string_view key,
                                      Callback callback) {
  absl::MutexLock l(&mu_);

  if (auto it = data_.find(key); it != data_.end()) {
    callback(it->second);
    return;
  }
  callbacks_[key].push_back(std::move(callback));
}

void KeyValueStore::NotifyCallbacksForKey(
    absl::string_view key, const absl::StatusOr<absl::string_view>& value) {
  if (auto it = callbacks_.find(key); it != callbacks_.end()) {
    for (Callback& callback : it->second) {
      callback(value);
    }
    callbacks_.erase(it);
  }
}

}  // namespace tsl

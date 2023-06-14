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
#include "tensorflow/core/kernels/batching_util/warmup.h"

#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/tsl/platform/logging.h"

namespace tensorflow {
namespace serving {

void WarmupStateRegistry::Handle::Release() {
  if (!key_.has_value()) {
    return;
  }

  DCHECK(registry_);
  registry_->Unregister(*key_);
}

absl::StatusOr<WarmupStateRegistry::Handle> WarmupStateRegistry::Register(
    const Key& model_key) {
  absl::MutexLock l(&mu_);
  VLOG(1) << "Registering model " << model_key.name << ":" << model_key.version
          << " to warm-up registry";
  if (!states_.insert(model_key).second) {
    return absl::AlreadyExistsError(
        absl::StrCat("Model ", model_key.name, ":", model_key.version,
                     " already exists in the warm-up registry"));
  }
  return Handle(model_key, this);
}

void WarmupStateRegistry::Unregister(const Key& model_key) {
  absl::MutexLock l(&mu_);

  VLOG(1) << "Unregistering model " << model_key.name << ":"
          << model_key.version << " from warm-up registry";
  states_.erase(model_key);
}

bool WarmupStateRegistry::Lookup(const Key& model_key) {
  absl::ReaderMutexLock l(&mu_);
  return states_.contains(model_key);
}

WarmupStateRegistry& GetGlobalWarmupStateRegistry() {
  static auto* const registry = new WarmupStateRegistry;
  return *registry;
}

}  // namespace serving
}  // namespace tensorflow

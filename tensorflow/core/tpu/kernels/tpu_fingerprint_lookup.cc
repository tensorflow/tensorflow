/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/tpu_fingerprint_lookup.h"

#include <optional>
#include <string>

namespace tensorflow {
namespace tpu {

TpuFingerprintLookup* TpuFingerprintLookup::Create() {
  return new TpuFingerprintLookup();
}

void TpuFingerprintLookup::RegisterKeyAndIntermediatePair(uint64 key,
                                                          uint64 intermediate) {
  absl::MutexLock lock(&mu_);
  auto [it, emplaced] = intermediate_to_key_.try_emplace(intermediate, key);
  if (it->second != key) {
    VLOG(2) << "The key (" << it->second
            << ") is associated with an existing intermediate ( " << it->first
            << "), which does not match the requesting key (" << key << ").";
  }
}

bool TpuFingerprintLookup::RegisterIntermediateAndValuePair(uint64 intermediate,
                                                            std::string value) {
  absl::MutexLock lock(&mu_);
  auto it = intermediate_to_key_.find(intermediate);
  if (it == intermediate_to_key_.end()) {
    VLOG(2) << "Cannot find the intermediate ( " << intermediate
            << "). A RegisterKeyAndIntermediatePair must precedes.";
    return false;
  } else {
    uint64 key = it->second;
    bool is_successful = false;
    VLOG(2) << "registering key (" << key << ") with value: " << value;
    auto it = key_to_value_.find(key);
    if (it == key_to_value_.end()) {
      // A new key. If the value is not seen before, register key-value and
      // value-key pairs. Otherwise, skip registration.
      auto maybe_existing_key = value_to_key_.find(value);
      if (maybe_existing_key == value_to_key_.end()) {
        key_to_value_.emplace(key, value);
        value_to_key_.emplace(value, key);
        is_successful = true;
      } else {
        // The value is registered before with a different key. Skip
        // registration.
        if (maybe_existing_key->second != key) {
          VLOG(2) << "The value (" << value
                  << ") is associated with an existing key ( "
                  << maybe_existing_key->second
                  << "), which does not match the requesting key (" << key
                  << ").";
        }
      }
    } else {
      // The key is registered before, no actions needed. For debugging purpose,
      // check if existing value agrees with the value.
      if (it->second != value) {
        VLOG(2) << "The key (" << key
                << ") has been registered and the requesting value ( " << value
                << " and the existing" << it->second << ") doesn't match.";
      }
    }
    DCHECK(key_to_value_.size() == value_to_key_.size());

    return is_successful;
  }
}

std::optional<absl::string_view> TpuFingerprintLookup::Lookup(uint64 key) {
  absl::MutexLock lock(&mu_);
  auto it = key_to_value_.find(key);
  if (it == key_to_value_.end()) {
    return std::optional<absl::string_view>{};
  } else {
    return it->second;
  }
}

}  // namespace tpu
}  // namespace tensorflow

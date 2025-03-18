// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_DISPATCH_REGISTRY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_DISPATCH_REGISTRY_H_

#include <vector>

#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert::qnn {

template <typename H, typename V>
class Registry {
 public:
  Expected<H> Register(const V& value) {
    // TODO: improve this linear search by keeping an index to the first unused
    // element.
    for (auto i = 0; i < entries_.size(); ++i) {
      auto& entry = entries_[i];
      if (!entry.used) {
        entry.value = value;
        entry.used = true;
        return static_cast<H>(i);
      }
    }
    // Grow the set of entries.
    H handle = static_cast<H>(entries_.size());
    entries_.emplace_back(value);
    return handle;
  }

  Expected<void> Unregister(H handle) {
    if (handle < 0 || handle >= entries_.size()) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Unexpected handle");
    }
    entries_[handle].used = false;
    return {};
  }

  Expected<V*> Get(H handle) {
    if (handle < 0 || handle >= entries_.size()) {
      return Unexpected(kLiteRtStatusErrorNotFound, "Unexpected handle");
    }
    return &entries_[handle].value;
  }

 private:
  struct Entry {
    V value;
    bool used;
    explicit Entry(const V& v) : value(v), used(true) {}
  };

  std::vector<Entry> entries_;
};

}  // namespace litert::qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_DISPATCH_REGISTRY_H_

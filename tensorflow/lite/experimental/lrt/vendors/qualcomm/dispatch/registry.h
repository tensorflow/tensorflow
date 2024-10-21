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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_DISPATCH_REGISTRY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_DISPATCH_REGISTRY_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace litert {
namespace qnn {

template <typename H, typename V>
class Registry {
 public:
  absl::StatusOr<H> Register(const V& value) {
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

  absl::Status Unregister(H handle) {
    if (handle < 0 || handle >= entries_.size()) {
      return absl::NotFoundError("Unexpected handle");
    }
    entries_[handle].used = false;
    return {};
  }

  absl::StatusOr<V*> Get(H handle) {
    if (handle < 0 || handle >= entries_.size()) {
      return absl::NotFoundError("Unexpected handle");
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

}  // namespace qnn
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_DISPATCH_REGISTRY_H_

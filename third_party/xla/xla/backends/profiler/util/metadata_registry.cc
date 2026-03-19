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

#include "xla/backends/profiler/util/metadata_registry.h"

#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace xla {
namespace profiler {

struct Metadata {
  absl::Mutex mu;
  absl::flat_hash_map<std::string, std::string> data ABSL_GUARDED_BY(mu);
};

Metadata& GetMetadata() {
  static Metadata* metadata = new Metadata();
  return *metadata;
}

void SetProfilerMetadata(absl::string_view key, absl::string_view value) {
  Metadata& metadata = GetMetadata();
  absl::MutexLock lock(metadata.mu);
  metadata.data[key] = value;
}

std::string GetProfilerMetadata(absl::string_view key) {
  Metadata& metadata = GetMetadata();
  absl::MutexLock lock(metadata.mu);
  auto it = metadata.data.find(key);
  if (it == metadata.data.end()) {
    return "";
  }
  return it->second;
}

absl::flat_hash_map<std::string, std::string> GetAllProfilerMetadata() {
  Metadata& metadata = GetMetadata();
  absl::MutexLock lock(metadata.mu);
  return metadata.data;
}

void ClearProfilerMetadata() {
  Metadata& metadata = GetMetadata();
  absl::MutexLock lock(metadata.mu);
  metadata.data.clear();
}

}  // namespace profiler
}  // namespace xla

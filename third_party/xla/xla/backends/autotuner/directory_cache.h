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

#ifndef XLA_BACKENDS_AUTOTUNER_DIRECTORY_CACHE_H_
#define XLA_BACKENDS_AUTOTUNER_DIRECTORY_CACHE_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/autotuning.pb.h"
#include "xla/backends/autotuner/persistent_cache.h"

namespace xla {

// DirectoryCache is a concrete implementation of PersistentCache that stores
// each entry as a separate file in a directory.
//
// The cache is keyed by the HLO fingerprint and the explicit version. The
// directory structure is:
//   <directory_path>/
//     <device>/
//       <explicit_version>/
//         <hlo_fingerprint>.pb
//
// If the explicit version is not set, the cache entry is stored in:
//   <directory_path>/
//     <device>/
//       <hlo_fingerprint>.pb
//
// The directory path is provided by the user and is expected to be an absolute
// path. The cache mode and key matching mode are the same as defined in
// PersistentCache.
class DirectoryCache : public PersistentCache {
 public:
  DirectoryCache(AutotuneScope scope, std::string directory_path,
                 CacheMode mode = CacheMode::kReadWrite,
                 KeyMatchingMode matching_mode = KeyMatchingMode::kStrict)
      : PersistentCache(std::move(scope), mode, matching_mode),
        directory_path_(std::move(directory_path)) {}

 protected:
  // Returns a vector of entries that match the target key, or an empty vector
  // if no match is found.
  absl::StatusOr<std::vector<autotuner::AutotuneEntry>> Read(
      const autotuner::AutotuneTargetKey& target_key) override;

  absl::Status Write(const autotuner::AutotuneEntry& entry) override;

 private:
  std::string directory_path_;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_DIRECTORY_CACHE_H_

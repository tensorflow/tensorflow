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

#ifndef XLA_BACKENDS_AUTOTUNER_DIRECTORY_STORE_H_
#define XLA_BACKENDS_AUTOTUNER_DIRECTORY_STORE_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/autotune_cache_store.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/autotuning.pb.h"

namespace xla {

// DirectoryStore implements AutotuneCacheStore by writing each autotune entry
// into its own protobuf file inside a structured directory layout:
//   <directory_path>/<device>/[<explicit_version>]/<hlo_fingerprint>.pb
//
// Renames are used during writes to ensure updates are atomic and thread-safe.
class DirectoryStore : public AutotuneCacheStore {
 public:
  DirectoryStore(std::string directory_path, CacheMode mode);
  ~DirectoryStore() override = default;

  absl::StatusOr<std::vector<autotuner::AutotuneEntry>> Read(
      const autotuner::AutotuneTargetKey& target_key) override;

  absl::Status Write(const autotuner::AutotuneEntry& entry) override;

  absl::StatusOr<std::vector<autotuner::AutotuneEntry>> ReadAll() override;

  CacheMode GetMode() const override { return mode_; }

 private:
  std::string GetFilePath(const autotuner::AutotuneTargetKey& target_key) const;

  std::string directory_path_;
  CacheMode mode_;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_DIRECTORY_STORE_H_

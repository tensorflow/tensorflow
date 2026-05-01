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

#ifndef XLA_STREAM_EXECUTOR_ABI_RUNTIME_ABI_VERSION_MANAGER_H_
#define XLA_STREAM_EXECUTOR_ABI_RUNTIME_ABI_VERSION_MANAGER_H_

#include <memory>
#include <string>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/abi/runtime_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version_resolver.h"

namespace stream_executor {

/** A registry which allows registration of RuntimeAbiVersionFactory functions
 * for each platform.
 *
 * This allows deserialization of a RuntimeAbiVersionProto to a
 * RuntimeAbiVersion object without directly linking the implementation to the
 * deserialization code for each platform.
 */
class RuntimeAbiVersionManager : public RuntimeAbiVersionResolver {
 public:
  RuntimeAbiVersionManager() = default;
  RuntimeAbiVersionManager(const RuntimeAbiVersionManager&) = delete;
  RuntimeAbiVersionManager& operator=(const RuntimeAbiVersionManager&) = delete;

  static RuntimeAbiVersionManager& GetInstance();

  absl::Status RegisterRuntimeAbiVersionFactory(
      std::string platform_name,
      RuntimeAbiVersionFactory runtime_abi_version_factory);

  absl::StatusOr<std::unique_ptr<RuntimeAbiVersion> absl_nonnull>
  GetRuntimeAbiVersion(const RuntimeAbiVersionProto& proto) const override;

 private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<std::string, RuntimeAbiVersionFactory>
      runtime_abi_version_factories_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ABI_RUNTIME_ABI_VERSION_MANAGER_H_

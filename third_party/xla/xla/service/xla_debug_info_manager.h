/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_XLA_DEBUG_INFO_MANAGER_H_
#define XLA_SERVICE_XLA_DEBUG_INFO_MANAGER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "tsl/platform/status.h"

namespace xla {

using ModuleIdentifier = int;

// XlaDebugInfoManager tracks all XLA programs (Executables) throughout their
// lifetime. Because the tracing period can start during an Executable's
// execution, we need to track Executables even when tracing is off.
// This class is thread-safe.
class XlaDebugInfoManager {
 public:
  static XlaDebugInfoManager* Get() {
    static XlaDebugInfoManager* singleton = new XlaDebugInfoManager();
    return singleton;
  }

  // Registers an active module to XlaDebugInfoManager.
  // The module_id of the module is expected to be unique per process.
  void RegisterModule(std::shared_ptr<const HloModule> hlo_module,
                      BufferAssignmentProto buffer_assignment);

  // Unregisters an active module.
  void UnregisterModule(ModuleIdentifier module_id);

  // Start tracing, began to collecting debug information for all the running
  // modules during the tracing period.
  void StartTracing();

  // Stops tracing.
  // If module_debug_info is not null, returns debug information for all the
  // modules that were alive since StartTracing().
  void StopTracing(
      std::vector<std::unique_ptr<HloProto>>* module_debug_info = nullptr);

  // Returns whether 'module_id' is tracked by XlaDebugInfoManager.
  bool TracksModule(ModuleIdentifier module_id) const;

  friend class XlaDebugInfoManagerTestPeer;

 private:
  XlaDebugInfoManager() = default;

  struct XlaModuleEntry {
    std::shared_ptr<const HloModule> hlo_module;
    BufferAssignmentProto buffer_assignment;
    bool active = false;
  };

  mutable absl::Mutex mutex_;
  bool tracing_active_ ABSL_GUARDED_BY(mutex_) = false;
  // Active modules are those still tracked by us. There could be much more
  // active modules than running modules, we will try to reduce the trace size
  // by only transfer those modules that were running during tracing period.
  absl::flat_hash_map<ModuleIdentifier, XlaModuleEntry> modules_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace xla

#endif  // XLA_SERVICE_XLA_DEBUG_INFO_MANAGER_H_

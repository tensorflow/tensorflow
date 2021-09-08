/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_XLA_DEBUG_INFO_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_XLA_DEBUG_INFO_MANAGER_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {

using ModuleIdentifier = string;

struct XlaModuleDebugInfo {
  ModuleIdentifier module_id;
  // The hlo proto associated with this xla program.
  std::unique_ptr<HloProto> hlo_proto;
  // TODO(b/133503446): We might need add performance info from cost analysis
  // and DeviceDescription which contains peak memory bandwidth, clock speed,
  // core count, and other device characteristics.
};

// Debug info manager keeps track of all the debug information (symbol table,
// HLO proto etc) during tracing period. Because tracing period can start
// during module execution, therefore even when tracing is off, we still need
// minimum level of monitoring (i.e. which program is running lately).
// We allow multiple programs with the same module_id, however from tracing
// debug information perspective, same module id implies the same debug
// information. We will only keep track unique debug information, identified
// by module_id.
// This class is thread-safe.
class XlaDebugInfoManager {
 public:
  static XlaDebugInfoManager* Get() {
    static XlaDebugInfoManager* singleton = new XlaDebugInfoManager();
    return singleton;
  }

  // Register an active module to XlaDebugInfoManager. We will keep track all
  // existing HloModules within the process.
  // Modules with same module id can be registered and tracked separately.
  void RegisterModule(
      const ModuleIdentifier& module_id, std::shared_ptr<HloModule> hlo_module,
      std::shared_ptr<const BufferAssignmentProto> buffer_assignment);

  // Unregister an active module. When the last active module of the same
  // module id is out of scope, we remove it from our database.
  // However during tracing, we will defer the cleanup after serialization.
  void UnregisterModule(
      const ModuleIdentifier& module_id, std::shared_ptr<HloModule> hlo_module,
      std::shared_ptr<const BufferAssignmentProto> buffer_assignment);

  // Register when the module start execution on certain device.
  // TODO(jiesun): Although we now track both running and compile time
  // metadata, let's keep the interface for now.
  void OnModuleStart(ModuleIdentifier module_id) {}
  // Register when the module stop execution on certain device.
  void OnModuleStop(ModuleIdentifier module_id) {}

  // Start tracing, began to collecting debug information for all the running
  // modules during the tracing period.
  void StartTracing();

  // Stop tracing and drop all instances that have been stoped during tracing,
  // Then drop all modules that have no instances registered. Dump debug
  // information for all the running modules to module_debug_info if specified.
  void StopTracing(
      std::vector<XlaModuleDebugInfo>* module_debug_info = nullptr);

  friend class XlaDebugInfoManagerTest;

 private:
  XlaDebugInfoManager() {}

  std::set<ModuleIdentifier> GetActiveModules() {
    tensorflow::mutex_lock lock(mutex_);
    std::set<ModuleIdentifier> active;
    for (const auto& id : active_modules_) {
      active.insert(id.first);
    }
    return active;
  }

  // We track each instance of GpuExecutable. Assuming multiple GpuExecutable
  // can have same unique id if they are actually same program. From the
  // perspective of symbol table, they are identical, but for the life time
  // tracking, they need to be tracked separately.
  struct XlaModuleInstance {
    XlaModuleInstance(std::shared_ptr<HloModule> m,
                      std::shared_ptr<const BufferAssignmentProto> b)
        : hlo_module(std::move(m)), buffer_assignment(std::move(b)) {}
    std::shared_ptr<HloModule> hlo_module;
    std::shared_ptr<const BufferAssignmentProto> buffer_assignment;
    bool active = true;
  };

  // Each XlaModuleEntry can have multiple XlaModuleInstance's if XlA registers
  // them with the same ModuleIdentifier.
  struct XlaModuleEntry {
    // The module symbol table/debug info that shared by all instances.
    ModuleIdentifier module_id;
    std::vector<XlaModuleInstance> instances;
  };

  tensorflow::mutex mutex_;
  bool tracing_active_ TF_GUARDED_BY(mutex_) = false;
  // Active modules are those still tracked by us. There could be much more
  // active modules than running modules, we will try to reduce the trace size
  // by only transfer those modules that were running during tracing period.
  absl::flat_hash_map<ModuleIdentifier, XlaModuleEntry> active_modules_
      TF_GUARDED_BY(mutex_);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_XLA_DEBUG_INFO_MANAGER_H_

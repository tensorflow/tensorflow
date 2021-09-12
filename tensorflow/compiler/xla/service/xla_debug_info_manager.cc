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

#include "tensorflow/compiler/xla/service/xla_debug_info_manager.h"

#include "tensorflow/compiler/xla/service/hlo_proto_util.h"

namespace xla {

void XlaDebugInfoManager::RegisterModule(
    const ModuleIdentifier& module_id, std::shared_ptr<HloModule> hlo_module,
    std::shared_ptr<const BufferAssignmentProto> buffer_assignment) {
  tensorflow::mutex_lock lock(mutex_);
  if (active_modules_.find(module_id) != active_modules_.end()) {
    active_modules_[module_id].instances.emplace_back(hlo_module,
                                                      buffer_assignment);
  } else {
    XlaModuleEntry m;
    m.module_id = module_id;
    m.instances.emplace_back(hlo_module, buffer_assignment);
    active_modules_[module_id] = std::move(m);
  }
}

// Unregister an active module, when the last active module of the same
// module id is out of scope, we remove it from our database.
// However during tracing, we will defer the cleanup after serialization.
void XlaDebugInfoManager::UnregisterModule(
    const ModuleIdentifier& module_id, std::shared_ptr<HloModule> hlo_module,
    std::shared_ptr<const BufferAssignmentProto> buffer_assignment) {
  tensorflow::mutex_lock lock(mutex_);
  CHECK(active_modules_.find(module_id) != active_modules_.end());
  XlaModuleEntry& active_module = active_modules_[module_id];
  auto instance_it =
      absl::c_find_if(active_module.instances, [&](XlaModuleInstance& e) {
        return e.hlo_module == hlo_module &&
               e.buffer_assignment == buffer_assignment;
      });

  CHECK(instance_it != active_module.instances.end());

  if (!tracing_active_) {
    active_module.instances.erase(instance_it);
    if (active_module.instances.empty()) {
      active_modules_.erase(module_id);
    }
  } else {
    instance_it->active = false;
  }
}

void XlaDebugInfoManager::StartTracing() {
  tensorflow::mutex_lock lock(mutex_);
  tracing_active_ = true;
}

void XlaDebugInfoManager::StopTracing(
    std::vector<XlaModuleDebugInfo>* module_debug_info) {
  std::vector<XlaModuleEntry> modules_to_serialize;
  {
    tensorflow::mutex_lock lock(mutex_);
    if (!tracing_active_) return;
    tracing_active_ = false;
    for (const auto& traced_module_id : active_modules_) {
      const XlaModuleEntry& active_module = traced_module_id.second;

      // Copy the instance so that we can serialize without holding the lock.
      // All instances are equivalent from the perspective of symbolization.
      // We only use the first one.
      if (!active_module.instances.empty()) {
        XlaModuleEntry e;
        e.module_id = active_module.module_id;
        e.instances.push_back(active_module.instances[0]);
        modules_to_serialize.push_back(std::move(e));
      }
    }

    // Remove all active modules which have an instance count equal to zero.
    for (auto it = active_modules_.begin(); it != active_modules_.end();) {
      auto& active_module = it->second;
      for (auto instance = active_module.instances.begin();
           instance != active_module.instances.end();) {
        if (instance->active) {
          ++instance;
        } else {
          instance = active_module.instances.erase(instance);
        }
      }

      if (active_module.instances.empty()) {
        active_modules_.erase(it++);
      } else {
        ++it;
      }
    }
  }

  if (module_debug_info) {
    module_debug_info->clear();
    for (const auto& m : modules_to_serialize) {
      XlaModuleDebugInfo info;
      info.module_id = m.module_id;
      // In real world, hlo_module and buffer_assignment will always be
      // non-nullptr. Due to the inconvenience of creation of buffer_assignment
      // object in test, we set it to nullptr and guard this for it.
      if (m.instances[0].hlo_module && m.instances[0].buffer_assignment) {
        info.hlo_proto = absl::make_unique<HloProto>(
            MakeHloProto(*m.instances[0].hlo_module));
        *info.hlo_proto->mutable_buffer_assignment() =
            *m.instances[0].buffer_assignment;
      }
      module_debug_info->emplace_back(std::move(info));
    }
  }
}

}  // namespace xla

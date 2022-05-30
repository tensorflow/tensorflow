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

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo_proto_util.h"

namespace xla {

void XlaDebugInfoManager::RegisterModule(
    ModuleIdentifier module_id, std::shared_ptr<const HloModule> hlo_module,
    std::shared_ptr<const BufferAssignmentProto> buffer_assignment) {
  CHECK(hlo_module != nullptr && module_id == hlo_module->unique_id());
  absl::MutexLock lock(&mutex_);
  auto result = modules_.try_emplace(module_id);
  CHECK(result.second);
  XlaModuleEntry& m = result.first->second;
  m.hlo_module = std::move(hlo_module);
  m.buffer_assignment = std::move(buffer_assignment);
  m.active = true;
}

// Unregister an active module, when the last active module of the same
// module id is out of scope, we remove it from our database.
// However during tracing, we will defer the cleanup after serialization.
void XlaDebugInfoManager::UnregisterModule(ModuleIdentifier module_id) {
  absl::MutexLock lock(&mutex_);
  auto it = modules_.find(module_id);
  CHECK(it != modules_.end());
  if (!tracing_active_) {
    modules_.erase(it);
  } else {
    XlaModuleEntry& m = it->second;
    m.active = false;
  }
}

void XlaDebugInfoManager::StartTracing() {
  absl::MutexLock lock(&mutex_);
  tracing_active_ = true;
}

void XlaDebugInfoManager::StopTracing(
    std::vector<std::unique_ptr<HloProto>>* module_debug_info) {
  std::vector<XlaModuleEntry> modules_to_serialize;
  {
    absl::MutexLock lock(&mutex_);
    if (!tracing_active_) return;
    tracing_active_ = false;

    // Copy all modules so we can serialize without holding the lock, and remove
    // all inactive modules.
    modules_to_serialize.reserve(modules_.size());
    for (auto it = modules_.begin(); it != modules_.end();) {
      auto& m = it->second;
      if (!m.active) {
        modules_to_serialize.emplace_back(std::move(m));
        modules_.erase(it++);
      } else {
        modules_to_serialize.emplace_back(m);
        ++it;
      }
    }
  }

  if (module_debug_info) {
    module_debug_info->clear();
    for (const auto& m : modules_to_serialize) {
      // In real world, hlo_module and buffer_assignment will always be
      // non-nullptr. Due to the inconvenience of creation of buffer_assignment
      // object in test, we set it to nullptr and guard this for it.
      auto hlo_proto = absl::make_unique<HloProto>(MakeHloProto(*m.hlo_module));
      if (m.buffer_assignment != nullptr) {
        *hlo_proto->mutable_buffer_assignment() = *m.buffer_assignment;
      }
      module_debug_info->emplace_back(std::move(hlo_proto));
    }
  }
}

}  // namespace xla

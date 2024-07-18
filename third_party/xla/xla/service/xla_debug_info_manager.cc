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

#include "xla/service/xla_debug_info_manager.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_proto_util.h"
#include "tsl/platform/logging.h"

namespace xla {

void XlaDebugInfoManager::RegisterModule(
    std::shared_ptr<const HloModule> hlo_module,
    BufferAssignmentProto buffer_assignment) {
  CHECK(hlo_module != nullptr);
  absl::MutexLock lock(&mutex_);
  auto result = modules_.try_emplace(hlo_module->unique_id());
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

static void print_msg_size(const std::string& name, const proto2::Message& msg,
                           int level, const std::string& prefix) {
  if (level <= 0) return;
  LOG(INFO) << prefix << name << ", ByteSizeLong()=" << msg.ByteSizeLong();
  if (level <= 1) return;

  std::string child_prefix = prefix + "----";
  const auto* desc = msg.GetDescriptor();
  const auto* refl = msg.GetReflection();
  auto field_count = desc->field_count();
  for (int i = 0; i < field_count; i++) {
    const proto2::FieldDescriptor* field = desc->field(i);
    const proto2::Message& child_msg = refl->GetMessage(msg, field);
    int field_size = field->is_repeated() ? refl->FieldSize(msg, field) : 0;
    LOG(INFO) << child_prefix << field->name()
              << ", element count=" << field_size
              << ", ByteSizeLong()=" << child_msg.ByteSizeLong();
    for (int j = 0; j < field_size; j++) {
      const proto2::Message& sub_ch_msg =
          refl->GetRepeatedMessage(msg, field, j);
      std::string sub_ch_name = field->name() + "[" + std::to_string(j) + "]";
      print_msg_size(sub_ch_name, sub_ch_msg, level - 1, child_prefix);
    }
  }
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
      auto cur_it = it++;
      if (!m.active) {
        modules_to_serialize.emplace_back(std::move(m));
        modules_.erase(cur_it);
      } else {
        modules_to_serialize.emplace_back(m);
      }
    }
  }

  if (module_debug_info) {
    module_debug_info->clear();
    for (const auto& m : modules_to_serialize) {
      auto hlo_proto = std::make_unique<HloProto>(MakeHloProto(*m.hlo_module));
      const auto& module_proto = hlo_proto->hlo_module();
      LOG(INFO) << "XlaModule size info for " << m.hlo_module->name()
                << ", id=" << m.hlo_module->unique_id();
      print_msg_size("xla_module", module_proto, 2, "----");
      print_msg_size("buffer_assignment", m.buffer_assignment, 2, "----");
      *hlo_proto->mutable_buffer_assignment() = m.buffer_assignment;
      module_debug_info->emplace_back(std::move(hlo_proto));
    }
  }
}

bool XlaDebugInfoManager::TracksModule(ModuleIdentifier module_id) const {
  absl::MutexLock lock(&mutex_);
  return modules_.find(module_id) != modules_.end();
}

}  // namespace xla

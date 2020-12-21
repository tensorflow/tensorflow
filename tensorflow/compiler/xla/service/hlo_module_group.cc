/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_module_group.h"

#include "tensorflow/core/lib/hash/hash.h"

namespace xla {

HloModuleGroup::HloModuleGroup(std::unique_ptr<HloModule> module)
    : name_(module->name()) {
  push_back(std::move(module));
}

HloModuleGroup::HloModuleGroup(absl::string_view name,
                               absl::Span<std::unique_ptr<HloModule>> modules)
    : name_(name) {
  for (auto& module : modules) {
    push_back(std::move(module));
  }
}

HloModuleGroup::HloModuleGroup(
    absl::string_view name, std::vector<std::unique_ptr<HloModule>>&& modules)
    : name_(name) {
  for (auto& module : modules) {
    push_back(std::move(module));
  }
}

std::vector<std::unique_ptr<HloModule>> HloModuleGroup::ConsumeModules() {
  std::vector<std::unique_ptr<HloModule>> ret_modules = std::move(modules_);

  // Clear everything so the object state is in a known (empty) state.
  modules_.clear();
  module_ptrs_.clear();
  return ret_modules;
}

string HloModuleGroup::ToString() const {
  std::ostringstream s;
  s << "HloModuleGroup " << name() << "\n\n";
  for (const HloModule* module : modules()) {
    s << module->ToString() << "\n";
  }
  return s.str();
}

HloModuleGroupProto HloModuleGroup::ToProto() const {
  HloModuleGroupProto proto;
  proto.set_name(name());
  for (const HloModule* module : modules()) {
    *proto.add_hlo_modules() = module->ToProto();
  }
  return proto;
}

uint64 HloModuleGroup::Hash() const {
  uint64 result = 0;
  for (auto& module : modules_) {
    result = tensorflow::Hash64Combine(result, module->Hash());
  }
  return result;
}

/* static */ StatusOr<HloModuleGroup> HloModuleGroup::CreateFromProto(
    const HloModuleGroupProto& proto,
    absl::Span<const HloModuleConfig> module_configs) {
  TF_RET_CHECK(!proto.name().empty()) << "Module group name cannot be empty";
  TF_RET_CHECK(proto.hlo_modules_size() > 0)
      << "Module group must have at least one HLO module";
  TF_RET_CHECK(proto.hlo_modules_size() == module_configs.size());

  std::vector<std::unique_ptr<HloModule>> modules;
  for (int i = 0; i < proto.hlo_modules_size(); ++i) {
    const HloModuleProto& module_proto = proto.hlo_modules(i);
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> module,
        HloModule::CreateFromProto(module_proto, module_configs[i]));
    modules.push_back(std::move(module));
  }

  return HloModuleGroup(proto.name(), absl::MakeSpan(modules));
}

void HloModuleGroup::push_back(std::unique_ptr<HloModule> module) {
  module->metadata()->set_module_group_name(name());
  modules_.push_back(std::move(module));
  module_ptrs_.push_back(modules_.back().get());
}

void HloModuleGroup::ReplaceModule(int index,
                                   std::unique_ptr<HloModule> module) {
  modules_.at(index)->MoveMetadataToModule(module.get());
  modules_.at(index) = std::move(module);
  module_ptrs_.at(index) = modules_.at(index).get();
}

std::ostream& operator<<(std::ostream& out, const HloModuleGroup& group) {
  out << group.ToString();
  return out;
}

}  // namespace xla

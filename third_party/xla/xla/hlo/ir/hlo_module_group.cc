/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_module_group.h"

#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

HloModuleGroup::HloModuleGroup(std::unique_ptr<HloModule> module)
    : name_(module->name()), module_(std::move(module)) {
  module_->metadata()->set_module_group_name(name_);
}

std::vector<std::unique_ptr<HloModule>> HloModuleGroup::ConsumeModules() {
  std::vector<std::unique_ptr<HloModule>> ret_modules;
  ret_modules.push_back(std::move(module_));
  return ret_modules;
}

std::string HloModuleGroup::ToString() const {
  std::ostringstream s;
  s << "HloModuleGroup " << name() << "\n\n";
  if (module_) {
    s << module_->ToString() << "\n";
  }
  return s.str();
}

void HloModuleGroup::AddModule(std::unique_ptr<HloModule> module) {
  CHECK_EQ(module_, nullptr);
  module_ = std::move(module);
}

std::ostream& operator<<(std::ostream& out, const HloModuleGroup& group) {
  out << group.ToString();
  return out;
}

}  // namespace xla

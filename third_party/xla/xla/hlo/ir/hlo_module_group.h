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

#ifndef XLA_HLO_IR_HLO_MODULE_GROUP_H_
#define XLA_HLO_IR_HLO_MODULE_GROUP_H_

#include <array>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"

namespace xla {

// An abstraction representing a ordered set of HLO module built to run
// concurrently across different devices.
class HloModuleGroup {
 public:
  // Construct a module group containing a single module.
  explicit HloModuleGroup(std::unique_ptr<HloModule> module);

  HloModuleGroup(const HloModuleGroup& other) = delete;
  HloModuleGroup(HloModuleGroup&& other) = default;
  HloModuleGroup& operator=(const HloModuleGroup& other) = delete;
  HloModuleGroup& operator=(HloModuleGroup&& other) = default;

  // Returns the modules contained in the group.
  std::array<HloModule*, 1> modules() const { return {module_.get()}; }

  // Returns a module at a particular index.
  HloModule& module() const { return *module_; }
  HloModule& module(int index) const {
    CHECK_EQ(index, 0);
    return *module_;
  }

  // Adds a module to the group, taking ownership of it.
  void AddModule(std::unique_ptr<HloModule> module);

  // Moves all modules from the group into the returned vector. After this
  // method runs, the module group will be empty.
  std::vector<std::unique_ptr<HloModule>> ConsumeModules();

  std::string name() const { return name_; }

  std::string ToString() const;

  // Deallocate removed instructions in each module.
  void Cleanup() {
    if (module_) {
      module_->Cleanup();
    }
  }

  template <typename H>
  friend H AbslHashValue(H h, const HloModuleGroup& group) {
    if (!group.module_) {
      return h;
    }
    return H::combine(std::move(h), group.module_);
  }


  // Returns the number of modules in the module group.
  int size() const { return module_ ? 1 : 0; }

  // Returns true if there are no modules in the module group.
  bool empty() const { return !module_; }

  std::string name_;

  // Vector of modules as std::unique_ptrs.
  std::unique_ptr<HloModule> module_;
};

std::ostream& operator<<(std::ostream& out, const HloModuleGroup& group);

}  // namespace xla

#endif  // XLA_HLO_IR_HLO_MODULE_GROUP_H_

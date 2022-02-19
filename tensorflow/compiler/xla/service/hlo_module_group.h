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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_H_

#include <iosfwd>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {

// An abstraction representing a ordered set of HLO module built to run
// concurrently across different devices.
class HloModuleGroup {
 public:
  // Construct an empty module group.
  explicit HloModuleGroup(absl::string_view name) : name_(name) {}

  // Construct a module group containing a single module.
  explicit HloModuleGroup(std::unique_ptr<HloModule> module);

  // Construct a module group containing any number of modules.
  HloModuleGroup(absl::string_view name,
                 absl::Span<std::unique_ptr<HloModule>> modules);
  HloModuleGroup(absl::string_view name,
                 std::vector<std::unique_ptr<HloModule>>&& modules);

  // Returns the modules contained in the group.
  const std::vector<HloModule*>& modules() const { return module_ptrs_; }

  // Returns a module at a particular index.
  HloModule& module(int index) const { return *module_ptrs_.at(index); }

  // Add a module to the back of vector of modules in the group.
  void push_back(std::unique_ptr<HloModule> module);

  // Replaces the existing module at the given index with the given module. The
  // existing module is discarded.
  void ReplaceModule(int index, std::unique_ptr<HloModule> module);

  // Moves all modules from the group into the returned vector. After this
  // method runs, the module group will be empty.
  std::vector<std::unique_ptr<HloModule>> ConsumeModules();

  std::string name() const { return name_; }

  std::string ToString() const;

  // Deallocate removed instructions in each module.
  void Cleanup() {
    for (auto& module : modules_) {
      module->Cleanup();
    }
  }

  template <typename H>
  friend H AbslHashValue(H h, const HloModuleGroup& group) {
    for (auto& module : group.modules_) {
      h = H::combine(std::move(h), *module);
    }
    return H::combine(std::move(h), group.modules_.size());
  }

  // Serialize the module group to/from a proto.
  HloModuleGroupProto ToProto() const;
  static StatusOr<HloModuleGroup> CreateFromProto(
      const HloModuleGroupProto& proto,
      absl::Span<const HloModuleConfig> module_configs);

  // Returns the number of modules in the module group.
  int size() const { return modules_.size(); }

  // Returns true if there are no modules in the module group.
  bool empty() const { return modules_.empty(); }

  absl::string_view cache_key() const { return cache_key_; }
  void set_cache_key(absl::string_view cache_key) {
    cache_key_ = std::string(cache_key);
  }

 private:
  std::string name_;

  // Vector of modules as std::unique_ptrs.
  std::vector<std::unique_ptr<HloModule>> modules_;

  // Vector of modules as normal pointers. This vector is kept in sync with
  // modules_ as modules are added to the group with push_back.
  std::vector<HloModule*> module_ptrs_;

  std::string cache_key_;
};

std::ostream& operator<<(std::ostream& out, const HloModuleGroup& group);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_GROUP_H_

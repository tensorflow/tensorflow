/* Copyright 2018 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/hlo_computation_name_uniquify.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"

#include <string>

namespace xla {
namespace poplarplugin {
namespace {
static absl::flat_hash_set<std::string> GetReservedPrefixes() {
  static absl::flat_hash_set<std::string> reserved_prefixes = {
      "__repeat", "__inline", "__arithmetic", "_pop_op"};
  return reserved_prefixes;
}

class ReservedNamespaceNameUniquer : public NameUniquer {
 public:
  std::string GetUniqueName(absl::string_view name) override {
    std::string name_str = std::string(name);
    auto starts_with_reserved_prefix = [&](std::string reserved) {
      return tensorflow::str_util::StartsWith(name_str, reserved);
    };
    if (absl::c_any_of(GetReservedPrefixes(), starts_with_reserved_prefix)) {
      name_str.insert(0, "a");
    }
    return name_str;
  }
};
}  // namespace

StatusOr<bool> HloComputationNameUniquify::Run(HloModule* module) {
  bool changed = false;
  ReservedNamespaceNameUniquer rnnu;
  for (auto* comp : module->computations()) {
    auto name_before = comp->name();
    comp->UniquifyName(&rnnu);
    auto name_after = comp->name();
    if (name_before != name_after) {
      changed = true;
    }
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla

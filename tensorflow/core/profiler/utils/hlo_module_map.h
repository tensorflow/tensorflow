/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_HLO_MODULE_MAP_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_HLO_MODULE_MAP_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"

namespace tensorflow {
namespace profiler {

// This wrapper allows caching the results of HloInstruction methods.
// This wrapper is not thread safe.
class HloInstructionWrapper {
 public:
  explicit HloInstructionWrapper(
      const xla::HloInstruction* instr,
      const xla::HloCostAnalysis* cost_analysis = nullptr);

  // Non copiable
  HloInstructionWrapper(const HloInstructionWrapper&) = delete;
  HloInstructionWrapper& operator=(const HloInstructionWrapper&) = delete;
  // Movable.
  HloInstructionWrapper(HloInstructionWrapper&&) = default;
  HloInstructionWrapper& operator=(HloInstructionWrapper&&) = default;

  absl::string_view Name() const { return instr_->name(); }

  xla::HloOpcode HloOpcode() const { return instr_->opcode(); }

  std::string HloOpcodeString() const {
    return std::string(xla::HloOpcodeString(instr_->opcode()));
  }

  const xla::OpMetadata& Metadata() const { return instr_->metadata(); }

  size_t flops() const { return flops_; }
  size_t bytes_accessed() const { return bytes_accessed_; }

  std::string_view op_full_name() const { return op_full_name_; }
  std::string source_info() const;

 private:
  const xla::HloInstruction* instr_;
  std::string op_full_name_;
  size_t flops_ = 0;
  size_t bytes_accessed_ = 0;
};

// Wrahps HLO module and provides an interface that maps HLO names to
// HloInstructionWrappers.
class HloModuleWrapper {
 public:
  explicit HloModuleWrapper(
      const xla::HloProto& hlo_proto,
      std::function<int64_t(const xla::Shape&)> shape_func = nullptr);

  explicit HloModuleWrapper(
      std::unique_ptr<xla::HloModule> module,
      std::function<int64_t(const xla::Shape&)> shape_func);

  const HloInstructionWrapper* GetHloInstruction(
      absl::string_view hlo_name) const;

  bool Empty() const { return instructions_by_name_.empty(); }

  absl::string_view Name() const { return module_->name(); }

 private:
  std::unique_ptr<xla::HloModule> module_;

  // Map of HloInstructionWrappers by name.
  using HloInstructionMap =
      absl::flat_hash_map<absl::string_view, HloInstructionWrapper>;
  HloInstructionMap instructions_by_name_;
};

// Map of HloModuleWrappers by program_id.
using HloModuleMap =
    absl::flat_hash_map<uint64_t /*program_id*/, HloModuleWrapper>;

void AddHloProto(HloModuleMap& hlo_module_map, uint64_t program_id,
                 const xla::HloProto& hlo_proto);

// WARNING: The returned pointer will be invalidated if HloModuleMap is mutated.
inline const HloModuleWrapper* GetHloModule(const HloModuleMap& hlo_module_map,
                                            uint64_t program_id) {
  auto iter = hlo_module_map.find(program_id);
  if (iter == hlo_module_map.end()) return nullptr;
  return &iter->second;
}

inline const HloInstructionWrapper* GetHloInstruction(
    const HloModuleMap& hlo_module_map, std::optional<uint64_t> program_id,
    absl::string_view hlo_name) {
  if (!program_id.has_value()) return nullptr;
  const auto* hlo_module = GetHloModule(hlo_module_map, *program_id);
  if (hlo_module == nullptr) return nullptr;
  return hlo_module->GetHloInstruction(hlo_name);
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_HLO_MODULE_MAP_H_

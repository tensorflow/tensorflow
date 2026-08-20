/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_INTERPRETER_INTERPRETER_EXECUTABLE_H_
#define XLA_PJRT_INTERPRETER_INTERPRETER_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/interpreter/interpreter_topology_description.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/dynamic_dimension_inference.h"

namespace xla {

class InterpreterExecutable final : public PjRtExecutable {
 public:
  InterpreterExecutable(
      std::shared_ptr<HloModule> hlo_module,
      std::optional<DynamicDimensionInference> dynamic_dimension_inference,
      CompileOptions compile_options,
      InterpreterTopologyDescription topology =
          InterpreterTopologyDescription());

  ~InterpreterExecutable() override = default;

  int num_replicas() const override;

  int num_partitions() const override;

  int64_t SizeOfGeneratedCodeInBytes() const override { return -1; }

  absl::string_view name() const override;

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override;

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetParameterMemoryKinds() const override;

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;

  absl::StatusOr<CompileOptions> GetCompileOptions() const override {
    return compile_options_;
  }

  const CompileOptions& compile_options() const { return compile_options_; }

  const std::shared_ptr<HloModule>& hlo_module() const { return hlo_module_; }

  const std::optional<DynamicDimensionInference>& dynamic_dimension_inference()
      const {
    return dynamic_dimension_inference_;
  }

  std::optional<DynamicDimensionInference>& dynamic_dimension_inference() {
    return dynamic_dimension_inference_;
  }

  const InterpreterTopologyDescription& topology() const { return topology_; }

  absl::StatusOr<std::string> FingerprintExecutable() const override;

  absl::StatusOr<std::string> SerializeExecutable() const override;

  static absl::StatusOr<std::unique_ptr<InterpreterExecutable>> Deserialize(
      absl::string_view serialized,
      std::optional<InterpreterTopologyDescription> topology = std::nullopt,
      std::optional<CompileOptions> options = std::nullopt);

 private:
  std::shared_ptr<HloModule> hlo_module_;
  std::optional<DynamicDimensionInference> dynamic_dimension_inference_;
  CompileOptions compile_options_;
  InterpreterTopologyDescription topology_;
};

absl::StatusOr<std::unique_ptr<HloModule>> RunInterpreterHloPasses(
    std::unique_ptr<HloModule> hlo_module);

absl::StatusOr<std::unique_ptr<InterpreterExecutable>>
CompileInterpreterExecutable(const XlaComputation& computation,
                             CompileOptions options,
                             const InterpreterTopologyDescription& topology);

absl::StatusOr<std::unique_ptr<InterpreterExecutable>>
CompileInterpreterExecutable(MaybeOwningMlirModule module,
                             CompileOptions options,
                             const InterpreterTopologyDescription& topology);

}  // namespace xla

#endif  // XLA_PJRT_INTERPRETER_INTERPRETER_EXECUTABLE_H_

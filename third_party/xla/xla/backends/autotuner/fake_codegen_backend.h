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

#ifndef XLA_BACKENDS_AUTOTUNER_FAKE_CODEGEN_BACKEND_H_
#define XLA_BACKENDS_AUTOTUNER_FAKE_CODEGEN_BACKEND_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/executable.h"

namespace xla {

class FakeCodegenBackend : public CodegenBackend {
 public:
  FakeCodegenBackend(autotuner::Backend backend, std::string version)
      : backend_(backend), version_(version) {}

  absl::string_view name() const override { return "fake"; }
  autotuner::Backend backend() const override { return backend_; }
  std::string version() const override { return version_; }

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  absl::StatusOr<std::unique_ptr<Executable>> Compile(
      const HloInstruction& instr, const BackendConfig& config) override {
    return absl::UnimplementedError("");
  }
  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override {
    return absl::OkStatus();
  }

  bool CanProduceWrongResults() const override { return false; }

 private:
  autotuner::Backend backend_;
  std::string version_;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_FAKE_CODEGEN_BACKEND_H_

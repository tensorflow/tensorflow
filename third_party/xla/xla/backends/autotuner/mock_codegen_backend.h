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

#ifndef XLA_BACKENDS_AUTOTUNER_MOCK_CODEGEN_BACKEND_H_
#define XLA_BACKENDS_AUTOTUNER_MOCK_CODEGEN_BACKEND_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/executable.h"

namespace xla {

inline int64_t GetAlgorithmId(absl::string_view name) {
  return static_cast<int64_t>(absl::HashOf(name) & 0x7FFFFFFF);
}

MATCHER_P(ConfigMatcher, name, "") {
  return arg.has_gemm() && arg.gemm().algorithm() == GetAlgorithmId(name);
}

inline std::unique_ptr<BackendConfig> GetTestConfig(absl::string_view name) {
  auto config = std::make_unique<BackendConfig>();
  config->mutable_gemm()->set_algorithm(GetAlgorithmId(name));
  return config;
}

class MockCodegenBackend : public CodegenBackend {
 public:
  MOCK_METHOD(absl::string_view, name, (), (const, override));
  MOCK_METHOD(autotuner::Backend, backend, (), (const, override));
  MOCK_METHOD(std::string, version, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>,
              GetSupportedConfigs, (const HloInstruction& instr), (override));
  MOCK_METHOD(absl::StatusOr<std::vector<EstimatedConfig>>,
              GetSupportedConfigsWithEstimates, (const HloInstruction& instr),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<BackendConfig>>, GetDefaultConfig,
              (const HloInstruction& instr), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Executable>>, Compile,
              (const HloInstruction& instr, const BackendConfig& config),
              (override));
  MOCK_METHOD(absl::Status, ApplyConfig,
              (HloInstruction & instr, const BackendConfig& config),
              (override));
  MOCK_METHOD(bool, CanProduceWrongResults, (), (const, override));
};

class MockCodegenBackendWithWrongResults : public MockCodegenBackend {
 public:
  bool CanProduceWrongResults() const override { return true; }
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_MOCK_CODEGEN_BACKEND_H_

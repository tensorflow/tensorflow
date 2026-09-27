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

#ifndef XLA_SERVICE_MOCK_COMPILER_H_
#define XLA_SERVICE_MOCK_COMPILER_H_

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/statusor.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_id.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

class MockCompiler : public Compiler {
 public:
  MOCK_METHOD(stream_executor::PlatformId, PlatformId, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<HloModule>>, RunHloPasses,
              (std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
               const CompileOptions& options),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Executable>>, RunBackend,
              (std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
               const CompileOptions& options),
              (override));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<Executable>>>, Compile,
              (std::unique_ptr<HloModule> module,
               std::vector<se::StreamExecutor*> stream_exec,
               const CompileOptions& options),
              (override));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>>,
              CompileAheadOfTime,
              (std::unique_ptr<HloModule> module,
               const AotCompilationOptions& options),
              (override));
  MOCK_METHOD(HloCostAnalysis::ShapeSizeFunction, ShapeSizeBytesFunction, (),
              (const, override));
};

}  // namespace xla

#endif  // XLA_SERVICE_MOCK_COMPILER_H_

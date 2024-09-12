/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_TESTS_CODEGEN_TEST_BASE_H_
#define XLA_TESTS_CODEGEN_TEST_BASE_H_

#include <memory>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {

// Provides access to both the JIT and the AOT compiler for testing.
class CodegenTestBase : public HloTestBase {
 protected:
  // Compiles hlo_module with the JIT compiler.
  absl::StatusOr<std::unique_ptr<Executable>> CompileToExecutable(
      std::unique_ptr<HloModule> hlo_module,
      bool run_optimization_passes = true);

  // Compiles hlo_module with the AOT compiler.
  absl::StatusOr<std::unique_ptr<AotCompilationResult>>
  CompileToAotCompilationResult(std::unique_ptr<HloModule> hlo_module,
                                const AotCompilationOptions& options);
};

}  // namespace xla

#endif  // XLA_TESTS_CODEGEN_TEST_BASE_H_

/* Copyright 2025 The OpenXLA Authors.

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

#include <memory>

#include <gtest/gtest.h>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/test/plugin_test_fixture.h"
#include "xla/tsl/platform/statusor.h"

namespace {

using ::xla::PluginTestFixture;

constexpr char kPassThroughStableHlo[] = R"(
  module {
    func.func @main(%arg0: tensor<i32>) -> tensor<i32> {
      return %arg0 : tensor<i32>
    }
  })";

TEST_F(PluginTestFixture, CompileWithSharedTopology) {
  TF_ASSERT_OK_AND_ASSIGN(const xla::PjRtTopologyDescription* topology,
                          client_->GetTopologyDescription());
  ASSERT_NE(topology, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtCompiler> compiler_client,
                          xla::GetCApiCompiler(plugin_name_));

  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      mlir::OwningOpRef<mlir::ModuleOp> module,
      xla::ParseMlirModuleString(kPassThroughStableHlo, context));

  xla::CompileOptions compile_options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::PjRtExecutable> executable,
      compiler_client->Compile(compile_options, module.get(), *topology,
                               nullptr));
}

}  // namespace

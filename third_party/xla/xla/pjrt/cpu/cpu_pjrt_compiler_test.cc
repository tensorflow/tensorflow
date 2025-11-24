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

#include "xla/pjrt/cpu/cpu_pjrt_compiler.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

constexpr absl::string_view kProgram = R"(HloModule Computation

ENTRY Computation() -> s32[] {
  ROOT result = s32[] constant(2)
})";

constexpr absl::string_view kMlirProgram = R"mlir(
  module {
    func.func @main() -> tensor<i32> {
      %0 = mhlo.constant dense<2> : tensor<i32>
      return %0 : tensor<i32>
    }
  })mlir";

using CpuPjrtCompilerTest = xla::HloHardwareIndependentTestBase;

TEST_F(CpuPjrtCompilerTest, CompileXlaComputationSuccess) {
  xla::CompileOptions options;
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kProgram));
  xla::XlaComputation computation(module->ToProto());

  // TODO(basioli): Temporary hack to get the cpu topology easily, will update
  // test once cross-compilation is supported.
  TF_ASSERT_OK_AND_ASSIGN(auto cpu_client, xla::GetXlaPjrtCpuClient({}));

  TF_ASSERT_OK_AND_ASSIGN(auto topology_description,
                          cpu_client->GetTopologyDescription());

  xla::cpu::CpuPjRtCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      compiler.Compile(options, computation, *topology_description,
                       /*client=*/nullptr));
}

TEST_F(CpuPjrtCompilerTest, CompileMlirOpSuccess) {
  xla::CompileOptions options;
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::mhlo::MhloDialect>();
  auto mlir_module =
      mlir::parseSourceString<mlir::ModuleOp>(kMlirProgram, &context);

  // TODO(basioli): Temporary hack to get the cpu topology easily, will update
  // test once cross-compilation is supported.
  TF_ASSERT_OK_AND_ASSIGN(auto cpu_client, xla::GetXlaPjrtCpuClient({}));

  TF_ASSERT_OK_AND_ASSIGN(auto topology_description,
                          cpu_client->GetTopologyDescription());

  xla::cpu::CpuPjRtCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      compiler.Compile(options, *mlir_module, *topology_description,
                       /*client=*/nullptr));
}

}  // namespace
}  // namespace xla::cpu

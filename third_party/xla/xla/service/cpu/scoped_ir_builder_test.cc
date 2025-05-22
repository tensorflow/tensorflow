/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/cpu/target_machine_features_stub.h"
#include "xla/service/logical_buffer.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

class IRBuilderGuardTest : public HloHardwareIndependentTestBase {
 public:
  IrEmitter MakeIrEmitter(llvm::LLVMContext& context) {
    auto module = std::make_unique<llvm::Module>("test", context);
    const char* hlo_text = R"(
    HloModule m
    ENTRY main {
      ROOT %zero = s32[] constant(0)
    })";

    auto hlo = ParseAndReturnVerifiedModule(hlo_text).value();

    std::unique_ptr<BufferAssignment> buffer_assignment =
        BufferAssigner::Run(
            hlo.get(), std::make_unique<DependencyHloOrdering>(hlo.get()),
            [](const BufferValue& buffer) {
              return CpuExecutable::ShapeSizeBytes(buffer.shape());
            },
            [](LogicalBuffer::Color) { return /*alignment=*/1; })
            .value();

    TargetMachineFeaturesStub target_machine([](int64_t size) { return 1; });
    return IrEmitter(/*mlir_context=*/nullptr, /*hlo_module=*/*hlo,
                     /*assignment=*/*buffer_assignment,
                     /*llvm_module=*/module.get(),
                     /*instruction_to_profile_idx=*/{},
                     /*computation_to_profile_idx=*/{},
                     /*computation_transitively_contains_custom_call=*/{},
                     /*target_machine=*/&target_machine,
                     /*emit_code_for_msan=*/false);
  }
};

TEST_F(IRBuilderGuardTest, OverwriteBuilder) {
  llvm::LLVMContext context;
  IrEmitter ir_emitter = MakeIrEmitter(context);

  // A temporary builder to replace the IR emitter's builder.
  llvm::IRBuilder<> substitute_builder(context);
  // Save the original builder.
  auto* original_builder = ir_emitter.builder();
  {
    auto builder_overwrite = ir_emitter.WithBuilder(substitute_builder);
    // The IR emitter's builder is now the temporary builder.
    EXPECT_EQ(ir_emitter.builder(), &substitute_builder);
  }
  // After the scope is exited, the IR emitter's builder is restored.
  EXPECT_EQ(ir_emitter.builder(), original_builder);
}

TEST_F(IRBuilderGuardTest, NestedIRBuilderGuards) {
  llvm::LLVMContext context;
  IrEmitter ir_emitter = MakeIrEmitter(context);

  // Temporary builders to replace the IR emitter's builder.
  llvm::IRBuilder<> substitute_builder(context);
  llvm::IRBuilder<> nested_substitute_builder(context);
  auto* original_builder = ir_emitter.builder();
  {
    // The first scope should use `substitute_builder`.
    auto scoped_builder_1 = ir_emitter.WithBuilder(substitute_builder);
    EXPECT_EQ(ir_emitter.builder(), &substitute_builder);
    {
      // The second scope (nested) should use `nested_substitute_builder`.
      auto scoped_builder_2 = ir_emitter.WithBuilder(nested_substitute_builder);
      EXPECT_EQ(ir_emitter.builder(), &nested_substitute_builder);
    }
    // Back to the first scope, which should still use `substitute_builder`.
    EXPECT_EQ(ir_emitter.builder(), &substitute_builder);
  }
  // After all scopes are exited, the IR emitter's builder is restored.
  EXPECT_EQ(ir_emitter.builder(), original_builder);
}

}  // namespace
}  // namespace xla::cpu

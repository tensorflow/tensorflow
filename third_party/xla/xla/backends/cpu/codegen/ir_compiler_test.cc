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

#include "xla/backends/cpu/codegen/ir_compiler.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

namespace {

using ::testing::HasSubstr;
using ::testing::Not;

constexpr absl::string_view kUnoptimizedIr = R"(
  define void @sum_vectors(ptr noalias %a, ptr noalias %b, ptr noalias %c, i64 %n) {
  entry:
    br label %loop.header

  loop.header:
    %i = phi i64 [ 0, %entry ], [ %i.next, %loop.body ]
    %cmp = icmp ult i64 %i, %n
    br i1 %cmp, label %loop.body, label %loop.exit

  loop.body:
    %a.ptr = getelementptr inbounds float, ptr %a, i64 %i
    %b.ptr = getelementptr inbounds float, ptr %b, i64 %i
    %c.ptr = getelementptr inbounds float, ptr %c, i64 %i

    %a.val = load float, ptr %a.ptr, align 4
    %b.val = load float, ptr %b.ptr, align 4

    %sum = fadd float %a.val, %b.val

    store float %sum, ptr %c.ptr, align 4

    %i.next = add nuw i64 %i, 1
    br label %loop.header

  loop.exit:
    ret void
  }
)";

// Parses the LLVM IR into a ThreadSafeModule.
static absl::StatusOr<std::unique_ptr<llvm::Module>> ParseModule(
    llvm::LLVMContext& context, absl::string_view ir, absl::string_view name) {
  llvm::SMDiagnostic diagnostic;
  llvm::MemoryBufferRef ir_buffer(ir, name);

  auto m = llvm::parseAssembly(ir_buffer, diagnostic, context);
  if (m == nullptr) {
    return Internal("Failed to parse LLVM IR: %s",
                    diagnostic.getMessage().str());
  }

  SetModuleMemoryRegionName(*m, "ir_compiler_test");

  return m;
}

TEST(IrCompilerTest, OverrideIrCompilerCompileOptions) {
  auto context = std::make_unique<llvm::LLVMContext>();
  IrCompiler::CompilationHooks compilation_hooks;

  std::unique_ptr<IrCompiler> ir_compiler = IrCompiler::Create(
      llvm::TargetOptions(),
      IrCompiler::Options{/*opt_level=*/llvm::CodeGenOptLevel::Aggressive},
      compilation_hooks);

  std::vector<std::unique_ptr<llvm::Module>> modules;

  auto add_module_with_options =
      [&](absl::string_view ir, absl::string_view name,
          const LlvmKernelOptions& options) -> absl::Status {
    TF_ASSIGN_OR_RETURN(modules.emplace_back(),
                        ParseModule(*context, ir, name));

    auto llvm_module = modules.back().get();
    SetXlaCpuBackendOptions(*llvm_module, options);

    return absl::OkStatus();
  };

  constexpr absl::string_view kModuleName = "test_module";
  {
    LlvmKernelOptions override_options;
    override_options.set_optimize_for_size(false);

    TF_ASSERT_OK(
        add_module_with_options(kUnoptimizedIr, kModuleName, override_options));
  }

  {
    LlvmKernelOptions override_options;
    override_options.set_optimize_for_size(true);
    TF_ASSERT_OK(
        add_module_with_options(kUnoptimizedIr, kModuleName, override_options));
  }

  EXPECT_EQ(modules.size(), 2);

  TF_ASSERT_OK_AND_ASSIGN(auto target_machine,
                          ir_compiler->build_target_machine());

  for (auto& llvm_module : modules) {
    llvm_module->setDataLayout(target_machine->createDataLayout());
    llvm_module->setTargetTriple(target_machine->getTargetTriple());
    cantFail((*ir_compiler)(*llvm_module));
  }

  auto vectorized_module_ir = llvm_ir::DumpToString(modules[0].get());
  auto non_vectorized_module_ir = llvm_ir::DumpToString(modules[1].get());

  // Check for presence/absence of vectorized instructions to ensure
  // vectorization happened/did not happen.
  EXPECT_THAT(vectorized_module_ir, ::testing::HasSubstr("wide.load"));
  EXPECT_THAT(non_vectorized_module_ir,
              ::testing::Not(::testing::HasSubstr("wide.load")))
      << non_vectorized_module_ir;

  EXPECT_THAT(vectorized_module_ir,
              ::testing::ContainsRegex("fadd <[0-9]+ x float>"))
      << vectorized_module_ir;
  EXPECT_THAT(non_vectorized_module_ir,
              ::testing::Not(::testing::ContainsRegex("fadd <[0-9]+ x float>")))
      << non_vectorized_module_ir;
}

TEST(IrCompilerTest, TestAdditionalFeatures) {
#if !defined(__x86_64__)
  GTEST_SKIP()
      << "Test only supported on native x86_64 (InitializeNativeTarget).";
#endif

  llvm::InitializeNativeTarget();

  bool has_avx512;
  auto builder =
      [&has_avx512]() -> absl::StatusOr<std::unique_ptr<llvm::TargetMachine>> {
    absl::string_view cpu_name = "skylake";
    absl::string_view features = has_avx512 ? "+avx512f" : "-avx512f";
    absl::string_view triple = "x86_64-unknown-linux-gnu";
    std::string error;
    const llvm::Target* target =
        llvm::TargetRegistry::lookupTarget(triple, error);
    if (target == nullptr) {
      return absl::InternalError("Failed to lookup target: " + error);
    }

    llvm::TargetOptions target_options;
    return absl::WrapUnique(target->createTargetMachine(
        llvm::Triple(triple), cpu_name, features, target_options,
        /*RM=*/std::nullopt));
  };

  IrCompiler ir_compiler(std::move(builder), IrCompiler::Options(),
                         IrCompiler::CompilationHooks());

  {
    has_avx512 = true;
    TF_ASSERT_OK_AND_ASSIGN(auto target_machine,
                            ir_compiler.build_target_machine());

    absl::string_view features = target_machine->getTargetFeatureString();
    EXPECT_THAT(features, HasSubstr("+prefer-no-scatter"));
    EXPECT_THAT(features, HasSubstr("+prefer-no-gather"));
  }

  {
    has_avx512 = false;
    TF_ASSERT_OK_AND_ASSIGN(auto target_machine,
                            ir_compiler.build_target_machine());

    absl::string_view features = target_machine->getTargetFeatureString();
    EXPECT_THAT(features, Not(HasSubstr("+prefer-no-scatter")));
    EXPECT_THAT(features, Not(HasSubstr("+prefer-no-gather")));
  }
}

}  // namespace

}  // namespace xla::cpu

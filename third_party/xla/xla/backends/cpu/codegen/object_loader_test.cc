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

#include "xla/backends/cpu/codegen/object_loader.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/CoreContainers.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/codegen/execution_engine.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

// Parses the LLVM IR into a ThreadSafeModule.
static absl::StatusOr<llvm::orc::ThreadSafeModule> ParseModule(
    llvm::orc::ThreadSafeContext& context, absl::string_view ir,
    absl::string_view name) {
  llvm::SMDiagnostic diagnostic;
  llvm::MemoryBufferRef ir_buffer(ir, name);

  auto m = llvm::parseAssembly(ir_buffer, diagnostic, *context.getContext());
  if (m == nullptr) {
    return Internal("Failed to parse LLVM IR: %s",
                    diagnostic.getMessage().str());
  }

  return llvm::orc::ThreadSafeModule(std::move(m), context);
}

static absl::StatusOr<std::unique_ptr<FunctionLibrary>> Compile(
    JitCompiler compiler, absl::Span<const FunctionLibrary::Symbol> symbols) {
  return std::move(compiler).Compile(symbols);
};

struct ObjectLoaderTestParams {
  std::string add_in_place_ir;
  ExecutionEngine::DefinitionGenerator definition_generator;
};

class ObjectLoaderTest
    : public ::testing::TestWithParam<ObjectLoaderTestParams> {};

TEST_P(ObjectLoaderTest, Load) {
  const ObjectLoaderTestParams& params = GetParam();
  constexpr size_t kNumDyLibs = 1;
  auto context = std::make_unique<llvm::LLVMContext>();
  llvm::orc::ThreadSafeContext tsc(std::move(context));

  std::vector<std::string> object_files;
  auto object_files_saver =
      [&object_files](const llvm::Module& /*module*/,
                      const llvm::object::ObjectFile& object_file) -> void {
    object_files.emplace_back(object_file.getData().data(),
                              object_file.getData().size());
  };

  JitCompiler::Options options;
  options.num_dylibs = kNumDyLibs;
  options.definition_generator = params.definition_generator;

  IrCompiler::CompilationHooks ir_compiler_hooks;
  ir_compiler_hooks.post_codegen = object_files_saver;

  std::unique_ptr<IrCompiler> ir_compiler = IrCompiler::Create(
      llvm::TargetOptions(), IrCompiler::Options(), ir_compiler_hooks);

  TF_ASSERT_OK_AND_ASSIGN(
      auto compiler,
      JitCompiler::Create(std::move(options), std::move(ir_compiler)));

  auto add_module = [&](absl::string_view ir, absl::string_view name,
                        size_t dylib_index) -> absl::Status {
    TF_ASSIGN_OR_RETURN(llvm::orc::ThreadSafeModule tsm,
                        ParseModule(tsc, ir, name));
    TF_RETURN_IF_ERROR(compiler.AddModule(std::move(tsm), dylib_index));
    return absl::OkStatus();
  };

  TF_ASSERT_OK(add_module(params.add_in_place_ir, "AddInplace", 0));

  using ScalarFn = void(float*);
  std::vector<FunctionLibrary::Symbol> symbols = {
      FunctionLibrary::Sym<ScalarFn>("AddInplace")};

  llvm::DataLayout data_layout = compiler.target_machine()->createDataLayout();
  TF_ASSERT_OK_AND_ASSIGN(auto function_library_compiled,
                          Compile(std::move(compiler), symbols));

  TF_ASSERT_OK_AND_ASSIGN(
      ScalarFn * add_in_place_compiled,
      function_library_compiled->ResolveFunction<ScalarFn>("AddInplace"));

  EXPECT_NE(add_in_place_compiled, nullptr);

  auto object_loader(std::make_unique<ObjectLoader>(
      /*num_dylibs=*/kNumDyLibs, data_layout, params.definition_generator));
  {
    size_t obj_file_index = 0;
    for (auto& obj_file : object_files) {
      llvm::StringRef data(obj_file.data(), obj_file.size());
      TF_ASSERT_OK(object_loader->AddObjFile(
          obj_file, absl::StrCat("loaded_obj_file_", obj_file_index++)));
    }
  }

  TF_ASSERT_OK_AND_ASSIGN(auto loaded_function_library,
                          std::move(*object_loader).Load(symbols));

  TF_ASSERT_OK_AND_ASSIGN(
      ScalarFn * loaded_add_in_place,
      loaded_function_library->ResolveFunction<ScalarFn>("AddInplace"));

  EXPECT_NE(loaded_add_in_place, nullptr);

  constexpr float kInputValue = 1.0f;
  constexpr float kExpectedOutput = kInputValue + kInputValue;

  float compiled_function_input = kInputValue;
  add_in_place_compiled(&compiled_function_input);
  EXPECT_EQ(compiled_function_input, kExpectedOutput);

  float loaded_function_input = 1.0f;
  loaded_add_in_place(&loaded_function_input);
  EXPECT_EQ(loaded_function_input, compiled_function_input);
}

class ExternalDefinitionGenerator : public llvm::orc::DefinitionGenerator {
 public:
  static void AddInplace(float* value) { *value += *value; }

  llvm::Error tryToGenerate(llvm::orc::LookupState&, llvm::orc::LookupKind,
                            llvm::orc::JITDylib& jit_dylib,
                            llvm::orc::JITDylibLookupFlags,
                            const llvm::orc::SymbolLookupSet& names) final {
    llvm::orc::SymbolMap new_defs;
    for (auto& [name, flags] : names) {
      std::string to_print((*name).begin(), (*name).end());
      if ((*name).contains("external_fn")) {
        new_defs[name] = llvm::orc::ExecutorSymbolDef{
            llvm::orc::ExecutorAddr(reinterpret_cast<uint64_t>(&AddInplace)),
            llvm::JITSymbolFlags::None};
      }
    }

    cantFail(jit_dylib.define(llvm::orc::absoluteSymbols(std::move(new_defs))));
    return llvm::Error::success();
  }
};

INSTANTIATE_TEST_SUITE_P(
    ObjectLoaderTestSuite, ObjectLoaderTest,
    ::testing::Values(  // List of test parameters
        ObjectLoaderTestParams{
            R"(
          define void @AddInplace(ptr %arg) {
            %v0 = load float, ptr %arg
            %v1 = fadd float %v0, %v0
            store float %v1, ptr %arg
            ret void
          })",
            nullptr},
        ObjectLoaderTestParams{
            R"(
          declare void @__external_fn(ptr %arg)

          define void @AddInplace(ptr %arg) {
            call void @__external_fn(ptr %arg)
            ret void
          })",
            [](const llvm::DataLayout& data_layout) {
              return std::make_unique<ExternalDefinitionGenerator>();
            }}));

}  // namespace
}  // namespace xla::cpu

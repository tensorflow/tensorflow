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

// Tests `cpu_aot_loader` helpers without linking ORC.

#include "xla/service/cpu/cpu_aot_loader.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/config.h"  // IWYU pragma: keep
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/cpu/function_library_loader.h"

#ifdef ABSL_HAVE_MEMORY_SANITIZER
#include <sanitizer/msan_interface.h>
#endif

namespace xla::cpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

// Gives `proto` this process's msan instrumentation, so that it loads.
void SetProcessMsan(CompilationResultProto& proto) {
#ifdef ABSL_HAVE_MEMORY_SANITIZER
  proto.set_msan_enabled(true);
  proto.set_msan_track_origins(__msan_get_track_origins());
#endif
}

TEST(CpuAotLoaderTest, GetCompiledSymbolsFromProto) {
  SymbolProto kernel;
  kernel.set_name("k");
  kernel.set_function_type_id(SymbolProto::KERNEL);
  SymbolProto comparator;
  comparator.set_name("c");
  comparator.set_function_type_id(SymbolProto::COMPARATOR);

  std::vector<SymbolProto> protos = {kernel, comparator};
  ASSERT_OK_AND_ASSIGN(std::vector<FunctionLibrary::Symbol> symbols,
                       GetCompiledSymbolsFromProto(protos));
  ASSERT_EQ(symbols.size(), 2);
  EXPECT_EQ(symbols[0].name, "k");
  EXPECT_EQ(symbols[0].type_id,
            FunctionLibrary::Sym<FunctionLibrary::Kernel>("k").type_id);
  EXPECT_EQ(symbols[1].name, "c");
  EXPECT_EQ(symbols[1].type_id,
            FunctionLibrary::Sym<FunctionLibrary::Comparator>("c").type_id);
}

TEST(CpuAotLoaderTest, GetCompiledSymbolsFromRepeatedProtoField) {
  CompilationResultProto aot_result_proto;
  SymbolProto* kernel = aot_result_proto.add_compiled_symbols();
  kernel->set_name("k");
  kernel->set_function_type_id(SymbolProto::KERNEL);
  SymbolProto* comparator = aot_result_proto.add_compiled_symbols();
  comparator->set_name("c");
  comparator->set_function_type_id(SymbolProto::COMPARATOR);

  ASSERT_OK_AND_ASSIGN(
      std::vector<FunctionLibrary::Symbol> symbols,
      GetCompiledSymbolsFromProto(aot_result_proto.compiled_symbols()));
  ASSERT_EQ(symbols.size(), 2);
  EXPECT_EQ(symbols[0].name, "k");
  EXPECT_EQ(symbols[0].type_id,
            FunctionLibrary::Sym<FunctionLibrary::Kernel>("k").type_id);
  EXPECT_EQ(symbols[1].name, "c");
  EXPECT_EQ(symbols[1].type_id,
            FunctionLibrary::Sym<FunctionLibrary::Comparator>("c").type_id);
}

TEST(CpuAotLoaderTest, GetCompiledSymbolsFromProtoRejectsUnknownType) {
  SymbolProto unknown;
  unknown.set_name("u");
  unknown.set_function_type_id(SymbolProto::UNKNOWN);

  std::vector<SymbolProto> protos = {unknown};
  EXPECT_THAT(GetCompiledSymbolsFromProto(protos),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Unknown function type id")));
}

TEST(CpuAotLoaderTest, LoadAotCompilationResultRejectsNullFunctionLibrary) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnUnverifiedModule(
          "HloModule test\nENTRY main {\n  ROOT c = f32[] constant(0)\n}"));

  CompilationResultProto aot_result_proto;
  *aot_result_proto.mutable_hlo_module()->mutable_hlo_module() =
      module->ToProto();
  *aot_result_proto.mutable_hlo_module()->mutable_config() =
      module->config().ToProto();
  *aot_result_proto.mutable_target_machine_options() =
      TargetMachineOptions().ToProto();
  SetProcessMsan(aot_result_proto);

  EXPECT_THAT(
      CpuAotLoader::LoadAotCompilationResult(
          aot_result_proto,
          [](const CompilationResultProto&)
              -> absl::StatusOr<std::unique_ptr<FunctionLibrary>> {
            return nullptr;
          }),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("null FunctionLibrary")));
}

TEST(CpuAotLoaderTest, LoadAotCompilationResultRejectsMsanMismatch) {
  CompilationResultProto matching;
  SetProcessMsan(matching);
  CompilationResultProto other_msan = matching;
  other_msan.set_msan_enabled(!matching.msan_enabled());
  CompilationResultProto other_track_origins = matching;
  other_track_origins.set_msan_track_origins(matching.msan_track_origins() + 1);

  for (const CompilationResultProto& proto :
       {other_msan, other_track_origins}) {
    EXPECT_THAT(
        CpuAotLoader::LoadAotCompilationResult(
            proto,
            [](const CompilationResultProto&)
                -> absl::StatusOr<std::unique_ptr<FunctionLibrary>> {
              return nullptr;
            }),
        StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("msan")));
  }
}

}  // namespace
}  // namespace xla::cpu

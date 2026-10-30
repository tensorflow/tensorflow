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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringMap.h"  // IWYU pragma: keep
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/cpu/test_target_triple_helper.h"
#include "xla/service/executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/restricted/hlo_runner_legacy.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/restricted/hlo_test_base_legacy.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "tsl/platform/casts.h"

namespace xla {
namespace cpu {
namespace {

using CpuAotCompilerTest = HloTestBaseLegacy;

// Separate from cpu_compiler_test.cc because we have to use HloTestBaseLegacy
// to get the HloRunnerLegacy since HloRunnerAgnosticTestBase doesn't support
// AOT compilation.
// TODO(basioli): Unify this test with the gpu_compiler one.
TEST_F(CpuAotCompilerTest, AheadOfTimeCompilation) {
  constexpr absl::string_view kHloAdd1 = R"(
add1 {
  p = s32[] parameter(0)
  c = s32[] constant(1)
  ROOT a = s32[] add(p, c)
}

ENTRY e {
  p = s32[] parameter(0)
  ROOT r = s32[] fusion(p), kind=kLoop, calls=add1
})";

  constexpr absl::string_view kHloAdd2 = R"(
add2 {
  p = s32[] parameter(0)
  c = s32[] constant(2)
  ROOT a = s32[] add(p, c)
}

ENTRY e {
  p = s32[] parameter(0)
  ROOT r = s32[] fusion(p), kind=kLoop, calls=add2
})";

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("host"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                       platform->ExecutorForDevice(0));

  Compiler* compiler = backend().compiler();
  ASSERT_NE(compiler, nullptr);

  auto aot_options = std::make_unique<CpuAotCompilationOptions>(
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::BigPic);
  aot_options->set_executor(stream_exec);

  auto test = [this, &compiler, aot_options = std::move(aot_options)](
                  absl::string_view test_name, absl::string_view hlo, int input,
                  int expected_result) {
    SCOPED_TRACE(test_name);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(
        std::vector<std::unique_ptr<CompiledModule>> aot_results,
        compiler->CompileAheadOfTime(std::move(module), *aot_options));

    ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                         aot_results[0]->SerializeAsString());
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<CompiledModule> aot_result,
        compiler->LoadAotCompilationResult(serialized_aot_result));

    ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                         std::move(*aot_result).LoadExecutable());
    std::unique_ptr<OpaqueExecutable> wrapped_executable =
        test_runner_as_hlo_runner().WrapExecutable(std::move(executable));

    const xla::Literal literal_input =
        xla::LiteralUtil::CreateR0<int32_t>(input);
    const xla::Literal literal_expected_result =
        xla::LiteralUtil::CreateR0<int32_t>(expected_result);

    ASSERT_OK_AND_ASSIGN(Literal result,
                         test_runner_as_hlo_runner().ExecuteWithExecutable(
                             wrapped_executable.get(), {&literal_input}));

    EXPECT_TRUE(LiteralTestUtil::Equal(result, literal_expected_result));
  };

  test("Test kHloAdd1", kHloAdd1, 1, 2);
  test("Test kHloAdd2", kHloAdd2, 1, 3);
}

TEST_F(CpuAotCompilerTest,
       ExportedExecutableTargetMachineOptionsMatchTheCompilationMachine) {
  absl::string_view module_string = R"(
HloModule module

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  a_plus_b = f32[] add(a, b)
  ROOT result = f32[] add(a_plus_b, b)
})";

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("host"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                       platform->ExecutorForDevice(0));

  Compiler* compiler = backend().compiler();
  ASSERT_NE(compiler, nullptr);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(module_string));

  xla::Compiler::CompileOptions compile_options;
  ASSERT_OK_AND_ASSIGN(
      hlo_module, compiler->RunHloPasses(std::move(hlo_module), stream_exec,
                                         compile_options));

  ASSERT_OK_AND_ASSIGN(auto executable,
                       compiler->RunBackend(std::move(hlo_module), stream_exec,
                                            compile_options));

  ASSERT_OK_AND_ASSIGN(auto aot_result, compiler->Export(executable.get()));

  CpuAotCompilationResult* cpu_aot_result =
      absl::down_cast<CpuAotCompilationResult*>(aot_result.get());
  ASSERT_NE(cpu_aot_result, nullptr);

  EXPECT_EQ(
      llvm::Triple(cpu_aot_result->proto().target_machine_options().triple())
          .getArchName(),
      llvm::Triple(kTargetTripleForHost).getArchName());

  auto host_machine_features = llvm::sys::getHostCPUFeatures();

  for (const auto& [feature, supported] : host_machine_features) {
    if (supported && absl::StartsWith(feature, "avx512")) {
      host_machine_features["prefer-no-scatter"] = true;
      host_machine_features["prefer-no-gather"] = true;
    }
  }

  for (const auto& feature : absl::StrSplit(
           cpu_aot_result->proto().target_machine_options().features(), ',')) {
    if (feature[0] == '+') {
      EXPECT_TRUE(host_machine_features[feature.substr(1)]);
    } else {
      EXPECT_EQ(feature[0], '-');
      // On some architectures, we disable features - like "sve" on AArch64
      // depending on the max isa. Once a cpp encapsulation of target options
      // lands we can remove this hack.
      auto xla_cpu_max_isa =
          CpuFeatureFromString(GetDebugOptionsFromFlags().xla_cpu_max_isa());
      EXPECT_FALSE(
          host_machine_features[feature.substr(1)] &&
          (xla_cpu_max_isa &&
           ShouldEnableCpuFeature(feature.substr(1), xla_cpu_max_isa.value())));
    }
  }
}

TEST_F(CpuAotCompilerTest,
       ExportedExecutableTargetMachineOptionsMatchTheOptionsSetInTargetConfig) {
  absl::string_view module_string = R"(
HloModule module

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  a_plus_b = f32[] add(a, b)
  ROOT result = f32[] add(a_plus_b, b)
})";

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("host"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                       platform->ExecutorForDevice(0));

  Compiler* compiler = backend().compiler();
  ASSERT_NE(compiler, nullptr);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(module_string));

  xla::Compiler::CompileOptions compile_options;
  TargetMachineOptions target_machine_options(
      kTargetTripleForHost, kTargetCpuForHost, "+foo-feature,-bar-feature");
  compile_options.cpu_target_config.emplace(target_machine_options);

  ASSERT_OK_AND_ASSIGN(
      hlo_module, compiler->RunHloPasses(std::move(hlo_module), stream_exec,
                                         compile_options));

  ASSERT_OK_AND_ASSIGN(auto executable,
                       compiler->RunBackend(std::move(hlo_module), stream_exec,
                                            compile_options));

  ASSERT_OK_AND_ASSIGN(auto aot_result, compiler->Export(executable.get()));

  CpuAotCompilationResult* cpu_aot_result =
      absl::down_cast<CpuAotCompilationResult*>(aot_result.get());
  ASSERT_NE(cpu_aot_result, nullptr);

  EXPECT_EQ(
      llvm::Triple(cpu_aot_result->proto().target_machine_options().triple())
          .getArchName(),
      llvm::Triple(kTargetTripleForHost).getArchName());

  std::vector<absl::string_view> aot_result_features = absl::StrSplit(
      cpu_aot_result->proto().target_machine_options().features(), ',');

  EXPECT_EQ(aot_result_features.size(), 2);
  EXPECT_EQ(aot_result_features[0], "+foo-feature");
  EXPECT_EQ(aot_result_features[1], "-bar-feature");
}

TEST_F(CpuAotCompilerTest, ExportedExecutableDataLayoutIsNotEmpty) {
  absl::string_view module_string = R"(
HloModule module

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT result = f32[] add(a, b)
})";

  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("host"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                       platform->ExecutorForDevice(0));

  Compiler* compiler = backend().compiler();
  ASSERT_NE(compiler, nullptr);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(module_string));

  xla::Compiler::CompileOptions compile_options;
  ASSERT_OK_AND_ASSIGN(
      hlo_module, compiler->RunHloPasses(std::move(hlo_module), stream_exec,
                                         compile_options));

  ASSERT_OK_AND_ASSIGN(auto executable,
                       compiler->RunBackend(std::move(hlo_module), stream_exec,
                                            compile_options));

  ASSERT_OK_AND_ASSIGN(auto aot_result, compiler->Export(executable.get()));

  CpuAotCompilationResult* cpu_aot_result =
      absl::down_cast<CpuAotCompilationResult*>(aot_result.get());
  ASSERT_NE(cpu_aot_result, nullptr);

  EXPECT_FALSE(cpu_aot_result->proto().data_layout().empty());
}

class CpuAotLoadExecutableTest : public HloTestBaseLegacy {
 protected:
  // Contains a constant so that an embedded HloProto has a literal to strip.
  static constexpr absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  p = f32[4] parameter(0)
  c = f32[4] constant({1, 2, 3, 4})
  ROOT add = f32[4] add(p, c)
})";

  // Compiles kHlo ahead of time, serializes the result and loads it back.
  absl::StatusOr<std::unique_ptr<CompiledModule>> CompileSerializeAndReload() {
    ABSL_ASSIGN_OR_RETURN(se::Platform * platform,
                     se::PlatformManager::PlatformWithName("host"));
    ABSL_ASSIGN_OR_RETURN(se::StreamExecutor * stream_exec,
                     platform->ExecutorForDevice(0));
    Compiler* compiler = backend().compiler();

    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(kHlo));

    CpuAotCompilationOptions aot_options(
        /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
        /*features=*/"",
        /*entry_point_name=*/"entry",
        /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::BigPic);
    aot_options.set_executor(stream_exec);

    ABSL_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<CompiledModule>> aot_results,
        compiler->CompileAheadOfTime(std::move(module), aot_options));
    if (aot_results.size() != 1) {
      return absl::InternalError("Expected exactly one AOT result.");
    }

    ABSL_ASSIGN_OR_RETURN(std::string serialized,
                     aot_results[0]->SerializeAsString());
    return compiler->LoadAotCompilationResult(serialized);
  }

  absl::StatusOr<std::unique_ptr<Executable>> LoadWithEmbedIr(
      bool embed_ir_in_executable) {
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<CompiledModule> loaded,
                     CompileSerializeAndReload());
    ABSL_ASSIGN_OR_RETURN(se::Platform * platform,
                     se::PlatformManager::PlatformWithName("host"));
    ABSL_ASSIGN_OR_RETURN(se::StreamExecutor * stream_exec,
                     platform->ExecutorForDevice(0));

    DebugOptions debug_options = GetDebugOptionsFromFlags();
    debug_options.set_xla_embed_ir_in_executable(embed_ir_in_executable);
    return std::move(*loaded).LoadExecutable(
        platform->id(), stream_exec->GetDeviceDescription(), debug_options);
  }
};

// The overload without DebugOptions is the one PjRt uses when deserializing.
TEST_F(CpuAotLoadExecutableTest, DoesNotEmbedHloProtoByDefault) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledModule> loaded,
                       CompileSerializeAndReload());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                       std::move(*loaded).LoadExecutable());
  EXPECT_EQ(executable->hlo_proto(), nullptr);
}

TEST_F(CpuAotLoadExecutableTest, DoesNotEmbedHloProtoWhenEmbedIrDisabled) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                       LoadWithEmbedIr(/*embed_ir_in_executable=*/false));
  EXPECT_EQ(executable->hlo_proto(), nullptr);
}

TEST_F(CpuAotLoadExecutableTest, EmbedsStrippedHloProtoWhenEmbedIrEnabled) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                       LoadWithEmbedIr(/*embed_ir_in_executable=*/true));

  const HloProto* hlo_proto = executable->hlo_proto();
  ASSERT_NE(hlo_proto, nullptr);
  EXPECT_TRUE(hlo_proto->has_hlo_module());
  EXPECT_TRUE(hlo_proto->has_buffer_assignment());

  int num_literals = 0;
  for (const auto& computation : hlo_proto->hlo_module().computations()) {
    for (const auto& instruction : computation.instructions()) {
      if (!instruction.has_literal()) {
        continue;
      }
      ++num_literals;
      EXPECT_TRUE(instruction.literal().has_shape());
      EXPECT_EQ(instruction.literal().f32s_size(), 0);
    }
  }
  // Guard against the constant being optimized away, which would make the
  // checks above vacuous.
  EXPECT_GT(num_literals, 0);
}

TEST_F(CpuAotLoadExecutableTest, EmbedsHloProtoWhenModuleCompiledWithEmbedIr) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("host"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                       platform->ExecutorForDevice(0));
  Compiler* compiler = backend().compiler();

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_embed_ir_in_executable(true);

  CpuAotCompilationOptions aot_options(
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::BigPic);
  aot_options.set_executor(stream_exec);

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<CompiledModule>> aot_results,
      compiler->CompileAheadOfTime(std::move(module), aot_options));
  ASSERT_EQ(aot_results.size(), 1);
  ASSERT_OK_AND_ASSIGN(std::string serialized,
                       aot_results[0]->SerializeAsString());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledModule> loaded,
                       compiler->LoadAotCompilationResult(serialized));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                       std::move(*loaded).LoadExecutable());
  EXPECT_NE(executable->hlo_proto(), nullptr);
}

TEST_F(CpuAotLoadExecutableTest, EmbedsFullHloProtoWhenDumpingSnapshots) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("host"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                       platform->ExecutorForDevice(0));
  Compiler* compiler = backend().compiler();

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  module->mutable_config().mutable_debug_options().set_xla_dump_hlo_snapshots(
      true);

  CpuAotCompilationOptions aot_options(
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::BigPic);
  aot_options.set_executor(stream_exec);

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<CompiledModule>> aot_results,
      compiler->CompileAheadOfTime(std::move(module), aot_options));
  ASSERT_EQ(aot_results.size(), 1);
  ASSERT_OK_AND_ASSIGN(std::string serialized,
                       aot_results[0]->SerializeAsString());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledModule> loaded,
                       compiler->LoadAotCompilationResult(serialized));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> executable,
                       std::move(*loaded).LoadExecutable());

  ASSERT_TRUE(executable->dumping_snapshot());
  const HloProto* hlo_proto = executable->hlo_proto();
  ASSERT_NE(hlo_proto, nullptr);

  // Literal payloads must be kept so the snapshot can be replayed.
  bool found_payload = false;
  for (const auto& computation : hlo_proto->hlo_module().computations()) {
    for (const auto& instruction : computation.instructions()) {
      if (instruction.has_literal() && instruction.literal().f32s_size() > 0) {
        found_payload = true;
      }
    }
  }
  EXPECT_TRUE(found_payload);
}

}  // namespace
}  // namespace cpu
}  // namespace xla

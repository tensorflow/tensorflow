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
#include <string>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/cuda/nvjitlink_support.h"
#include "xla/stream_executor/cuda/ptx_compilation_method.h"
#include "xla/stream_executor/cuda/ptx_compiler_support.h"
#include "xla/stream_executor/cuda/ptx_linking_method.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {

constexpr absl::string_view kSimpleHlo = R"(
HloModule simple

ENTRY main {
  p = f32[10]{0} parameter(0)
  ROOT neg = f32[10]{0} negate(p)
}
)";
constexpr absl::string_view kParallelCompilationHlo = R"(
HloModule parallel_compilation

ENTRY main {
  p1 = f32[10,20,30] parameter(0)
  p2 = f32[40,30,10] parameter(1)
  // With the new MLIR emitters, each indexing change leads to a new function.
  // So adding 2 transposes and a concatenate will results in 3 LLVM IR
  // functions that can be compiled in parallel.
  t1 = f32[20,10,30] transpose(p1), dimensions={1,0,2}
  t2 = f32[40,10,30] transpose(p2), dimensions={0,2,1}
  ROOT c = f32[60,10,30] concatenate(t1, t2), dimensions={0}
}
)";

constexpr absl::string_view kSM90AHlo = R"(
gemm_fusion_dot {
  %p0 = f16[64,1024]{1,0} parameter(0)
  %p1 = f16[1024,32,32]{2,1,0} parameter(1)
  %bitcast.74246 = f16[1024,1024]{0,1} bitcast(f16[1024,32,32]{2,1,0} %p1)
  ROOT %dot.1302 = f16[64,1024]{1,0} dot(f16[64,1024]{1,0} %p0, f16[1024,1024]{0,1} %bitcast.74246), lhs_contracting_dims={1}, rhs_contracting_dims={0}, frontend_attributes={grad_x="false",grad_y="false"}
}

ENTRY e {
  p0 = f16[64,1024]{1,0} parameter(0)
  p1 = f16[1024,32,32]{2,1,0} parameter(1)
  // This Triton fusion generates a wgmma instruction which allows us to test
  // whether we properly enable SM 9.0A in all compilation and linking paths.
  ROOT triton_gemm_fusion_dot = f16[64,1024]{1,0} fusion(p0, p1), kind=kCustom,
    calls=gemm_fusion_dot,
    backend_config={"fusion_backend_config": {kind: "__triton_gemm",
      triton_gemm_config:
        {"block_m":64,"block_n":32,"block_k":32,
         "split_k":1,"num_stages":1,"num_warps":4,
         "num_ctas":1}}}
})";

constexpr absl::string_view kResultsInNoPtxHlo = R"(
  ENTRY e {
    a = f32[5,5] parameter(0)
    ROOT _ = f32[5,5] custom-call(a, a), custom_call_target="__cublas$gemm",
      backend_config="{ \"gemm_backend_config\": {\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}}"
  })";

absl::string_view GetHlo(absl::string_view name) {
  static const absl::flat_hash_map<absl::string_view, absl::string_view>* const
      kHloMap = new absl::flat_hash_map<absl::string_view, absl::string_view>(
          {{"simple", kSimpleHlo},
           {"parallel_compilation", kParallelCompilationHlo},
           {"requires_sm90a", kSM90AHlo},
           {"results_in_no_ptx", kResultsInNoPtxHlo}});
  return kHloMap->at(name);
}

void DumpArtifactIfEnabled(absl::string_view name,
                           absl::Span<const uint8_t> data) {
  if (std::string output_dir;
      tsl::io::GetTestUndeclaredOutputsDir(&output_dir)) {
    (void)tsl::WriteStringToFile(
        tsl::Env::Default(), tsl::io::JoinPath(output_dir, name),
        absl::string_view(reinterpret_cast<const char*>(data.data()),
                          data.size()));
  }
}

using stream_executor::PtxCompilationMethod;
using stream_executor::PtxLinkingMethod;

std::string GenerateParametrizedTestname(
    absl::string_view name, PtxCompilationMethod compilation_method,
    PtxLinkingMethod linking_method) {
  return absl::StrFormat("%v_CompilationMethod_%v_LinkingMethod_%v", name,
                         compilation_method, linking_method);
}

class NVPTXCompilationTests
    : public HloTestBase,
      public ::testing::WithParamInterface<std::tuple<
          absl::string_view, PtxCompilationMethod, PtxLinkingMethod>> {
 public:
  void SkipTestIfUnsupported(absl::string_view name,
                             PtxCompilationMethod compilation_method,
                             PtxLinkingMethod linking_method) {
    using CudaComputeCapability = stream_executor::CudaComputeCapability;
    if (!::testing::Value(backend()
                              .default_stream_executor()
                              ->GetDeviceDescription()
                              .gpu_compute_capability(),
                          ::testing::VariantWith<CudaComputeCapability>(
                              CudaComputeCapability{9, 0})) &&
        name == "requires_sm90a") {
      GTEST_SKIP() << "This test requires SM 9.0a";
    }

    if (!stream_executor::IsLibNvPtxCompilerSupported() &&
        compilation_method == PtxCompilationMethod::kNvPtxCompiler) {
      // Compiled without libnvptxcompiler support
      GTEST_SKIP() << "libnvptxcompiler is not supported in this build.";
    }

    if (!stream_executor::IsLibNvJitLinkSupported() &&
        (compilation_method == PtxCompilationMethod::kNvJitLink ||
         linking_method == PtxLinkingMethod::kNvJitLink)) {
      // Compiled without libnvjitlink support
      GTEST_SKIP() << "libnvjitlink is not supported in this build.";
    }

    if (compilation_method == PtxCompilationMethod::kNvJitLink &&
        linking_method != PtxLinkingMethod::kNvJitLink) {
      // When compilation method is NvJitLink, linking method must be NvJitLink
      // as well.
      GTEST_SKIP() << "Compilation method NvJitLink is only supported if the "
                      "linking method is NvJitLink as well.";
    }

    if (compilation_method == PtxCompilationMethod::kPtxas &&
        linking_method == PtxLinkingMethod::kNvJitLink) {
      // We could support this combination, but it would require some
      // refactoring of the flags.
      GTEST_SKIP() << "Compilation method Ptxas is not supported with linking "
                      "method NvJitLink.";
    }
  }

  void SetDebugOptionsFromPtxSettings(DebugOptions* debug_options,
                                      PtxCompilationMethod compilation_method,
                                      PtxLinkingMethod linking_method) {
    debug_options->set_xla_gpu_enable_libnvptxcompiler(
        compilation_method == PtxCompilationMethod::kNvPtxCompiler);

    debug_options->set_xla_gpu_libnvjitlink_mode(
        (compilation_method == PtxCompilationMethod::kNvJitLink ||
         linking_method == PtxLinkingMethod::kNvJitLink)
            ? DebugOptions::LIB_NV_JIT_LINK_MODE_ENABLED
            : DebugOptions::LIB_NV_JIT_LINK_MODE_DISABLED);

    debug_options->set_xla_gpu_enable_llvm_module_compilation_parallelism(
        linking_method != PtxLinkingMethod::kNone);
    debug_options->set_xla_gpu_force_compilation_parallelism(12);

    if (linking_method == PtxLinkingMethod::kDriver) {
      debug_options->set_xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found(
          true);
      debug_options->set_xla_gpu_cuda_data_dir("/does/not/exist");
    }

    tsl::setenv("TF_USE_NVLINK_FOR_PARALLEL_COMPILATION",
                linking_method == PtxLinkingMethod::kNvLink ? "true" : "false",
                1);

    // We need individual functions to test parallel compilation.
    debug_options->set_xla_llvm_force_inline_before_split(false);
  }

  DebugOptions GetDebugOptionsForTest() const override {
    auto debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_autotune_level(0);
    return debug_options;
  }

  void SetUp() override {
    HloTestBase::SetUp();
    absl::string_view name = std::get<0>(GetParam());
    PtxCompilationMethod compilation_method = std::get<1>(GetParam());
    PtxLinkingMethod linking_method = std::get<2>(GetParam());
    SkipTestIfUnsupported(name, compilation_method, linking_method);
  }

  absl::StatusOr<std::unique_ptr<Executable>> CompileExecutable(
      std::unique_ptr<HloModule> module) {
    NVPTXCompiler compiler{module->config().debug_options()};

    return compiler.RunBackend(std::move(module),
                               backend().default_stream_executor(),
                               {/*device_allocator=*/nullptr,
                                /*thread_pool=*/nullptr,
                                /*layout_canonicalization_callback=*/{},
                                /*is_autotuning_compilation=*/false});
  }
};

TEST_P(NVPTXCompilationTests, CompileProgram) {
  absl::string_view name = std::get<0>(GetParam());
  absl::string_view hlo_text = GetHlo(name);
  auto module = ParseAndReturnVerifiedModule(hlo_text).value();

  HloModuleConfig hlo_module_config = module->config();
  DebugOptions debug_options = hlo_module_config.debug_options();
  PtxCompilationMethod compilation_method = std::get<1>(GetParam());
  PtxLinkingMethod linking_method = std::get<2>(GetParam());
  SetDebugOptionsFromPtxSettings(&debug_options, compilation_method,
                                 linking_method);
  hlo_module_config.set_debug_options(debug_options);
  module->set_config(hlo_module_config);

  EXPECT_THAT(CompileExecutable(std::move(module)),
              tsl::testing::IsOkAndHolds(::testing::NotNull()));
}

MATCHER(MatchesSectionNameAndBinarySize, "") {
  return std::get<0>(arg).first == std::get<1>(arg).first &&
         std::get<0>(arg).second.size() == std::get<1>(arg).second.size();
}

TEST_P(NVPTXCompilationTests, CompareBinaryOutput) {
  absl::string_view name = std::get<0>(GetParam());
  absl::string_view hlo_text = GetHlo(name);
  auto compile = [&](PtxCompilationMethod compilation_method,
                     PtxLinkingMethod linking_method) {
    auto module = ParseAndReturnVerifiedModule(hlo_text).value();

    HloModuleConfig hlo_module_config = module->config();
    DebugOptions debug_options = hlo_module_config.debug_options();
    SetDebugOptionsFromPtxSettings(&debug_options, compilation_method,
                                   linking_method);
    hlo_module_config.set_debug_options(debug_options);
    module->set_config(hlo_module_config);

    return CompileExecutable(std::move(module));
  };

  PtxCompilationMethod compilation_method = std::get<1>(GetParam());
  PtxLinkingMethod linking_method = std::get<2>(GetParam());
  absl::StatusOr<std::unique_ptr<Executable>> executable =
      compile(compilation_method, linking_method);

  // Binaries produced in a separate linking step differ from binaries produced
  // with combined compilation/linking. Therefore we only enable linking in the
  // reference build when the build under test also uses a separate linking
  // step.
  const PtxLinkingMethod reference_linking_method =
      (linking_method == PtxLinkingMethod::kNone) ? PtxLinkingMethod::kNone
                                                  : PtxLinkingMethod::kNvLink;

  absl::StatusOr<std::unique_ptr<Executable>> reference =
      compile(PtxCompilationMethod::kPtxas, reference_linking_method);

  ASSERT_THAT(executable, tsl::testing::IsOkAndHolds(::testing::NotNull()));
  ASSERT_THAT(reference, tsl::testing::IsOkAndHolds(::testing::NotNull()));

  absl::Span<const uint8_t> executable_binary =
      static_cast<GpuExecutable*>(executable.value().get())->binary();
  absl::Span<const uint8_t> reference_binary =
      static_cast<GpuExecutable*>(reference.value().get())->binary();

  if (executable_binary == reference_binary) {
    // If the binaries are exactly the same, we can short circuit and don't need
    // to parse them.
    SUCCEED();
    return;
  }

  std::string test_name =
      GenerateParametrizedTestname(name, compilation_method, linking_method);
  DumpArtifactIfEnabled(absl::StrCat(test_name, "_executable.bin"),
                        executable_binary);
  DumpArtifactIfEnabled(absl::StrCat(test_name, "_reference.bin"),
                        reference_binary);

  auto get_text_sections = [&](absl::Span<const uint8_t> binary)
      -> absl::StatusOr<absl::btree_map<std::string, std::string>> {
    auto buffer = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(reinterpret_cast<const char*>(binary.data()),
                        binary.size()),
        /*BufferName=*/"", /*RequiresNullTerminator=*/false);
    auto object_file =
        llvm::object::ObjectFile::createObjectFile(buffer->getMemBufferRef());

    if (!object_file) {
      return absl::InternalError(llvm::toString(object_file.takeError()));
    }

    auto executable_elf_object_file =
        llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(
            object_file.get().get());

    if (!executable_elf_object_file) {
      return absl::InternalError(
          "Generated executable binary is not a 64bit ELF file.");
    }

    absl::btree_map<std::string, std::string> text_sections;

    for (const auto& section : executable_elf_object_file->sections()) {
      if (absl::StartsWith(section.getName().get().str(), ".text")) {
        text_sections[section.getName().get().str()] =
            section.getContents().get().str();
      }
    }

    return text_sections;
  };

  TF_ASSERT_OK_AND_ASSIGN(auto executable_text_sections,
                          get_text_sections(executable_binary));
  TF_ASSERT_OK_AND_ASSIGN(auto reference_text_sections,
                          get_text_sections(reference_binary));

  if (linking_method == reference_linking_method) {
    EXPECT_THAT(executable_text_sections,
                ::testing::Eq(reference_text_sections));
    return;
  }

  // Different linking methods lead to slightly different code (different
  // register assignment, different instruction ordering). Ideally we would
  // disassemble the code and check for equivalence, but for now let's only
  // compare the text section names and their sizes. If it turns out that
  // this doesn't bring the necessary coverage or that it's too unstable
  // we have to revisit that.
  EXPECT_THAT(executable_text_sections,
              ::testing::Pointwise(MatchesSectionNameAndBinarySize(),
                                   reference_text_sections));
}

INSTANTIATE_TEST_SUITE_P(
    NVPTXCompilationTest, NVPTXCompilationTests,
    ::testing::Combine(::testing::Values("simple", "parallel_compilation",
                                         "requires_sm90a", "results_in_no_ptx"),
                       ::testing::Values(PtxCompilationMethod::kNvPtxCompiler,
                                         PtxCompilationMethod::kPtxas,
                                         PtxCompilationMethod::kNvJitLink),
                       ::testing::Values(PtxLinkingMethod::kNone,
                                         PtxLinkingMethod::kNvLink,
                                         PtxLinkingMethod::kDriver,
                                         PtxLinkingMethod::kNvJitLink)),
    [](const ::testing::TestParamInfo<std::tuple<
           absl::string_view, PtxCompilationMethod, PtxLinkingMethod>>& info) {
      return GenerateParametrizedTestname(std::get<0>(info.param),
                                          std::get<1>(info.param),
                                          std::get<2>(info.param));
    });

}  // namespace
}  // namespace xla::gpu

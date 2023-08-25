/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/autotune_results.pb.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/gpu/horizontal_loop_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/metrics.h"
#include "tensorflow/compiler/xla/service/xla_debug_info_manager.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;

using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::TempDir;

using GpuCompilerTest = HloTestBase;

TEST_F(GpuCompilerTest, CompiledProgramsCount) {
  const char* hlo_text = R"(
HloModule test

ENTRY main {
  p = f32[10]{0} parameter(0)
  ROOT neg = f32[10]{0} negate(p)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).value();
  std::unique_ptr<Executable> executable =
      backend()
          .compiler()
          ->RunBackend(std::move(module), backend().default_stream_executor(),
                       {/*device_allocator=*/nullptr,
                        /*thread_pool=*/nullptr,
                        /*layout_canonicalization_callback=*/{},
                        /*is_autotuning_compilation=*/false})
          .value();
  EXPECT_EQ(GetCompiledProgramsCount(), 1);
}

TEST_F(GpuCompilerTest, GenerateDebugInfoForNonAutotuningCompilations) {
  const char* hlo_text = R"(
HloModule test

ENTRY main {
  p = f32[10]{0} parameter(0)
  ROOT neg = f32[10]{0} negate(p)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).value();
  std::unique_ptr<Executable> executable =
      backend()
          .compiler()
          ->RunBackend(std::move(module), backend().default_stream_executor(),
                       {/*device_allocator=*/nullptr,
                        /*thread_pool=*/nullptr,
                        /*layout_canonicalization_callback=*/{},
                        /*is_autotuning_compilation=*/false})
          .value();
  EXPECT_TRUE(XlaDebugInfoManager::Get()->TracksModule(
      executable->module().unique_id()));
}

TEST_F(GpuCompilerTest, DoesNotGenerateDebugInfoForAutotuningCompilations) {
  const char* hlo_text = R"(
HloModule test

ENTRY main {
  p = f32[10]{0} parameter(0)
  ROOT neg = f32[10]{0} negate(p)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).value();
  int module_id = module->unique_id();
  std::unique_ptr<Executable> executable =
      backend()
          .compiler()
          ->RunBackend(std::move(module), backend().default_stream_executor(),
                       {/*device_allocator=*/nullptr,
                        /*thread_pool=*/nullptr,
                        /*layout_canonicalization_callback=*/{},
                        /*is_autotuning_compilation=*/true})
          .value();
  EXPECT_FALSE(XlaDebugInfoManager::Get()->TracksModule(module_id));
}

TEST_F(GpuCompilerTest, CopyInsertionFusion) {
  const char* hlo_text = R"(
HloModule cluster

ENTRY main {
  cst = f32[1]{0} constant({0})
  ROOT tuple_out = (f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}) tuple(cst, cst, cst, cst)
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{0, 0}));

  auto module = ParseAndReturnVerifiedModule(hlo_text).value();
  std::unique_ptr<HloModule> compiled_module =
      backend()
          .compiler()
          ->RunHloPasses(module->Clone(), backend().default_stream_executor(),
                         /*device_allocator=*/nullptr)
          .value();
  VLOG(2) << compiled_module->ToString();

  // Verify that the total number of fusion instructions is 1.
  size_t total_fusion_instrs = 0;
  for (const HloInstruction* instr :
       compiled_module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kFusion) {
      ++total_fusion_instrs;
    }
  }
  EXPECT_EQ(total_fusion_instrs, 1);

  const HloInstruction* entry_root =
      compiled_module->entry_computation()->root_instruction();
  // Check that we add bitcast when needed.
  EXPECT_THAT(entry_root, op::Tuple(op::GetTupleElement(op::Fusion()),
                                    op::GetTupleElement(op::Fusion()),
                                    op::GetTupleElement(op::Fusion()),
                                    op::GetTupleElement(op::Fusion())));
}

class PersistedAutotuningTest : public HloTestBase {
 protected:
  static constexpr absl::string_view kHloText = R"(
HloModule t

ENTRY e {
  p0 = f16[1,16,17,3] parameter(0)
  p1 = s8[16,17,3] parameter(1)
  cp1 = f16[16,17,3] convert(p1)
  ROOT _ = f16[1,16,16] dot(p0, cp1),
    lhs_contracting_dims={2,3}, rhs_contracting_dims={1,2}
})";

  std::string GetUniqueTempFilePath(absl::string_view suffix) {
    std::string filename = TempDir();
    CHECK(tsl::Env::Default()->CreateUniqueFileName(&filename,
                                                    std::string(suffix)));
    return filename;
  }

  std::string ExpectToReadNonEmptyFile(absl::string_view file_path) {
    std::string str;
    tsl::Env* env = tsl::Env::Default();
    TF_EXPECT_OK(tsl::ReadFileToString(env, std::string(file_path), &str));
    EXPECT_THAT(str, Not(IsEmpty()));
    return str;
  }

  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions options = HloTestBase::GetDebugOptionsForTest();
    options.set_xla_gpu_dump_autotune_results_to(
        xla_gpu_dump_autotune_results_to_);
    options.set_xla_gpu_load_autotune_results_from(
        xla_gpu_load_autotune_results_from_);
    return options;
  }

  std::string xla_gpu_dump_autotune_results_to_;
  std::string xla_gpu_load_autotune_results_from_;
};

TEST_F(PersistedAutotuningTest, WriteResultsOnEachCompilation) {
  constexpr absl::string_view kInvalidTextProto = "Invalid!";
  xla_gpu_dump_autotune_results_to_ = GetUniqueTempFilePath(".txt");

  // Check that it writes the results on the first compilation.
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());
  {
    std::string autotune_results_str =
        ExpectToReadNonEmptyFile(xla_gpu_dump_autotune_results_to_);
    AutotuneResults results;
    EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(autotune_results_str,
                                                           &results));
  }

  // Overwrite results with an invalid textproto.
  tsl::Env* env = tsl::Env::Default();
  TF_EXPECT_OK(tsl::WriteStringToFile(env, xla_gpu_dump_autotune_results_to_,
                                      kInvalidTextProto));

  // Check that it writes the results on the second compilation.
  TF_EXPECT_OK(GetOptimizedModule(kHloText).status());
  {
    std::string autotune_results_str =
        ExpectToReadNonEmptyFile(xla_gpu_dump_autotune_results_to_);
    AutotuneResults results;
    EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(autotune_results_str,
                                                           &results));
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla

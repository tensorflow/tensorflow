/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_compiler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/autotune_results.pb.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/lib/monitoring/collected_metrics.h"
#include "xla/tsl/lib/monitoring/collection_registry.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/regexp.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

using ::testing::IsEmpty;
using ::testing::IsSupersetOf;
using ::testing::Matches;
using ::testing::Not;
using ::testing::TempDir;

class GpuCompilerTest : public HloTestBase {
 public:
  absl::Status Schedule(HloModule* module) {
    auto compiler = backend().compiler();
    const se::DeviceDescription& gpu_device_info =
        backend().default_stream_executor()->GetDeviceDescription();
    TF_RETURN_IF_ERROR(ScheduleGpuModule(module, 4, gpu_device_info).status());
    return tensorflow::down_cast<GpuCompiler*>(compiler)
        ->RunPostSchedulingPipelines(module, 4 * 1024 * 1024, gpu_device_info);
  }
};

// TODO(b/399912696): Fix and enable this test.
TEST_F(GpuCompilerTest, DISABLED_CompiledProgramsCount) {
  const char* hlo_text = R"(
HloModule test

ENTRY main {
  p = f32[10]{0} parameter(0)
  ROOT neg = f32[10]{0} negate(p)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).value();
  ResetCompiledProgramsCountForTesting();
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

TEST_F(GpuCompilerTest, RecordsStreamzStackTrace) {
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

  const std::string kGpuCompilerStacktraceMetricName =
      "/xla/service/gpu/compiler_stacktrace_count";
  tsl::monitoring::CollectionRegistry::CollectMetricsOptions options;
  std::unique_ptr<tsl::monitoring::CollectedMetrics> metrics =
      tsl::monitoring::CollectionRegistry::Default()->CollectMetrics(options);

  EXPECT_TRUE(metrics->point_set_map.find(kGpuCompilerStacktraceMetricName) !=
              metrics->point_set_map.end());

  // Since Streamz is recorded every call, we expect at least one point.
  // All other callers may increment the counter as well.
  EXPECT_GT(
      metrics->point_set_map[kGpuCompilerStacktraceMetricName]->points.size(),
      0);
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
  EXPECT_THAT(
      entry_root,
      GmockMatch(m::Tuple(
          m::GetTupleElement(m::Fusion()), m::GetTupleElement(m::Fusion()),
          m::GetTupleElement(m::Fusion()), m::GetTupleElement(m::Fusion()))));
}

TEST_F(GpuCompilerTest, CanRunScheduledModules) {
  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_disable_all_hlo_passes(true);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m, is_scheduled=true

w {
  p = s8[] parameter(0)
  ROOT n = s8[] negate(p)
}

ENTRY e {
  p = s8[] parameter(0)
  ROOT _ = s8[] fusion(p), kind=kLoop, calls=w
})",
                                                       config));
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
}

TEST_F(GpuCompilerTest, NonFusedInstructionsAreWrapped) {
  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p = f32[2,4,4] parameter(0)
  ROOT _ = f32[2,4,4]{2,1,0} transpose(p), dimensions={0,2,1}
})",
                                                       config));

  config.set_debug_options(debug_options);
  std::unique_ptr<Executable> executable =
      backend()
          .compiler()
          ->RunBackend(std::move(module), backend().default_stream_executor(),
                       {/*device_allocator=*/nullptr,
                        /*thread_pool=*/nullptr,
                        /*layout_canonicalization_callback=*/{},
                        /*is_autotuning_compilation=*/false})
          .value();

  HloModule& compiled_module = executable->module();
  const HloInstruction* entry_root =
      compiled_module.entry_computation()->root_instruction();
  EXPECT_THAT(entry_root, GmockMatch(m::Fusion()));
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

  DebugOptions GetDebugOptionsForTest() const override {
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

int64_t CountCopies(const HloComputation& computation) {
  int64_t count = 0;
  for (const auto& instruction : computation.instructions()) {
    if (instruction->opcode() == HloOpcode::kCopy) {
      count++;
    }
  }
  return count;
}

int64_t CountCopies(const HloModule& module) {
  int64_t count = 0;
  for (const auto& computation : module.computations()) {
    count += CountCopies(*computation);
  }
  return count;
}

TEST_F(GpuCompilerTest, AnnotatesPipelinedInstructions) {
  // Simple IR with AllReduce subjectible to pipelining.
  absl::string_view kHloString = R"(
     HloModule module
      add {
        lhs = bf16[] parameter(0)
        rhs = bf16[] parameter(1)
        ROOT add = bf16[] add(lhs, rhs)
      }

      while_cond {
        param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
        gte = s32[] get-tuple-element(param), index=0
        constant.1 = s32[] constant(3)
        ROOT cmp = pred[] compare(gte, constant.1), direction=LT
      }

      while_body {
        param = (s32[], bf16[3,8,128], bf16[3,8,128]) parameter(0)
        current-loop-index = s32[] get-tuple-element(param), index=0
        output-buffer = bf16[3,8,128] get-tuple-element(param), index=1
        input-buffer = bf16[3,8,128] get-tuple-element(param), index=2
        constant.1 = s32[] constant(1)
        next-loop-index = s32[] add(current-loop-index, constant.1)
        constant.0 = s32[] constant(0)
        sliced-input-buffer = bf16[1,8,128] dynamic-slice(input-buffer,
          current-loop-index, constant.0, constant.0),
            dynamic_slice_sizes={1,8,128}
        all-reduce = bf16[1,8,128] all-reduce(sliced-input-buffer),
          replica_groups={}, to_apply=add, channel_id=1
        dynamic-update-slice = bf16[3,8,128] dynamic-update-slice(output-buffer,
          all-reduce, current-loop-index, constant.0, constant.0)
        ROOT tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(
        next-loop-index, dynamic-update-slice, input-buffer)
      }

      ENTRY entry {
        c0 = s32[] constant(0)
        p0 = bf16[3,8,128] parameter(0)
        tuple = (s32[], bf16[3,8,128], bf16[3,8,128]) tuple(c0, p0, p0)
        while = (s32[], bf16[3,8,128], bf16[3,8,128]) while(tuple),
          condition=while_cond, body=while_body
        ROOT gte1 = bf16[3,8,128] get-tuple-element(while), index=1
      }
  )";

  HloModuleConfig config = GetModuleConfigForTest();
  auto& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_enable_pipelined_all_reduce(true);
  debug_options.set_xla_gpu_all_reduce_combine_threshold_bytes(0);
  TF_ASSERT_OK_AND_ASSIGN(auto parsed,
                          ParseAndReturnVerifiedModule(kHloString, config));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(std::move(parsed)));

  absl::string_view kExpected = R"(
    CHECK: all-reduce-start{{.*}}"is_pipelined":true
  )";
  HloPrintOptions options;
  options.set_print_operand_shape(false);
  options.set_print_result_shape(false);
  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                          RunFileCheck(module->ToString(options), kExpected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(GpuCompilerTest, RemovesUnnecessaryCopyAfterScheduling) {
  const absl::string_view hlo_string = R"(
HloModule all_gather_overlapping
condition {
  input_tuple = (f32[1,128], f32[2,128], pred[]) parameter(0)
  ROOT cond = pred[] get-tuple-element(input_tuple), index=2
}

body {
  input_tuple = (f32[1,128], f32[2,128], pred[]) parameter(0)
  param_0 = f32[1,128] get-tuple-element(input_tuple), index=0
  param_1 = f32[2,128] get-tuple-element(input_tuple), index=1
  cond = pred[] get-tuple-element(input_tuple), index=2

  c0 = f32[] constant(0)
  splat_c0 = f32[1,128] broadcast(c0), dimensions={}
  add = f32[1,128] add(splat_c0, param_0)

  // Start all-gather communication
  all-gather-start = (f32[1,128], f32[2,128]) all-gather-start(add), channel_id=1337, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true

  // Intertwined with the all-gather communication, an operation happens which
  // depends on param_1, but crucially has a different output shape (which
  // excludes reusing param_1's buffer for its output).
  c1_s32 = s32[] constant(1)
  c0_s32 = s32[] constant(0)
  dynamic-slice = f32[1,128] dynamic-slice(param_1, c1_s32, c0_s32), dynamic_slice_sizes={1,128}

  // The all-gather communication finishes
  all-gather-done = f32[2,128] all-gather-done(all-gather-start)

  ROOT output_tuple = (f32[1,128], f32[2,128], pred[]) tuple(dynamic-slice, all-gather-done, cond)
}

ENTRY main {
  param_0 = f32[1,128] parameter(0)
  param_1 = f32[2,128] parameter(1)
  param_2 = pred[] parameter(2)
  tuple = (f32[1,128], f32[2,128], pred[]) tuple(param_0, param_1, param_2)
  ROOT while = (f32[1,128], f32[2,128], pred[]) while(tuple), condition=condition, body=body
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(hlo_string));

  EXPECT_EQ(CountCopies(*module), 7);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* while_op = root->operand(0)->operand(0);
  EXPECT_EQ(while_op->while_body()->root_instruction()->operand(1)->opcode(),
            HloOpcode::kCopy);

  TF_ASSERT_OK(Schedule(module.get()));
  EXPECT_EQ(CountCopies(*module), 4);
  module->entry_computation()->root_instruction();
  while_op = root->operand(0)->operand(0);
  // Make sure that the copy of AllGatherDone has been removed.
  EXPECT_EQ(while_op->while_body()->root_instruction()->operand(1)->opcode(),
            HloOpcode::kAllGatherDone);
}

class GpuCompilerTestWithAutotuneDb : public GpuCompilerTest {
 public:
  static void SetUpTestSuite() {
    std::string path =
        tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service", "gpu",
                          "gpu_compiler_test_autotune_db.textproto");
    TF_EXPECT_OK(AutotunerUtil::LoadAutotuneResultsFromFile(path));
  }

  static void TearDownTestSuite() { AutotunerUtil::ClearAutotuneResults(); }
};

TEST_F(GpuCompilerTestWithAutotuneDb,
       GemmFusionIsNoOpWhenGemmFusionAutotunerFallsBackToCublas) {
  auto cc = backend()
                .default_stream_executor()
                ->GetDeviceDescription()
                .cuda_compute_capability();
  if (!cc.IsAtLeastAmpere()) {
    GTEST_SKIP() << "Autotuning results have only been generated for Ampere "
                 << "and later GPUs";
  }
  const absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  param_0 = bf16[3,32,1024,4,1024]{4,3,2,1,0} parameter(0)
  param_1 = bf16[4,3,32,1024]{3,2,1,0} parameter(1)
  param_2 = s32[] parameter(2)
  constant_0 = s32[] constant(0)
  dynamic-slice_0 = bf16[1,3,32,1024]{3,2,1,0} dynamic-slice(param_1, param_2, constant_0, constant_0, constant_0), dynamic_slice_sizes={1,3,32,1024}
  reshape_0 = bf16[3,32,1024]{2,1,0} reshape(dynamic-slice_0)
  broadcast_0 = bf16[3,32,1024,4,1024]{2,1,4,3,0} broadcast(reshape_0), dimensions={0,1,2}
  add_0 = bf16[3,32,1024,4,1024]{4,3,2,1,0} add(param_0, broadcast_0)
  transpose_0 = bf16[3,4,1024,32,1024]{2,1,4,3,0} transpose(add_0), dimensions={0,3,4,1,2}
  slice_0 = bf16[1,4,1024,32,1024]{4,3,2,1,0} slice(transpose_0), slice={[0:1], [0:4], [0:1024], [0:32], [0:1024]}
  reshape_1 = bf16[4,1024,32,1024]{3,2,1,0} reshape(slice_0)
  copy_0 = bf16[4,1024,32,1024]{3,2,1,0} copy(reshape_1)
  constant_1 = bf16[] constant(0.08838)
  broadcast_1 = bf16[4,1024,32,1024]{3,2,1,0} broadcast(constant_1), dimensions={}
  multiply_0 = bf16[4,1024,32,1024]{3,2,1,0} multiply(copy_0, broadcast_1)
  slice_1 = bf16[1,4,1024,32,1024]{4,3,2,1,0} slice(transpose_0), slice={[1:2], [0:4], [0:1024], [0:32], [0:1024]}
  reshape_2 = bf16[4,1024,32,1024]{3,2,1,0} reshape(slice_1)
  copy_1 = bf16[4,1024,32,1024]{3,2,1,0} copy(reshape_2)
  ROOT dot_0 = bf16[4,32,1024,1024]{3,2,1,0} dot(multiply_0, copy_1), lhs_batch_dims={0,2}, lhs_contracting_dims={3}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}
}
)";

  HloModuleConfig config;
  DebugOptions triton_enabled_debug_options = GetDebugOptionsForTest();
  triton_enabled_debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  triton_enabled_debug_options
      .set_xla_gpu_require_complete_aot_autotune_results(true);
  config.set_debug_options(triton_enabled_debug_options);
  config.set_replica_count(1);
  config.set_num_partitions(1);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> triton_enabled_module,
                          GetOptimizedModule(std::move(module)));
  DebugOptions triton_disabled_debug_options = GetDebugOptionsForTest();
  triton_disabled_debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  triton_disabled_debug_options.set_xla_gpu_enable_triton_gemm(false);
  config.set_debug_options(triton_disabled_debug_options);
  TF_ASSERT_OK_AND_ASSIGN(module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> triton_disabled_module,
                          GetOptimizedModule(std::move(module)));
  // Make sure autotuner falls back to cuBLAS when enabling triton gemm
  const HloInstruction* root =
      triton_enabled_module->entry_computation()->root_instruction();
  const HloInstruction* custom_op = root->operand(0)->operand(0);
  EXPECT_TRUE(custom_op->IsCustomCall("__cublas$gemm"));
  // Make sure that the module has the same number of computations with/without
  // enabling triton gemm
  EXPECT_EQ(triton_enabled_module->computation_count(),
            triton_disabled_module->computation_count());
}

TEST_F(GpuCompilerTestWithAutotuneDb,
       CublasF8NumericallySameWithTritonFallbackAndWithoutTriton) {
  auto cc = backend()
                .default_stream_executor()
                ->GetDeviceDescription()
                .cuda_compute_capability();
  if (!cc.IsAtLeastHopper()) {
    GTEST_SKIP()
        << "Autotuning results have only been generated for Hopper GPUs";
  }
  const absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  p0 = f8e4m3fn[12288,4096]{0,1} parameter(0)
  p1 = f8e4m3fn[4096,16384]{0,1} parameter(1)
  dot = bf16[12288,16384]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  bitcast = bf16[] constant(0.956)
  broadcast = bf16[12288,16384]{1,0} broadcast(bitcast), dimensions={}
  ROOT multiply = bf16[12288,16384]{1,0} multiply(dot, broadcast)
  })";

  HloModuleConfig config;
  DebugOptions triton_enabled_debug_options = GetDebugOptionsForTest();
  triton_enabled_debug_options
      .set_xla_gpu_require_complete_aot_autotune_results(true);
  config.set_debug_options(triton_enabled_debug_options);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> triton_enabled_module,
                          GetOptimizedModule(std::move(module)));

  DebugOptions triton_disabled_debug_options = GetDebugOptionsForTest();
  triton_disabled_debug_options.set_xla_gpu_enable_triton_gemm(false);
  triton_disabled_debug_options.set_xla_gpu_cublas_fallback(true);
  config.set_debug_options(triton_disabled_debug_options);

  TF_ASSERT_OK_AND_ASSIGN(module,
                          ParseAndReturnVerifiedModule(hlo_string, config));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> triton_disabled_module,
                          GetOptimizedModule(std::move(module)));

  EXPECT_TRUE(RunAndCompareTwoModules(std::move(triton_enabled_module),
                                      std::move(triton_disabled_module),
                                      ErrorSpec{1e-6, 1e-6}, false));
}

class FloatNormalizationTest : public GpuCompilerTest,
                               public ::testing::WithParamInterface<
                                   std::pair<PrimitiveType, PrimitiveType>> {};

INSTANTIATE_TEST_SUITE_P(
    Fp8s, FloatNormalizationTest,
    ::testing::Values(
        std::make_pair(PrimitiveType::F8E4M3FN, PrimitiveType::F8E4M3FN),
        std::make_pair(PrimitiveType::F8E5M2, PrimitiveType::F8E4M3FN),
        std::make_pair(PrimitiveType::F8E4M3FN, PrimitiveType::F8E5M2),
        std::make_pair(PrimitiveType::F8E5M2, PrimitiveType::F8E5M2)));

TEST_P(FloatNormalizationTest, Fp8Normalization) {
  const PrimitiveType lhs_type = GetParam().first;
  const PrimitiveType rhs_type = GetParam().second;
  const std::string lhs_name =
      primitive_util::LowercasePrimitiveTypeName(lhs_type);
  const std::string rhs_name =
      primitive_util::LowercasePrimitiveTypeName(rhs_type);
  const std::string module_str = absl::Substitute(R"(
HloModule sch

ENTRY main {
  parameter = $0[1600,1600]{1,0} parameter(0)
  parameter.1 = $1[1600,1600]{1,0} parameter(1)
  neg = $1[1600,1600]{1,0} negate(parameter.1)
  dot = f16[1600,1600]{1,0} dot(parameter,neg), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  constant = f16[] constant(0)
  broadcast = f16[1600,1600]{1,0} broadcast(constant), dimensions={}
  ROOT maximum = f16[1600,1600]{1,0} maximum(dot,broadcast)
})",
                                                  lhs_name, rhs_name);

  auto optimize_module = [&](bool enable_triton, bool enable_blas,
                             bool enable_blas_fallback)
      -> absl::StatusOr<std::unique_ptr<HloModule>> {
    HloModuleConfig config;
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_cublas_fallback(enable_blas_fallback);
    debug_options.set_xla_gpu_enable_triton_gemm(enable_triton);
    debug_options.set_xla_gpu_cudnn_gemm_fusion_level(0);
    if (!enable_blas) {
      debug_options.add_xla_disable_hlo_passes("cublas-gemm-rewriter");
    }
    config.set_debug_options(debug_options);
    config.set_num_partitions(1);

    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                        ParseAndReturnVerifiedModule(module_str, config));
    return GetOptimizedModule(std::move(module));
  };

  auto cc = backend()
                .default_stream_executor()
                ->GetDeviceDescription()
                .cuda_compute_capability();

  const std::string triton_keep_types = absl::Substitute(
      R"(CHECK: fusion($0{{[^)]*}}, $1{{[^)]*}}){{.*}}"kind":"__triton_gemm")",
      lhs_name, rhs_name);
  const std::string cublaslt_keep_types = absl::Substitute(
      R"(CHECK: custom-call($0{{[^)]*}}, $1{{[^)]*}}){{.*}}custom_call_target="__cublas$$lt$$matmul$$f8")",
      lhs_name, rhs_name);
  const std::string cublas_convert_to_f16 =
      R"(CHECK: custom-call(f16{{[^)]*}}, f16{{[^)]*}}){{.*}}custom_call_target="__cublas$gemm")";
  const std::string fallback_convert_to_f16 =
      R"(CHECK: dot(f16{{[^)]*}}, f16{{[^)]*}}))";

  HloPrintOptions print_options =
      HloPrintOptions().set_print_operand_shape(true);

  {
    // Triton enabled, no fallback.
    TF_ASSERT_OK_AND_ASSIGN(auto optimized_module_no_fallback,
                            optimize_module(/*enable_triton=*/true,
                                            /*enable_blas=*/true,
                                            /*enable_blas_fallback=*/false));
    // Triton supports f8e4m3fn on Hopper and f8e5m2 on Ampere.
    const std::string triton_expected_check =
        (cc.IsAtLeastHopper() ||
         (cc.IsAtLeastAmpere() && lhs_type == F8E5M2 && rhs_type == F8E5M2))
            ? triton_keep_types
            : cublas_convert_to_f16;
    TF_ASSERT_OK_AND_ASSIGN(
        bool filecheck_matched,
        RunFileCheck(optimized_module_no_fallback->ToString(print_options),
                     triton_expected_check));
    EXPECT_TRUE(filecheck_matched);
  }

  {
    // Triton disabled, BLAS enabled.
    TF_ASSERT_OK_AND_ASSIGN(auto optimized_module_no_triton,
                            optimize_module(/*enable_triton=*/false,
                                            /*enable_blas=*/true,
                                            /*enable_blas_fallback=*/true));
    // cuBLASlt is only available on Hopper and it doesn't support
    // f8e5m2×f8e5m2.
    const std::string blas_expected_check =
        (cc.IsAtLeastHopper() && !(lhs_type == F8E5M2 && rhs_type == F8E5M2))
            ? cublaslt_keep_types
            : cublas_convert_to_f16;

    TF_ASSERT_OK_AND_ASSIGN(
        bool filecheck_matched,
        RunFileCheck(optimized_module_no_triton->ToString(print_options),
                     blas_expected_check));
    EXPECT_TRUE(filecheck_matched);
  }

  {
    // Neither Triton nor BLAS enabled, always fall back.
    TF_ASSERT_OK_AND_ASSIGN(auto optimized_module_nothing,
                            optimize_module(/*enable_triton=*/false,
                                            /*enable_blas=*/false,
                                            /*enable_blas_fallback=*/false));
    TF_ASSERT_OK_AND_ASSIGN(
        bool filecheck_matched,
        RunFileCheck(optimized_module_nothing->ToString(print_options),
                     fallback_convert_to_f16));
    EXPECT_TRUE(filecheck_matched);
  }
}

TEST_F(GpuCompilerTest, CollectivePermuteDecompositionAndPipelining) {
  const char* kModuleStr = R"(
HloModule cp

cond {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(%param), index=0
    ub = u32[] constant(11)
    ROOT result = pred[] compare(count, ub), direction=LT
 }

body {
    param = (u32[], f32[1, 1024, 1024]) parameter(0)
    count = get-tuple-element(%param), index=0
    send-data = get-tuple-element(%param), index=1

    recv-data = f32[1, 1024, 1024] collective-permute(send-data),
      source_target_pairs={{0,1}, {1,2}, {2,3}, {3,4}}, channel_id=1

    // The computation code that uses the current recv-data and
    // produces the send-data for the next iteration.
    c1 = u32[] constant(1)
    new_count = u32[] add(count, c1)
    replica = u32[] replica-id()
    c10 = u32[] constant(10)
    sum = u32[] add(replica, c10)
    sum2 = u32[] add(sum, count)
    conv = f32[] convert(sum2)
    p = f32[1, 1024, 1024] broadcast(conv), dimensions={}
    b = f32[1, 1024, 1024] add(p, recv-data)
    c = f32[1, 1024, 1024] multiply(b, b)
    d = f32[1, 1024, 1024] tan(c)
    s = f32[1, 1024, 1024] dot(c, d), lhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}

    ROOT result = (u32[], f32[1, 1024, 1024]) tuple(new_count, s)
}

ENTRY test_computation {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}
    while_init = (u32[], f32[1, 1024, 1024]) tuple(c0, init)
    while_result = (u32[], f32[1, 1024, 1024]) while(while_init), body=body, condition=cond
    ROOT result = f32[1, 1024, 1024] get-tuple-element(while_result), index=1
}
)";

  const char* kExpected = R"(
CHECK:       recv-done
CHECK-SAME:    channel_id=[[CHANNEL_ID:[0-9]+]]
CHECK-SAME:    frontend_attributes={_xla_send_recv_pipeline="0"}
CHECK:       %[[AFTER_ALL:.*]] = after-all
CHECK:       send-done
CHECK-SAME:    channel_id=[[CHANNEL_ID]]
CHECK-SAME:    frontend_attributes={_xla_send_recv_pipeline="0"}
CHECK:       %[[CUSTOM_CALL:.*]] = custom-call
CHECK:       %[[RESULT_RECV:.*]] = recv(%[[AFTER_ALL]])
CHECK-SAME:    channel_id=[[CHANNEL_ID]]
CHECK-SAME:    frontend_attributes={_xla_send_recv_pipeline="0",
CHECK-SAME{LITERAL}:                _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}},
CHECK-SAME:                         control-predecessors={%[[CUSTOM_CALL:.*]]}
CHECK:       %[[RESULT_SEND:.*]] = send(%[[SOME_SEND_ARG:.*]], %[[AFTER_ALL]])
CHECK-SAME:    channel_id=1
CHECK-SAME:    frontend_attributes={_xla_send_recv_pipeline="0",
CHECK-SAME{LITERAL}:                _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}},
CHECK-SAME:                         control-predecessors={%[[RESULT_RECV]]}
CHECK:       ROOT
// We actually expect both RESULT_RECV and RESULT_SEND to match on this line.
// However, despite popular belief, CHECK-DAG-SAME is not actually a valid
// directive. Checking for both without using a DAG would be inherently flaky,
// so we take the hit and only check for one of them.
CHECK-SAME:    %[[RESULT_RECV]]

CHECK: ENTRY
CHECK:       %[[ENTRY_AFTER_ALL:.*]] = after-all
CHECK:       %[[ENTRY_RECV:.*]] = recv(%[[ENTRY_AFTER_ALL]])
CHECK-SAME:    channel_id=[[CHANNEL_ID]]
CHECK-SAME:    frontend_attributes={_xla_send_recv_pipeline="0",
CHECK-SAME{LITERAL}:                _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}}
CHECK:       %[[ENTRY_SEND:.*]] = send(%[[SOME_SEND_ARG:.*]], %[[ENTRY_AFTER_ALL]])
CHECK-SAME:    channel_id=1
CHECK-SAME:    frontend_attributes={_xla_send_recv_pipeline="0",
CHECK-SAME{LITERAL}:                _xla_send_recv_source_target_pairs={{0,1},{1,2},{2,3},{3,4}}},
CHECK-SAME:                         control-predecessors={%[[ENTRY_RECV]]}
CHECK:       %[[WHILE_INIT:.*]] = tuple
// Check here that the send argument is likewise passed to the while loop, as
// a counterpart to the check in the child computation above.
CHECK-SAME:    %[[ENTRY_SEND]]
CHECK:       while(%[[WHILE_INIT]])
CHECK:       recv-done
CHECK-SAME:    channel_id=[[CHANNEL_ID]]
CHECK-SAME:    frontend_attributes={_xla_send_recv_pipeline="0"}
CHECK:       send-done
CHECK-SAME:    channel_id=[[CHANNEL_ID]]
CHECK-SAME:    frontend_attributes={_xla_send_recv_pipeline="0"}
)";

  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_latency_hiding_scheduler(true);
  debug_options.set_xla_gpu_collective_permute_decomposer_threshold(1);
  debug_options.set_xla_gpu_enable_pipelined_p2p(true);
  debug_options.set_xla_gpu_enable_triton_gemm(false);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(module)));
  TF_ASSERT_OK(Schedule(optimized_module.get()));

  HloPrintOptions options;
  options.set_print_operand_shape(false);
  options.set_print_result_shape(false);
  TF_ASSERT_OK_AND_ASSIGN(
      bool filecheck_matched,
      RunFileCheck(optimized_module->ToString(options), kExpected));
  EXPECT_TRUE(filecheck_matched);
}

class KernelCacheTest : public HloTestBase {
 public:
  void SetUp() override {
    CHECK(tsl::Env::Default()->LocalTempFilename(&cache_file_name_));
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(
        bool can_use_link_modules,
        dynamic_cast<GpuCompiler*>(backend().compiler())
            ->CanUseLinkModules(
                config,
                backend().default_stream_executor()->GetDeviceDescription()));
    if (!can_use_link_modules) {
      GTEST_SKIP() << "Caching compiled kernels requires support of linking.";
    }
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_kernel_cache_file(cache_file_name_);
    debug_options.set_xla_gpu_enable_llvm_module_compilation_parallelism(true);
    return debug_options;
  }

  bool CacheFileExists() {
    if (!tsl::Env::Default()->FileExists(cache_file_name_).ok()) {
      return false;
    }
    return true;
  }

  int CacheEntryCount() {
    if (!CacheFileExists()) {
      return 0;
    }
    std::string serialized;
    TF_EXPECT_OK(tsl::ReadFileToString(tsl::Env::Default(), cache_file_name_,
                                       &serialized));
    CompilationCacheProto proto;
    EXPECT_TRUE(proto.ParseFromString(std::string(serialized)));
    return proto.entries_size();
  }

  std::string cache_file_name_;
  static constexpr absl::string_view kHloText = R"(
  ENTRY e {
    p = s8[] parameter(0)
    c = s8[] constant(8)
    ROOT _ = s8[] add(p, c)
  })";
};

TEST_F(KernelCacheTest, CacheIsGenerated) {
  // First run - no cache file
  EXPECT_FALSE(CacheFileExists());
  EXPECT_TRUE(Run(kHloText, /*run_hlo_passes=*/false));
  // First run generates a cache
  EXPECT_EQ(CacheEntryCount(), 1);
  // Second run - with cache file
  EXPECT_TRUE(Run(kHloText, /*run_hlo_passes=*/false));
  EXPECT_EQ(CacheEntryCount(), 1);
}

TEST_F(KernelCacheTest, NoCacheIsGeneratedWithoutCompiledKernels) {
  EXPECT_FALSE(CacheFileExists());
  EXPECT_TRUE(Run(R"(
  ENTRY e {
    a = f32[5,5] parameter(0)
    ROOT _ = f32[5,5] custom-call(a, a), custom_call_target="__cublas$gemm",
      backend_config="{ \"gemm_backend_config\": {\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}}"
  })",
                  /*run_hlo_passes=*/false));
  EXPECT_FALSE(CacheFileExists());
}

TEST_F(KernelCacheTest, CacheGrowsWithNewKernels) {
  EXPECT_FALSE(CacheFileExists());
  EXPECT_TRUE(Run(kHloText, /*run_hlo_passes=*/false));
  EXPECT_EQ(CacheEntryCount(), 1);
  // Second run - with cache file and another HLO
  EXPECT_TRUE(Run(R"(
  ENTRY e {
    p = s8[] parameter(0)
    ROOT _ = s8[] multiply(p, p)
  })",
                  /*run_hlo_passes=*/false));
  EXPECT_EQ(CacheEntryCount(), 2);
}

TEST_F(KernelCacheTest, AllKernelsAreCachedBecauseSplitModuleUsesRoundRobin) {
  EXPECT_FALSE(CacheFileExists());
  EXPECT_TRUE(Run(R"(
  ENTRY e {
    p = s8[] parameter(0)
    n = s8[] negate(p)
    a = s8[] add(n, n)
    s = s8[] subtract(p, a)
    ROOT _ = s8[] multiply(s, p)
  })",
                  /*run_hlo_passes=*/false));
  EXPECT_EQ(CacheEntryCount(), 4);
}

TEST_F(KernelCacheTest, CachingWorksWithLoadedExecutables) {
  const std::string kHloAdd1 = R"(
add1 {
  p = s32[] parameter(0)
  c = s32[] constant(1)
  ROOT a = s32[] add(p, c)
}

ENTRY e {
  p = s32[] parameter(0)
  ROOT r = s32[] fusion(p), kind=kLoop, calls=add1
})";

  const std::string kHloAdd2 = R"(
add2 {
  p = s32[] parameter(0)
  c = s32[] constant(2)
  ROOT a = s32[] add(p, c)
}

ENTRY e {
  p = s32[] parameter(0)
  ROOT r = s32[] fusion(p), kind=kLoop, calls=add2
})";

  TF_ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                          se::PlatformManager::PlatformWithName("cuda"));
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_exec,
                          platform->ExecutorForDevice(0));

  Compiler* compiler = backend().compiler();
  AotCompilationOptions aot_options(compiler->PlatformId());
  aot_options.set_executor(stream_exec);

  auto test = [this, &compiler, &aot_options](absl::string_view hlo, int input,
                                              int expected_result) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(hlo));
    auto module_group = std::make_unique<HloModuleGroup>(std::move(module));
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
        compiler->CompileAheadOfTime(std::move(module_group), aot_options));

    TF_ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                            aot_results[0]->SerializeAsString());
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<AotCompilationResult> aot_result,
        compiler->LoadAotCompilationResult(serialized_aot_result));

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Executable> executable,
        std::move(*aot_result)
            .LoadExecutable(compiler, aot_options.executor()));
    std::unique_ptr<OpaqueExecutable> wrapped_executable =
        test_runner_as_hlo_runner().WrapExecutable(std::move(executable));

    const xla::Literal literal_input =
        xla::LiteralUtil::CreateR0<int32_t>(input);
    const xla::Literal literal_expected_result =
        xla::LiteralUtil::CreateR0<int32_t>(expected_result);

    TF_ASSERT_OK_AND_ASSIGN(Literal result,
                            test_runner_as_hlo_runner().ExecuteWithExecutable(
                                wrapped_executable.get(), {&literal_input}));

    EXPECT_TRUE(LiteralTestUtil::Equal(result, literal_expected_result));
  };

  test(kHloAdd1, 1, 2);
  test(kHloAdd2, 1, 3);
  // The test used to fail on the second execution of the second module when it
  // was already cached.
  test(kHloAdd2, 1, 3);
}

class KernelCacheTestSingleThreaded : public KernelCacheTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = KernelCacheTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_force_compilation_parallelism(1);
    return debug_options;
  }
};

TEST_F(KernelCacheTestSingleThreaded, CacheIsGenerated) {
  EXPECT_FALSE(CacheFileExists());
  EXPECT_TRUE(Run(kHloText, /*run_hlo_passes=*/false));
  EXPECT_EQ(CacheEntryCount(), 1);
  EXPECT_TRUE(Run(kHloText, /*run_hlo_passes=*/false));
  EXPECT_EQ(CacheEntryCount(), 1);
}

class NoKernelCacheTest : public KernelCacheTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = KernelCacheTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_llvm_module_compilation_parallelism(false);
    return debug_options;
  }
};

TEST_F(NoKernelCacheTest, NoCacheWithoutCompilationParallelism) {
  EXPECT_TRUE(Run(kHloText, /*run_hlo_passes=*/false));
  EXPECT_FALSE(CacheFileExists());
}

TEST_F(GpuCompilerTest, TestFlag_xla_gpu_unsafe_pipelined_loop_annotator) {
  const char* hlo = R"(
  HloModule test, entry_computation_layout={()->(s32[], s32[])}
    %Body (param: (s32[], s32[])) -> (s32[], s32[]) {
      %param = (s32[], s32[]) parameter(0)
      %i = s32[] get-tuple-element((s32[], s32[]) %param), index=1
      %one = s32[] constant(1)
      %i_plus_one = s32[] add(s32[] %i, s32[] %one)
      %permute = s32[] collective-permute(%i_plus_one), channel_id=1, source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
      ROOT %tuple = (s32[], s32[]) tuple(s32[] %permute, s32[] %i_plus_one)
    }
    %Cond (param.1: (s32[], s32[])) -> pred[] {
      %param.1 = (s32[], s32[]) parameter(0)
      %i.1 = s32[] get-tuple-element((s32[], s32[]) %param.1), index=1
      %trip_count = s32[] constant(10)
      ROOT %done = pred[] compare(s32[] %i.1, s32[] %trip_count), direction=LT
    }
    ENTRY %test () -> (s32[], s32[]) {
      %i_start = s32[] constant(0)
      %p_start = s32[] constant(0)
      %initial_tuple = (s32[], s32[]) tuple(s32[] %i_start, s32[] %p_start)
      ROOT %while = (s32[], s32[]) while((s32[], s32[]) %initial_tuple), condition=%Cond, body=%Body, frontend_attributes={is_pipelined_while_loop="true"}
    })";

  const char* kExpected = R"(
  // CHECK: {{.+}} = recv({{.+}}), {{.+}}, frontend_attributes={_xla_send_recv_source_target_pairs={{[{]}}{3,0}},_xla_send_recv_validation={{[{]}}{3,9}}}
  // CHECK: {{.+}} = send({{.+}}), {{.+}}, frontend_attributes={_xla_send_recv_source_target_pairs={{[{]}}{3,0}},_xla_send_recv_validation={{[{]}}{3,9}}}
  // CHECK: {{.+}} = recv({{.+}}), {{.+}}, frontend_attributes={_xla_send_recv_source_target_pairs={{[{]}}{0,1},{1,2},{2,3}},_xla_send_recv_validation={{[{]}}{0,6},{1,7},{2,8}}}
  // CHECK: {{.+}} = send({{.+}}), {{.+}}, frontend_attributes={_xla_send_recv_source_target_pairs={{[{]}}{0,1},{1,2},{2,3}},_xla_send_recv_validation={{[{]}}{0,6},{1,7},{2,8}}}
  )";

  DebugOptions debug_options;
  HloModuleConfig config;
  debug_options.set_xla_gpu_unsafe_pipelined_loop_annotator(true);
  debug_options.set_xla_gpu_enable_pipelined_p2p(true);
  config.set_debug_options(debug_options);
  config.set_num_partitions(4);
  config.set_use_spmd_partitioning(true);
  TF_ASSERT_OK_AND_ASSIGN(auto unoptimized_module,
                          ParseAndReturnVerifiedModule(hlo, config));
  TF_ASSERT_OK_AND_ASSIGN(auto optimized_module,
                          GetOptimizedModule(std::move(unoptimized_module)));
  HloPrintOptions options;
  options.set_print_operand_shape(false);
  options.set_print_result_shape(false);
  TF_ASSERT_OK_AND_ASSIGN(
      bool filecheck_matched,
      RunFileCheck(optimized_module->ToString(options), kExpected));
  EXPECT_TRUE(filecheck_matched);
}

bool HasBlockLevelFusionConfig(const HloInstruction* fusion) {
  return fusion->opcode() == HloOpcode::kFusion &&
         fusion->has_backend_config() &&
         fusion->backend_config<GpuBackendConfig>().ok() &&
         fusion->backend_config<GpuBackendConfig>()
             ->fusion_backend_config()
             .has_block_level_fusion_config();
}

TEST_F(GpuCompilerTest,
       LoopFusionRootedInTransposeIsRewrittenToBlockLevelByDefaultPostAmpere) {
  auto cc = backend()
                .default_stream_executor()
                ->GetDeviceDescription()
                .cuda_compute_capability();

  constexpr absl::string_view transpose_fusion_module = R"(
transpose {
  p0 = f32[1024,1024,1024] parameter(0)
  ROOT transpose = f32[1024,1024,1024] transpose(p0), dimensions={2,1,0}
}

ENTRY main {
  p0 = f32[1024,1024,1024] parameter(0)
  ROOT fusion = f32[1024,1024,1024] fusion(p0), kind=kLoop, calls=transpose
})";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(transpose_fusion_module));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(module)));

  if (cc.IsAtLeastAmpere()) {
    EXPECT_TRUE(HasBlockLevelFusionConfig(
        optimized_module->entry_computation()->root_instruction()));
  } else {
    EXPECT_FALSE(HasBlockLevelFusionConfig(
        optimized_module->entry_computation()->root_instruction()));
  }
}

TEST_F(
    GpuCompilerTest,
    FusionBlockLevelRewriterRewritesKLoopTransposeWithBitcastIfTheSmallMinorDimIsAPowerOfTwo) {  // NOLINT(whitespace/line_length)
  auto cc = backend()
                .default_stream_executor()
                ->GetDeviceDescription()
                .cuda_compute_capability();
  if (!cc.IsAtLeastAmpere()) {
    GTEST_SKIP() << "FusionBlockLevelRewriter requires Ampere+ to run.";
  }

  // If this test starts failing, then it's likely that this no longer generates
  // a kLoop transpose. That's great---it probably means the rewrite in question
  // is no longer necessary!
  //
  // The small minor dimension here is a power of two, so the rewrite should
  // succeed.
  constexpr absl::string_view rewritable_transpose_string = R"(
ENTRY main {
  p0 = f32[1024,4096]{1,0} parameter(0)
  reshape = f32[1024,1024,4]{2,1,0} reshape(p0)
  ROOT transpose = f32[4,1024,1024]{2,1,0} transpose(reshape), dimensions={2,1,0}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> rewritable_transpose_module,
      ParseAndReturnVerifiedModule(rewritable_transpose_string));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> rewritable_transpose_optimized_module,
      GetOptimizedModule(std::move(rewritable_transpose_module)));
  EXPECT_TRUE(HasBlockLevelFusionConfig(
      rewritable_transpose_optimized_module->entry_computation()
          ->root_instruction()));

  // The small minor dimension here is not a power of two, so the rewrite should
  // fail.
  constexpr absl::string_view unrewritable_transpose_string = R"(
ENTRY main {
  p0 = f32[1024,6144]{1,0} parameter(0)
  reshape = f32[1024,1024,6]{2,1,0} reshape(p0)
  ROOT transpose = f32[6,1024,1024]{2,1,0} transpose(reshape), dimensions={2,1,0}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> unrewritable_transpose_module,
      ParseAndReturnVerifiedModule(unrewritable_transpose_string));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> unrewritable_transpose_optimized_module,
      GetOptimizedModule(std::move(unrewritable_transpose_module)));
  EXPECT_FALSE(HasBlockLevelFusionConfig(
      unrewritable_transpose_optimized_module->entry_computation()
          ->root_instruction()));
}

TEST_F(GpuCompilerTest, NoRaceConditionInParallelCompilation) {
  // This test will fail under TSAN if there is a race condition somewhere.

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_pool", 2);

  // Running two compilations on different threads is enough.
  // If there is some unsynchronized memory access, TSAN will report it.
  constexpr int kNumOfParallelCompilations = 2;

  for (int i = 0; i < kNumOfParallelCompilations; ++i) {
    thread_pool.Schedule([&]() {
      HloModuleConfig config;
      DebugOptions debug_options = GetDebugOptionsForTest();
      config.set_debug_options(debug_options);
      // The contents on this module don't matter that much, but it should
      // be something going through the autotuner.
      TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                              ParseAndReturnVerifiedModule(R"(
HloModule module

triton_gemm_dot {
  p0 = s8[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  c0 = f32[10,10] convert(p0)
  ROOT dot.0 = f32[10,10] dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY entry {
  p0 = s8[10,10] parameter(0)
  p1 = f32[10,10] parameter(1)
  s = f32[10,10] sqrt(p1)
  d = f32[10,10] fusion(p0, p1), kind=kCustom, calls=triton_gemm_dot
  ROOT r = f32[10,10] add(d, s)
})",
                                                           config));
      std::unique_ptr<HloModule> compiled_module =
          backend()
              .compiler()
              ->RunHloPasses(module->Clone(),
                             backend().default_stream_executor(),
                             /*device_allocator=*/nullptr)
              .value();
      std::unique_ptr<Executable> executable =
          backend()
              .compiler()
              ->RunBackend(std::move(compiled_module),
                           backend().default_stream_executor(),
                           {/*device_allocator=*/nullptr,
                            /*thread_pool=*/nullptr,
                            /*layout_canonicalization_callback=*/{},
                            /*is_autotuning_compilation=*/false})
              .value();
    });
  }
}

MATCHER_P(ThunkKindIs, kind, "") {
  return ExplainMatchResult(::testing::Eq(kind), arg->kind(), result_listener);
}

TEST_F(GpuCompilerTest, StreamAnnotationThunkTest) {
  const absl::string_view hlo_text = R"(
HloModule composite

async_call {
  p0 = f32[32,32] parameter(0)
  p1 = f32[32,32] parameter(1)
  gemm = (f32[32,32], s8[8192]) custom-call(p0, p1), custom_call_target="__cublas$gemm",
    backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],
      "gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,
      "dot_dimension_numbers":
        {"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"]},
      "precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},
      "lhs_stride":"1024","rhs_stride":"1024"}}
  ROOT get-tuple-element = f32[32,32] get-tuple-element(gemm), index=0
}, execution_thread="explicit"

ENTRY main {
  p0 = f32[32,32] parameter(0)
  p1 = f32[32,32] parameter(1)
  call-start = ((f32[32,32], f32[32,32]), f32[32,32]) call-start(p0, p1),
    async_execution_thread="explicit", to_apply=async_call,
    frontend_attributes={_xla_stream_annotation="1"}
  ROOT call-done = f32[32,32]{1,0} call-done(call-start),
    frontend_attributes={_xla_stream_annotation="1"},
    backend_config={"operation_queue_id":"0"}
})";
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
  std::unique_ptr<GpuExecutable> gpu_exec(
      static_cast<GpuExecutable*>(executable.release()));

  EXPECT_EQ(gpu_exec->GetThunk().thunks().size(), 3);
  EXPECT_THAT(gpu_exec->GetThunk().thunks(),
              ::testing::ElementsAre(ThunkKindIs(Thunk::kWaitForStreams),
                                     ThunkKindIs(Thunk::kSequential),
                                     ThunkKindIs(Thunk::kWaitForStreams)));

  // Within the sequential thunk, there should only be a single gemm
  // thunk with an explicitly set execution stream id.
  auto sequential_thunk =
      static_cast<SequentialThunk*>(gpu_exec->GetThunk().thunks()[1].get());
  EXPECT_EQ(sequential_thunk->thunks().size(), 1);
  EXPECT_THAT(sequential_thunk->thunks(),
              ::testing::ElementsAre(ThunkKindIs(Thunk::kGemm)));
  // Ensure the gemm is run on the explicitly set stream.
  EXPECT_EQ(sequential_thunk->thunks()[0]->execution_stream_id(), 1);
}

using GpuCompilerPassTest = GpuCompilerTest;

TEST_F(GpuCompilerPassTest,
       GpuCompilerRunsTritonGemmRewriterByDefaultFromAmpere) {
  auto cc = backend()
                .default_stream_executor()
                ->GetDeviceDescription()
                .cuda_compute_capability();

  bool is_rocm = std::holds_alternative<stream_executor::RocmComputeCapability>(
      backend()
          .default_stream_executor()
          ->GetDeviceDescription()
          .gpu_compute_capability());

  bool expect_triton_gemm_rewriter_has_run = cc.IsAtLeastAmpere() || is_rocm;

  constexpr absl::string_view constant_module = R"(
HloModule noop

ENTRY main {
  ROOT constant = f32[] constant(0)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(constant_module));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(module)));
  const HloModuleMetadataProto& module_metadata =
      optimized_module->metadata()->proto();

  bool triton_gemm_rewriter_has_run = false;
  for (const HloPassMetadata& pass_metadata : module_metadata.pass_metadata()) {
    triton_gemm_rewriter_has_run |=
        pass_metadata.pass_name() == "triton-gemm-rewriter";
  }

  EXPECT_EQ(triton_gemm_rewriter_has_run, expect_triton_gemm_rewriter_has_run);
}

TEST_F(GpuCompilerPassTest,
       GpuCompilerRunsCustomKernelFusionByDefaultFromVolta) {
  auto cc = backend()
                .default_stream_executor()
                ->GetDeviceDescription()
                .cuda_compute_capability();

  bool expect_custom_kernel_fusion_rewriter_has_run =
      cc.major == se::CudaComputeCapability::kVolta;

  constexpr absl::string_view constant_module = R"(
HloModule noop

ENTRY main {
  ROOT constant = f32[] constant(0)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(constant_module));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(std::move(module)));
  const HloModuleMetadataProto& module_metadata =
      optimized_module->metadata()->proto();

  bool custom_kernel_fusion_rewriter_has_run = false;
  for (const HloPassMetadata& pass_metadata : module_metadata.pass_metadata()) {
    custom_kernel_fusion_rewriter_has_run |=
        pass_metadata.pass_name() == "custom-kernel-fusion-rewriter";
  }

  EXPECT_EQ(custom_kernel_fusion_rewriter_has_run,
            expect_custom_kernel_fusion_rewriter_has_run);
}

class PassOrderTest : public GpuCompilerTest {
 public:
  struct PassRange {
    int first_pass_run_index;
    int second_pass_run_index;
  };
  void SetDebugOptions(const DebugOptions& options) {
    HloModuleConfig config = GetModuleConfigForTest();
    config.set_debug_options(options);
    CompileModule(config);
  }

  void SetAndCompileEfficiencyEffort(float exec_effort) {
    HloModuleConfig config = GetModuleConfigForTest();
    config.set_exec_time_optimization_effort(exec_effort);
    CompileModule(config);
  }

  // Fails if any of the passes matching `other_pass_regex` runs before
  // the first occurrence of the pass matching `first_pass_regex`.
  void VerifyPassRunsAtLeastOnceBefore(absl::string_view first_pass_regex,
                                       absl::string_view other_pass_regex) {
    if (!optimized_module_) {
      CompileModule(GetModuleConfigForTest());
    }
    int first_pass_first_run = std::numeric_limits<int>::max();
    int other_pass_first_run = std::numeric_limits<int>::max();
    int run_index = 0;
    for (const HloPassMetadata& pass_metadata :
         optimized_module_->metadata()->proto().pass_metadata()) {
      if (RE2::FullMatch(pass_metadata.pass_name(), first_pass_regex)) {
        VLOG(2) << "Pass " << pass_metadata.pass_name()
                << " matches first_pass_regex." << std::endl;
        first_pass_first_run = std::min(first_pass_first_run, run_index);
      }
      if (RE2::FullMatch(pass_metadata.pass_name(), other_pass_regex)) {
        VLOG(2) << "Pass " << pass_metadata.pass_name()
                << " matches other_pass_regex." << std::endl;
        other_pass_first_run = std::min(other_pass_first_run, run_index);
      }
      ++run_index;
    }

    EXPECT_NE(first_pass_first_run, std::numeric_limits<int>::max())
        << "Did not run a pass matching " << first_pass_regex;
    EXPECT_NE(other_pass_first_run, std::numeric_limits<int>::max())
        << "Did not run a pass matching " << other_pass_regex;
    EXPECT_LE(first_pass_first_run, other_pass_first_run)
        << "A pass matching " << first_pass_regex
        << " did not run before passes matching " << other_pass_regex;
  }

  // Fails if any of the passes with names matching the regular expression
  // `first_pass_regex` run after any of the passes matching `last_pass_regex`
  // or if none of the executed passes matches `first_pass_regex` or
  // `last_pass_regex`. Returns a PassRange with the latest run index of any
  // passes with names matching `first_pass_regex` and the earliest run index of
  // any passes with names matching 'last_pass_regex'. Passes matching both
  // regexes will be counted towards last_pass (i.e., overlap of the two ranges
  // is allowed).
  PassRange VerifyPassOrder(absl::string_view first_pass_regex,
                            absl::string_view last_pass_regex,
                            bool include_pipeline_name = false) {
    if (!optimized_module_) {
      CompileModule(GetModuleConfigForTest());
    }
    int first_pass_latest_run = -1;
    int last_pass_earliest_run = std::numeric_limits<int>::max();
    int run_index = 0;
    for (const HloPassMetadata& pass_metadata :
         optimized_module_->metadata()->proto().pass_metadata()) {
      std::string name = pass_metadata.pass_name();
      if (include_pipeline_name) {
        name = absl::StrCat(pass_metadata.pipeline_name(), ".",
                            pass_metadata.pass_name());
      }
      if (RE2::FullMatch(name, last_pass_regex)) {
        VLOG(2) << "Pass " << pass_metadata.pass_name()
                << " matches last_pass_regex." << std::endl;
        last_pass_earliest_run = std::min(last_pass_earliest_run, run_index);
      } else if (RE2::FullMatch(name, first_pass_regex)) {
        VLOG(2) << "Pass " << pass_metadata.pass_name()
                << " matches first_pass_regex." << std::endl;
        first_pass_latest_run = std::max(first_pass_latest_run, run_index);
      }
      ++run_index;
    }

    EXPECT_GT(first_pass_latest_run, -1)
        << "Did not run a pass matching " << first_pass_regex;
    EXPECT_LT(last_pass_earliest_run, std::numeric_limits<int>::max())
        << "Did not run a pass matching " << last_pass_regex;
    EXPECT_LT(first_pass_latest_run, last_pass_earliest_run)
        << "One or more passes matching " << first_pass_regex
        << " ran after passes matching " << last_pass_regex;
    return {first_pass_latest_run, last_pass_earliest_run};
  }

  // Checks that no pass that matches `pass_regex` runs strictly in between
  // `pass_range.first_pass_run_index` and `pass_range.second_pass_run_index`.
  void VerifyNotRunInBetween(const PassRange& pass_range,
                             absl::string_view pass_regex) {
    int run_index = 0;
    for (const HloPassMetadata& pass_metadata :
         optimized_module_->metadata()->proto().pass_metadata()) {
      if (run_index >= pass_range.second_pass_run_index) {
        break;
      }
      if (run_index++ <= pass_range.first_pass_run_index) {
        continue;
      }
      EXPECT_FALSE(RE2::FullMatch(pass_metadata.pass_name(), pass_regex))
          << "Ran " << pass_metadata.pass_name() << " in the given range";
    }
  }

 protected:
  absl::Status ScheduleModule() { return Schedule(optimized_module_.get()); }

  void CompileModule(const HloModuleConfig& config) {
    constexpr absl::string_view constant_module = R"(
ENTRY main {
  ROOT constant = f32[] constant(0)
})";
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<VerifiedHloModule> module,
        ParseAndReturnVerifiedModule(constant_module, config));
    TF_ASSERT_OK_AND_ASSIGN(optimized_module_,
                            GetOptimizedModule(std::move(module)));
  }

  std::unique_ptr<HloModule> optimized_module_;
};

TEST_F(PassOrderTest, PassesAreRunInCorrectOrder) {
  VerifyPassOrder(/*first_pass_regex=*/"layout-assignment",
                  /*last_pass_regex=*/"priority-fusion");
  VerifyPassOrder(/*first_pass_regex=*/"layout-assignment",
                  /*last_pass_regex=*/"layout_normalization");
}

TEST_F(PassOrderTest, OffloadingPassesAreRunInCorrectOrder) {
  // HostOffloadLegalize must run before LayoutNormalization to prevent
  // the creation of invalid transpose/bitcast operations within
  // host memory offloading segments.
  VerifyPassRunsAtLeastOnceBefore(/*first_pass_regex=*/"host-offload-legalize",
                                  /*other_pass_regex=*/"layout_normalization");

  // CSE should not run between HostOffloadLegalize and HostOffloader
  // because it could break the invariants established
  // by the legalize pass, such as the buffer initialization broadcasts
  // before loops having only a single use
  // (see https://github.com/openxla/xla/issues/20373).
  auto pass_range =
      VerifyPassOrder(/*first_pass_regex=*/"host-offload-legalize",
                      /*last_pass_regex=*/"host-offloader");
  VerifyNotRunInBetween(pass_range, /*pass_regex=*/"cse");
}

TEST_F(PassOrderTest, FusionDispatchRunsAfterAllFusionPasses) {
  auto cc = backend()
                .default_stream_executor()
                ->GetDeviceDescription()
                .cuda_compute_capability();
  if (!cc.IsAtLeastAmpere()) {
    GTEST_SKIP() << "fusion-dispatch requires Ampere+ to run.";
  }

  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_enable_fusion_block_level_rewriter(
      true);
  SetDebugOptions(debug_options);

  VerifyPassOrder(/*first_pass_regex=*/".*fusion.*",
                  /*last_pass_regex=*/"fusion-dispatch-pipeline.*",
                  /*include_pipeline_name=*/true);
}

TEST_F(PassOrderTest, CollectivePipelinerRunsAfterCollectiveQuantizer) {
  DebugOptions options = GetDebugOptionsForTest();
  options.set_xla_gpu_enable_pipelined_collectives(true);
  SetDebugOptions(options);

  VerifyPassOrder(/*first_pass_regex=*/"collective-quantizer",
                  /*last_pass_regex=*/"collective-pipeliner.*");
}

TEST_F(PassOrderTest,
       AllGatherDynamicSliceSimplifierRunsAfterAllGatherOptimizer) {
  VerifyPassOrder(
      /*first_pass_regex=*/".*all-gather-optimizer.*",
      /*last_pass_regex=*/".*all-gather-dynamic-slice-simplifier.*");
}

TEST_F(PassOrderTest, StableSortExpanderRunsAfterDynamicPadder) {
  VerifyPassOrder(
      /*first_pass_regex=*/"dynamic_padder",
      /*last_pass_regex=*/"stable-sort-expander");
}

MATCHER_P(HasExpectedPasses, expected_pass_names, "") {
  std::vector<absl::string_view> run_pass_names;
  auto metadata = arg->metadata()->proto();
  run_pass_names.reserve(metadata.pass_metadata_size());
  for (auto& pass_metadata : metadata.pass_metadata()) {
    run_pass_names.push_back(pass_metadata.pass_name());
  }
  return Matches(IsSupersetOf(expected_pass_names))(run_pass_names);
}

TEST_F(PassOrderTest, ExecEffortAt0point2RunsSpecifiedPasses) {
  HloModuleConfig config = GetModuleConfigForTest();
  CompileModule(config);
  TF_ASSERT_OK(ScheduleModule());

  // Make sure passes are not enabled by default.
  std::vector<std::string> kExpectedPasses = {
      "loop-double-buffer-transformer",
      "collective-pipeliner-forward",
      "collective-pipeliner-backward",
      "latency-hiding-scheduler",
  };
  EXPECT_THAT(optimized_module_, Not(HasExpectedPasses(kExpectedPasses)));

  // Make sure only after setting the correct optimization effort they are
  // enabled.
  config.set_exec_time_optimization_effort(0.2);
  CompileModule(config);
  TF_ASSERT_OK(ScheduleModule());
  EXPECT_THAT(optimized_module_, HasExpectedPasses(kExpectedPasses));
}

TEST_F(PassOrderTest, LHSRunsIfProfileDataIsAvailable) {
  HloModuleConfig config = GetModuleConfigForTest();

  // Make sure LHS is off by default.
  std::vector<std::string> kExpectedPasses = {
      "latency-hiding-scheduler",
  };
  CompileModule(config);
  TF_ASSERT_OK(ScheduleModule());
  EXPECT_THAT(optimized_module_, Not(HasExpectedPasses(kExpectedPasses)));

  // Make sure we turn the LHS on with we schedule with profile data.
  const absl::string_view kProfile = R"pb(
    costs { name: "cp" cost_us: 100.0 }
  )pb";
  config.set_fdo_profile(kProfile);
  CompileModule(config);
  TF_ASSERT_OK(ScheduleModule());

  EXPECT_THAT(optimized_module_, HasExpectedPasses(kExpectedPasses));
}

TEST_F(PassOrderTest, GemmFusionRunsAfterDotNormalizer) {
  auto cc = backend()
                .default_stream_executor()
                ->GetDeviceDescription()
                .cuda_compute_capability();
  if (!cc.IsAtLeastAmpere()) {
    GTEST_SKIP() << "GemmFusion requires Ampere+ to run.";
  }
  DebugOptions options = GetDebugOptionsForTest();
  options.set_xla_gpu_enable_triton_gemm(true);
  SetDebugOptions(options);
  auto pass_range = VerifyPassOrder(
      /*first_pass_regex=*/"dot_normalizer",
      /*last_pass_regex=*/"triton-gemm-rewriter");
  VerifyNotRunInBetween(pass_range, /*pass_regex=*/"algsimp");
}

TEST_F(PassOrderTest, GemmRewriterRunsAfterDotNormalizer) {
  auto pass_range = VerifyPassOrder(
      /*first_pass_regex=*/"dot_normalizer",
      /*last_pass_regex=*/"cublas-gemm-rewriter");
  VerifyNotRunInBetween(pass_range, /*pass_regex=*/"algsimp");
}

TEST_F(PassOrderTest, NestGemmFusionRunsAfterGemmFusionAutotuner) {
  // NestGemmFusion expect to see __triton_gemm custom call with a backend
  // config created by gemm_fusion_autotuner.
  DebugOptions options = GetDebugOptionsForTest();
  options.set_xla_gpu_unsupported_enable_generic_triton_emitter_for_gemms(true);
  SetDebugOptions(options);
  VerifyPassOrder("gemm-fusion-autotuner", "nest_gemm_fusion");
}

TEST_F(PassOrderTest,
       ReducePrecisionIsRemovedAfterAllCallsToSimplifyFPConversions) {
  // Because of an issue with JAX remat and `SimplifyFPConversions` (see PR:
  // https://github.com/jax-ml/jax/pull/22244), we can only eliminate the
  // no-op reduce-precision operations after the last call to
  // `SimplifyFPConversions`. No-op reduce-precisions are removed within
  // algebraic simplifier, if the option to remove them is set. In the compiler
  // pipeline, this is done as a subpipeline, which should be after the last
  // invocation of SimplifyFPConversions.
  VerifyPassOrder("simplify-fp-conversions",
                  "remove-no-op-reduce-precision-algebraic-simplifier");
}

// Tests that passes are converging and pipelines reach a fix point.
class FixPointTest : public HloTestBase {
 public:
  void ExpectPipelinesReachFixedPoint(absl::string_view module_text) {
    std::unique_ptr<HloModule> optimized_module;
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<VerifiedHloModule> module,
        ParseAndReturnVerifiedModule(module_text, GetModuleConfigForTest()));
    TF_ASSERT_OK_AND_ASSIGN(optimized_module,
                            GetOptimizedModule(std::move(module)));

    std::string last_pipeline_name;
    int count = 0;
    for (const HloPassMetadata& pass_metadata :
         optimized_module->metadata()->proto().pass_metadata()) {
      if (pass_metadata.pass_name() != "pipeline-start") {
        continue;
      }
      VLOG(2) << "pipeline: " << pass_metadata.pipeline_name();
      if (pass_metadata.pipeline_name() != last_pipeline_name) {
        count = 0;
        last_pipeline_name = pass_metadata.pipeline_name();
      }
      count++;
      // 25 is a default iteration limit of HloPassFix.
      EXPECT_LT(count, 25) << "Pipeline '" << pass_metadata.pipeline_name()
                           << "' ran " << count
                           << " times. That is likely an indication that the "
                              "pipeline is not reaching a fixed point.";
    }
  }
};

TEST_F(FixPointTest, Constant) {
  ExpectPipelinesReachFixedPoint(R"(ENTRY main {
  ROOT constant = f32[] constant(0)
})");
}

TEST_F(FixPointTest, ReshapeTranspose) {
  ExpectPipelinesReachFixedPoint(R"(ENTRY main {
p0 = f32[1024,4096]{1,0} parameter(0)
reshape = f32[1024,1024,4]{2,1,0} reshape(p0)
ROOT transpose = f32[4,1024,1024]{2,1,0} transpose(reshape), dimensions={2,1,0}
})");
}

TEST_F(FixPointTest, DotWithBatchDims) {
  // Reduced test case for b/383729716.
  ExpectPipelinesReachFixedPoint(R"(ENTRY main {
p0 = f32[8,4,64]{2,1,0} parameter(0)
p1 = f32[4,64,1024] parameter(1)
ROOT dot = f32[4,8,1024]{2,1,0} dot(p0, p1), lhs_batch_dims={1}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
})");
}

TEST_F(FixPointTest, DotWithReshapes) {
  // Reduced test case for b/383729716.
  ExpectPipelinesReachFixedPoint(
      R"(ENTRY main {
tmp_0 = f64[3]{0} parameter(0)
tmp_1 = f64[3,1]{1,0} reshape(tmp_0)
tmp_2 = f64[3]{0} reshape(tmp_1)
tmp_3 = f64[3]{0} transpose(tmp_2), dimensions={0}
tmp_4 = f64[3,1]{1,0} reshape(tmp_3)
tmp_5 = f64[2]{0} parameter(1)
tmp_6 = f64[1,2]{1,0} reshape(tmp_5)
tmp_7 = f64[2]{0} reshape(tmp_6)
tmp_8 = f64[2]{0} transpose(tmp_7), dimensions={0}
tmp_9 = f64[1,2]{1,0} reshape(tmp_8)
tmp_10 = f64[3,2]{1,0} dot(tmp_4, tmp_9), lhs_contracting_dims={1}, rhs_contracting_dims={0}
ROOT tmp_11 = f64[3,2]{1,0} reshape(tmp_10)
})");
}

TEST_F(GpuCompilerTest,
       DynamicSliceFusionWithCollectiveShouldWrapInAsyncAndTestE2E) {
  const char* hlo = R"(
    HloModule test, replica_count=2
    add {
      x = s32[] parameter(0)
      y = s32[] parameter(1)
      ROOT add = s32[] add(x, y)
    }
    ENTRY main {
      destination = s32[2,2,32] parameter(0)
      c1 = s32[] constant(1)
      c0 = s32[] constant(0)
      c4 = s32[] constant(4)
      source = s32[8,32] parameter(1)
      a = s32[1024,1024] parameter(2)
      b = s32[1024,1024] parameter(3)
      slice = s32[4,32] slice(source), slice={[4:8], [0:32]}
      rs = s32[2,32] reduce-scatter(slice), replica_groups={{0,1}}, dimensions={0}, to_apply=add
      reshape = s32[1,2,32] reshape(rs)
      dus = s32[2,2,32] dynamic-update-slice(destination, reshape, c1, c0, c0)
      dot = s32[1024,1024] dot(a,b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      ROOT tuple = tuple(dus,dot)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));
  std::unique_ptr<HloModule> m_ref = m->Clone();
  m->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_dynamic_slice_fusion(true);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpaqueExecutable> wrapped_exec,
      CreateExecutable(m->Clone(), /*run_hlo_passes=*/true));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Executable> exec,
                          test_runner_as_hlo_runner().ExecutableFromWrapped(
                              std::move(wrapped_exec)));
  const char* kExpected = R"(
    // CHECK:      dynamic-slice-fusion{{.+}} {
    // CHECK:        %[[slice:.+]] = {{.+}} slice({{.+}}), slice={[4:8], [0:32]}
    // CHECK:        %[[rs:.+]] = {{.+}} reduce-scatter(%[[slice]]),
    // CHECK-SAME{LITERAL}:              replica_groups={{0,1}}, dimensions={0}
    // CHECK:        %[[bitcast:.+]] = {{.+}} bitcast(%[[rs]])
    // CHECK:        ROOT {{.+}} = {{.+}} dynamic-update-slice({{.+}}, %[[bitcast]], {{.+}})
    // CHECK:      ENTRY
    // CHECK:        %[[fusion_start:.+]] = {{.+}} fusion-start({{.+}}), kind=kCustom, {{.+}}"name":"dynamic_address_computation"
    // CHECK-NEXT:   %[[wrapped_dot:.+]] = {{.+}} fusion({{.+}}), kind=kLoop
    // CHECK-NEXT:   %[[fusion_done:.+]] = {{.+}} fusion-done(%[[fusion_start]]), {{.+}}"name":"dynamic_address_computation"
    // CHECK:        ROOT {{.+}} = {{.+}} tuple(%[[fusion_done]], %[[wrapped_dot]])
  )";
  EXPECT_THAT(
      RunFileCheck(exec->module().ToString(HloPrintOptions{}
                                               .set_print_operand_shape(false)
                                               .set_print_metadata(false)),
                   kExpected),
      ::tsl::testing::IsOkAndHolds(true));

  if (test_runner().device_count() < 2) {
    GTEST_SKIP() << "Skipping test as it requires at least 2 devices.";
  }
  EXPECT_TRUE(RunAndCompareTwoModulesReplicated(std::move(m), std::move(m_ref),
                                                /*run_hlo_passes=*/true,
                                                /*use_threads=*/true,
                                                std::nullopt));
}

TEST_F(GpuCompilerTest, DynamicSliceFusionReduceScatterMultipleBuffers) {
  const char* hlo = R"(
    HloModule test, replica_count=2
    add {
      x = s32[] parameter(0)
      y = s32[] parameter(1)
      ROOT add = s32[] add(x, y)
    }
    ENTRY main {
      p0 = s32[2,2,32] parameter(0)
      p1 = s32[8,32] parameter(1)
      slice = s32[4,32] slice(p1), slice={[4:8], [0:32]}
      rs1 = s32[2,32] reduce-scatter(slice), replica_groups={{0,1}}, dimensions={0}, to_apply=add
      slice2 = s32[4,32] slice(p1), slice={[0:4], [0:32]}
      rs2 = s32[2,32] reduce-scatter(slice2), replica_groups={{0,1}}, dimensions={0}, to_apply=add
      ROOT tuple = tuple(rs1, rs2)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));
  m->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_enable_dynamic_slice_fusion(true);
  TF_ASSERT_OK_AND_ASSIGN(m, GetOptimizedModule(std::move(m)));
  const char* kExpected = R"(
    // CHECK: dynamic-slice-fusion{{.*}} {
    // CHECK-DAG: %[[slice1:.+]] = {{.+}} slice({{.+}}), slice={[4:8], [0:32]}
    // CHECK-DAG: %[[slice2:.+]] = {{.+}} slice({{.+}}), slice={[0:4], [0:32]}
    // CHECK-DAG: ROOT %[[rs:.+]] = {{.+}} reduce-scatter(%[[slice1]], %[[slice2]]),
    // CHECK-SAME{LITERAL}:                                      replica_groups={{0,1}}, dimensions={0}, to_apply=%add
    // CHECK: ENTRY
  )";
  EXPECT_THAT(RunFileCheck(m->ToString(), kExpected),
              ::tsl::testing::IsOkAndHolds(true));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

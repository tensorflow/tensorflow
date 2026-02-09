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
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xla/autotune_results.pb.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/gpu/metrics.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/lib/gtl/value_or_die.h"
#include "xla/tsl/lib/monitoring/collected_metrics.h"
#include "xla/tsl/lib/monitoring/collection_registry.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/path.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/regexp.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

using ::testing::AssertionResult;
using ::testing::EndsWith;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::IsSupersetOf;
using ::testing::Matches;
using ::testing::Not;
using ::testing::NotNull;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::SizeIs;
using ::testing::StartsWith;
using ::testing::TempDir;
using ::tsl::gtl::ValueOrDie;

class GpuCompilerTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtGpuTestBase> {
 public:
  se::CudaComputeCapability get_cuda_cc() const {
    return device_description().cuda_compute_capability();
  }
};

absl::StatusOr<std::string> ReadNonEmptyFile(absl::string_view file_path) {
  std::string str;
  tsl::Env* env = tsl::Env::Default();
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(env, std::string(file_path), &str));
  if (str.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("File is empty: ", file_path));
  }
  return str;
}

TEST_F(GpuCompilerTest, CompiledProgramsCount) {
  const char* hlo_text = R"(
HloModule test

ENTRY main {
  p = f32[10]{0} parameter(0)
  ROOT neg = f32[10]{0} negate(p)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  int64_t before = GetCompiledProgramsCount();
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpaqueExecutable> executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/false));
  EXPECT_EQ(GetCompiledProgramsCount(), before + 1);
}

TEST_F(GpuCompilerTest, CatchCollectiveDeadlocksPostScheduling) {
  constexpr absl::string_view kHloText = R"(
HloModule test, is_scheduled=true

ENTRY test_computation {
  c0 = u32[] constant(0)
  c1 = u32[] constant(1)
  replica = u32[] replica-id()
  a = u32[] add(c1, replica)
  send-data = u32[2] broadcast(a), dimensions={}

  after-all.0 = token[] after-all()
  recv.0 = (u32[2], u32[], token[]) recv(after-all.0), channel_id=0,
  frontend_attributes={
      _xla_send_recv_source_target_pairs="{{1,0}}",
      _xla_send_recv_pipeline="1"
    }
  send.0 = (u32[2], u32[], token[]) send(send-data, after-all.0),
    channel_id=0, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{1,0}}",
      _xla_send_recv_pipeline="1"
    }

  after-all.1 = token[] after-all()
  recv.1 = (u32[2], u32[], token[]) recv(after-all.1), channel_id=0,
  frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0,1}, {1,0}}"
    }
  send.1 = (u32[2], u32[], token[]) send(send-data, after-all.1),
    channel_id=0, frontend_attributes={
      _xla_send_recv_source_target_pairs="{{0,1}, {1,0}}"
    }

  recv-done.0 = (u32[2], token[]) recv-done(recv.0), channel_id=0,
  frontend_attributes={
      _xla_send_recv_pipeline="1"
    }
  recv-data.0 = u32[2] get-tuple-element(recv-done.0), index=0
  recv-done.1 = (u32[2], token[]) recv-done(recv.1), channel_id=0,
  frontend_attributes={
      _xla_send_recv_pipeline="0"
    }
  recv-data.1 = u32[2] get-tuple-element(recv-done.1), index=0

  compare0 = pred[] compare(replica, c0), direction=EQ
  compare = pred[2] broadcast(compare0), dimensions={}
  recv-data = u32[2] select(compare, recv-data.0, recv-data.1)

  send-done.0 = token[] send-done(send.0), channel_id=0,
  frontend_attributes={
      _xla_send_recv_pipeline="1"
    }
  send-done.1 = token[] send-done(send.1), channel_id=0,
  frontend_attributes={
      _xla_send_recv_pipeline="0"
    }
  c1b = u32[2] broadcast(c1), dimensions={}
  ROOT result = u32[2] add(c1b, recv-data)
}
)";
  AssertionResult run_result =
      Run(std::move(ValueOrDie(ParseAndReturnVerifiedModule(kHloText))),
          /*run_hlo_passes=*/true);
  EXPECT_THAT(run_result.failure_message(),
              HasSubstr("Expected send and recv instructions to have "
                        "non-cyclical source-target pairs"));
}

TEST_F(GpuCompilerTest, RecordsStreamzStackTrace) {
  if (tsl::kIsOpenSource) {
    GTEST_SKIP() << "Streamz is not supported in OSS.";
  }

  const char* hlo_text = R"(
HloModule test

ENTRY main {
  p = f32[10]{0} parameter(0)
  ROOT neg = f32[10]{0} negate(p)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpaqueExecutable> executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/false));

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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpaqueExecutable> executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/false));
  ASSERT_OK_AND_ASSIGN(const HloModule* optimized_module,
                       test_runner().HloModuleFromWrapped(executable.get()));
  EXPECT_TRUE(
      XlaDebugInfoManager::Get()->TracksModule(optimized_module->unique_id()));
}

TEST_F(GpuCompilerTest, DoesNotGenerateDebugInfoForAutotuningCompilations) {
  const char* hlo_text = R"(
HloModule test

ENTRY main {
  p = f32[10]{0} parameter(0)
  ROOT neg = f32[10]{0} negate(p)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  int module_id = module->unique_id();
  Compiler::CompileOptions compile_options;
  compile_options.embed_hlo_module = false;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpaqueExecutable> executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/false));
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
}

TEST_F(GpuCompilerTest, CanRunScheduledModules) {
  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_disable_all_hlo_passes(true);
  config.set_debug_options(debug_options);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p = f32[2,4,4] parameter(0)
  ROOT _ = f32[2,4,4]{2,1,0} transpose(p), dimensions={0,2,1}
})",
                                                    config));

  config.set_debug_options(debug_options);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpaqueExecutable> executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/false));

  ASSERT_OK_AND_ASSIGN(const HloModule* compiled_module,
                       test_runner().HloModuleFromWrapped(executable.get()));
  const HloInstruction* entry_root =
      compiled_module->entry_computation()->root_instruction();
  EXPECT_THAT(entry_root, GmockMatch(m::Fusion()));
}

class PersistedAutotuningTest : public HloPjRtTestBase {
 protected:
  void SetUp() override {
    AutotunerUtil::ClearAutotuneResults();
    xla_gpu_dump_autotune_results_to_ = GetUniqueTempFilePath(".txt");
  }

  void TearDown() override { AutotunerUtil::ClearAutotuneResults(); }

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

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options =
        HloHardwareIndependentTestBase::GetDebugOptionsForTest();
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

  HloModuleConfig config = GetModuleConfigForTest();
  // Check that it writes the results on the first compilation.
  TF_EXPECT_OK(GetOptimizedModuleForExecutable(kHloText, config).status());
  {
    ASSERT_OK_AND_ASSIGN(std::string autotune_results_str,
                         ReadNonEmptyFile(xla_gpu_dump_autotune_results_to_));
    AutotuneResults results;
    EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(autotune_results_str,
                                                           &results));
  }

  // Overwrite results with an invalid textproto.
  tsl::Env* env = tsl::Env::Default();
  TF_EXPECT_OK(tsl::WriteStringToFile(env, xla_gpu_dump_autotune_results_to_,
                                      kInvalidTextProto));

  // Check that it writes the results on the second compilation.
  TF_EXPECT_OK(GetOptimizedModuleForExecutable(kHloText, config).status());
  {
    ASSERT_OK_AND_ASSIGN(std::string autotune_results_str,
                         ReadNonEmptyFile(xla_gpu_dump_autotune_results_to_));
    AutotuneResults results;
    EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(autotune_results_str,
                                                           &results));
  }
}

TEST_F(PersistedAutotuningTest, SingleOperationGetsAutotuned) {
  TF_EXPECT_OK(GetOptimizedModuleForExecutable(R"(
e {
  a = f32[64,128] parameter(0)
  t = f32[128,64] transpose(a), dimensions={1,0}
})",
                                               GetModuleConfigForTest())
                   .status());

  ASSERT_OK_AND_ASSIGN(std::string autotune_results_str,
                       ReadNonEmptyFile(xla_gpu_dump_autotune_results_to_));
  AutotuneResults results;
  EXPECT_TRUE(tsl::protobuf::TextFormat::ParseFromString(autotune_results_str,
                                                         &results));
  EXPECT_THAT(results.results(), Not(IsEmpty()));
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
        c0 = s32[] constant(1)
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
  ASSERT_OK_AND_ASSIGN(auto module_and_executable,
                       GetOptimizedModuleForExecutable(kHloString, config));
  const HloModule* module = module_and_executable.first;

  absl::string_view kExpected = R"(
    CHECK: all-reduce-start{{.*}}"is_pipelined":true
  )";
  HloPrintOptions options;
  options.set_print_operand_shape(false);
  options.set_print_result_shape(false);
  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
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
  HloModuleConfig config = GetModuleConfigForTest();
  auto& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_enable_analytical_sol_latency_estimator(false);
  ASSERT_OK_AND_ASSIGN(auto module_and_executable,
                       GetOptimizedModuleForExecutable(hlo_string, config));
  const HloModule* module = module_and_executable.first;

  const HloInstruction* root = module->entry_computation()->root_instruction();

  EXPECT_EQ(CountCopies(*module), 4);
  // Make sure that there is no copy of AllGatherDone.
  const HloInstruction* while_op =
      root->operand(0)->operand(0)->operand(0)->operand(0);
  EXPECT_EQ(while_op->while_body()->root_instruction()->operand(1)->opcode(),
            HloOpcode::kAllGatherDone);
}

class GpuCompilerTestWithAutotuneDb : public GpuCompilerTest {
 public:
  void SetUp() override {
    std::string path =
        tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service", "gpu",
                          "gpu_compiler_test_autotune_db.textproto");

    tsl::Env* env = tsl::Env::Default();
    std::string tmp_filepath = ::testing::TempDir();
    ASSERT_TRUE(env->CreateUniqueFileName(&tmp_filepath, ".textproto"));

    absl::Cleanup cleanup = [&] { CHECK_OK(env->DeleteFile(tmp_filepath)); };

    std::string contents;
    CHECK_OK(tsl::ReadFileToString(env, path, &contents));

    // The autotuning cache entries depend on the DNN library version, but this
    // is not relevant for these tests. Therefore we replace the DNN version
    // with the actual version of the DNN library so that the cache entries
    // match.
    stream_executor::SemanticVersion dnn_version =
        device_description().dnn_version();
    constexpr absl::string_view kCudnnVersionPlaceholder = "1.2.3";
    contents = absl::StrReplaceAll(
        contents, {{kCudnnVersionPlaceholder, dnn_version.ToString()}});

    TF_EXPECT_OK(tsl::WriteStringToFile(env, tmp_filepath, contents));
    AutotunerUtil::ClearAutotuneResults();
    TF_EXPECT_OK(AutotunerUtil::LoadAutotuneResultsFromFile(tmp_filepath));
  }

  static void TearDownTestSuite() { AutotunerUtil::ClearAutotuneResults(); }
};

TEST_F(GpuCompilerTestWithAutotuneDb,
       GemmFusionIsNoOpWhenGemmFusionAutotunerFallsBackToCublas) {
  if (!get_cuda_cc().IsAtLeastAmpere()) {
    GTEST_SKIP() << "Autotuning results have only been generated for Ampere "
                 << "and later GPUs";
  }
  if (get_cuda_cc().IsAtLeastBlackwell()) {
    // TODO(b/445172709): Re-enable once fixed.
    GTEST_SKIP();
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

  ASSERT_OK_AND_ASSIGN(auto triton_enabled_module_and_executable,
                       GetOptimizedModuleForExecutable(hlo_string, config));
  const HloModule* triton_enabled_module =
      triton_enabled_module_and_executable.first;
  DebugOptions triton_disabled_debug_options = GetDebugOptionsForTest();
  triton_disabled_debug_options.set_xla_gpu_enable_dynamic_slice_fusion(false);
  triton_disabled_debug_options.set_xla_gpu_enable_triton_gemm(false);
  config.set_debug_options(triton_disabled_debug_options);
  ASSERT_OK_AND_ASSIGN(auto triton_disabled_module_and_executable,
                       GetOptimizedModuleForExecutable(hlo_string, config));
  const HloModule* triton_disabled_module =
      triton_disabled_module_and_executable.first;
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
  if (!get_cuda_cc().IsAtLeastHopper()) {
    GTEST_SKIP()
        << "Autotuning results have only been generated for Hopper GPUs";
  }
  if (get_cuda_cc().IsAtLeastBlackwell()) {
    // TODO(b/445172709): Re-enable once fixed.
    GTEST_SKIP();
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

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> triton_enabled_module,
                       ParseAndReturnVerifiedModule(hlo_string, config));

  DebugOptions triton_disabled_debug_options = GetDebugOptionsForTest();
  triton_disabled_debug_options.set_xla_gpu_enable_triton_gemm(false);
  triton_disabled_debug_options.set_xla_gpu_cublas_fallback(true);
  config.set_debug_options(triton_disabled_debug_options);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> triton_disabled_module,
      ParseAndReturnVerifiedModule(hlo_string, config));

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
        std::make_pair(PrimitiveType::F8E5M2, PrimitiveType::F8E5M2)),
    [](const ::testing::TestParamInfo<FloatNormalizationTest::ParamType>&
           info) {
      return absl::StrCat(
          primitive_util::LowercasePrimitiveTypeName(info.param.first), "_",
          primitive_util::LowercasePrimitiveTypeName(info.param.second));
    });

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
                             bool enable_blas_fallback) {
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

    return GetOptimizedModuleForExecutable(module_str, config);
  };

  se::GpuComputeCapability gpu_cc =
      device_description().gpu_compute_capability();
  bool is_cuda = gpu_cc.IsCuda();
  se::CudaComputeCapability cuda_cc = get_cuda_cc();
  se::RocmComputeCapability rocm_cc =
      device_description().rocm_compute_capability();

  const std::string triton_keep_types = absl::Substitute(
      R"(CHECK: fusion($0{{[^)]*}}, $1{{[^)]*}}){{.*}}"kind":"{{__triton_gemm|__triton_nested_gemm_fusion}}")",
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
  if (is_cuda) {
    // Triton enabled, no fallback.
    ASSERT_OK_AND_ASSIGN(auto optimized_module_no_fallback_and_executable,
                         optimize_module(/*enable_triton=*/true,
                                         /*enable_blas=*/true,
                                         /*enable_blas_fallback=*/false));
    // Triton supports f8e4m3fn on Hopper and f8e5m2 on Ampere.
    const std::string triton_expected_check =
        (cuda_cc.IsAtLeastHopper() ||
         (cuda_cc.IsAtLeastAmpere() && lhs_type == F8E5M2 &&
          rhs_type == F8E5M2))
            ? triton_keep_types
            : cublas_convert_to_f16;
    ASSERT_OK_AND_ASSIGN(
        bool filecheck_matched,
        RunFileCheck(
            optimized_module_no_fallback_and_executable.first->ToString(
                print_options),
            triton_expected_check));
    EXPECT_TRUE(filecheck_matched);
  }

  {
    // Triton disabled, BLAS enabled.
    ASSERT_OK_AND_ASSIGN(auto optimized_module_no_triton_and_executable,
                         optimize_module(/*enable_triton=*/false,
                                         /*enable_blas=*/true,
                                         /*enable_blas_fallback=*/true));
    // cuBLASlt is only available on Hopper and it doesn't support
    // f8e5m2×f8e5m2.
    const std::string blas_expected_check =
        ((rocm_cc.has_ocp_fp8_support() || cuda_cc.IsAtLeastHopper()) &&
         !(lhs_type == F8E5M2 && rhs_type == F8E5M2))
            ? cublaslt_keep_types
            : cublas_convert_to_f16;

    ASSERT_OK_AND_ASSIGN(
        bool filecheck_matched,
        RunFileCheck(optimized_module_no_triton_and_executable.first->ToString(
                         print_options),
                     blas_expected_check));
    EXPECT_TRUE(filecheck_matched);
  }

  {
    // Neither Triton nor BLAS enabled, always fall back.
    ASSERT_OK_AND_ASSIGN(auto optimized_module_nothing_and_executable,
                         optimize_module(/*enable_triton=*/false,
                                         /*enable_blas=*/false,
                                         /*enable_blas_fallback=*/false));
    ASSERT_OK_AND_ASSIGN(
        bool filecheck_matched,
        RunFileCheck(optimized_module_nothing_and_executable.first->ToString(
                         print_options),
                     fallback_convert_to_f16));
    EXPECT_TRUE(filecheck_matched);
  }
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
  constexpr absl::string_view transpose_fusion_module = R"(
transpose {
  p0 = f32[1024,1024,1024] parameter(0)
  ROOT transpose = f32[1024,1024,1024] transpose(p0), dimensions={2,1,0}
}

ENTRY main {
  p0 = f32[1024,1024,1024] parameter(0)
  ROOT fusion = f32[1024,1024,1024] fusion(p0), kind=kLoop, calls=transpose
})";

  // Disable autotuning as this test is attempting to test a heuristic, but
  // autotuning tests both cases, and is not guaranteed to be deterministic.
  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options().set_xla_gpu_autotune_level(0);
  ASSERT_OK_AND_ASSIGN(
      auto optimized_module_and_executable,
      GetOptimizedModuleForExecutable(transpose_fusion_module, config));
  const HloModule* optimized_module = optimized_module_and_executable.first;

  if (get_cuda_cc().IsAtLeastAmpere()) {
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
  if (!get_cuda_cc().IsAtLeastAmpere()) {
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
  // Disable autotuning as this test is attempting to test a heuristic, but
  // autotuning tests both cases, and is not guaranteed to be deterministic.
  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options().set_xla_gpu_autotune_level(0);
  ASSERT_OK_AND_ASSIGN(
      auto rewritable_transpose_optimized_module_and_executable,
      GetOptimizedModuleForExecutable(rewritable_transpose_string, config));
  EXPECT_TRUE(HasBlockLevelFusionConfig(
      rewritable_transpose_optimized_module_and_executable.first
          ->entry_computation()
          ->root_instruction()));

  // The small minor dimension here is not a power of two, so the rewrite should
  // fail.
  constexpr absl::string_view unrewritable_transpose_string = R"(
ENTRY main {
  p0 = f32[1024,6144]{1,0} parameter(0)
  reshape = f32[1024,1024,6]{2,1,0} reshape(p0)
  ROOT transpose = f32[6,1024,1024]{2,1,0} transpose(reshape), dimensions={2,1,0}
})";

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> unrewritable_transpose_module,
      ParseAndReturnVerifiedModule(unrewritable_transpose_string));

  ASSERT_OK_AND_ASSIGN(
      auto unrewritable_transpose_module_and_executable,
      GetOptimizedModuleForExecutable(unrewritable_transpose_string, config));
  const HloModule* unrewritable_transpose_optimized_module =
      unrewritable_transpose_module_and_executable.first;
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
      ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
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
      ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> executable,
                           CreateExecutable(module->Clone(),
                                            /*run_hlo_passes=*/true));
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
    backend_config={
      "gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,
      "dot_dimension_numbers":
        {"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"]},
      "precision_config":{"operand_precision":["DEFAULT","DEFAULT"]},
      "lhs_stride":"1024","rhs_stride":"1024"}}
  ROOT get-tuple-element = f32[32,32] get-tuple-element(gemm), index=0
}

ENTRY main {
  p0 = f32[32,32] parameter(0)
  p1 = f32[32,32] parameter(1)
  call-start = ((f32[32,32], f32[32,32]), f32[32,32]) call-start(p0, p1),
    to_apply=async_call,
    frontend_attributes={_xla_stream_annotation="1"}
  ROOT call-done = f32[32,32]{1,0} call-done(call-start),
    frontend_attributes={_xla_stream_annotation="1"}
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));

  Compiler::CompileOptions compile_options;
  compile_options.gpu_topology =
      GetSingleDeviceGpuTopology(/*platform_version=*/"", gpu_target_config());
  compile_options.early_exit_with_layouts = false;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      compiler()->RunBackend(std::move(module), /*executor=*/nullptr,
                             compile_options));
  std::unique_ptr<GpuExecutable> gpu_exec(
      static_cast<GpuExecutable*>(executable.release()));

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

TEST_F(GpuCompilerTest, StreamAnnotationThunkTestFDO) {
  constexpr absl::string_view hlo_text = R"(
HloModule composite

async_call {
  p0 = f32[32,32] parameter(0)
  p1 = f32[32,32] parameter(1)
  gemm = (f32[32,32], s8[8192]) custom-call(p0, p1), custom_call_target="__cublas$gemm",
    backend_config={
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
    to_apply=async_call,
    frontend_attributes={_xla_stream_annotation="1"}
  ROOT call-done = f32[32,32]{1,0} call-done(call-start),
    frontend_attributes={_xla_stream_annotation="1"}
})";

  const absl::string_view fdo_profile = R"pb(
    costs { name: "cp" cost_us: 100.0 }
  )pb";

  HloModuleConfig config = GetModuleConfigForTest(1, 1);  // Default values.
  config.set_fdo_profile(fdo_profile);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text, config));

  Compiler::CompileOptions compile_options;
  compile_options.gpu_topology =
      GetSingleDeviceGpuTopology(/*platform_version=*/"", gpu_target_config());
  compile_options.early_exit_with_layouts = false;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      compiler()->RunBackend(std::move(module), /*executor=*/nullptr,
                             compile_options));
  std::unique_ptr<GpuExecutable> gpu_exec(
      static_cast<GpuExecutable*>(executable.release()));

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
  bool is_rocm = device_description().gpu_compute_capability().IsRocm();

  bool expect_triton_gemm_rewriter_has_run =
      get_cuda_cc().IsAtLeastAmpere() || is_rocm;

  constexpr absl::string_view constant_module = R"(
HloModule noop

ENTRY main {
  ROOT constant = f32[] constant(0)
})";

  HloModuleConfig config = GetModuleConfigForTest();
  ASSERT_OK_AND_ASSIGN(
      auto optimized_module_and_executable,
      GetOptimizedModuleForExecutable(constant_module, config));
  const HloModule* optimized_module = optimized_module_and_executable.first;
  const HloModuleMetadataProto& module_metadata =
      optimized_module->metadata().proto();

  bool triton_gemm_rewriter_has_run = false;
  for (const HloPassMetadata& pass_metadata : module_metadata.pass_metadata()) {
    triton_gemm_rewriter_has_run |=
        pass_metadata.pass_name() == "triton-gemm-rewriter";
  }

  EXPECT_EQ(triton_gemm_rewriter_has_run, expect_triton_gemm_rewriter_has_run);
}

TEST_F(GpuCompilerPassTest,
       GpuCompilerRunsCustomKernelFusionByDefaultFromVolta) {
  bool expect_custom_kernel_fusion_rewriter_has_run =
      get_cuda_cc().major == se::CudaComputeCapability::kVolta;

  constexpr absl::string_view constant_module = R"(
HloModule noop

ENTRY main {
  ROOT constant = f32[] constant(0)
})";

  HloModuleConfig config = GetModuleConfigForTest();
  ASSERT_OK_AND_ASSIGN(
      auto optimized_module_and_executable,
      GetOptimizedModuleForExecutable(constant_module, config));
  const HloModule* optimized_module = optimized_module_and_executable.first;
  const HloModuleMetadataProto& module_metadata =
      optimized_module->metadata().proto();

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
         optimized_module_->metadata().proto().pass_metadata()) {
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
         optimized_module_->metadata().proto().pass_metadata()) {
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
    CHECK(optimized_module_);
    int run_index = 0;
    for (const HloPassMetadata& pass_metadata :
         optimized_module_->metadata().proto().pass_metadata()) {
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
  // Compiles a dummy module with the given configuration, running all passes,
  // including the ones in RunBackend. This is important because otherwise, we
  // might miss some passes when verifying pass order.
  void CompileModule(const HloModuleConfig& config) {
    constexpr absl::string_view constant_module = R"(
        ENTRY main {
          ROOT constant = f32[] constant(0)
        })";
    ASSERT_OK_AND_ASSIGN(
        std::tie(optimized_module_, compiled_executable_),
        GetOptimizedModuleForExecutable(constant_module, config));
  }

  // Owns the optimized_module_ below.
  std::unique_ptr<OpaqueExecutable> compiled_executable_ = nullptr;
  const HloModule* optimized_module_ = nullptr;
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
}

TEST_F(PassOrderTest, FusionDispatchRunsAfterAllFusionPasses) {
  if (!get_cuda_cc().IsAtLeastAmpere()) {
    GTEST_SKIP() << "fusion-dispatch requires Ampere+ to run.";
  }

  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_enable_fusion_block_level_rewriter(
      true);
  SetDebugOptions(debug_options);

  VerifyPassOrder(
      /*first_pass_regex=*/".*(fusion|stream-attribute-annotator).*",
      /*last_pass_regex=*/"fusion-dispatch-pipeline.*",
      /*include_pipeline_name=*/true);
}

TEST_F(PassOrderTest,
       SortRewriterRunsBeforeStableSortExpanderAndComparisonExpander) {
  VerifyPassOrder(/*first_pass_regex=*/"sort-rewriter",
                  /*last_pass_regex=*/"stable-sort-expander");
  VerifyPassRunsAtLeastOnceBefore(
      /*first_pass_regex=*/"sort-rewriter",
      /*other_pass_regex=*/"comparison-expander");
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
  HloModuleMetadataProto metadata = arg->metadata().proto();
  run_pass_names.reserve(metadata.pass_metadata_size());
  for (auto& pass_metadata : metadata.pass_metadata()) {
    run_pass_names.push_back(pass_metadata.pass_name());
  }
  return Matches(IsSupersetOf(expected_pass_names))(run_pass_names);
}

TEST_F(PassOrderTest, ExecEffortAt0point2RunsSpecifiedPasses) {
  HloModuleConfig config = GetModuleConfigForTest();
  CompileModule(config);

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
  EXPECT_THAT(optimized_module_, HasExpectedPasses(kExpectedPasses));
}

TEST_F(PassOrderTest, LHSRunsIfProfileDataIsAvailable) {
  HloModuleConfig config = GetModuleConfigForTest();

  // Make sure LHS is off by default.
  std::vector<std::string> kExpectedPasses = {
      "latency-hiding-scheduler",
  };
  CompileModule(config);

  // Make sure we turn the LHS on with we schedule with profile data.
  const absl::string_view kProfile = R"pb(
    costs { name: "cp" cost_us: 100.0 }
  )pb";
  config.set_fdo_profile(kProfile);
  CompileModule(config);

  EXPECT_THAT(optimized_module_, HasExpectedPasses(kExpectedPasses));
}

TEST_F(PassOrderTest, GemmFusionRunsAfterDotNormalizer) {
  if (!get_cuda_cc().IsAtLeastAmpere()) {
    GTEST_SKIP() << "GemmFusion requires Ampere+ to run.";
  }
  DebugOptions options = GetDebugOptionsForTest();
  options.set_xla_gpu_enable_triton_gemm(true);
  SetDebugOptions(options);
  PassRange pass_range = VerifyPassOrder(
      /*first_pass_regex=*/"dot_normalizer",
      /*last_pass_regex=*/"triton-gemm-rewriter");
  VerifyNotRunInBetween(pass_range, /*pass_regex=*/"algsimp");
}

TEST_F(PassOrderTest, GemmRewriterRunsAfterDotNormalizer) {
  PassRange pass_range = VerifyPassOrder(
      /*first_pass_regex=*/"dot_normalizer",
      /*last_pass_regex=*/"cublas-gemm-rewriter");
  VerifyNotRunInBetween(pass_range, /*pass_regex=*/"algsimp");
}

TEST_F(PassOrderTest, HoistFusedBitcastsRunsAfterAutotuner) {
  VerifyPassRunsAtLeastOnceBefore("autotuner", "hoist-fused-bitcasts");
}

TEST_F(PassOrderTest, NestGemmFusionRunsAfterHoistFusedBitcasts) {
  // NestGemmFusion expect to see __triton_gemm custom call with a backend
  // config created by gemm_fusion_autotuner.
  VerifyPassOrder("hoist-fused-bitcasts", "nest_gemm_fusion");
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
class FixPointTest : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase> {
 public:
  void ExpectPipelinesReachFixedPoint(absl::string_view module_text) {
    ASSERT_OK_AND_ASSIGN(
        auto optimized_module_and_executable,
        GetOptimizedModuleForExecutable(module_text, GetModuleConfigForTest()));
    const HloModule* optimized_module = optimized_module_and_executable.first;

    std::string last_pipeline_name;
    int count = 0;
    for (const HloPassMetadata& pass_metadata :
         optimized_module->metadata().proto().pass_metadata()) {
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
  HloModuleConfig config = GetModuleConfigForTest();
  config.set_replica_count(2);
  xla::DeviceAssignment device_assignment(2, 1);
  device_assignment(0, 0) = 0;
  device_assignment(1, 0) = 1;
  config.set_static_device_assignment(device_assignment);
  config.mutable_debug_options().set_xla_gpu_shard_autotuning(false);
  config.mutable_debug_options().set_xla_gpu_enable_dynamic_slice_fusion(true);
  ASSERT_OK_AND_ASSIGN(auto optimized_module_and_executable,
                       GetOptimizedModuleForExecutable(hlo, config));
  const HloModule* optimized_module = optimized_module_and_executable.first;
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
  EXPECT_THAT(RunFileCheck(
                  optimized_module->ToString(HloPrintOptions{}
                                                 .set_print_operand_shape(false)
                                                 .set_print_metadata(false)),
                  kExpected),
              absl_testing::IsOkAndHolds(true));

  if (test_runner().device_count() < 2) {
    GTEST_SKIP() << "Skipping test as it requires at least 2 devices.";
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                       ParseAndReturnVerifiedModule(hlo, config));
  HloModuleConfig reference_config = config;
  reference_config.mutable_debug_options()
      .set_xla_gpu_enable_dynamic_slice_fusion(false);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_ref,
                       ParseAndReturnVerifiedModule(hlo, reference_config));
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
  HloModuleConfig config = GetModuleConfigForTest();
  config.set_replica_count(2);
  xla::DeviceAssignment device_assignment(2, 1);
  device_assignment(0, 0) = 0;
  device_assignment(1, 0) = 1;
  config.set_static_device_assignment(device_assignment);
  config.mutable_debug_options().set_xla_gpu_shard_autotuning(false);
  config.mutable_debug_options().set_xla_gpu_enable_dynamic_slice_fusion(true);
  ASSERT_OK_AND_ASSIGN(auto module_and_executable,
                       GetOptimizedModuleForExecutable(hlo, config));
  const HloModule* module = module_and_executable.first;
  const char* kExpected = R"(
    // CHECK: dynamic-slice-fusion{{.*}} {
    // CHECK-DAG: %[[slice1:.+]] = {{.+}} slice({{.+}}), slice={[4:8], [0:32]}
    // CHECK-DAG: %[[slice2:.+]] = {{.+}} slice({{.+}}), slice={[0:4], [0:32]}
    // CHECK-DAG: ROOT %[[rs:.+]] = {{.+}} reduce-scatter(%[[slice1]], %[[slice2]]),
    // CHECK-SAME{LITERAL}:                                      replica_groups={{0,1}}, dimensions={0}, to_apply=%add
    // CHECK: ENTRY
  )";
  EXPECT_THAT(RunFileCheck(module->ToString(), kExpected),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(GpuCompilerTest, CompilingSortsWorksWithoutDevice) {
  constexpr absl::string_view kHlo = R"(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[1000] parameter(0)
  ROOT %sort = f32[1000] sort(%input), dimensions={0}, to_apply=%compare
})";

  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_enable_cub_radix_sort(true);

  std::string target_file;
  ASSERT_TRUE(tsl::Env::Default()->LocalTempFilename(&target_file));
  TF_ASSERT_OK(tsl::WriteTextProto(tsl::Env::Default(), target_file,
                                   gpu_target_config().ToProto()));
  debug_options.set_xla_gpu_target_config_filename(target_file);
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo, config));

  absl::ScopedMockLog mock_log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(mock_log,
              Log(absl::LogSeverity::kWarning, EndsWith("/gpu_compiler.cc"),
                  StartsWith("Using fallback sort algorithm")))
      .Times(1);

  // StartCapturingLogs has to be called even if we expect not to capture any
  // logs.
  mock_log.StartCapturingLogs();
  TF_ASSERT_OK(compiler()->RunHloPasses(std::move(module), nullptr, nullptr));
}

TEST_F(GpuCompilerTest, CompilingAndCollectingMetadata) {
  constexpr absl::string_view kHlo = R"(
    HloModule cluster

    ENTRY main {
      cst = f32[1]{0} constant({0})
      ROOT tuple_out = (f32[1]{0}, f32[1]{0}, f32[1]{0}, f32[1]{0}) tuple(cst, cst, cst, cst)
    }
  )";

  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();

  std::string target_file;
  ASSERT_TRUE(tsl::Env::Default()->LocalTempFilename(&target_file));
  TF_ASSERT_OK(tsl::WriteTextProto(tsl::Env::Default(), target_file,
                                   gpu_target_config().ToProto()));
  debug_options.set_xla_gpu_target_config_filename(target_file);
  config.set_debug_options(debug_options);
  ASSERT_OK_AND_ASSIGN(auto exe_module_and_executable,
                       GetOptimizedModuleForExecutable(kHlo, config));
  const HloModule* exe_module = exe_module_and_executable.first;
  const HloModuleMetadataProto& exe_metadata = exe_module->metadata().proto();
  for (int pass = 0; pass < exe_metadata.pass_metadata().size(); pass++) {
    const HloPassMetadata& pass_metadata = exe_metadata.pass_metadata(pass);
    EXPECT_NE(pass_metadata.pass_id(), 0);
    EXPECT_FALSE(pass_metadata.pass_name().empty());
    EXPECT_FALSE(pass_metadata.pipeline_name().empty());
    EXPECT_EQ(pass_metadata.module_id(), exe_module->unique_id());
    EXPECT_GT(pass_metadata.start_timestamp_usec(), 0);
    EXPECT_LE(pass_metadata.start_timestamp_usec(),
              pass_metadata.end_timestamp_usec());
  }
}

TEST_F(GpuCompilerTest, CommandBufferConversionPassRuns) {
  const char* hlo_text = R"(
HloModule test

ENTRY main {
  a = f32[2,2] parameter(0)
  b = f32[2,2] parameter(1)
  ROOT dot = f32[2,2] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  auto hlo_module = ParseAndReturnVerifiedModule(hlo_text).value();

  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  debug_options.set_xla_gpu_graph_min_graph_size(1);

  hlo_module->mutable_config().set_debug_options(debug_options);

  Compiler::CompileOptions compile_options;
  compile_options.gpu_topology =
      GetSingleDeviceGpuTopology(/*platform_version=*/"", gpu_target_config());
  compile_options.early_exit_with_layouts = false;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      compiler()->RunBackend(std::move(hlo_module), /*executor=*/nullptr,
                             compile_options));
  std::unique_ptr<GpuExecutable> gpu_exec(
      static_cast<GpuExecutable*>(executable.release()));
  const ThunkSequence& thunks = gpu_exec->GetThunk().thunks();
  ASSERT_EQ(thunks.size(), 1);
  EXPECT_EQ(thunks[0]->kind(), Thunk::Kind::kCommandBuffer);
}

TEST_F(GpuCompilerTest, NoCudnnVectorizationOnHopperAndBeyond) {
  bool is_hopper_or_beyond = get_cuda_cc().IsAtLeastHopper();

  constexpr absl::string_view kHlo = R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = f32[10,20,30,64] parameter(0)
    filter = f32[2,2,64,64] parameter(1)
    ROOT result = f32[10,19,29,64] convolution(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f
  })";

  ASSERT_OK_AND_ASSIGN(
      auto optimized_module_and_executable,
      GetOptimizedModuleForExecutable(kHlo, GetModuleConfigForTest()));
  const HloModule* optimized_module = optimized_module_and_executable.first;

  constexpr absl::string_view kVectorizationdExpected = R"(
    CHECK: (f32[10,64,19,29]{3,2,1,0}, u8[{{[0-9]*}}]{0}) custom-call
  )";
  constexpr absl::string_view kNoVectorizationExpected = R"(
    CHECK: (f32[10,19,29,64]{3,2,1,0}, u8[{{[0-9]*}}]{0}) custom-call
  )";
  absl::string_view expected =
      is_hopper_or_beyond ? kNoVectorizationExpected : kVectorizationdExpected;

  EXPECT_THAT(RunFileCheck(optimized_module->ToString(), expected),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(GpuCompilerTest, BitcastConvertSimplificationToBitcastIsValid) {
  const std::string kHloText = R"(
m {
  a = s4[3,5,2]{2,1,0} parameter(0)
  b = s8[3,5]{1,0} bitcast-convert(a)
  c = s8[3,5]{1,0} copy(b)
})";

  ASSERT_OK_AND_ASSIGN(
      auto optimized_module_and_executable,
      GetOptimizedModuleForExecutable(kHloText, GetModuleConfigForTest()));
  const HloModule* optimized_module = optimized_module_and_executable.first;
  OpaqueExecutable* executable = optimized_module_and_executable.second.get();
  EXPECT_THAT(optimized_module->entry_computation()->root_instruction(),
              GmockMatch(m::Copy(m::Bitcast(m::Parameter()))));

  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.add_xla_disable_hlo_passes("algsimp");
  config.set_debug_options(debug_options);
  ASSERT_OK_AND_ASSIGN(auto ref_module_and_ref_executable,
                       GetOptimizedModuleForExecutable(kHloText, config));
  const HloModule* ref_module = ref_module_and_ref_executable.first;
  OpaqueExecutable* ref_executable = ref_module_and_ref_executable.second.get();
  EXPECT_THAT(ref_module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));

  EXPECT_TRUE(
      RunAndCompareTwoExecutables(executable, ref_executable, std::nullopt));
}

// Define a test-specific enum for expected TopK implementations.
enum class TopKImpl {
  kCustomKernel,  // Custom GPU kernel
  kSelectK,       // raft::select_k
  kSort           // Fallback Sort+Slice
};

// Test fixture for verifying GPU TopK lowering to SelectK or custom kernel.
class GpuCompilerSelectKTest
    : public GpuCompilerTest,
      public ::testing::WithParamInterface<std::tuple<int, int, TopKImpl>> {};

// Test lowering of TopK to different GPU implementations
// (CustomKernel, raft::select_k, or Sort+Slice (LLVM/CUBSort)).
TEST_P(GpuCompilerSelectKTest, SelectKOrCustomKernelThunk) {
  auto [n, k, expected_impl] = GetParam();

  bool is_rocm = device_description().gpu_compute_capability().IsRocm();

  if (is_rocm && expected_impl == TopKImpl::kSelectK) {
    GTEST_SKIP() << "raft::select_k is not supported in ROCm.";
  }
  // Generate HLO text with parameters substituted.
  std::string hlo_text = absl::Substitute(R"(
HloModule m

ENTRY main {
  p = f32[8,$0]{1,0} parameter(0)
  ROOT t = (f32[8,$1]{1,0}, s32[8,$1]{1,0}) topk(p), k=$1, largest=true
}
)",
                                          n, k);

  // Configure module with debug options for experimental raft select_k.
  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_experimental_use_raft_select_k(true);
  config.set_debug_options(debug_options);

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_text, config));

  Compiler::CompileOptions compile_options;
  compile_options.gpu_topology =
      GetSingleDeviceGpuTopology(/*platform_version=*/"", gpu_target_config());
  compile_options.early_exit_with_layouts = false;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> compiled_module,
      compiler()->RunHloPasses(module->Clone(), /*executor=*/nullptr,
                               compile_options));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      compiler()->RunBackend(std::move(compiled_module), /*executor=*/nullptr,
                             compile_options));

  // Downcast to GPU executable
  xla::gpu::GpuExecutable* gpu_executable =
      tensorflow::down_cast<xla::gpu::GpuExecutable*>(executable.get());
  ASSERT_NE(gpu_executable, nullptr);

  // Get the thunk sequence and check its size and type
  const SequentialThunk& seq_thunk = gpu_executable->GetThunk();
  std::vector<Thunk::Kind> kinds;
  kinds.reserve(seq_thunk.thunks().size());
  for (const auto& thunk : seq_thunk.thunks()) {
    kinds.push_back(thunk->kind());
  }

  using ::testing::ElementsAre;

  switch (expected_impl) {
    case TopKImpl::kCustomKernel:
      EXPECT_THAT(kinds, ElementsAre(Thunk::Kind::kCustomKernel));
      break;

    case TopKImpl::kSelectK:
      EXPECT_THAT(kinds, ElementsAre(Thunk::Kind::kSelectK));
      break;

    case TopKImpl::kSort: {
      if (kinds.size() == 1) {
        // LLVM
        EXPECT_THAT(kinds, ElementsAre(Thunk::Kind::kCommandBuffer));
      } else if (kinds.size() == 4) {
        // CUBSort
        EXPECT_THAT(kinds,
                    ElementsAre(Thunk::Kind::kKernel, Thunk::Kind::kCubSort,
                                Thunk::Kind::kKernel, Thunk::Kind::kKernel));
      } else {
        FAIL() << "Unexpected thunk sequence size: " << kinds.size();
      }
      break;
    }

    default:
      FAIL() << "Unexpected TopKImpl: " << static_cast<int>(expected_impl);
  }
}

auto SelectKTestParams() {
  // Depending on N and K, XLA chooses different TopK implementations:
  // CustomKernel, raft::select_k, or Sort+Slice.
  // The heuristic for selecting between TopK CustomKernel and
  // raft::matrix::select_k was developed as part of the initial research
  // described in b/409009349.
  return ::testing::Values(std::make_tuple(1023, 4, TopKImpl::kSelectK),
                           std::make_tuple(1024, 4, TopKImpl::kCustomKernel),
                           std::make_tuple(1024, 16, TopKImpl::kSelectK),
                           std::make_tuple(8192, 24, TopKImpl::kSelectK),
                           std::make_tuple(8192, 512, TopKImpl::kSort));
}
// Instantiate the test suite with (n, k, expected_kind) pairs.
INSTANTIATE_TEST_SUITE_P(SelectKOrCustomKernel, GpuCompilerSelectKTest,
                         SelectKTestParams());
}  // namespace

XLA_FFI_DEFINE_HANDLER(
    kAttemptsToAcquireLockInstantiate,
    []() {
      xla::llvm_ir::LLVMCommandLineOptionsLock lock(
          /*client_options=*/{"--frame-pointer=all"});
      return absl::OkStatus();
    },
    ffi::Ffi::BindInstantiate());
XLA_FFI_DEFINE_HANDLER(
    kAttemptsToAcquireLock,
    [](se::Stream* stream, ffi::AnyBuffer, ffi::Result<ffi::AnyBuffer> result) {
      constexpr int32_t kReturnValue = 42;
      se::DeviceAddressBase device_memory = result->device_memory();
      return stream->Memset32(&device_memory, kReturnValue,
                              sizeof(kReturnValue));
    },
    ffi::Ffi::Bind()
        .Ctx<ffi::Stream>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());

TEST_F(GpuCompilerTest, GlobalLLVMLockGetsReleasedForCustomCallThunkCreation) {
  XLA_FFI_Handler_Bundle bundle = {
      /*instantiate=*/kAttemptsToAcquireLockInstantiate,
      /*prepare=*/nullptr,
      /*initialize=*/nullptr,
      /*execute=*/kAttemptsToAcquireLock,
  };
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(),
                                       "xla.gpu.acquire_lock", "CUDA", bundle);
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(),
                                       "xla.gpu.acquire_lock", "ROCM", bundle);
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(),
                                       "xla.gpu.acquire_lock", "SYCL", bundle);

  constexpr absl::string_view kModuleStr = R"hlo(
  HloModule test
  ENTRY test_computation {
    p = s32[] parameter(0)
    ROOT v = s32[] custom-call(p), custom_call_target="xla.gpu.acquire_lock", api_version=API_VERSION_TYPED_FFI
  })hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kModuleStr));
  Literal input = LiteralUtil::Zero(S32);
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&input},
                                               /*run_hlo_passes=*/true));
  // Checking the result ensures that the custom call thunk was executed.
  EXPECT_EQ(result.GetLinear<int32_t>(0), 42);
}
}  // namespace gpu
}  // namespace xla

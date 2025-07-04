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

#include "xla/service/gpu/autotuning/conv_algorithm_picker.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "google/protobuf/duration.pb.h"
#include <gmock/gmock.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/autotune_results.pb.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/transforms/conv_rewriter.h"
#include "xla/service/gpu/transforms/cudnn_fused_conv_rewriter.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {
using ::testing::_;
using ::testing::AtLeast;
using ::testing::Eq;
using ::testing::Ne;
using ::testing::Property;
using ::tsl::testing::IsOkAndHolds;

namespace m = ::xla::match;

class MockGpuConvAlgorithmPicker : public GpuConvAlgorithmPicker {
 public:
  explicit MockGpuConvAlgorithmPicker(AutotuneConfig config)
      : GpuConvAlgorithmPicker(config) {}
  // absl::StatusOr<AutotuneResult> AutotuneOneConvRunner(
  //     GenericConvRunner* runner,
  //     std::optional<ReferenceResult>* reference_result,
  //     absl::Span<const stream_executor::dnn::AlgorithmDesc> disabled_algos,
  //     absl::string_view instr_str,
  //     const AutotuneRuntimeArguments& runtime_arguments) override;

  MOCK_METHOD(
      absl::StatusOr<AutotuneResult>, AutotuneOneConvRunner,
      (GenericConvRunner * runner,
       std::optional<ReferenceResult>* reference_result,
       absl::Span<const stream_executor::dnn::AlgorithmDesc> disabled_algos,
       absl::string_view instr_str,
       const AutotuneRuntimeArguments& runtime_arguments),
      (override));
};

class GpuConvAlgorithmPickerTest : public HloTestBase {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
  stream_executor::dnn::VersionInfo GetDnnVersion() {
    return GetDnnVersionInfoOrDefault(backend().default_stream_executor());
  }

  GpuConvAlgorithmPickerTest() { AutotunerUtil::ClearAutotuneResults(); }
};

TEST_F(GpuConvAlgorithmPickerTest, SetAlgorithm) {
  constexpr absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  %arg0 = f32[3,56,56,16]{2,1,0,3} parameter(0)
  %arg1 = f32[3,3,3,64]{2,1,0,3} parameter(1)
  ROOT %conv = f32[54,54,16,64]{1,0,3,2} convolution(%arg0, %arg1), window={size=3x3}, dim_labels=f01b_i01o->01bf
})";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHlo));

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                          PlatformUtil::GetStreamExecutors(platform));
  ASSERT_GT(executors.size(), 0);
  se::StreamExecutor* stream_exec = executors[0];

  const se::GpuComputeCapability& cc = backend()
                                           .default_stream_executor()
                                           ->GetDeviceDescription()
                                           .gpu_compute_capability();
  bool changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(ConvRewriter(cc), m.get()));
  changed = false;
  DebugOptions opts = DefaultDebugOptionsIgnoringFlags();

  AutotuneConfig cfg = AutotuneConfig::FromDebugOptions(
      DeviceOrDevicelessConfig{DeviceConfig{stream_exec, nullptr}}, opts);
  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GpuConvAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  AutotuneResults results;
  TF_ASSERT_OK(AutotunerUtil::SerializeAutotuneResults(&results));
  ASSERT_EQ(results.results_size(), 1);
  auto& result = *results.mutable_results(0)->mutable_result();
  int64_t old_scratch_bytes = result.scratch_bytes();
  int64_t new_scratch_bytes = old_scratch_bytes + 1;
  result.set_scratch_bytes(new_scratch_bytes);

  AutotunerUtil::ClearAutotuneResults();
  TF_ASSERT_OK(AutotunerUtil::LoadAutotuneResults(results));

  // Now send the same module through GpuConvAlgorithmPicker again.  The conv
  // should have the new scratch bytes.
  TF_ASSERT_OK_AND_ASSIGN(m, ParseAndReturnVerifiedModule(kHlo));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(ConvRewriter(cc), m.get()));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GpuConvAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  // TupleSimplifier cleans this up a bit before we pattern-match
  TF_ASSERT_OK(RunHloPass(TupleSimplifier(), m.get()).status());

  SCOPED_TRACE(m->ToString());
  HloInstruction* conv;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(m::CustomCall(&conv))));
  EXPECT_THAT(
      conv->shape(),
      GmockMatch(m::Shape().WithSubshape(
          {1}, m::Shape().WithElementType(U8).WithDims({new_scratch_bytes}))));

  // Algorithm 14 is disabled for cuDNN 9 on V100
  TF_ASSERT_OK_AND_ASSIGN(auto dnn_version, GetDnnVersionInfo(stream_exec));
  if (dnn_version.major_version() >= 9 && dnn_version.major_version() < 10 &&
      std::holds_alternative<stream_executor::CudaComputeCapability>(cc) &&
      std::get<stream_executor::CudaComputeCapability>(cc).major == 7 &&
      std::get<stream_executor::CudaComputeCapability>(cc).minor == 0) {
    EXPECT_TRUE(conv->backend_config<GpuBackendConfig>()
                    ->has_cudnn_conv_backend_config() &&
                conv->backend_config<GpuBackendConfig>()
                        ->cudnn_conv_backend_config()
                        .algorithm()
                        .algo_id() != 14);
  }
}

TEST_F(GpuConvAlgorithmPickerTest, SetAlgorithmGraphConvF8) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::kHopper)) {
    GTEST_SKIP() << "FP8 convolutions require Hopper or newer architecture.";
  }
  constexpr absl::string_view kHlo = R"(
HloModule module
apply {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT c = f32[] maximum(a, b)
}
ENTRY main {
  input = f8e4m3fn[1,6,6,128] parameter(0)
  filter = f8e4m3fn[16,3,3,128] parameter(1)
  input_scale = f32[] parameter(2)
  input_scale_bcast = f32[1,6,6,128] broadcast(input_scale), dimensions={}
  filter_scale = f32[] parameter(3)
  filter_scale_bcast = f32[16,3,3,128] broadcast(filter_scale), dimensions={}
  input_f32 = f32[1,6,6,128] convert(input)
  input_unscaled = f32[1,6,6,128] multiply(input_f32, input_scale_bcast)
  filter_f32 = f32[16,3,3,128] convert(filter)
  filter_unscaled = f32[16,3,3,128] multiply(filter_f32, filter_scale_bcast)
  conv_a = f32[1,6,6,16] convolution(input_unscaled, filter_unscaled), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, feature_group_count=1
  z_scale = f32[] parameter(4)
  z_scale_bcast = f32[1,6,6,16] broadcast(z_scale), dimensions={}
  conv_a_scaled = f32[1,6,6,16] multiply(conv_a, z_scale_bcast)
  c1 = f32[] constant(-448.)
  c1_bcast = f32[1,6,6,16] broadcast(c1), dimensions={}
  c2 = f32[] constant(448.)
  c2_bcast = f32[1,6,6,16] broadcast(c2), dimensions={}
  conv_a_clamped = f32[1,6,6,16] clamp(c1_bcast, conv_a_scaled, c2_bcast)
  conv_a_clamped_f8 = f8e4m3fn[1,6,6,16] convert(conv_a_clamped)
  abs_conv_a = f32[1,6,6,16] abs(conv_a)
  c0 = f32[] constant(-inf)
  amax = f32[] reduce(abs_conv_a, c0), dimensions={0,1,2,3}, to_apply=apply
  ROOT conv_f8 = (f8e4m3fn[1,6,6,16], f32[]) tuple(conv_a_clamped_f8, amax)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHlo));

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                          PlatformUtil::GetStreamExecutors(platform));
  ASSERT_GT(executors.size(), 0);
  se::StreamExecutor* stream_exec = executors[0];

  const se::GpuComputeCapability& cc = GetCudaComputeCapability();
  bool changed;
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(ConvRewriter(cc), m.get()));
  ASSERT_TRUE(changed);
  TF_ASSERT_OK_AND_ASSIGN(
      changed,
      RunHloPass(CudnnFusedConvRewriter(
                     GetCudaComputeCapability(), GetDnnVersion(),
                     stream_exec->GetDeviceDescription().runtime_version()),
                 m.get()));
  ASSERT_TRUE(changed);

  DebugOptions opts = DefaultDebugOptionsIgnoringFlags();
  AutotuneConfig cfg = AutotuneConfig::FromDebugOptions(
      DeviceOrDevicelessConfig{DeviceConfig{stream_exec, nullptr}}, opts);
  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GpuConvAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);
}

TEST_F(GpuConvAlgorithmPickerTest, DoesntTryEngine0IfAlternativeIsAvailable) {
  if (GetCudaComputeCapability().ToPair() == std::make_pair(0, 0)) {
    GTEST_SKIP() << "This test only makes sense for cuDNN.";
  }

  constexpr absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  %arg0 = f32[7,2500,3072]{2,1,0} parameter(0)
  %arg1 = f32[3072,513,512]{2,1,0} parameter(1)
  %conv = (f32[7,2500,3072]{2,1,0}, u8[0]{0}) custom-call(%arg0, %arg1), window={size=513 pad=256_256}, dim_labels=b0f_o0i->b0f, feature_group_count=6, custom_call_target="__cudnn$convForward", backend_config={"cudnn_conv_backend_config":{"activation_mode":"kNone", "conv_result_scale":1, "leakyrelu_alpha":0, "side_input_scale":0}, "force_earliest_schedule":false, "operation_queue_id":"0", "reification_cost":[], "wait_on_operation_queues":[]}
  ROOT %conv_result = f32[7,2500,3072]{2,1,0} get-tuple-element(%conv), index=0
})";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHlo));

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                          PlatformUtil::GetStreamExecutors(platform));
  ASSERT_GT(executors.size(), 0);
  se::StreamExecutor* stream_exec = executors[0];

  MockGpuConvAlgorithmPicker algorithm_picker(AutotuneConfig::FromDebugOptions(
      DeviceOrDevicelessConfig{DeviceConfig{stream_exec, nullptr}},
      DefaultDebugOptionsIgnoringFlags()));

  // We expect that algorithm 0 is not autotuned.
  EXPECT_CALL(
      algorithm_picker,
      AutotuneOneConvRunner(
          Property(
              &GenericConvRunner::ToAlgorithmDesc,
              Property(&stream_executor::dnn::AlgorithmDesc::algo_id, Eq(0))),
          _, _, _, _))
      .Times(0);

  AutotuneResult generic_result;
  generic_result.mutable_run_time()->set_seconds(1);

  // In all other cases (algo_id != 0), we return some generic result.
  EXPECT_CALL(
      algorithm_picker,
      AutotuneOneConvRunner(
          Property(
              &GenericConvRunner::ToAlgorithmDesc,
              Property(&stream_executor::dnn::AlgorithmDesc::algo_id, Ne(0))),
          _, _, _, _))
      .Times(AtLeast(1))
      .WillRepeatedly(
          [](GenericConvRunner* runner,
             std::optional<GpuConvAlgorithmPicker::ReferenceResult>*
                 reference_result,
             absl::Span<const stream_executor::dnn::AlgorithmDesc>
                 disabled_algos,
             absl::string_view instr_str,
             const GpuConvAlgorithmPicker::AutotuneRuntimeArguments&
                 runtime_arguments) -> absl::StatusOr<AutotuneResult> {
            AutotuneResult result;
            result.mutable_run_time()->set_seconds(1);
            *result.mutable_algorithm() = runner->ToAlgorithmDesc().ToProto();

            if (!reference_result->has_value()) {
              reference_result->emplace();
              reference_result->value().algorithm = runner->ToAlgorithmDesc();
            }
            return result;
          });

  AutotuneConfig cfg = AutotuneConfig::FromDebugOptions(
      DeviceOrDevicelessConfig{DeviceConfig{stream_exec, nullptr}},
      DefaultDebugOptionsIgnoringFlags());
  EXPECT_THAT(RunHloPass(&algorithm_picker, m.get()), IsOkAndHolds(true));
}

}  // namespace
}  // namespace xla::gpu

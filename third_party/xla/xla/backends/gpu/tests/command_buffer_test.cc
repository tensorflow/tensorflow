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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/error_spec.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

bool IsAtLeastCuda12900(const se::StreamExecutor* stream_executor) {
  const auto& device_description = stream_executor->GetDeviceDescription();
  const auto* cuda_cc =
      device_description.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc != nullptr) {
    if (device_description.driver_version() >=
            stream_executor::SemanticVersion(12, 9, 0) &&
        device_description.runtime_version() >=
            stream_executor::SemanticVersion(12, 9, 0)) {
      return true;
    }
  }
  return false;
}

class CommandBufferTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>,
      public ::testing::WithParamInterface<
          DebugOptions::CommandBufferSchedulingMode> {
 protected:
  bool IsRocm() {
    return test_runner().HasProperty(HloRunnerPropertyTag::kUsingGpuRocm);
  }
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_command_buffer_scheduling_mode(GetParam());
    return debug_options;
  }

  // Execute compiled module three times to exercise warm-up, create, and
  // update paths. Third run uses cloned arguments to encourage device buffer
  // address changes.
  void ExecuteThreePhasesAndExpect(std::unique_ptr<HloModule> module,
                                   absl::Span<const Literal* const> arguments,
                                   const Literal& expected,
                                   bool run_hlo_passes) {
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<OpaqueExecutable> executable,
        CreateExecutable(std::move(module), run_hlo_passes));

    // 1) Warm-up (may run thunks)
    TF_ASSERT_OK_AND_ASSIGN(
        Literal result1,
        test_runner().ExecuteWithExecutable(executable.get(), arguments));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result1));

    // 2) Create (record and execute command buffer)
    TF_ASSERT_OK_AND_ASSIGN(
        Literal result2,
        test_runner().ExecuteWithExecutable(executable.get(), arguments));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result2));

    // 3) Update (execute with cloned arguments to attempt buffer changes)
    std::vector<Literal> cloned_args_storage;
    cloned_args_storage.reserve(arguments.size());
    std::vector<const Literal*> cloned_args;
    cloned_args.reserve(arguments.size());
    for (const Literal* arg : arguments) {
      cloned_args_storage.push_back(arg->Clone());
      cloned_args.push_back(&cloned_args_storage.back());
    }

    TF_ASSERT_OK_AND_ASSIGN(Literal result3,
                            test_runner().ExecuteWithExecutable(
                                executable.get(), absl::MakeSpan(cloned_args)));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected, result3));
  }

  // Same as above, but generates fake inputs and compares results to a
  // reference execution. Useful for tests originally using RunAndCompare.
  void RunAndCompareThreeIterations(std::unique_ptr<HloModule> module,
                                    bool run_hlo_passes,
                                    const std::optional<ErrorSpec>& error) {
    // Verify module then clone for reference.
    CHECK_OK(this->verifier().Run(module.get()).status());
    std::unique_ptr<HloModule> reference_module = module->Clone();

    // Prepare fake args for both runners.
    TF_ASSERT_OK_AND_ASSIGN(auto fake_args, MakeFakeArguments(module.get()));
    auto arg_ptrs = LiteralUtil::MakePointers(fake_args);

    // Store input_layouts and untuple_results before the module is consumed
    // by CreateExecutable.
    auto input_layouts = module->entry_computation_layout().parameter_layouts();
    bool untuple_results = module->result_shape().IsTuple();

    // Reference once.
    TF_ASSERT_OK_AND_ASSIGN(
        auto reference,
        reference_runner().Execute(std::move(reference_module),
                                   absl::MakeSpan(arg_ptrs), run_hlo_passes));

    TF_ASSERT_OK_AND_ASSIGN(
        auto exec, CreateExecutable(std::move(module), run_hlo_passes));

    auto* pjrt_runner = tsl::down_cast<HloRunnerPjRt*>(&test_runner());
    ASSERT_TRUE(pjrt_runner != nullptr);

    // Create two copies of device buffers to make sure command buffer saved
    // pointers really get updated during the last "update run".
    enum BufferSet { kInitial = 0, kUpdated = 1 };
    std::array<std::vector<std::unique_ptr<PjRtBuffer>>, 2> argument_handles;
    for (auto& handle : argument_handles) {
      TF_ASSERT_OK_AND_ASSIGN(handle,
                              pjrt_runner->TransferLiteralsToDefaultDevice(
                                  input_layouts, absl::MakeSpan(arg_ptrs)));
    }

    static constexpr absl::string_view kPhases[] = {"warm-up", "create",
                                                    "update"};
    for (size_t i = 0; i < std::size(kPhases); i++) {
      BufferSet current_set = (i < 2) ? kInitial : kUpdated;
      TF_ASSERT_OK_AND_ASSIGN(auto output_buffers,
                              pjrt_runner->ExecuteWithDeviceBuffers(
                                  exec.get(), argument_handles[current_set]));

      TF_ASSERT_OK_AND_ASSIGN(auto result,
                              pjrt_runner->TransferLiteralsFromDevice(
                                  output_buffers, untuple_results));
      EXPECT_TRUE(LiteralTestUtil::NearOrEqual(reference, result, error))
          << "Mismatch on " << kPhases[i] << " run (iteration " << i << ")";
    }  // for
  }
};

// Test fixture that enables loop unrolling for command buffers.
class CommandBufferUnrollTest : public CommandBufferTest {
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_command_buffer_scheduling_mode(GetParam());
    debug_options.set_xla_gpu_command_buffer_unroll_loops(true);
    return debug_options;
  }
};

TEST_P(CommandBufferTest, Fusions) {
  constexpr absl::string_view hlo_text = R"(
  HloModule m, is_scheduled=true

  double {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] add(p0, p0)
  }

  square {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] multiply(p0, p0)
  }

  sum {
    p0 = f32[2,2] parameter(0)
    p1 = f32[2,2] parameter(1)
    ROOT sum = f32[2,2] add(p0, p1)
  }

  command_buffer {
    p0 = f32[2,2] parameter(0)
    f0 = f32[2,2] fusion(p0), kind=kLoop, calls=double
    f1 = f32[2,2] fusion(p0), kind=kLoop, calls=square
    ROOT f3 = f32[2,2] fusion(f0, f1), kind=kLoop, calls=sum
  }

  ENTRY main {
    p0 = f32[2,2] parameter(0)
    ROOT call = f32[2,2] call(p0), to_apply=command_buffer
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  Literal argument = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  Literal expected = LiteralUtil::CreateR2<float>({{3.0, 8.0}, {15.0, 24.0}});

  ExecuteThreePhasesAndExpect(std::move(module), {&argument}, expected,
                              /*run_hlo_passes=*/false);
}

static absl::Status Memcpy(se::Stream* stream, ffi::AnyBuffer src,
                           ffi::Result<ffi::AnyBuffer> dst) {
  se::DeviceAddressBase dst_mem = dst->device_memory();
  se::DeviceAddressBase src_mem = src.device_memory();
  return stream->MemcpyD2D(&dst_mem, src_mem, src_mem.size());
}

XLA_FFI_DEFINE_HANDLER(kMemcpyExecuteNoState, Memcpy,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()   // src
                           .Ret<ffi::AnyBuffer>(),  // dst
                       {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", "gpu",
                         kMemcpyExecuteNoState);

TEST_P(CommandBufferTest, TracedCustomCalls) {
  constexpr absl::string_view hlo_text = R"(
  HloModule m, is_scheduled=true

  command_buffer {
    p0 = f32[2,2] parameter(0)
    ROOT f3 = f32[2,2] custom-call(p0),
      custom_call_target="__xla_test$$memcpy",
      api_version=API_VERSION_TYPED_FFI
  }

  ENTRY main {
    p0 = f32[2,2] parameter(0)
    ROOT call = f32[2,2] call(p0), to_apply=command_buffer
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  Literal argument = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  Literal expected = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});

  ExecuteThreePhasesAndExpect(std::move(module), {&argument}, expected,
                              /*run_hlo_passes=*/false);
}

// Empty memcpy state to test stateful custom calls.
struct MemcpyState {};

static absl::StatusOr<std::unique_ptr<MemcpyState>> MemcpyInstantiate() {
  return std::make_unique<MemcpyState>();
}

static absl::StatusOr<std::unique_ptr<MemcpyState>> MemcpyInitialize() {
  return std::make_unique<MemcpyState>();
}

static absl::Status StatefulMemcpy(
    se::Stream* stream, const xla::gpu::CollectiveParams* collective_params,
    const xla::gpu::CollectiveMemory* collective_memory, MemcpyState* state,
    MemcpyState* device_state,

    ffi::AnyBuffer src, ffi::Result<ffi::AnyBuffer> dst) {
  EXPECT_NE(state, nullptr);
  EXPECT_NE(device_state, nullptr);
  EXPECT_NE(collective_params, nullptr);
  EXPECT_NE(collective_memory, nullptr);
  se::DeviceAddressBase dst_mem = dst->device_memory();
  se::DeviceAddressBase src_mem = src.device_memory();
  return stream->MemcpyD2D(&dst_mem, src_mem, src_mem.size());
}

XLA_FFI_DEFINE_HANDLER(kMemcpyInstantiate, MemcpyInstantiate,
                       ffi::Ffi::BindInstantiate());

XLA_FFI_DEFINE_HANDLER(kMemcpyInitialize, MemcpyInitialize,
                       ffi::Ffi::BindInitialize());

XLA_FFI_DEFINE_HANDLER(kMemcpyExecute, StatefulMemcpy,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::CollectiveParams>()
                           .Ctx<ffi::CollectiveMemory>()
                           .Ctx<ffi::Initialized<MemcpyState>>()
                           .Ctx<ffi::State<MemcpyState>>()
                           .Arg<ffi::AnyBuffer>()   // src
                           .Ret<ffi::AnyBuffer>(),  // dst
                       {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$stateful_memcpy",
                         "gpu",
                         {
                             /*instantiate=*/kMemcpyInstantiate,
                             /*prepare=*/nullptr,
                             /*initialize=*/kMemcpyInitialize,
                             /*execute=*/kMemcpyExecute,
                         });

TEST_P(CommandBufferTest, TracedStatefulCustomCalls) {
  constexpr absl::string_view hlo_text = R"(
  HloModule m, is_scheduled=true

  command_buffer {
    p0 = f32[2,2] parameter(0)
    ROOT f3 = f32[2,2] custom-call(p0),
      custom_call_target="__xla_test$$stateful_memcpy",
      api_version=API_VERSION_TYPED_FFI
  }

  ENTRY main {
    p0 = f32[2,2] parameter(0)
    ROOT call = f32[2,2] call(p0), to_apply=command_buffer
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  Literal argument = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  Literal expected = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});

  ExecuteThreePhasesAndExpect(std::move(module), {&argument}, expected,
                              /*run_hlo_passes=*/false);
}

TEST_P(CommandBufferTest, TrueFalseConditional) {
  if (IsRocm()) {
    GTEST_SKIP() << "Graph conditionals are not yet supported on HIP graphs";
  }
  constexpr absl::string_view hlo_text = R"(
  HloModule m, is_scheduled=true

  double {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] add(p0, p0)
  }

  square {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] multiply(p0, p0)
  }

  double_computation {
    p0 = f32[2,2] parameter(0)
    ROOT double = f32[2,2] fusion(p0), kind=kLoop, calls=double
  }

  square_computation {
    p0 = f32[2,2] parameter(0)
    ROOT square = f32[2,2] fusion(p0), kind=kLoop, calls=square
  }

  command_buffer {
    p0 = pred[] parameter(0)
    p1 = f32[2,2] parameter(1)
    ROOT conditional = f32[2,2] conditional(p0, p1, p1),
                                true_computation=double_computation,
                                false_computation=square_computation
  }

  ENTRY main {
    p0 = pred[] parameter(0)
    p1 = f32[2,2] parameter(1)
    ROOT call = f32[2,2] call(p0, p1), to_apply=command_buffer
  })";

  Literal p1 = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});

  {  // Execute `true` branch.
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_text));

    Literal pred = LiteralUtil::CreateR0<bool>(true);
    Literal expected = LiteralUtil::CreateR2<float>({{2.0, 4.0}, {6.0, 8.0}});
    ExecuteThreePhasesAndExpect(std::move(m), {&pred, &p1}, expected,
                                /*run_hlo_passes=*/false);
  }

  {  // Execute `false` branch.
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_text));

    Literal pred = LiteralUtil::CreateR0<bool>(false);
    Literal expected = LiteralUtil::CreateR2<float>({{1.0, 4.0}, {9.0, 16.0}});
    ExecuteThreePhasesAndExpect(std::move(m), {&pred, &p1}, expected,
                                /*run_hlo_passes=*/false);
  }
}

TEST_P(CommandBufferTest, IndexConditional) {
  if (IsRocm()) {
    GTEST_SKIP() << "Graph conditionals are not yet supported on HIP graphs";
  }
  constexpr absl::string_view hlo_text = R"(
  HloModule m, is_scheduled=true

  double {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] add(p0, p0)
  }

  square {
    p0 = f32[2,2] parameter(0)
    ROOT add = f32[2,2] multiply(p0, p0)
  }

  double_computation {
    p0 = f32[2,2] parameter(0)
    ROOT double = f32[2,2] fusion(p0), kind=kLoop, calls=double
  }

  square_computation {
    p0 = f32[2,2] parameter(0)
    ROOT square = f32[2,2] fusion(p0), kind=kLoop, calls=square
  }

  command_buffer {
    p0 = s32[] parameter(0)
    p1 = f32[2,2] parameter(1)
    ROOT conditional = f32[2,2] conditional(p0, p1, p1),
      branch_computations={double_computation, square_computation}
  }

  ENTRY main {
    p0 = s32[] parameter(0)
    p1 = f32[2,2] parameter(1)
    ROOT call = f32[2,2] call(p0, p1), to_apply=command_buffer
  })";

  Literal p1 = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});

  {  // Execute `0` branch.
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_text));

    Literal index = LiteralUtil::CreateR0<int32_t>(0);
    Literal expected = LiteralUtil::CreateR2<float>({{2.0, 4.0}, {6.0, 8.0}});
    ExecuteThreePhasesAndExpect(std::move(m), {&index, &p1}, expected,
                                /*run_hlo_passes=*/false);
  }

  {  // Execute `1` branch.
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_text));

    Literal index = LiteralUtil::CreateR0<int32_t>(1);
    Literal expected = LiteralUtil::CreateR2<float>({{1.0, 4.0}, {9.0, 16.0}});
    ExecuteThreePhasesAndExpect(std::move(m), {&index, &p1}, expected,
                                /*run_hlo_passes=*/false);
  }

  {  // Execute `1024` branch (our of bound index executes N-1 branch).
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_text));

    Literal index = LiteralUtil::CreateR0<int32_t>(1024);
    Literal expected = LiteralUtil::CreateR2<float>({{1.0, 4.0}, {9.0, 16.0}});
    ExecuteThreePhasesAndExpect(std::move(m), {&index, &p1}, expected,
                                /*run_hlo_passes=*/false);
  }
}

TEST_P(CommandBufferTest, WhileLoop) {
  if (IsRocm()) {
    GTEST_SKIP() << "Graph conditionals are not yet supported on HIP graphs";
  }
  constexpr absl::string_view hlo_text = R"(
  HloModule m, is_scheduled=true

  compare_fusion {
    p0 = s32[] parameter(0)
    ten = s32[] constant(10)
    ROOT compare = compare(p0, ten), direction=LT
  }

  add_one {
    p0 = s32[] parameter(0)
    one = s32[] constant(1)
    ROOT add = add(p0, one)
  }

  add_two {
    p0 = f32[] parameter(0)
    two = f32[] constant(2.0)
    ROOT add = add(p0, two)
  }

  body {
    p0 = (s32[], f32[]) parameter(0)
    cnt = get-tuple-element(p0), index=0
    val = get-tuple-element(p0), index=1
    add_cnt = s32[] fusion(cnt), kind=kLoop, calls=add_one
    add_val = f32[] fusion(val), kind=kLoop, calls=add_two
    ROOT tuple = (s32[], f32[]) tuple(add_cnt, add_val)
  }

  cond {
    p0 = (s32[], f32[]) parameter(0)
    cnt = get-tuple-element(p0), index=0
    ROOT compare = pred[] fusion(cnt), kind=kLoop, calls=compare_fusion
  }

  command_buffer {
    p0 = (s32[], f32[]) parameter(0)
    ROOT while = while(p0), condition=cond, body=body
  }

  ENTRY main {
    p0 = (s32[], f32[]) parameter(0)
    ROOT call = (s32[], f32[]) call(p0), to_apply=command_buffer
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  Literal cnt = LiteralUtil::CreateR0<int32_t>(0);
  Literal value = LiteralUtil::CreateR0<float>(0.0);
  Literal argument = LiteralUtil::MakeTuple({&cnt, &value});

  Literal expected_cnt = LiteralUtil::CreateR0<int32_t>(10);
  Literal expected_value = LiteralUtil::CreateR0<float>(20.0);
  Literal expected = LiteralUtil::MakeTuple({&expected_cnt, &expected_value});

  ExecuteThreePhasesAndExpect(std::move(module), {&argument}, expected,
                              /*run_hlo_passes=*/false);
}

TEST_P(CommandBufferTest, ControlDependencyTest) {
  constexpr absl::string_view module_str = R"(
HloModule m

%x (a: f32[3200,6400]) -> f32[3200,6400] {
  %a = f32[3200,6400]{1,0} parameter(0)
  ROOT %b = f32[3200,6400]{1,0} negate(%a)
}

%y (a.1: f32[3200,6400]) -> f32[3200,6400] {
  %a.1 = f32[3200,6400]{1,0} parameter(0)
  ROOT %b.1 = f32[3200,6400]{1,0} add(%a.1, %a.1)
}

%command_buffer (p: f32[3200,6400], p.1: f32[3200,6400]) -> (f32[3200,6400], f32[3200,6400]) {
  %p = f32[3200,6400]{1,0} parameter(0)
  %p.1 = f32[3200,6400]{1,0} parameter(1)
  %b.2 = f32[3200,6400]{1,0} fusion(%p), kind=kLoop, calls=%x
  %c = f32[3200,6400]{1,0} fusion(%p.1), kind=kLoop, calls=%y, control-predecessors={%b.2}
  ROOT %tuple = (f32[3200,6400]{1,0}, f32[3200,6400]{1,0}) tuple(%b.2, %c)
}

ENTRY %e (m: f32[3200,6400], n: f32[3200,6400]) -> (f32[3200,6400], f32[3200,6400]) {
  %m = f32[3200,6400]{1,0} parameter(0)
  %n = f32[3200,6400]{1,0} parameter(1)
  %call = (f32[3200,6400]{1,0}, f32[3200,6400]{1,0}) call(%m, %n), to_apply=%command_buffer
  %get-tuple-element = f32[3200,6400]{1,0} get-tuple-element(%call), index=0
  %get-tuple-element.1 = f32[3200,6400]{1,0} get-tuple-element(%call), index=1
  ROOT %t = (f32[3200,6400]{1,0}, f32[3200,6400]{1,0}) tuple(%get-tuple-element, %get-tuple-element.1)
}
  )";

  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_disable_all_hlo_passes(true);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{1e-3, 2e-3}));
}

TEST_P(CommandBufferUnrollTest, WhileLoop) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  if (!IsAtLeastCuda12900(stream_executor)) {
    GTEST_SKIP() << "Child command is not supported for CUDA < 12.9";
  }

  constexpr absl::string_view hlo_text = R"(
  HloModule m, is_scheduled=true

  compare_fusion {
    p0 = s32[] parameter(0)
    ten = s32[] constant(10)
    ROOT compare = compare(p0, ten), direction=LT
  }

  add_one {
    p0 = s32[] parameter(0)
    one = s32[] constant(1)
    ROOT add = add(p0, one)
  }

  add_two {
    p0 = f32[] parameter(0)
    two = f32[] constant(2.0)
    ROOT add = add(p0, two)
  }

  body {
    p0 = (s32[], f32[]) parameter(0)
    cnt = get-tuple-element(p0), index=0
    val = get-tuple-element(p0), index=1
    add_cnt = s32[] fusion(cnt), kind=kLoop, calls=add_one
    add_val = f32[] fusion(val), kind=kLoop, calls=add_two
    ROOT tuple = (s32[], f32[]) tuple(add_cnt, add_val)
  }

  cond {
    p0 = (s32[], f32[]) parameter(0)
    cnt = get-tuple-element(p0), index=0
    ROOT compare = pred[] fusion(cnt), kind=kLoop, calls=compare_fusion
  }

  command_buffer {
    p0 = (s32[], f32[]) parameter(0)
    ROOT while = while(p0), condition=cond, body=body, backend_config={"known_trip_count":{"n":"10"}}
  }

  ENTRY main {
    p0 = (s32[], f32[]) parameter(0)
    ROOT call = (s32[], f32[]) call(p0), to_apply=command_buffer
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  Literal cnt = LiteralUtil::CreateR0<int32_t>(0);
  Literal value = LiteralUtil::CreateR0<float>(0.0);
  Literal argument = LiteralUtil::MakeTuple({&cnt, &value});

  Literal expected_cnt = LiteralUtil::CreateR0<int32_t>(10);
  Literal expected_value = LiteralUtil::CreateR0<float>(20.0);
  Literal expected = LiteralUtil::MakeTuple({&expected_cnt, &expected_value});

  ExecuteThreePhasesAndExpect(std::move(module), {&argument}, expected,
                              /*run_hlo_passes=*/false);
}

TEST_P(CommandBufferUnrollTest, WhileLoopMultiDevice) {
  se::StreamExecutor* stream_executor = GpuExecutor();

  if (!IsAtLeastCuda12900(stream_executor)) {
    GTEST_SKIP() << "Child command is not supported for CUDA < 12.9";
  }

  // Require at least two visible GPU devices for multi-device execution.
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  if (platform->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires >= 2 visible GPU devices";
  }

  constexpr absl::string_view hlo_text = R"(
  HloModule m, is_scheduled=true

  compare_fusion {
    p0 = s32[] parameter(0)
    ten = s32[] constant(10)
    ROOT compare = compare(p0, ten), direction=LT
  }

  add_one {
    p0 = s32[] parameter(0)
    one = s32[] constant(1)
    ROOT add = add(p0, one)
  }

  add_two {
    p0 = f32[] parameter(0)
    two = f32[] constant(2.0)
    ROOT add = add(p0, two)
  }

  body {
    p0 = (s32[], f32[]) parameter(0)
    cnt = get-tuple-element(p0), index=0
    val = get-tuple-element(p0), index=1
    add_cnt = s32[] fusion(cnt), kind=kLoop, calls=add_one
    add_val = f32[] fusion(val), kind=kLoop, calls=add_two
    ROOT tuple = (s32[], f32[]) tuple(add_cnt, add_val)
  }

  cond {
    p0 = (s32[], f32[]) parameter(0)
    cnt = get-tuple-element(p0), index=0
    ROOT compare = pred[] fusion(cnt), kind=kLoop, calls=compare_fusion
  }

  command_buffer {
    a = s32[] parameter(0)
    b = f32[] parameter(1)
    tuple = (s32[], f32[]) tuple(a, b)
    ROOT while = while(tuple), condition=cond, body=body, backend_config={"known_trip_count":{"n":"10"}}
  }

  ENTRY main {
    a = s32[] parameter(0)
    b = f32[] parameter(1)
    ROOT call = (s32[], f32[]) call(a, b), to_apply=command_buffer
  })";

  // Parse with replica_count=2 to run on two devices.
  HloModuleConfig config = GetModuleConfigForTest(/*replica_count=*/2);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  Literal cnt = LiteralUtil::CreateR0<int32_t>(0);
  Literal value = LiteralUtil::CreateR0<float>(0.0);

  Literal expected_cnt = LiteralUtil::CreateR0<int32_t>(10);
  Literal expected_value = LiteralUtil::CreateR0<float>(20.0);
  Literal expected = LiteralUtil::MakeTuple({&expected_cnt, &expected_value});

  // Flatten tuple parameter into individual leaves for PJRT replicated execute.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module), {&cnt, &value}, /*num_devices=*/2,
                        /*use_threads=*/true, /*run_hlo_passes=*/false));
  ASSERT_EQ(results.size(), 2);
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, results[0]));
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, results[1]));
}

INSTANTIATE_TEST_SUITE_P(CommandBufferTests, CommandBufferTest,
                         ::testing::Values(DebugOptions::LHS,
                                           DebugOptions::CONCURRENT));
INSTANTIATE_TEST_SUITE_P(CommandBufferTestsUnroll, CommandBufferUnrollTest,
                         ::testing::Values(DebugOptions::LHS,
                                           DebugOptions::CONCURRENT));

}  // namespace
}  // namespace xla::gpu

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
#include "xla/error_spec.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
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
  ::testing::AssertionResult RunAndCompareThreeIterations(
      std::unique_ptr<HloModule> module, bool run_hlo_passes,
      const std::optional<ErrorSpec>& error) {
    // Verify module then clone for reference.
    CHECK_OK(this->verifier().Run(module.get()).status());
    std::unique_ptr<HloModule> reference_module = module->Clone();

    // Prepare fake args for both runners.
    absl::StatusOr<std::vector<Literal>> fake_args_or =
        MakeFakeArguments(module.get());
    if (!fake_args_or.ok()) {
      return ::testing::AssertionFailure() << fake_args_or.status().message();
    }
    std::vector<Literal> fake_args = std::move(*fake_args_or);
    std::vector<const Literal*> arg_ptrs = LiteralUtil::MakePointers(fake_args);

    // Reference once.
    absl::StatusOr<Literal> reference = reference_runner().Execute(
        std::move(reference_module), absl::MakeSpan(arg_ptrs), run_hlo_passes);
    if (!reference.ok()) {
      return ::testing::AssertionFailure() << reference.status();
    }

    // Compile once on test backend and run three iterations.
    absl::StatusOr<std::unique_ptr<OpaqueExecutable>> exec_or =
        CreateExecutable(std::move(module), run_hlo_passes);
    if (!exec_or.ok()) {
      return ::testing::AssertionFailure() << exec_or.status();
    }
    std::unique_ptr<OpaqueExecutable> exec = std::move(*exec_or);

    // 1) Warm-up
    absl::StatusOr<Literal> r1 = test_runner().ExecuteWithExecutable(
        exec.get(), absl::MakeSpan(arg_ptrs));
    if (!r1.ok()) return ::testing::AssertionFailure() << r1.status();
    if (!LiteralTestUtil::NearOrEqual(*reference, *r1, error))
      return ::testing::AssertionFailure() << "Mismatch on warm-up run";

    // 2) Create
    absl::StatusOr<Literal> r2 = test_runner().ExecuteWithExecutable(
        exec.get(), absl::MakeSpan(arg_ptrs));
    if (!r2.ok()) return ::testing::AssertionFailure() << r2.status();
    if (!LiteralTestUtil::NearOrEqual(*reference, *r2, error))
      return ::testing::AssertionFailure() << "Mismatch on create run";

    // 3) Update with cloned args
    std::vector<Literal> cloned_args_storage;
    cloned_args_storage.reserve(arg_ptrs.size());
    std::vector<const Literal*> cloned_arg_ptrs;
    cloned_arg_ptrs.reserve(arg_ptrs.size());
    for (const Literal* a : arg_ptrs) {
      cloned_args_storage.push_back(a->Clone());
      cloned_arg_ptrs.push_back(&cloned_args_storage.back());
    }

    absl::StatusOr<Literal> r3 = test_runner().ExecuteWithExecutable(
        exec.get(), absl::MakeSpan(cloned_arg_ptrs));
    if (!r3.ok()) return ::testing::AssertionFailure() << r3.status();
    if (!LiteralTestUtil::NearOrEqual(*reference, *r3, error))
      return ::testing::AssertionFailure() << "Mismatch on update run";

    return ::testing::AssertionSuccess();
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

// Empty memcpy state to test stateful custom calls.
struct MemcpyState {};

static absl::StatusOr<std::unique_ptr<MemcpyState>> MemcpyInstantiate() {
  return std::make_unique<MemcpyState>();
}

static absl::StatusOr<std::unique_ptr<MemcpyState>> MemcpyInitialize() {
  return std::make_unique<MemcpyState>();
}

static absl::Status Memcpy(se::Stream* stream, MemcpyState* state,
                           MemcpyState* device_state, ffi::AnyBuffer src,
                           ffi::Result<ffi::AnyBuffer> dst) {
  EXPECT_NE(state, nullptr);
  EXPECT_NE(device_state, nullptr);
  se::DeviceAddressBase dst_mem = dst->device_memory();
  se::DeviceAddressBase src_mem = src.device_memory();
  return stream->MemcpyD2D(&dst_mem, src_mem, src_mem.size());
}

XLA_FFI_DEFINE_HANDLER(kMemcpyInstantiate, MemcpyInstantiate,
                       ffi::Ffi::BindInstantiate());

XLA_FFI_DEFINE_HANDLER(kMemcpyInitialize, MemcpyInitialize,
                       ffi::Ffi::BindInitialize());

XLA_FFI_DEFINE_HANDLER(kMemcpyExecute, Memcpy,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::State<MemcpyState>>()
                           .Ctx<ffi::Initialized<MemcpyState>>()
                           .Arg<ffi::AnyBuffer>()   // src
                           .Ret<ffi::AnyBuffer>(),  // dst
                       {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", "gpu",
                         {
                             /*instantiate=*/kMemcpyInstantiate,
                             /*prepare=*/nullptr,
                             /*initialize=*/kMemcpyInitialize,
                             /*execute=*/kMemcpyExecute,
                         });

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

TEST_P(CommandBufferTest, DynamicSliceFusionCmd) {
  // Hlo generated by the following JAX program:
  // def scan_body(carry, x):
  //     sliced_x = lax.slice(x, (0, 0), (128, 128))
  //     result = jnp.dot(carry, sliced_x)
  //     new_carry = result
  //     return new_carry, result
  // @jax.jit
  // def run_scan(initial_carry, xs):
  //     final_carry, outputs = lax.scan(scan_body, initial_carry, xs, length=2)
  //     return final_carry, outputs

  constexpr absl::string_view module_str = R"(
HloModule jit_run_scan

None.7 {
  Arg_0.8 = f32[128,128]{1,0} parameter(0)
  Arg_1.9 = f32[128,128]{1,0} parameter(1)
  dot.10 = f32[128,128]{1,0} dot(Arg_0.8, Arg_1.9), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT tuple.11 = (f32[128,128]{1,0}, f32[128,128]{1,0}) tuple(dot.10, dot.10)
}

region_0.12 {
  arg_tuple.13 = (s32[], f32[128,128]{1,0}, f32[2,128,128]{2,1,0}, f32[2,128,128]{2,1,0}) parameter(0)
  get-tuple-element.14 = s32[] get-tuple-element(arg_tuple.13), index=0
  constant.18 = s32[] constant(1)
  add.34 = s32[] add(get-tuple-element.14, constant.18)
  get-tuple-element.15 = f32[128,128]{1,0} get-tuple-element(arg_tuple.13), index=1
  get-tuple-element.17 = f32[2,128,128]{2,1,0} get-tuple-element(arg_tuple.13), index=3
  constant.20 = s32[] constant(0)
  compare.21 = pred[] compare(get-tuple-element.14, constant.20), direction=LT
  constant.19 = s32[] constant(2)
  add.22 = s32[] add(get-tuple-element.14, constant.19)
  select.23 = s32[] select(compare.21, add.22, get-tuple-element.14)
  dynamic-slice.24 = f32[1,128,128]{2,1,0} dynamic-slice(get-tuple-element.17, select.23, constant.20, constant.20), dynamic_slice_sizes={1,128,128}
  reshape.25 = f32[128,128]{1,0} reshape(dynamic-slice.24)
  call.26 = (f32[128,128]{1,0}, f32[128,128]{1,0}) call(get-tuple-element.15, reshape.25), to_apply=None.7
  get-tuple-element.27 = f32[128,128]{1,0} get-tuple-element(call.26), index=0
  get-tuple-element.16 = f32[2,128,128]{2,1,0} get-tuple-element(arg_tuple.13), index=2
  get-tuple-element.28 = f32[128,128]{1,0} get-tuple-element(call.26), index=1
  reshape.29 = f32[1,128,128]{2,1,0} reshape(get-tuple-element.28)
  compare.30 = pred[] compare(get-tuple-element.14, constant.20), direction=LT
  add.31 = s32[] add(get-tuple-element.14, constant.19)
  select.32 = s32[] select(compare.30, add.31, get-tuple-element.14)
  dynamic-update-slice.33 = f32[2,128,128]{2,1,0} dynamic-update-slice(get-tuple-element.16, reshape.29, select.32, constant.20, constant.20)
  ROOT tuple.35 = (s32[], f32[128,128]{1,0}, f32[2,128,128]{2,1,0}, f32[2,128,128]{2,1,0}) tuple(add.34, get-tuple-element.27, dynamic-update-slice.33, get-tuple-element.17)
} // region_0.12

region_1.36 {
  arg_tuple.37 = (s32[], f32[128,128]{1,0}, f32[2,128,128]{2,1,0}, f32[2,128,128]{2,1,0}) parameter(0)
  get-tuple-element.39 = f32[128,128]{1,0} get-tuple-element(arg_tuple.37), index=1
  get-tuple-element.40 = f32[2,128,128]{2,1,0} get-tuple-element(arg_tuple.37), index=2
  get-tuple-element.41 = f32[2,128,128]{2,1,0} get-tuple-element(arg_tuple.37), index=3
  get-tuple-element.38 = s32[] get-tuple-element(arg_tuple.37), index=0
  constant.42 = s32[] constant(2)
  ROOT compare.43 = pred[] compare(get-tuple-element.38, constant.42), direction=LT
} // region_1.36

ENTRY main.49 {
  constant.3 = s32[] constant(0)
  Arg_0.1 = f32[128,128]{1,0} parameter(0)
  constant.4 = f32[] constant(0)
  broadcast.5 = f32[2,128,128]{2,1,0} broadcast(constant.4), dimensions={}
  Arg_1.2 = f32[2,128,128]{2,1,0} parameter(1)
  tuple.6 = (s32[], f32[128,128]{1,0}, f32[2,128,128]{2,1,0}, f32[2,128,128]{2,1,0}) tuple(constant.3, Arg_0.1, broadcast.5, Arg_1.2)
  while.44 = (s32[], f32[128,128]{1,0}, f32[2,128,128]{2,1,0}, f32[2,128,128]{2,1,0}) while(tuple.6), condition=region_1.36, body=region_0.12
  get-tuple-element.45 = s32[] get-tuple-element(while.44), index=0
  get-tuple-element.46 = f32[128,128]{1,0} get-tuple-element(while.44), index=1
  get-tuple-element.47 = f32[2,128,128]{2,1,0} get-tuple-element(while.44), index=2
  ROOT tuple.48 = (f32[128,128]{1,0}, f32[2,128,128]{2,1,0}) tuple(get-tuple-element.46, get-tuple-element.47)
}
  )";

  // Run twice toggling exclusive lock to match original test behavior.
  HloModuleConfig config;
  auto debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_require_exclusive_lock(false);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLASLT);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUSTOM_CALL);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUDNN);
  debug_options.add_xla_gpu_enable_command_buffer(
      DebugOptions::DYNAMIC_SLICE_FUSION);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{1e-3, 2e-3}));

  debug_options.set_xla_gpu_require_exclusive_lock(true);
  config.set_debug_options(debug_options);
  TF_ASSERT_OK_AND_ASSIGN(module,
                          ParseAndReturnVerifiedModule(module_str, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{1e-3, 2e-3}));
}

TEST_P(CommandBufferTest, DynamicSliceCopyFusionCmd) {
  if (GpuExecutor() != nullptr) {
    GTEST_SKIP() << "This test leads to segfault in CommandBuffer #36087";
  }

  constexpr absl::string_view hlo_text = R"(
    dynamic_slice {
      p0 = s32[4,8,8]{2,1,0} parameter(0)
      p1 = s32[] parameter(1)
      c1 = s32[] constant(1)
      p2 = s32[] parameter(2)

      p1p1 = s32[] add(p1, c1)

      // Test all supported kinds of offsets: derived from the while loop's
      // induction variable (p1p1), constant (c1) and always clamped to 0, so
      // the value is irrelevant (p2).
      ROOT slice = s32[1,1,8] dynamic-slice(p0, p1p1, c1, p2),
          dynamic_slice_sizes={1,1,8}
    }

    remainder {
      p0 = s32[] parameter(0)
      c5 = s32[] constant(5)
      // We take the value modulo 5 to test for correct clamping (the offset 4
      // must get clamped to 3, since it's greater or equal than the dimension
      // size).
      ROOT remainder = s32[] remainder(p0, c5)
    }

    add {
      p0 = s32[] parameter(0)
      c1 = s32[] constant(1)
      ROOT sum = s32[] add(p0, c1)
    }

    add_slices {
      p0 = s32[1,1,8] parameter(0)
      p1 = s32[1,1,8] parameter(1)
      ROOT sum = s32[1,1,8] add(p0, p1)
    }

    times_two {
      p0 = s32[] parameter(0)
      ROOT sum = s32[] add(p0, p0)
    }

    body {
      p0 = (s32[], s32[4,8,8]{2,1,0}, s32[1,1,8], s32[]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[4,8,8]{2,1,0} get-tuple-element(p0), index=1

      ivar_copy = s32[] copy(ivar)
      acc = s32[1,1,8] get-tuple-element(p0), index=2
      acc_copy = s32[1,1,8] copy(acc)

      offset1 = s32[] fusion(ivar_copy), kind=kLoop, calls=remainder
      offset2 = s32[] get-tuple-element(p0), index=3

      slice = s32[1,1,8] fusion(input, offset1, offset2), kind=kLoop, calls=dynamic_slice,
          backend_config={"fusion_backend_config":{
              "kind":"__dynamic_memcpy",
              "dynamic_memcpy_config":{
                  "depends_on_loop":true,
                  "src_offset_bytes":["288","544","800","800","800","288"],
                  "dst_offset_bytes":["0","0","0","0","0","0"]}}}
      next_ivar = s32[] fusion(ivar_copy), kind=kLoop, calls=add
      next_offset_2 = s32[] fusion(offset2), kind=kLoop, calls=times_two

      next_acc = s32[1,1,8] fusion(acc_copy, slice), kind=kLoop, calls=add_slices
      ROOT result = (s32[], s32[4,8,8]{2,1,0}, s32[1,1,8], s32[])
          tuple(next_ivar, input, next_acc, next_offset_2)
    }

    compare {
      p0 = s32[] parameter(0)
      c6 = s32[] constant(6)
      ROOT cmp = pred[] compare(p0, c6), direction=LT
    }

    condition {
      p0 = (s32[], s32[4,8,8]{2,1,0}, s32[1,1,8], s32[]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      ROOT cmp = pred[] fusion(ivar), kind=kLoop, calls=compare
    }

    zero {
      c0 = s32[] constant(0)
      ROOT bc = s32[1,1,8] broadcast(c0), dimensions={}
    }

    input {
      iota = s32[256] iota(), iota_dimension=0
      ROOT bc = s32[4,8,8]{2,1,0} bitcast(iota)
    }

    ENTRY main {
      input = s32[4,8,8]{2,1,0} fusion(), kind=kLoop, calls=input
      init_acc = s32[1,1,8] fusion(), kind=kLoop, calls=zero
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      tuple = (s32[], s32[4,8,8]{2,1,0}, s32[1,1,8], s32[]) tuple(c0, input, init_acc, c1)
      ROOT while = (s32[], s32[4,8,8]{2,1,0}, s32[1,1,8], s32[]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"6"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    }
  )";

  HloModuleConfig config;
  DebugOptions debug_options = GetDebugOptionsForTest();
  debug_options.set_xla_gpu_require_exclusive_lock(false);
  debug_options.add_xla_gpu_enable_command_buffer(
      DebugOptions::DYNAMIC_SLICE_COPY_FUSION);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  EXPECT_TRUE(RunAndCompareThreeIterations(
      std::move(module), /*run_hlo_passes=*/false, ErrorSpec{1e-3, 2e-3}));

  if (!IsAtLeastCuda12900(GpuExecutor())) {
    GTEST_SKIP() << "While loop unrolling is not supported for CUDA < 12.9";
  }

  debug_options.add_xla_gpu_enable_command_buffer(
      DebugOptions::DYNAMIC_SLICE_COPY_FUSION);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLASLT);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUSTOM_CALL);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUDNN);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::WHILE);
  debug_options.set_xla_gpu_command_buffer_unroll_loops(true);
  config.set_debug_options(debug_options);

  TF_ASSERT_OK_AND_ASSIGN(auto unrolled_module,
                          ParseAndReturnVerifiedModule(hlo_text, config));

  EXPECT_TRUE(RunAndCompareThreeIterations(std::move(unrolled_module),
                                           /*run_hlo_passes=*/false,
                                           ErrorSpec{1e-3, 2e-3}));
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

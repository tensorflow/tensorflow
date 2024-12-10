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

#include "xla/backends/cpu/nanort/nanort_client.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::cpu {
namespace {

using Arguments = absl::InlinedVector<NanoRtExecutable::Argument, 8>;
using Results = absl::InlinedVector<NanoRtExecutable::Result, 8>;

TEST(NanoRtClientTest, CompileAndRunScalarComputation) {
  constexpr absl::string_view hlo = R"(
    HloModule add

    ENTRY e {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  XlaComputation computation(module->ToProto());

  NanoRtClient client;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<NanoRtExecutable> executable,
                          client.Compile(computation));

  // Storage for executable parameters and results.
  alignas(32) float p0_value = 1.0f;
  alignas(32) float p1_value = 2.0f;
  alignas(32) float r0_value = 0.0f;

  // Prepare executable parameters, results and temp storage.
  Arguments arguments = {{&p0_value, 1}, {&p1_value, 1}};
  Results results = {{&r0_value, 1}};

  auto event = executable->Execute(arguments, results);
  tsl::BlockUntilReady(event);

  ASSERT_TRUE(event.IsConcrete());
  EXPECT_EQ(r0_value, 3.0f);
}

TEST(NanoRtClientTest, CompileAndRunTupledComputation) {
  constexpr absl::string_view hlo = R"(
    HloModule add_and_mul

    ENTRY e {
      p = (f32[], f32[]) parameter(0)
      p0 = f32[] get-tuple-element(p), index=0
      p1 = f32[] get-tuple-element(p), index=1
      add = f32[] add(p0, p1)
      mul = f32[] multiply(p0, p1)
      ROOT add_and_mul = (f32[], f32[]) tuple(add, mul)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  XlaComputation computation(module->ToProto());

  NanoRtClient client;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<NanoRtExecutable> executable,
                          client.Compile(computation));

  // Storage for executable parameters and results.
  alignas(32) float p0_value = 2.0f;
  alignas(32) float p1_value = 3.0f;
  alignas(32) float r0_value = 0.0f;
  alignas(32) float r1_value = 0.0f;

  // Prepare executable parameters, results and temp storage.
  Arguments arguments = {{&p0_value, 1}, {&p1_value, 1}};
  Results results = {{&r0_value, 1}, {&r1_value, 1}};

  auto event = executable->Execute(arguments, results);
  tsl::BlockUntilReady(event);

  ASSERT_TRUE(event.IsConcrete());
  EXPECT_EQ(r0_value, 5.0f);
  EXPECT_EQ(r1_value, 6.0f);
}

TEST(NanoRtClientTest, CompileAndRunConstantComputation) {
  absl::string_view hlo = R"(
    HloModule cst

    ENTRY e {
      ROOT cst = f32[] constant(42.0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  XlaComputation computation(module->ToProto());

  NanoRtClient client;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<NanoRtExecutable> executable,
                          client.Compile(computation));

  // Storage for executable results.
  alignas(32) float r0_value = 0.0f;

  // Prepare executable parameters, results and temp storage.
  Arguments arguments;
  Results results = {{&r0_value, 1}};

  auto event = executable->Execute(arguments, results);
  tsl::BlockUntilReady(event);

  ASSERT_TRUE(event.IsConcrete());
  EXPECT_EQ(r0_value, 42.0f);
}

TEST(NanoRtClientTest, CompileAndRunConditionalComputation) {
  absl::string_view hlo = R"(
    HloModule conditional

    %add (x: f32[]) -> f32[] {
      %p = f32[] parameter(0)
      ROOT %add = f32[] add(%p, %p)
    }

    %mul (x: f32[]) -> f32[] {
      %p = f32[] parameter(0)
      ROOT %mul = f32[] multiply(%p, %p)
    }

    ENTRY e {
      p0 = s32[] parameter(0)
      p1 = f32[] parameter(1)
      c0 = f32[] conditional(p0, p1, p1), branch_computations={%add, %mul}
      ROOT add = f32[] add(c0, c0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  XlaComputation computation(module->ToProto());

  NanoRtClient client;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<NanoRtExecutable> executable,
                          client.Compile(computation));

  // Storage for executable parameters and results.
  alignas(32) int32_t p0_value = 0;
  alignas(32) float p1_value = 2.0f;
  alignas(32) float r0_value = 0.0f;

  // Prepare executable parameters, results and temp storage.
  Arguments arguments = {{&p0_value, 1}, {&p1_value, 1}};
  Results results = {{&r0_value, 1}};
  NanoRtExecutable::ManagedTemp<32> temp(executable->temp_buffer_size());

  auto event = executable->Execute(arguments, results, temp);
  tsl::BlockUntilReady(event);

  ASSERT_TRUE(event.IsConcrete());
  EXPECT_EQ(r0_value, 8.0f);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

static absl::StatusOr<XlaComputation> CreateAddScalarsComputation() {
  XlaBuilder b("add");

  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {}), "p1");
  Add(p0, p1);

  return b.Build();
}

static absl::StatusOr<XlaComputation> CreateFibonacciComputation() {
  XlaBuilder b("fib");

  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {}), "p1");

  std::vector<XlaOp> vars = {p0, p1};

  static constexpr int kFibonacciNumber = 20;
  for (int i = 2; i < kFibonacciNumber; ++i) {
    vars.push_back(Add(vars[i - 2], vars[i - 1]));
  }

  return b.Build();
}

static void BM_NanoRtAddScalars(benchmark::State& state) {
  NanoRtClient client;

  auto computation = CreateAddScalarsComputation();
  auto executable = client.Compile(*computation);

  // Storage for executable arguments and results.
  alignas(32) float p0_value = 1.0f;
  alignas(32) float p1_value = 2.0f;
  alignas(32) float r0_value = 0.0f;

  for (auto _ : state) {
    Arguments arguments = {{&p0_value, 1}, {&p1_value, 1}};
    Results results = {{&r0_value, 1}};

    auto event = (*executable)->Execute(arguments, results);
    tsl::BlockUntilReady(event);
  }
}

BENCHMARK(BM_NanoRtAddScalars);

static void BM_NanoRtFibonacci(benchmark::State& state) {
  NanoRtClient client;

  auto computation = CreateFibonacciComputation();
  auto executable = client.Compile(*computation);

  // Storage for executable arguments and results.
  alignas(32) float p0_value = 1.0f;
  alignas(32) float p1_value = 2.0f;
  alignas(32) float r0_value = 0.0f;

  for (auto _ : state) {
    Arguments arguments = {{&p0_value, 1}, {&p1_value, 1}};
    Results results = {{&r0_value, 1}};

    auto event = (*executable)->Execute(arguments, results);
    tsl::BlockUntilReady(event);
  }
}

BENCHMARK(BM_NanoRtFibonacci);

static void BM_PjRtAddScalars(benchmark::State& state) {
  auto client = GetXlaPjrtCpuClient(/*options=*/{});
  PjRtDevice* device = (*client)->devices().front();

  auto computation = CreateAddScalarsComputation();

  CompileOptions compile_options;
  auto executable = (*client)->Compile(*computation, compile_options);

  // Storage for executable arguments.
  alignas(32) float p0_value = 1.0f;
  alignas(32) float p1_value = 2.0f;

  ExecuteOptions execute_options;

  for (auto _ : state) {
    auto p0 = (*client)->BufferFromHostBuffer(
        &p0_value, PrimitiveType::F32, {}, std::nullopt,
        PjRtClient::HostBufferSemantics::kImmutableZeroCopy, nullptr, device);

    auto p1 = (*client)->BufferFromHostBuffer(
        &p1_value, PrimitiveType::F32, {}, std::nullopt,
        PjRtClient::HostBufferSemantics::kImmutableZeroCopy, nullptr, device);

    absl::InlinedVector<PjRtBuffer*, 2> arguments = {p0->get(), p1->get()};
    CHECK_OK((*executable)->ExecuteSharded(arguments, device, execute_options));
  }
}

BENCHMARK(BM_PjRtAddScalars);

static void BM_PjRtFibonacci(benchmark::State& state) {
  auto client = GetXlaPjrtCpuClient(/*options=*/{});
  PjRtDevice* device = (*client)->devices().front();

  auto computation = CreateFibonacciComputation();

  CompileOptions compile_options;
  auto executable = (*client)->Compile(*computation, compile_options);

  // Storage for executable arguments.
  alignas(32) float p0_value = 1.0f;
  alignas(32) float p1_value = 2.0f;

  ExecuteOptions execute_options;

  for (auto _ : state) {
    auto p0 = (*client)->BufferFromHostBuffer(
        &p0_value, PrimitiveType::F32, {}, std::nullopt,
        PjRtClient::HostBufferSemantics::kImmutableZeroCopy, nullptr, device);

    auto p1 = (*client)->BufferFromHostBuffer(
        &p1_value, PrimitiveType::F32, {}, std::nullopt,
        PjRtClient::HostBufferSemantics::kImmutableZeroCopy, nullptr, device);

    absl::InlinedVector<PjRtBuffer*, 2> arguments = {p0->get(), p1->get()};
    CHECK_OK((*executable)->ExecuteSharded(arguments, device, execute_options));
  }
}

BENCHMARK(BM_PjRtFibonacci);

}  // namespace
}  // namespace xla::cpu

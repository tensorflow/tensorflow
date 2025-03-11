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
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

#define EIGEN_USE_THREADS

#include "Eigen/ThreadPool"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

using Arguments = absl::InlinedVector<NanoRtExecutable::Argument, 8>;
using Results = absl::InlinedVector<NanoRtExecutable::Result, 8>;

TEST(NanoRtClientTest, ManagedTempAlignment) {
  NanoRtExecutable::ManagedTemp<3> temp0(1);
  NanoRtExecutable::ManagedTemp<3> temp1(2);
  NanoRtExecutable::ManagedTemp<3> temp2(3);
  NanoRtExecutable::ManagedTemp<3> temp3(1024);

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&temp0.data()[0]) % Align(), 0);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&temp1.data()[0]) % Align(), 0);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&temp2.data()[0]) % Align(), 0);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&temp3.data()[0]) % Align(), 0);
}

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

TEST(NanoRtClientTest, CompileAndRunModelWithThreadPool) {
  // Implements matmul(A, C) + matmul(B, C)
  absl::string_view hlo = R"(
    HloModule test_module

ENTRY test_module {
  first = f32[1024,4096] parameter(0)
  second = f32[1024,4096] parameter(1)
  mul_par = f32[4096,4096] parameter(2)
  matmul_1 = f32[1024,4096] dot(first, mul_par), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  matmul_2 = f32[1024,4096] dot(second, mul_par), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT add = f32[1024,4096] add(matmul_1, matmul_2)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  XlaComputation computation(module->ToProto());

  NanoRtClient client;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<NanoRtExecutable> executable,
                          client.Compile(computation));

  xla::Literal first_literal =
      LiteralUtil::CreateR2FromArray2D<float>(Array2D<float>(1024, 4096, 1.0f));
  xla::Literal second_literal =
      LiteralUtil::CreateR2FromArray2D<float>(Array2D<float>(1024, 4096, 1.0f));
  xla::Literal mul_literal =
      LiteralUtil::CreateR2FromArray2D<float>(Array2D<float>(4096, 4096, 1.0f));
  xla::Literal result_literal =
      LiteralUtil::CreateR2FromArray2D<float>(Array2D<float>(1024, 4096, 0.0f));

  const float expected_result = 4096 * 2;

  absl::Span<float> first_span = first_literal.data<float>();
  absl::Span<float> second_span = second_literal.data<float>();
  absl::Span<float> mul_span = mul_literal.data<float>();
  absl::Span<float> result_span = result_literal.data<float>();

  // Prepare executable parameters, results and temp storage.
  Arguments arguments = {
      {first_span.data(), static_cast<int64_t>(first_span.size())},
      {second_span.data(), static_cast<int64_t>(second_span.size())},
      {mul_span.data(), static_cast<int64_t>(mul_span.size())}};
  Results results = {
      {result_span.data(), static_cast<int64_t>(result_span.size())}};
  NanoRtExecutable::ManagedTemp<32> temp(executable->temp_buffer_size());

  Eigen::ThreadPool tp(2);
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

  NanoRtExecutable::ExecuteOptions execute_options;
  execute_options.set_intra_op_thread_pool(&device);
  auto event = executable->Execute(arguments, results, temp, execute_options);
  tsl::BlockUntilReady(event);

  EXPECT_TRUE(event.IsConcrete());
  EXPECT_EQ(result_span[0], expected_result);
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

static void BM_NanoRtAddScalars(benchmark::State& state,
                                std::optional<Eigen::ThreadPool> tp) {
  NanoRtClient client;

  auto computation = CreateAddScalarsComputation();
  auto executable = client.Compile(*computation);

  // Storage for executable arguments and results.
  alignas(32) float p0_value = 1.0f;
  alignas(32) float p1_value = 2.0f;
  alignas(32) float r0_value = 0.0f;

  NanoRtExecutable::ExecuteOptions execute_options;
  if (tp) {
    Eigen::ThreadPoolDevice device(&tp.value(), tp->NumThreads());
    execute_options.set_intra_op_thread_pool(&device);
  }

  for (auto _ : state) {
    Arguments arguments = {{&p0_value, 1}, {&p1_value, 1}};
    Results results = {{&r0_value, 1}};

    auto event =
        (*executable)->Execute(arguments, results, {}, execute_options);
    tsl::BlockUntilReady(event);
  }
}

BENCHMARK_CAPTURE(BM_NanoRtAddScalars, no_thread_pool, std::nullopt);
BENCHMARK_CAPTURE(BM_NanoRtAddScalars, thread_pool,
                  std::make_optional<Eigen::ThreadPool>(2));

static void BM_NanoRtFibonacci(benchmark::State& state,
                               std::optional<Eigen::ThreadPool> tp) {
  NanoRtClient client;

  auto computation = CreateFibonacciComputation();
  auto executable = client.Compile(*computation);

  NanoRtExecutable::ExecuteOptions execute_options;
  if (tp) {
    Eigen::ThreadPoolDevice device(&tp.value(), tp->NumThreads());
    execute_options.set_intra_op_thread_pool(&device);
  }

  // Storage for executable arguments and results.
  alignas(32) float p0_value = 1.0f;
  alignas(32) float p1_value = 2.0f;
  alignas(32) float r0_value = 0.0f;

  for (auto _ : state) {
    Arguments arguments = {{&p0_value, 1}, {&p1_value, 1}};
    Results results = {{&r0_value, 1}};

    auto event =
        (*executable)->Execute(arguments, results, {}, execute_options);
    tsl::BlockUntilReady(event);
  }
}

BENCHMARK_CAPTURE(BM_NanoRtFibonacci, no_thread_pool, std::nullopt);
BENCHMARK_CAPTURE(BM_NanoRtFibonacci, thread_pool,
                  std::make_optional<Eigen::ThreadPool>(2));

static void BM_PjRtAddScalars(benchmark::State& state) {
  auto client = GetXlaPjrtCpuClient(/*options=*/{});
  PjRtDevice* device = (*client)->devices().front();
  PjRtMemorySpace* memory_space = *device->default_memory_space();

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
        PjRtClient::HostBufferSemantics::kImmutableZeroCopy, nullptr,
        memory_space, /*device_layout=*/nullptr);

    auto p1 = (*client)->BufferFromHostBuffer(
        &p1_value, PrimitiveType::F32, {}, std::nullopt,
        PjRtClient::HostBufferSemantics::kImmutableZeroCopy, nullptr,
        memory_space, /*device_layout=*/nullptr);

    absl::InlinedVector<PjRtBuffer*, 2> arguments = {p0->get(), p1->get()};
    CHECK_OK((*executable)->ExecuteSharded(arguments, device, execute_options));
  }
}

BENCHMARK(BM_PjRtAddScalars);

static void BM_PjRtFibonacci(benchmark::State& state) {
  auto client = GetXlaPjrtCpuClient(/*options=*/{});
  PjRtDevice* device = (*client)->devices().front();
  PjRtMemorySpace* memory_space = *device->default_memory_space();

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
        PjRtClient::HostBufferSemantics::kImmutableZeroCopy, nullptr,
        memory_space, /*device_layout=*/nullptr);

    auto p1 = (*client)->BufferFromHostBuffer(
        &p1_value, PrimitiveType::F32, {}, std::nullopt,
        PjRtClient::HostBufferSemantics::kImmutableZeroCopy, nullptr,
        memory_space, /*device_layout=*/nullptr);

    absl::InlinedVector<PjRtBuffer*, 2> arguments = {p0->get(), p1->get()};
    CHECK_OK((*executable)->ExecuteSharded(arguments, device, execute_options));
  }
}

BENCHMARK(BM_PjRtFibonacci);

}  // namespace
}  // namespace xla::cpu

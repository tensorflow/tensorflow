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

#include <array>
#include <cstddef>
#include <memory>
#include <optional>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace xla {

using HostBufferSemantics = PjRtClient::HostBufferSemantics;

TEST(StreamExecutorGpuClientBenchmarkTest, NoOp) {
  // An empty test is needed to avoid build error.
}

static void BM_AddTwoScalars(benchmark::State& state) {
  constexpr absl::string_view program = R"(
    module {
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %0 = stablehlo.add %arg0, %arg0 : tensor<f32>
        return %0 : tensor<f32>
    }
  })";

  mlir::MLIRContext context;
  auto module = xla::ParseMlirModuleString(program, context);
  CHECK_OK(module) << "Failed to parse MLIR program";

  GpuClientOptions client_option;
  client_option.allocator_config.kind = GpuAllocatorConfig::Kind::kBFC;
  client_option.allocator_config.preallocate = true;
  client_option.allocator_config.memory_fraction = 0.005;
  client_option.allowed_devices = {0};

  auto client = GetStreamExecutorGpuClient(client_option);
  CHECK_OK(client) << "Failed to create StreamExecutorGpuClient";
  PjRtDevice* device = (*client)->addressable_devices().front();

  xla::CompileOptions compile_options;
  auto executable = (*client)->CompileAndLoad(**module, compile_options);
  CHECK_OK(executable) << "Failed to compile executable";

  float input = 42.0f;

  for (auto _ : state) {
    auto input_buffer = (*client)->BufferFromHostBuffer(
        &input, F32, {},
        /*byte_strides=*/std::nullopt,
        HostBufferSemantics::kImmutableOnlyDuringCall,
        /*on_done_with_host_buffer=*/nullptr, *device->default_memory_space(),
        /*device_layout=*/nullptr);
    CHECK_OK(input_buffer) << "Failed to create input buffer";
    std::array<PjRtBuffer*, 1> args = {input_buffer->get()};

    auto result = (*executable)->ExecuteSharded(args, device, ExecuteOptions());
    CHECK_OK(result) << "Failed to execute executable";
    CHECK_EQ(result->size(), 1) << "Expected 1 result buffer";

    PjRtBuffer* result_buffer = result->front().get();
    auto literal = result_buffer->ToLiteralSync();
    CHECK_OK(literal) << "Failed to convert buffer to literal";
    VLOG(10) << "Result: " << **literal;
  }
}

BENCHMARK(BM_AddTwoScalars);

static void BM_AddManyScalars(benchmark::State& state) {
  constexpr absl::string_view program = R"(
    module {
      func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>,
                      %arg2: tensor<f32>, %arg3: tensor<f32>,
                      %arg4: tensor<f32>, %arg5: tensor<f32>,
                      %arg6: tensor<f32>, %arg7: tensor<f32>,
                      %arg8: tensor<f32>, %arg9: tensor<f32>)
    -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
    {
        %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
        %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
        %2 = stablehlo.add %arg4, %arg5 : tensor<f32>
        %3 = stablehlo.add %arg6, %arg7 : tensor<f32>
        %4 = stablehlo.add %arg8, %arg9 : tensor<f32>
        return %0, %1, %2, %3, %4 : tensor<f32>, tensor<f32>, tensor<f32>,
                                    tensor<f32>, tensor<f32>
    }
  })";

  mlir::MLIRContext context;
  auto module = xla::ParseMlirModuleString(program, context);
  CHECK_OK(module) << "Failed to parse MLIR program";

  GpuClientOptions client_option;
  client_option.allocator_config.kind = GpuAllocatorConfig::Kind::kBFC;
  client_option.allocator_config.preallocate = true;
  client_option.allocator_config.memory_fraction = 0.005;
  client_option.allowed_devices = {0};

  auto client = GetStreamExecutorGpuClient(client_option);
  CHECK_OK(client) << "Failed to create StreamExecutorGpuClient";
  CHECK_EQ((*client)->addressable_devices().size(), 1);
  PjRtDevice* device = (*client)->addressable_devices().front();

  xla::CompileOptions compile_options;
  auto executable = (*client)->CompileAndLoad(**module, compile_options);
  CHECK_OK(executable) << "Failed to compile executable";

  float inputs[10] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f};

  for (auto _ : state) {
    std::array<absl::StatusOr<std::unique_ptr<PjRtBuffer>>, 10> input_buffers;
    std::array<PjRtBuffer*, 10> args;

    for (size_t i = 0; i < 10; ++i) {
      input_buffers[i] = (*client)->BufferFromHostBuffer(
          &inputs[i], F32, {},
          /*byte_strides=*/std::nullopt,
          HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, *device->default_memory_space(),
          /*device_layout=*/nullptr);
      CHECK_OK(input_buffers[i]) << "Failed to create input buffer";
      args[i] = input_buffers[i]->get();
    }

    auto result = (*executable)->ExecuteSharded(args, device, ExecuteOptions());
    CHECK_OK(result) << "Failed to execute executable";
    CHECK_EQ(result->size(), 5) << "Expected 5 result buffer";

    for (size_t i = 0; i < 5; ++i) {
      PjRtBuffer* result_buffer = (*result)[i].get();
      auto literal = result_buffer->ToLiteralSync();
      CHECK_OK(literal) << "Failed to convert buffer to literal";
      VLOG(10) << "Result [" << i << "]: " << **literal;
    }
  }
}

BENCHMARK(BM_AddManyScalars);

}  // namespace xla

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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/se/stream_executor_executable.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace xla {

using HostBufferSemantics = PjRtClient::HostBufferSemantics;

static void RunAddTwoScalars(benchmark::State* state = nullptr) {
  constexpr absl::string_view program = R"(
    module {
      func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        %0 = stablehlo.add %arg0, %arg0 : tensor<f32>
        return %0 : tensor<f32>
    }
  })";

  auto context = std::make_unique<mlir::MLIRContext>();
  auto module = xla::ParseMlirModuleString(program, *context);
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
  auto executable = (*client)->CompileAndLoad(
      MaybeOwningMlirModule(std::move(context), std::move(*module)),
      compile_options);
  CHECK_OK(executable) << "Failed to compile executable";

  float input = 42.0f;

  auto run_iteration = [&]() {
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
    auto literal = result_buffer->ToLiteral().Await();
    CHECK_OK(literal) << "Failed to convert buffer to literal";
    VLOG(10) << "Result: " << **literal;
  };

  if (state != nullptr) {
    for (auto _ : *state) {
      run_iteration();
    }
  } else {
    run_iteration();
  }
}

TEST(StreamExecutorGpuClientBenchmarkTest, AddTwoScalars) {
  RunAddTwoScalars();
}

static void BM_AddTwoScalars(benchmark::State& state) {
  RunAddTwoScalars(&state);
}

BENCHMARK(BM_AddTwoScalars);

static void RunAddManyScalars(benchmark::State* state = nullptr) {
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

  auto context = std::make_unique<mlir::MLIRContext>();
  auto module = xla::ParseMlirModuleString(program, *context);
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
  auto executable = (*client)->CompileAndLoad(
      MaybeOwningMlirModule(std::move(context), std::move(*module)),
      compile_options);
  CHECK_OK(executable) << "Failed to compile executable";

  float inputs[10] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f};

  auto run_iteration = [&]() {
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
      auto literal = result_buffer->ToLiteral().Await();
      CHECK_OK(literal) << "Failed to convert buffer to literal";
      VLOG(10) << "Result [" << i << "]: " << **literal;
    }
  };

  if (state != nullptr) {
    for (auto _ : *state) {
      run_iteration();
    }
  } else {
    run_iteration();
  }
}

TEST(StreamExecutorGpuClientBenchmarkTest, AddManyScalars) {
  RunAddManyScalars();
}

static void BM_AddManyScalars(benchmark::State& state) {
  RunAddManyScalars(&state);
}

BENCHMARK(BM_AddManyScalars);

//===----------------------------------------------------------------------===//
// Benchmarks for loading serialized (AOT compiled) executables.
//===----------------------------------------------------------------------===//

namespace {

constexpr absl::string_view kAddModuleName = "add";
constexpr absl::string_view kDotChainModuleName = "dot_chain_500";

// HLO modules from the XLA benchmark suite, relative to the XLA source root.
constexpr std::pair<absl::string_view, absl::string_view> kHloFileModules[] = {
    {"gemma4_2b_bf16", "tools/benchmarks/hlo/hlo_gemma4_2b_bf16.hlo"},
    {"maxtext_1n1g_train_step",
     "tools/benchmarks/hlo/"
     "nv_maxtext_1n1g_jit_train_step_before_optimization.hlo"},
};

// Returns a module with `num_dots` chained dots. After optimization every dot
// still is a separate instruction (and thunk), so this gives us a module with
// a large optimized HLO graph that still executes quickly.
std::string MakeDotChainHlo(int num_dots) {
  std::string hlo = R"(
    HloModule dot_chain

    ENTRY main {
      x0 = f32[64,64] parameter(0)
      w = f32[64,64] parameter(1)
  )";
  for (int i = 0; i < num_dots; ++i) {
    absl::StrAppendFormat(
        &hlo,
        "  dot%d = f32[64,64] dot(x%d, w), lhs_contracting_dims={1}, "
        "rhs_contracting_dims={0}\n"
        "  tanh%d = f32[64,64] tanh(dot%d)\n"
        "  x%d = f32[64,64] add(tanh%d, x%d)\n",
        i, i, i, i, i + 1, i, i);
  }
  absl::StrAppendFormat(&hlo, "  ROOT result = f32[64,64] copy(x%d)\n}\n",
                        num_dots);
  return hlo;
}

absl::StatusOr<std::string> GetHloText(absl::string_view module_name) {
  if (module_name == kAddModuleName) {
    return std::string(R"(
      HloModule add

      ENTRY main {
        p0 = f32[] parameter(0)
        ROOT add = f32[] add(p0, p0)
      })");
  }
  if (module_name == kDotChainModuleName) {
    return MakeDotChainHlo(500);
  }
  for (const auto& [name, path] : kHloFileModules) {
    if (name == module_name) {
      std::string hlo;
      ABSL_RETURN_IF_ERROR(tsl::ReadFileToString(
          tsl::Env::Default(),
          tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), path), &hlo));
      return hlo;
    }
  }
  return absl::NotFoundError(absl::StrCat("Unknown module: ", module_name));
}

PjRtClient& GetLoadBenchmarkClient() {
  static PjRtClient* const client = [] {
    GpuClientOptions options;
    options.allocator_config.kind = GpuAllocatorConfig::Kind::kBFC;
    options.allocator_config.preallocate = false;
    options.allowed_devices = {0};
    absl::StatusOr<std::unique_ptr<PjRtClient>> client =
        GetStreamExecutorGpuClient(options);
    CHECK_OK(client) << "Failed to create StreamExecutorGpuClient";
    return client->release();
  }();
  return *client;
}

// Deserializes AOT compilation results and loads them into Executables
// directly, bypassing PjRt.
class AotResultLoader {
 public:
  AotResultLoader() {
    absl::StatusOr<stream_executor::Platform*> platform =
        PlatformUtil::GetPlatform("gpu");
    CHECK_OK(platform);
    platform_id_ = (*platform)->id();
    absl::StatusOr<stream_executor::StreamExecutor*> executor =
        (*platform)->ExecutorForDevice(0);
    CHECK_OK(executor);
    device_description_ = &(*executor)->GetDeviceDescription();
    absl::StatusOr<std::unique_ptr<Compiler>> compiler =
        Compiler::GetForPlatform(platform_id_);
    CHECK_OK(compiler);
    compiler_ = *std::move(compiler);
  }

  std::unique_ptr<CompiledModule> Deserialize(
      const std::string& serialized_aot_result) {
    absl::StatusOr<std::unique_ptr<CompiledModule>> aot_result =
        compiler_->LoadAotCompilationResult(serialized_aot_result);
    CHECK_OK(aot_result) << "Failed to deserialize AOT compilation result";
    return *std::move(aot_result);
  }

  std::unique_ptr<Executable> Load(const std::string& serialized_aot_result,
                                   const DebugOptions& debug_options) {
    absl::StatusOr<std::unique_ptr<Executable>> executable =
        std::move(*Deserialize(serialized_aot_result))
            .LoadExecutable(platform_id_, *device_description_, debug_options);
    CHECK_OK(executable) << "Failed to load executable";
    return *std::move(executable);
  }

 private:
  stream_executor::Platform::Id platform_id_;
  const stream_executor::DeviceDescription* device_description_;
  std::unique_ptr<Compiler> compiler_;
};

struct CompiledBenchmarkModule {
  // Output of PjRtExecutable::SerializeExecutable().
  std::string serialized_executable;
  // The serialized CompiledModule (GpuAotCompilationResult) that is embedded
  // in `serialized_executable`.
  std::string serialized_aot_result;
  // The debug options of the compile options embedded in
  // `serialized_executable`. PjRt uses these when loading the executable.
  DebugOptions debug_options;
  // The optimized HLO module of the executable (as it gets deserialized from
  // `serialized_aot_result`).
  HloModuleProtoWithConfig optimized_module;
  std::vector<Shape> parameter_shapes;
};

// Compiles `module_name` on first use and caches the serialized executable,
// so that the compilation is not part of any measurement.
const CompiledBenchmarkModule& GetCompiledModule(
    absl::string_view module_name) {
  // Benchmarks run sequentially, so we don't need to synchronize access.
  static absl::NoDestructor<
      absl::node_hash_map<std::string, CompiledBenchmarkModule>>
      cache;
  if (auto it = cache->find(module_name); it != cache->end()) {
    return it->second;
  }

  absl::StatusOr<std::string> hlo_text = GetHloText(module_name);
  CHECK_OK(hlo_text);
  absl::StatusOr<std::unique_ptr<HloModule>> hlo_module =
      ParseAndReturnUnverifiedModule(*hlo_text);
  CHECK_OK(hlo_module) << "Failed to parse " << module_name;

  CompiledBenchmarkModule compiled;
  for (const HloInstruction* parameter :
       (*hlo_module)->entry_computation()->parameter_instructions()) {
    compiled.parameter_shapes.push_back(parameter->shape());
  }

  CompileOptions compile_options;
  // Autotuning doesn't change what needs to be loaded, but it makes the
  // compilation of the large modules much slower.
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_autotune_level(0);
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> executable =
      GetLoadBenchmarkClient().CompileAndLoad(
          XlaComputation((*hlo_module)->ToProto()), compile_options);
  CHECK_OK(executable) << "Failed to compile " << module_name;

  absl::StatusOr<std::string> serialized = (*executable)->SerializeExecutable();
  CHECK_OK(serialized);
  compiled.serialized_executable = *std::move(serialized);

  absl::StatusOr<ExecutableAndOptionsProto> proto =
      SerializedGpuExecutableFromString(compiled.serialized_executable);
  CHECK_OK(proto);
  compiled.serialized_aot_result = proto->serialized_executable();
  compiled.debug_options =
      proto->compile_options().executable_build_options().debug_options();

  std::unique_ptr<CompiledModule> aot_result =
      AotResultLoader().Deserialize(compiled.serialized_aot_result);
  CHECK_NE(aot_result->optimized_module(), nullptr);
  compiled.optimized_module =
      aot_result->optimized_module()->ToProtoWithConfig();

  return cache->emplace(module_name, std::move(compiled)).first->second;
}

// Returns the number of heap bytes currently in use by the application, or
// std::nullopt if the allocator doesn't tell us. With glibc malloc this is the
// sum of the allocated bytes in the arenas (`uordblks`) and in mmapped chunks
// (`hblkhd`). TCMalloc overrides mallinfo() and mallinfo2(), reports the same
// value as the "generic.current_allocated_bytes" property in `uordblks` and
// sets `hblkhd` to zero.
//
// Older glibc versions only have mallinfo() which reports `int` fields, so in
// that case only the lower 32 bits of the result are meaningful. This is fine
// as long as we only look at differences smaller than 2 GiB (see
// `HeapBytesDelta`).
std::optional<uint64_t> GetHeapBytesInUse() {
#if defined(__GLIBC__) && \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
  struct mallinfo2 info = mallinfo2();
  return info.uordblks + info.hblkhd;
#elif defined(__GLIBC__)
  struct mallinfo info = mallinfo();
  return static_cast<uint32_t>(info.uordblks) +
         static_cast<uint32_t>(info.hblkhd);
#else
  return std::nullopt;
#endif
}

// Returns `after - before` for two values returned by `GetHeapBytesInUse`.
int64_t HeapBytesDelta(uint64_t before, uint64_t after) {
#if defined(__GLIBC__) && \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33))
  return static_cast<int64_t>(after - before);
#else
  return static_cast<int32_t>(static_cast<uint32_t>(after - before));
#endif
}

// Reports the heap memory retained by one result of `create` as the
// `heap_bytes_retained` counter. This runs outside of the timed loop.
template <typename CreateFn>
void ReportRetainedHeapBytes(benchmark::State& state, CreateFn create) {
  // The first call can populate global caches, which we don't want to count.
  create();
  std::optional<uint64_t> heap_bytes_before = GetHeapBytesInUse();
  if (!heap_bytes_before.has_value()) {
    return;
  }
  auto created = create();
  benchmark::DoNotOptimize(created);
  std::optional<uint64_t> heap_bytes_after = GetHeapBytesInUse();
  if (!heap_bytes_after.has_value()) {
    return;
  }
  state.counters["heap_bytes_retained"] = static_cast<double>(
      HeapBytesDelta(*heap_bytes_before, *heap_bytes_after));
}

// Calls `create` in every iteration of the benchmark loop. The destruction of
// the result is not part of the measurement.
template <typename CreateFn>
void MeasureCreation(benchmark::State& state, CreateFn create) {
  for (auto _ : state) {
    auto created = create();
    benchmark::DoNotOptimize(created);
    state.PauseTiming();
    created.reset();
    state.ResumeTiming();
  }
}

std::unique_ptr<PjRtLoadedExecutable> LoadSerializedExecutable(
    const CompiledBenchmarkModule& module) {
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> executable =
      GetLoadBenchmarkClient().LoadSerializedExecutable(
          module.serialized_executable, /*options=*/std::nullopt,
          LoadOptions());
  CHECK_OK(executable) << "Failed to load serialized executable";
  return *std::move(executable);
}

// Creates zero-initialized device buffers for the given parameter shapes.
absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> CreateArguments(
    absl::Span<const Shape> parameter_shapes, PjRtDevice& device) {
  ABSL_ASSIGN_OR_RETURN(PjRtMemorySpace * memory_space,
                   device.default_memory_space());
  std::vector<std::unique_ptr<PjRtBuffer>> arguments;
  for (const Shape& shape : parameter_shapes) {
    Literal literal = Literal::CreateFromShape(shape);
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<PjRtBuffer> buffer,
        GetLoadBenchmarkClient().BufferFromHostLiteral(literal, memory_space));
    ABSL_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
    arguments.push_back(std::move(buffer));
  }
  return arguments;
}

void ExecuteAndWait(PjRtLoadedExecutable& executable, PjRtDevice& device,
                    absl::Span<const std::unique_ptr<PjRtBuffer>> arguments) {
  std::vector<PjRtBuffer*> argument_ptrs;
  for (const std::unique_ptr<PjRtBuffer>& argument : arguments) {
    argument_ptrs.push_back(argument.get());
  }
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> results =
      executable.ExecuteSharded(argument_ptrs, &device, ExecuteOptions());
  CHECK_OK(results) << "Failed to execute executable";
  for (const std::unique_ptr<PjRtBuffer>& result : *results) {
    CHECK_OK(result->GetReadyFuture().Await());
  }
}

PjRtDevice& GetDevice() {
  return *GetLoadBenchmarkClient().addressable_devices().front();
}

TEST(StreamExecutorGpuClientBenchmarkTest, LoadAndExecuteSerializedExecutable) {
  const CompiledBenchmarkModule& module = GetCompiledModule(kAddModuleName);
  std::unique_ptr<PjRtLoadedExecutable> executable =
      LoadSerializedExecutable(module);
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<PjRtBuffer>> arguments,
                       CreateArguments(module.parameter_shapes, GetDevice()));
  ExecuteAndWait(*executable, GetDevice(), arguments);
  EXPECT_NE(AotResultLoader().Load(module.serialized_aot_result,
                                   module.debug_options),
            nullptr);
  EXPECT_OK(HloModule::CreateFromProtoWithConfig(module.optimized_module));
}

// Makes sure that the HLO files used by the benchmarks exist and parse, since
// the benchmarks themselves don't run on TAP.
TEST(StreamExecutorGpuClientBenchmarkTest, HloFileModulesParse) {
  for (const auto& [name, path] : kHloFileModules) {
    SCOPED_TRACE(name);
    absl::StatusOr<std::string> hlo_text = GetHloText(name);
    ASSERT_OK(hlo_text);
    EXPECT_OK(ParseAndReturnUnverifiedModule(*hlo_text));
  }
}

// Measures PjRtClient::LoadSerializedExecutable.
void BM_LoadSerializedExecutable(benchmark::State& state,
                                 absl::string_view module_name) {
  const CompiledBenchmarkModule& module = GetCompiledModule(module_name);
  auto load = [&] { return LoadSerializedExecutable(module); };
  ReportRetainedHeapBytes(state, load);
  MeasureCreation(state, load);
}

// Measures deserializing a GpuAotCompilationResult and loading it into an
// Executable, which is the part of LoadSerializedExecutable below the PjRt
// layer.
void BM_GpuAotFromSerializedAndLoad(benchmark::State& state,
                                    absl::string_view module_name) {
  const CompiledBenchmarkModule& module = GetCompiledModule(module_name);
  AotResultLoader loader;
  auto load = [&] {
    return loader.Load(module.serialized_aot_result, module.debug_options);
  };
  ReportRetainedHeapBytes(state, load);
  MeasureCreation(state, load);
}

// Measures creating the optimized HloModule from its proto and computing its
// fingerprint, which is what loading an executable spends on the HloModule.
// This is an upper bound of what we can save by not materializing the module.
void BM_HloModuleFromProto(benchmark::State& state,
                           absl::string_view module_name) {
  const CompiledBenchmarkModule& module = GetCompiledModule(module_name);
  auto create = [&] {
    absl::StatusOr<std::unique_ptr<HloModule>> hlo_module =
        HloModule::CreateFromProtoWithConfig(module.optimized_module);
    CHECK_OK(hlo_module);
    std::string fingerprint = (*hlo_module)->GetFingerprint128();
    benchmark::DoNotOptimize(fingerprint);
    return *std::move(hlo_module);
  };
  ReportRetainedHeapBytes(state, create);
  MeasureCreation(state, create);
}

// Measures the first execution of a freshly loaded executable. Loading and
// destroying the executable are not part of the measurement. The
// `heap_bytes_retained` counter reports the heap memory retained by an
// executable after loading and executing it once, so that we can see if the
// first execution materializes additional data (compare with
// BM_LoadSerializedExecutable).
void BM_FirstExecuteAfterLoad(benchmark::State& state,
                              absl::string_view module_name) {
  const CompiledBenchmarkModule& module = GetCompiledModule(module_name);
  PjRtDevice& device = GetDevice();
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> arguments =
      CreateArguments(module.parameter_shapes, device);
  if (!arguments.ok()) {
    // Typically this means that the device doesn't have enough memory.
    state.SkipWithMessage(absl::StrCat("Failed to create arguments: ",
                                       arguments.status().ToString()));
    return;
  }
  absl::Span<const std::unique_ptr<PjRtBuffer>> argument_buffers = *arguments;
  ReportRetainedHeapBytes(state, [&] {
    std::unique_ptr<PjRtLoadedExecutable> executable =
        LoadSerializedExecutable(module);
    ExecuteAndWait(*executable, device, argument_buffers);
    return executable;
  });
  for (auto _ : state) {
    state.PauseTiming();
    std::unique_ptr<PjRtLoadedExecutable> executable =
        LoadSerializedExecutable(module);
    state.ResumeTiming();
    ExecuteAndWait(*executable, device, argument_buffers);
    state.PauseTiming();
    executable.reset();
    state.ResumeTiming();
  }
}

// Measures the steady state execution of a deserialized executable.
void BM_ExecuteAfterDeserialize(benchmark::State& state,
                                absl::string_view module_name) {
  const CompiledBenchmarkModule& module = GetCompiledModule(module_name);
  PjRtDevice& device = GetDevice();
  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> arguments =
      CreateArguments(module.parameter_shapes, device);
  if (!arguments.ok()) {
    // Typically this means that the device doesn't have enough memory.
    state.SkipWithMessage(absl::StrCat("Failed to create arguments: ",
                                       arguments.status().ToString()));
    return;
  }
  absl::Span<const std::unique_ptr<PjRtBuffer>> argument_buffers = *arguments;
  std::unique_ptr<PjRtLoadedExecutable> executable =
      LoadSerializedExecutable(module);
  ExecuteAndWait(*executable, device, argument_buffers);  // Warm up.
  for (auto _ : state) {
    ExecuteAndWait(*executable, device, argument_buffers);
  }
}

// One iteration of the gemma4 benchmarks takes a few hundred milliseconds, so
// we give them more time to get a meaningful number of iterations.
constexpr double kGemmaMinTimeSeconds = 5.0;

// The load benchmarks run on all modules.
BENCHMARK_CAPTURE(BM_LoadSerializedExecutable, add, kAddModuleName)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_LoadSerializedExecutable, dot_chain_500,
                  kDotChainModuleName)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_LoadSerializedExecutable, gemma4_2b_bf16, "gemma4_2b_bf16")
    ->UseRealTime()
    ->MinTime(kGemmaMinTimeSeconds);
BENCHMARK_CAPTURE(BM_LoadSerializedExecutable, maxtext_1n1g_train_step,
                  "maxtext_1n1g_train_step")
    ->UseRealTime();

BENCHMARK_CAPTURE(BM_GpuAotFromSerializedAndLoad, add, kAddModuleName)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_GpuAotFromSerializedAndLoad, dot_chain_500,
                  kDotChainModuleName)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_GpuAotFromSerializedAndLoad, gemma4_2b_bf16,
                  "gemma4_2b_bf16")
    ->UseRealTime()
    ->MinTime(kGemmaMinTimeSeconds);
BENCHMARK_CAPTURE(BM_GpuAotFromSerializedAndLoad, maxtext_1n1g_train_step,
                  "maxtext_1n1g_train_step")
    ->UseRealTime();

BENCHMARK_CAPTURE(BM_HloModuleFromProto, add, kAddModuleName)->UseRealTime();
BENCHMARK_CAPTURE(BM_HloModuleFromProto, dot_chain_500, kDotChainModuleName)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_HloModuleFromProto, gemma4_2b_bf16, "gemma4_2b_bf16")
    ->UseRealTime()
    ->MinTime(kGemmaMinTimeSeconds);
BENCHMARK_CAPTURE(BM_HloModuleFromProto, maxtext_1n1g_train_step,
                  "maxtext_1n1g_train_step")
    ->UseRealTime();

// The execution benchmarks use zero-initialized arguments. They don't run on
// maxtext_1n1g_train_step because it donates its arguments, so they would
// have to be recreated for every execution.
BENCHMARK_CAPTURE(BM_FirstExecuteAfterLoad, add, kAddModuleName)->UseRealTime();
BENCHMARK_CAPTURE(BM_FirstExecuteAfterLoad, dot_chain_500, kDotChainModuleName)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_FirstExecuteAfterLoad, gemma4_2b_bf16, "gemma4_2b_bf16")
    ->UseRealTime()
    ->MinTime(kGemmaMinTimeSeconds);

BENCHMARK_CAPTURE(BM_ExecuteAfterDeserialize, add, kAddModuleName)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_ExecuteAfterDeserialize, dot_chain_500,
                  kDotChainModuleName)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_ExecuteAfterDeserialize, gemma4_2b_bf16, "gemma4_2b_bf16")
    ->UseRealTime();

}  // namespace
}  // namespace xla

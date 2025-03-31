/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/cpu/nanort/ifrt_client.h"

#include <array>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace xla::cpu {

// Many of the tests we run are provided by IFRT, they use NanoIfrtClient via
// the "register_nanort_for_ifrt_tests" target, which can also be used to run
// NanoIfrtClient in other tests. see the BUILD file for the list. We need a
// main function to filter out one test that doesn't seem worth supporting.

TEST(NanoIfrtClientTest, BigResult) {
  // A program that is likely to need some temporary buffers to be allocated.
  absl::string_view kBigResult =
      R"(module {
      func.func @main(%arg0: tensor<f32>) -> tensor<1024x1024xf32> {
        %0 = stablehlo.broadcast %arg0, sizes = [1024, 1024] : (tensor<f32>) -> tensor<1024x1024xf32>
        %1 = stablehlo.add %0, %0 : tensor<1024x1024xf32>
        %2 = stablehlo.dot %1, %1 : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
        return %2 : tensor<1024x1024xf32>
      }
    })";
  auto client = NanoIfrtClient::Create();
  auto compiler = client->GetDefaultCompiler();

  mlir::MLIRContext context;
  auto module = xla::ParseMlirModuleString(kBigResult, context);

  auto executable =
      compiler->Compile(std::make_unique<ifrt::HloProgram>(**module), nullptr);
  CHECK_OK(executable);

  ifrt::DType dtype(ifrt::DType::kF32);
  ifrt::Shape shape({});
  const float a = 13.0f;

  auto a_array = client->MakeArrayFromHostBuffer(
      &a, dtype, shape, std::nullopt, client->default_sharding(),
      ifrt::Client::HostBufferSemantics::kImmutableZeroCopy,
      /*on_done_with_host_buffer=*/nullptr,
      tsl::RCReference<xla::ifrt::UserContext>());
  CHECK_OK(a_array);

  auto result =
      (*executable)->Execute(absl::MakeSpan(&*a_array, 1), {}, std::nullopt);
  CHECK_EQ(result->outputs.size(), 1);

  std::vector<float> result_data(1024 * 1024);
  CHECK_OK(result->outputs[0]
               ->CopyToHostBuffer(result_data.data(), std::nullopt,
                                  ifrt::ArrayCopySemantics::kAlwaysCopy)
               .Await());
  for (const auto& v : result_data) {
    // Should be (a+a)^2 * 1024
    EXPECT_EQ(v, 692224.0);
  }
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

static absl::StatusOr<std::unique_ptr<ifrt::LoadedExecutable>> Compile(
    NanoIfrtClient* client, absl::string_view program) {
  auto compiler = client->GetDefaultCompiler();

  mlir::MLIRContext context;
  auto module = xla::ParseMlirModuleString(program, context);

  auto compile_options =
      std::make_unique<ifrt::XlaCompileOptions>(xla::CompileOptions());
  compile_options->compile_options.compile_portable_executable = true;

  return compiler->Compile(std::make_unique<ifrt::HloProgram>(**module),
                           std::move(compile_options));
}

static ifrt::DType DTypeFromPrimitiveType(PrimitiveType type) {
  switch (type) {
    case PrimitiveType::F32:
      return ifrt::DType(ifrt::DType::kF32);
    default:
      return ifrt::DType(ifrt::DType::kInvalid);
  }
}

static absl::StatusOr<tsl::RCReference<ifrt::Array>> MakeArrayFromLiteral(
    NanoIfrtClient* client, const Literal& literal,
    std::shared_ptr<const ifrt::Sharding> sharding) {
  return client->MakeArrayFromHostBuffer(
      literal.untyped_data(),
      DTypeFromPrimitiveType(literal.shape().element_type()),
      ifrt::Shape(literal.shape().dimensions()),
      /*byte_strides=*/std::nullopt, std::move(sharding),
      ifrt::Client::HostBufferSemantics::kImmutableZeroCopy,
      /*on_done_with_host_buffer=*/{},
      tsl::RCReference<xla::ifrt::UserContext>());
}

static void BM_IfRtAddScalars(benchmark::State& state) {
  constexpr absl::string_view program =
      R"(module {
        func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
          %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
          return %0 : tensor<f32>
        }
      })";

  auto client = NanoIfrtClient::Create();
  auto executable = Compile(client.get(), program);

  ifrt::Device* device = client->addressable_devices().at(0);
  std::shared_ptr<const ifrt::Sharding> sharding =
      ifrt::SingleDeviceSharding::Create(device, ifrt::MemoryKind());

  ifrt::ExecuteOptions execute_options;
  execute_options.fill_status = true;

  Literal a = LiteralUtil::CreateR0<float>(1.0f);
  Literal b = LiteralUtil::CreateR0<float>(2.0f);

  for (auto _ : state) {
    auto a_array = MakeArrayFromLiteral(client.get(), a, sharding);
    auto b_array = MakeArrayFromLiteral(client.get(), b, sharding);
    CHECK(a_array.ok() && b_array.ok());

    std::array<tsl::RCReference<ifrt::Array>, 2> args = {std::move(*a_array),
                                                         std::move(*b_array)};

    auto result = (*executable)
                      ->Execute(absl::MakeSpan(args), execute_options,
                                /*devices=*/std::nullopt);
    CHECK_OK(result->status.Await());
    CHECK_EQ(result->outputs.size(), 1);
  }
}

BENCHMARK(BM_IfRtAddScalars);

static void BM_IfRtAddManyScalars(benchmark::State& state) {
  constexpr absl::string_view program =
      R"(module {
        func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>)
           -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>,
               tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>)
        {
          %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
          %1 = stablehlo.add %arg0, %0 : tensor<f32>
          %2 = stablehlo.add %arg0, %1 : tensor<f32>
          %3 = stablehlo.add %arg0, %2 : tensor<f32>
          %4 = stablehlo.add %arg0, %3 : tensor<f32>
          %5 = stablehlo.add %arg0, %4 : tensor<f32>
          %6 = stablehlo.add %arg0, %5 : tensor<f32>
          %7 = stablehlo.add %arg0, %6 : tensor<f32>
          %8 = stablehlo.add %arg0, %7 : tensor<f32>
          %9 = stablehlo.add %arg0, %8 : tensor<f32>
          return %0, %1, %2, %3, %4, %5, %6, %7, %8, %9
            : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>,
              tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
        }
      })";

  auto client = NanoIfrtClient::Create();
  auto executable = Compile(client.get(), program);

  ifrt::Device* device = client->addressable_devices().at(0);
  std::shared_ptr<const ifrt::Sharding> sharding =
      ifrt::SingleDeviceSharding::Create(device, ifrt::MemoryKind());

  ifrt::ExecuteOptions execute_options;
  execute_options.fill_status = true;

  Literal a = LiteralUtil::CreateR0<float>(1.0f);
  Literal b = LiteralUtil::CreateR0<float>(2.0f);

  for (auto _ : state) {
    auto a_array = MakeArrayFromLiteral(client.get(), a, sharding);
    auto b_array = MakeArrayFromLiteral(client.get(), b, sharding);
    CHECK(a_array.ok() && b_array.ok());

    std::array<tsl::RCReference<ifrt::Array>, 2> args = {std::move(*a_array),
                                                         std::move(*b_array)};

    auto result = (*executable)
                      ->Execute(absl::MakeSpan(args), execute_options,
                                /*devices=*/std::nullopt);
    CHECK_OK(result->status.Await());
    CHECK_EQ(result->outputs.size(), 10);
  }
}

BENCHMARK(BM_IfRtAddManyScalars);

}  // namespace xla::cpu

int main(int argc, char** argv) {
  // This test expects copies to multiple devices to fail, but we only have one
  // device and it doesn't seem worth pretending that we have more.
  static constexpr absl::string_view kFilter =
      "-ArrayImplTest.CopyMixedSourceDevices";
  xla::ifrt::test_util::SetTestFilterIfNotUserSpecified(kFilter);

  for (int i = 1; i < argc; i++) {
    if (absl::StartsWith(argv[i], "--benchmark_filter=")) {
      ::benchmark::Initialize(&argc, argv);

      testing::InitGoogleTest(&argc, argv);
      ::benchmark::RunSpecifiedBenchmarks();
      return 0;
    }
  }

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

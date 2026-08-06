/* Copyright 2026 The OpenXLA Authors.

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
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/resource_loader.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {
namespace {

TEST(XlaFfiAotCustomCallTest, LoadAndRunAotCustomCall) {
  std::string path = tsl::GetDataDependencyFilepath(
      "tensorflow/compiler/xla/backends/gpu/ffi/"
      "xla_ffi_aot_custom_call_executable_h100");

  std::string serialized_aot_result;
  ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), path, &serialized_aot_result));

  ASSERT_OK_AND_ASSIGN(stream_executor::Platform * platform,
                       stream_executor::PlatformManager::PlatformWithId(
                           stream_executor::cuda::kCudaPlatformId));

  ASSERT_OK_AND_ASSIGN(stream_executor::StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<stream_executor::Stream> stream,
                       executor->CreateStream());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Compiler> compiler,
      Compiler::GetForPlatform(stream_executor::cuda::kCudaPlatformId));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CompiledModule> aot_result,
      compiler->LoadAotCompilationResult(serialized_aot_result));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      std::move(*aot_result)
          .LoadExecutable(compiler->PlatformId(),
                          executor->GetDeviceDescription(), DebugOptions()));

  ASSERT_OK_AND_ASSIGN(TransferManager * transfer_manager,
                       TransferManager::GetForPlatform(platform));

  stream_executor::StreamExecutorAddressAllocator allocator(executor);

  ExecutableRunOptions run_options;
  run_options.set_stream(stream.get());
  run_options.set_allocator(&allocator);

  ServiceExecutableRunOptions service_run_options(run_options);

  ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer result_buffer,
      executable->ExecuteOnStream(&service_run_options,
                                  absl::Span<const ShapedBuffer* const>()));

  ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager->TransferLiteralFromDevice(stream.get(), result_buffer));

  Literal expected = LiteralUtil::CreateR1<int32_t>({42, 42, 42, 42});
  EXPECT_EQ(expected, result);
}

}  // namespace
}  // namespace xla::gpu

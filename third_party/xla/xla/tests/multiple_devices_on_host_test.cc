/* Copyright 2018 The OpenXLA Authors.

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

#include "absl/synchronization/mutex.h"
#include "xla/client/client_library.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {
absl::StatusOr<XlaComputation> BuildComputation() {
  XlaBuilder b("computation");
  Shape scalar_s32 = ShapeUtil::MakeShape(S32, {});
  XlaOp infeed = InfeedWithToken(CreateToken(&b), scalar_s32);
  return b.Build(OutfeedWithToken(
      GetTupleElement(infeed, 0) +
          ConstantLiteral(&b, LiteralUtil::CreateR0<int32_t>(1)),
      GetTupleElement(infeed, 1), scalar_s32, ""));
}

void CompileAndExecute(
    LocalExecutable* executable, int device_ordinal, LocalClient* client,
    absl::Mutex* results_mutex,
    std::vector<std::pair<int, absl::StatusOr<ScopedShapedBuffer>>>* results) {
  xla::ExecutableRunOptions execute_options;
  execute_options.set_intra_op_thread_pool(
      client->backend().eigen_intra_op_thread_pool_device());
  execute_options.set_device_ordinal(device_ordinal);
  execute_options.set_allocator(
      xla::ClientLibrary::GetXlaService(client->platform())
          ->backend()
          .memory_allocator());
  absl::StatusOr<ScopedShapedBuffer> result =
      executable->Run(absl::Span<const ShapedBuffer* const>(), execute_options);
  {
    absl::MutexLock lock(results_mutex);
    results->emplace_back(device_ordinal, std::move(result));
  }
}

void TestWithDeviceCount(const int device_count) {
  // Run `device_count` copies of the XLA program built by BuildComputation.
  TF_ASSERT_OK_AND_ASSIGN(
      se::Platform* const platform,
      stream_executor::PlatformManager::PlatformWithName("Host"));
  xla::LocalClientOptions client_options;
  client_options.set_platform(platform);
  TF_ASSERT_OK_AND_ASSIGN(
      LocalClient* const client,
      xla::ClientLibrary::GetOrCreateLocalClient(client_options));

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation xla_computation, BuildComputation());
  TF_ASSERT_OK_AND_ASSIGN(
      auto executables,
      client->Compile(xla_computation, {}, xla::ExecutableBuildOptions{}));
  std::unique_ptr<LocalExecutable> executable = std::move(executables[0]);
  std::vector<tsl::Thread*> threads;
  absl::Mutex results_mutex;
  std::vector<std::pair<int, absl::StatusOr<ScopedShapedBuffer>>> results;
  tsl::Env* env = tsl::Env::Default();
  for (int device_ordinal = 0; device_ordinal < device_count;
       device_ordinal++) {
    tsl::Thread* t = env->StartThread(
        tsl::ThreadOptions{}, absl::StrCat("thread-", device_ordinal),
        [&executable, device_ordinal, client, &results_mutex, &results] {
          CompileAndExecute(executable.get(), device_ordinal, client,
                            &results_mutex, &results);
        });
    threads.push_back(t);
  }

  for (int device_ordinal = 0; device_ordinal < device_count;
       device_ordinal++) {
    TF_ASSERT_OK(client->TransferToInfeedLocal(
        LiteralUtil::CreateR0<int32_t>(device_ordinal * 100), device_ordinal));
  }

  for (int device_ordinal = 0; device_ordinal < device_count;
       device_ordinal++) {
    Literal outfeed(ShapeUtil::MakeShape(S32, {}));
    TF_ASSERT_OK(client->TransferFromOutfeedLocal(device_ordinal, &outfeed));
    EXPECT_EQ(outfeed,
              LiteralUtil::CreateR0<int32_t>(device_ordinal * 100 + 1));
  }

  for (int device_ordinal = 0; device_ordinal < device_count;
       device_ordinal++) {
    delete threads[device_ordinal];
  }

  for (int device_ordinal = 0; device_ordinal < device_count;
       device_ordinal++) {
    TF_ASSERT_OK(results[device_ordinal].second.status());
  }
}

// NB!  This test requires --xla_force_host_platform_device_count=4

TEST(MultipleDeviceOnHostTest, OneDevice) { TestWithDeviceCount(1); }

TEST(MultipleDeviceOnHostTest, TwoDevices) { TestWithDeviceCount(2); }

TEST(MultipleDeviceOnHostTest, ThreeDevices) { TestWithDeviceCount(3); }

TEST(MultipleDeviceOnHostTest, FourDevices) { TestWithDeviceCount(4); }
}  // namespace
}  // namespace xla

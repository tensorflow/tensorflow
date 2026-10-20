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

#include "xla/pjrt/pjrt_client_test.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/flags/commandlineflag.h"
#include "absl/flags/reflection.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/transpose.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class TestClientFactory {
 public:
  void Register(
      std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>()> factory) {
    absl::MutexLock lock(mu_);
    CHECK(!factory_);
    factory_ = std::move(factory);
  }

  std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>()> Get() const {
    absl::MutexLock lock(mu_);
    return factory_;
  }

 private:
  mutable absl::Mutex mu_;
  std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>()> factory_
      ABSL_GUARDED_BY(mu_);
};

TestClientFactory& GetGlobalTestClientFactory() {
  static auto* const factory = new TestClientFactory;
  return *factory;
}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetClient() {
  return GetGlobalTestClientFactory().Get()();
}

}  // namespace

void RegisterTestClientFactory(
    std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>()> factory) {
  GetGlobalTestClientFactory().Register(std::move(factory));
}

namespace {

std::unique_ptr<PjRtLoadedExecutable> MakeIncrementProgram(
    PjRtClient* client, bool alias, int device, bool tuplize_arg = false) {
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  XlaBuilder builder("inc");
  if (tuplize_arg) {
    shape = ShapeUtil::MakeTupleShape({shape});
  }
  auto inp = Parameter(&builder, 0, shape, "inp");
  if (tuplize_arg) {
    inp = GetTupleElement(inp, 0);
  }
  auto one = ConstantR0<int32_t>(&builder, 1);
  auto inc = Add(inp, one);
  if (alias) {
    builder.SetUpAlias({}, 0, {});
  }
  XlaComputation computation = builder.Build(inc).value();
  DeviceAssignment assignment(1, 1);
  assignment(0, 0) = device;
  CompileOptions options;
  options.parameter_is_tupled_arguments = tuplize_arg;
  options.executable_build_options.set_device_assignment(assignment);
  return client->CompileAndLoad(computation, options).value();
}

class PjRtClientTest
    : public ::testing::TestWithParam<ExecuteOptions::ExecutionMode> {};

TEST_P(PjRtClientTest, Execute) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  auto executable =
      MakeIncrementProgram(client.get(), /*alias=*/false, /*device=*/0);

  std::vector<int32_t> data(4, 0);
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  ExecuteOptions options;
  options.execution_mode = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executable->Execute({{buffer.get()}}, options));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(auto literal, results[0][0]->ToLiteral().Await());

  std::vector<int32_t> expected(4, 1);
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST_P(PjRtClientTest, ExecuteWithImmutableUntilTransferCompletes) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  auto executable =
      MakeIncrementProgram(client.get(), /*alias=*/false, /*device=*/0);

  std::vector<int32_t> data(4, 0);
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
          nullptr, client->memory_spaces()[0], /*device_layout=*/nullptr));

  ExecuteOptions options;
  options.execution_mode = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executable->Execute({{buffer.get()}}, options));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(auto literal, results[0][0]->ToLiteral().Await());

  std::vector<int32_t> expected(4, 1);
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST_P(PjRtClientTest, ExecuteWithTupleZeroCopy) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  auto executable = MakeIncrementProgram(client.get(), /*alias=*/false,
                                         /*device=*/0, /*tuplize_arg=*/true);

  std::vector<int32_t> data(4, 0);
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer, client->BufferFromHostBuffer(
                       data.data(), shape.element_type(), shape.dimensions(),
                       /*byte_strides=*/std::nullopt,
                       // Use kZeroCopy to test the correctness of
                       // `on_done_with_host_buffer`.
                       PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
                       /*on_done_with_host_buffer=*/
                       [&data]() {
                         // Deliberately modifying the content of `data`. A
                         // correct implementation of PjRt should not use `data`
                         // after `on_done_with_host_buffer` is called.
                         std::fill(data.begin(), data.end(), 1);
                       },
                       client->memory_spaces()[0], /*device_layout=*/nullptr));

  ExecuteOptions options;
  options.execution_mode = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executable->Execute({{buffer.get()}}, options));
  // Immediately release the input buffer. A correct implementation will not
  // invoke `on_done_with_host_buffer` until the execution, which can be in a
  // separate thread, finishes.
  buffer.reset();

  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(auto literal, results[0][0]->ToLiteral().Await());

  std::vector<int32_t> expected(4, 1);
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST_P(PjRtClientTest, ExecuteWithDonation) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  auto executable =
      MakeIncrementProgram(client.get(), /*alias=*/true, /*device=*/0);

  std::vector<int32_t> data(4, 0);
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableZeroCopy, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  ExecuteOptions options;
  options.execution_mode = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executable->Execute({{buffer.get()}}, options));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(auto literal, results[0][0]->ToLiteral().Await());

  std::vector<int32_t> expected(4, 1);
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST_P(PjRtClientTest, ExecuteWithDonationAbort) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  if (client->platform_id() == CpuId()) {
    // The CPU platform currently copies donated buffers if there is an
    // external reference.
    return;
  }
  auto executable =
      MakeIncrementProgram(client.get(), /*alias=*/true, /*device=*/0);

  std::vector<int32_t> data(4, 0);
  auto shared_data = std::make_shared<std::vector<int32_t>>(data);
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          shared_data->data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
          [shared_data]() {}, client->memory_spaces()[0],
          /*device_layout=*/nullptr));

  auto external_reference = buffer->AcquireExternalReference();

  ExecuteOptions options;
  options.execution_mode = GetParam();

  auto resultsor = executable->Execute({{buffer.get()}}, options);
  ASSERT_FALSE(resultsor.ok());
  EXPECT_THAT(resultsor.status().message(),
              ::testing::HasSubstr(
                  "Donation requested for buffer with external reference"));
}

TEST_P(PjRtClientTest, ExecuteWithConcurrentUsage) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  auto executable =
      MakeIncrementProgram(client.get(), /*alias=*/false, /*device=*/0);

  std::vector<int32_t> data(4, 0);
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  ExecuteOptions options;
  options.execution_mode = GetParam();

  constexpr int kNumThreads = 4;
  tsl::thread::ThreadPool thread_pool(
      tsl::Env::Default(), "ExecuteWithConcurrentUsage", kNumThreads);

  constexpr int kConcurrency = 16;
  absl::BlockingCounter blocking_counter(kConcurrency);
  std::vector<std::unique_ptr<PjRtBuffer>> results(kConcurrency);
  for (int i = 0; i < kConcurrency; ++i) {
    thread_pool.Schedule([&, &result = results[i]]() {
      auto results = executable->Execute({{buffer.get()}}, options).value();
      CHECK_EQ(results.size(), 1);
      CHECK_EQ(results[0].size(), 1);
      result = std::move(results[0][0]);
      blocking_counter.DecrementCount();
    });
  }

  blocking_counter.Wait();

  std::vector<int32_t> expected(4, 1);
  for (const auto& result : results) {
    TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteral().Await());
    EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                       *literal));
  }
}

TEST_P(PjRtClientTest, ExecuteWithConcurrentUsageAndDonation) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  auto executable =
      MakeIncrementProgram(client.get(), /*alias=*/false, /*device=*/0);
  auto executable_with_donation =
      MakeIncrementProgram(client.get(), /*alias=*/true, /*device=*/0);

  std::vector<int32_t> data(4, 0);
  std::vector<int32_t> expected(4, 1);
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableZeroCopy, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  ExecuteOptions options;
  options.execution_mode = GetParam();

  constexpr int kNumThreads = 4;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(),
                                      "ExecuteWithConcurrentUsageAndDonation",
                                      kNumThreads);

  constexpr int kConcurrentUsage = 16;
  absl::BlockingCounter blocking_counter(kConcurrentUsage + 1);

  for (int i = 0; i < kConcurrentUsage; ++i) {
    thread_pool.Schedule([&]() {
      auto results_or = executable->Execute({{buffer.get()}}, options);
      // For this test, we don't care whether this execution will fail or not,
      // as this test is to test donation logic. But if the execution succeeds,
      // the result should be correct.
      if (results_or.ok()) {
        auto& results = *results_or;
        CHECK_EQ(results.size(), 1);
        CHECK_EQ(results[0].size(), 1);
        auto literal_or = results[0][0]->ToLiteral().Await();
        if (literal_or.ok()) {
          CHECK(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                       *literal_or.value()));
        }
      }
      blocking_counter.DecrementCount();
    });
  }

  std::unique_ptr<PjRtBuffer> result;
  // The donation must succeed with concurrent usages.
  thread_pool.Schedule([&]() {
    auto results =
        executable_with_donation->Execute({{buffer.get()}}, options).value();
    CHECK_EQ(results.size(), 1);
    CHECK_EQ(results[0].size(), 1);
    result = std::move(results[0][0]);
    blocking_counter.DecrementCount();
  });

  blocking_counter.Wait();

  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteral().Await());
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST_P(PjRtClientTest, ExecuteWithWrongNumArgs) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  auto executable =
      MakeIncrementProgram(client.get(), /*alias=*/false, /*device=*/0);

  std::vector<int32_t> data(4, 0);
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  ExecuteOptions options;
  options.execution_mode = GetParam();

  // Expected 1 argument, but passing 2.
  {
    auto results_or =
        executable->Execute({{buffer.get(), buffer.get()}}, options);
    EXPECT_FALSE(results_or.ok());
    EXPECT_THAT(
        results_or.status().message(),
        ::testing::HasSubstr(
            "Execution supplied 2 arguments but compiled program expected 1"));
  }

  // Expected 1 argument, but passing 0.
  {
    auto results_or = executable->Execute({{}}, options);
    EXPECT_FALSE(results_or.ok());
    EXPECT_THAT(
        results_or.status().message(),
        ::testing::HasSubstr(
            "Execution supplied 0 arguments but compiled program expected 1"));
  }
}

INSTANTIATE_TEST_SUITE_P(
    PjRtClientTestSuite, PjRtClientTest,
    ::testing::Values(ExecuteOptions::ExecutionMode::kSynchronous,
                      ExecuteOptions::ExecutionMode::kAsynchronous));

TEST(PjRtClientTest, CopyToDevice) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  ASSERT_GT(client->addressable_devices().size(), 1);

  std::vector<int32_t> data(4, 0);
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  auto* device_1 = client->addressable_devices()[1];

  TF_ASSERT_OK_AND_ASSIGN(auto result, buffer->CopyToMemorySpace(
                                           *device_1->default_memory_space()));

  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteral().Await());

  std::vector<int32_t> expected(4, 0);
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST(PjRtClientTest, CopyToDeviceWithDest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  ASSERT_GT(client->addressable_devices().size(), 1);

  std::vector<int32_t> data(4, 0);
  Shape shape = ShapeUtil::MakeShape(S32, {4});

  TF_ASSERT_OK_AND_ASSIGN(
      auto scratchpad0,
      client->CreateUninitializedBuffer(shape, client->memory_spaces()[0]));

  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          scratchpad0.get(), /*device_layout=*/nullptr));

  auto* device_1 = client->addressable_devices()[1];

  TF_ASSERT_OK_AND_ASSIGN(
      auto scratchpad1,
      client->CreateUninitializedBuffer(buffer->on_device_shape(),
                                        *device_1->default_memory_space()));

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          buffer->CopyToMemorySpace(scratchpad1.get()));

  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteral().Await());

  std::vector<int32_t> expected(4, 0);
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
}

TEST(PjRtClientTest, CopyToDeviceAsync) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  ASSERT_GT(client->addressable_devices().size(), 1);

  std::vector<int32_t> data(4, 0);
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  auto* device_1 = client->addressable_devices()[1];

  constexpr int kNumThreads = 4;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "CopyToDeviceAsync",
                                      kNumThreads);

  constexpr int kConcurrentCopy = 16;
  std::vector<std::unique_ptr<PjRtBuffer>> results(kConcurrentCopy);
  for (int i = 0; i < kConcurrentCopy; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(results[i], buffer->CopyToMemorySpace(
                                            *device_1->default_memory_space()));
  }

  // The destructor of PjRtCpuBuffer should wait for outstanding copy.
  buffer.reset();

  for (const auto& result : results) {
    ASSERT_TRUE(result);
    TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteral().Await());

    std::vector<int32_t> expected(4, 0);
    EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                       *literal));
  }
}

TEST(PjRtClientTest, CopyToDeviceAsyncExternalCpuOnly) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  ASSERT_GT(client->addressable_devices().size(), 1);

  // Skip non-CPU platforms.
  if (client->platform_id() != CpuId()) {
    GTEST_SKIP() << "This test is for CPU only.";
  }

  alignas(cpu::MinAlign()) std::array<int32_t, 4> data;
  data.fill(0);
  auto* data_ptr = data.data();
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->CreateViewOfDeviceBuffer(
          data_ptr, shape, client->memory_spaces()[0],
          /*on_delete_callback=*/[data = std::move(data)]() mutable {
            (void)data;
          }));

  auto* device_1 = client->addressable_devices()[1];

  constexpr int kNumThreads = 4;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(),
                                      "CopyToDeviceAsyncExternal", kNumThreads);

  constexpr int kConcurrentCopy = 16;
  std::vector<std::unique_ptr<PjRtBuffer>> results(kConcurrentCopy);
  for (int i = 0; i < kConcurrentCopy; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(results[i], buffer->CopyToMemorySpace(
                                            *device_1->default_memory_space()));
  }

  // The destructor of PjRtCpuBuffer should wait for outstanding copy.
  buffer.reset();

  for (const auto& result : results) {
    ASSERT_TRUE(result);
    TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteral().Await());

    std::vector<int32_t> expected(4, 0);
    EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                       *literal));
  }
}

TEST(PjRtClientTest, CreateViewOfUnalignedBufferReturnsErrorCpuOnly) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  ASSERT_GT(client->addressable_devices().size(), 1);

  // Skip non-CPU platforms.
  if (client->platform_id() != CpuId()) {
    GTEST_SKIP() << "This test is for CPU only.";
  }

  alignas(cpu::MinAlign()) std::array<int32_t, 5> data;
  auto* data_ptr = data.data();

  // Pointer to the second element is always unaligned, because it's shifted by
  // 4 bytes (size of int32_t) from the original pointer.
  auto* unaligned_ptr = data_ptr + 1;

  // Shape with a size smaller than the original data vector, because the
  // 'unaligned_ptr' points to the second element.
  Shape shape = ShapeUtil::MakeShape(S32, {4});

  // Attempt to create a view of the unaligned buffer. Expect an error.
  auto result = client->CreateViewOfDeviceBuffer(
      unaligned_ptr, shape, client->memory_spaces()[0],
      /*on_delete_callback=*/std::function<void()>());

  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr("unaligned data"));
}

TEST(PjRtClientTest, FulfillAliasBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());

  std::vector<int32_t> data{1, 2, 3, 4, 5, 6};
  Shape shape = ShapeUtil::MakeShape(S32, {2, 3});
  TF_ASSERT_OK_AND_ASSIGN(
      *shape.mutable_layout(),
      client->GetDefaultLayout(shape.element_type(), shape.dimensions()));
  TF_ASSERT_OK_AND_ASSIGN(
      auto alias_buffer,
      client->CreateAliasBuffer(shape, client->memory_spaces()[0]));
  auto future = alias_buffer.first->ToLiteral();

  TF_ASSERT_OK_AND_ASSIGN(
      auto param,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  ASSERT_NE(alias_buffer.second, nullptr);
  TF_ASSERT_OK(std::move(alias_buffer.second)(param.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto shared_literal, future.Await());

  std::vector<int32_t> expected = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(shared_literal->data<int32_t>(), expected);
}

TEST(PjRtClientTest, Bitcast) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());

  std::vector<int32_t> data{3};
  Shape shape = ShapeUtil::MakeShape(S32, {});
  TF_ASSERT_OK_AND_ASSIGN(
      *shape.mutable_layout(),
      client->GetDefaultLayout(shape.element_type(), shape.dimensions()));

  Shape new_shape = ShapeUtil::MakeShape(S32, {1});
  TF_ASSERT_OK_AND_ASSIGN(*new_shape.mutable_layout(),
                          client->GetDefaultLayout(new_shape.element_type(),
                                                   new_shape.dimensions()));
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(
      auto bitcast_buffer,
      buffer->Bitcast(new_shape.element_type(), new_shape.dimensions(),
                      &new_shape.layout()));
  EXPECT_TRUE(buffer->IsDeleted());
  ASSERT_EQ(bitcast_buffer->on_device_shape(), new_shape);

  auto future = bitcast_buffer->ToLiteral();
  TF_ASSERT_OK_AND_ASSIGN(auto shared_literal, future.Await());
  std::vector<int32_t> expected = {3};
  EXPECT_EQ(shared_literal->data<int32_t>(), expected);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> MakeFloatBuffer(
    PjRtClient* client, const std::vector<float>& data,
    absl::Span<const int64_t> dimensions) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  return client->BufferFromHostBuffer(
      data.data(), shape.element_type(), shape.dimensions(),
      /*byte_strides=*/std::nullopt,
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
      client->memory_spaces()[0], /*device_layout=*/nullptr);
}

TEST(PjRtClientTest, DuplicateDonationError) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  constexpr char kProgram[] =
      R"(HloModule DuplicateDonationError, input_output_alias={ {0}: (1, {}, must-alias), {1}: (2, {}, must-alias) }
ENTRY DuplicateDonationError() -> (f32[2, 2], f32[2, 2]) {
    %input0 = f32[2, 2] parameter(0)
    %input1 = f32[2, 2] parameter(1) // donated
    %input2 = f32[2, 2] parameter(2) // donated
    %input3 = f32[2, 2] parameter(3)
    %tmp1 = f32[2, 2] add(%input0, %input1)
    %tmp2 = f32[2, 2] add(%input2, %input3)
    ROOT %result = (f32[2, 2], f32[2, 2]) tuple(%tmp1, %tmp2)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kProgram, {}));
  XlaComputation xla_computation(hlo_module->ToProto());
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_executable,
                          client->CompileAndLoad(xla_computation, {}));

  std::vector<float> data(4, 0);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer0,
                          MakeFloatBuffer(client.get(), data, {2, 2}));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer1,
                          MakeFloatBuffer(client.get(), data, {2, 2}));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer2,
                          MakeFloatBuffer(client.get(), data, {2, 2}));

  xla::ExecuteOptions options;
  {
    auto result = pjrt_executable->Execute(/*argument_handles=*/{{
                                               buffer0.get(),
                                               buffer1.get(),
                                               buffer1.get(),
                                               buffer0.get(),
                                           }},
                                           /*options=*/options);
    ASSERT_FALSE(result.ok());
    EXPECT_THAT(result.status().message(),
                ::testing::HasSubstr("f(donate(a), donate(a))"));
  }
  {
    auto result = pjrt_executable->Execute(/*argument_handles=*/{{
                                               buffer1.get(),
                                               buffer1.get(),
                                               buffer2.get(),
                                               buffer0.get(),
                                           }},
                                           /*options=*/options);
    ASSERT_FALSE(result.ok());
    EXPECT_THAT(result.status().message(),
                ::testing::HasSubstr("f(a, donate(a))"));
  }
  {
    auto result = pjrt_executable->Execute(/*argument_handles=*/{{
                                               buffer0.get(),
                                               buffer1.get(),
                                               buffer2.get(),
                                               buffer2.get(),
                                           }},
                                           /*options=*/options);
    ASSERT_FALSE(result.ok());
    EXPECT_THAT(result.status().message(),
                ::testing::HasSubstr("f(donate(a), a)"));
  }
}

TEST(PjRtClientTest, RuntimeDonationDenialMustAliasFails) {
  static constexpr char kProgram[] =
      R"(
HloModule RuntimeDonationDenialMustAliasFails,
          input_output_alias={ {}: (0, {}, must-alias) }

ENTRY RuntimeDonationDenialMustAliasFails() -> f32[2, 2] {
    ROOT %param = f32[2, 2] parameter(0)
})";

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kProgram, {}));
  XlaComputation xla_computation(hlo_module->ToProto());
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_executable,
                          client->CompileAndLoad(xla_computation, {}));

  std::vector<float> data(4, 0);
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
          client->memory_spaces()[0], /*device_layout=*/nullptr));

  ExecuteOptions options;
  options.non_donatable_input_indices.insert(0);
  auto result =
      pjrt_executable->Execute(/*argument_handles=*/{{buffer.get()}}, options);
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(),
              ::testing::HasSubstr(
                  "An input was configured to be must-alias at compile time "
                  "but not donated at runtime"));
}

TEST(PjRtClientTest, GetDefaultLayout) {}

TEST(PjRtClientTest, IsCApiTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  EXPECT_FALSE(client->IsCApi());
}

TEST(PjRtClientTest, ClearPeakMemory) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  PjRtDevice* device = client->addressable_devices()[0];

  if (absl::IsUnimplemented(device->ClearMemoryStats())) {
    GTEST_SKIP() << "ClearMemoryStats not supported on this platform.";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto initial_stats, device->GetAllocatorStats());

  // alloc
  const int64_t kAllocSize = 16 * 1024;
  const int64_t num_elements = kAllocSize / sizeof(float);
  const std::vector<float> data(num_elements, 1.0f);
  const Shape shape = ShapeUtil::MakeShape(F32, {num_elements});

  TF_ASSERT_OK_AND_ASSIGN(PjRtMemorySpace * memory_space,
                          device->default_memory_space());

  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->BufferFromHostBuffer(
          data.data(), shape.element_type(), shape.dimensions(),
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/nullptr, memory_space,
          /*device_layout=*/nullptr));

  ASSERT_OK(buffer->GetReadyFuture().Await());

  TF_ASSERT_OK_AND_ASSIGN(auto alloc_stats, device->GetAllocatorStats());
  ASSERT_EQ(alloc_stats.bytes_in_use, initial_stats.bytes_in_use + kAllocSize);
  ASSERT_EQ(alloc_stats.peak_bytes_in_use,
            initial_stats.peak_bytes_in_use + kAllocSize);
  ASSERT_EQ(alloc_stats.bytes_in_use, alloc_stats.peak_bytes_in_use);
  ASSERT_EQ(alloc_stats.peak_allocated_bytes,
            initial_stats.peak_allocated_bytes + kAllocSize);

  // dealloc
  buffer.reset();

  TF_ASSERT_OK_AND_ASSIGN(auto dealloc_stats, device->GetAllocatorStats());
  ASSERT_EQ(initial_stats.bytes_in_use, dealloc_stats.bytes_in_use);
  ASSERT_EQ(dealloc_stats.peak_bytes_in_use, alloc_stats.peak_bytes_in_use);
  ASSERT_EQ(dealloc_stats.peak_allocated_bytes,
            alloc_stats.peak_allocated_bytes);

  absl::Status clear_status = device->ClearMemoryStats();
  if (!absl::IsUnimplemented(clear_status)) {
    ASSERT_OK(clear_status);
    TF_ASSERT_OK_AND_ASSIGN(auto clear_stats, device->GetAllocatorStats());
    EXPECT_EQ(clear_stats.bytes_in_use, dealloc_stats.bytes_in_use);
    EXPECT_EQ(clear_stats.peak_bytes_in_use, dealloc_stats.bytes_in_use);
    EXPECT_EQ(clear_stats.peak_allocated_bytes,
              dealloc_stats.bytes_in_use + dealloc_stats.bytes_reserved);
  }
}
struct LinearizePackTestParam {
  PjRtClient::HostBufferSemantics host_buffer_semantics;
  bool need_transpose;
  std::string test_name;
};

inline std::string PrintLinearizePackTestParam(
    const ::testing::TestParamInfo<LinearizePackTestParam>& info) {
  return info.param.test_name;
}

class PjRtClientLinearizePackTest
    : public ::testing::TestWithParam<LinearizePackTestParam> {};

TEST_P(PjRtClientLinearizePackTest, PackSubbyteType) {
  const LinearizePackTestParam& param = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());

  std::vector<int8_t> data{1, 2, 3, 4};
  Shape shape = ShapeUtil::MakeShape(S4, {2, 2});

  ASSERT_GE(client->addressable_devices().size(), 1);
  auto device = client->addressable_devices()[0];
  auto* memory_space = *device->default_memory_space();

  std::optional<absl::Span<int64_t const>> byte_strides;
  std::vector<int64_t> custom_strides;
  if (param.need_transpose) {
    custom_strides = {1, 2};
    byte_strides = custom_strides;
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer, client->BufferFromHostBuffer(
                       data.data(), shape.element_type(), shape.dimensions(),
                       byte_strides, param.host_buffer_semantics, nullptr,
                       memory_space, /*device_layout=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(auto literal, buffer->ToLiteral().Await());

  std::vector<s4> expected;
  if (param.need_transpose) {
    expected = {s4(1), s4(3), s4(2), s4(4)};
  } else {
    expected = {s4(1), s4(2), s4(3), s4(4)};
  }
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::CreateR2<s4>(
          {{expected[0], expected[1]}, {expected[2], expected[3]}}),
      *literal));
}

INSTANTIATE_TEST_SUITE_P(
    PjRtClientLinearizePackTestSuite, PjRtClientLinearizePackTest,
    ::testing::Values(
        LinearizePackTestParam{
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, false,
            "ImmutableOnlyDuringCall_NoTranspose"},
        LinearizePackTestParam{
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, true,
            "ImmutableOnlyDuringCall_Transpose"},
        LinearizePackTestParam{
            PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
            false, "ImmutableUntilTransferCompletes_NoTranspose"},
        LinearizePackTestParam{
            PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes,
            true, "ImmutableUntilTransferCompletes_Transpose"}),
    PrintLinearizePackTestParam);

absl::StatusOr<std::shared_ptr<Literal>> RoundTripHostBuffer(
    PjRtClient* client, PjRtMemorySpace* memory_space, const void* data,
    const Shape& shape,
    std::optional<absl::Span<int64_t const>> byte_strides = std::nullopt,
    PjRtClient::HostBufferSemantics semantics =
        PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
    const Layout* device_layout = nullptr) {
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostBuffer(data, shape.element_type(),
                                   shape.dimensions(), byte_strides, semantics,
                                   /*on_done_with_host_buffer=*/nullptr,
                                   memory_space, device_layout));
  ABSL_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
  return buffer->ToLiteral().Await();
}

TEST(PjRtClientTest, ChunkedLinearizeAndDelinearizeRoundTrip) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client, GetClient());
  auto* common_client = dynamic_cast<CommonPjRtClient*>(client.get());
  if (common_client == nullptr) {
    GTEST_SKIP() << "Requires CommonPjRtClient.";
  }
  ASSERT_FALSE(client->addressable_devices().empty());
  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());

  constexpr absl::string_view kTargetChunkBytes = "2097152";  // 2 MiB
  absl::CommandLineFlag* chunk_flag = nullptr;
  std::string prev_chunk_bytes;
  if (IsTpuId(common_client->platform_id())) {
    chunk_flag =
        absl::FindCommandLineFlag("pjrt_tpu_chunked_transfer_target_bytes");
    ASSERT_NE(chunk_flag, nullptr);
    prev_chunk_bytes = chunk_flag->CurrentValue();
    std::string err;
    ASSERT_TRUE(chunk_flag->ParseFrom(kTargetChunkBytes, &err)) << err;
  }
  absl::Cleanup restore_chunk_flag = [&] {
    if (chunk_flag != nullptr) {
      std::string err;
      CHECK(chunk_flag->ParseFrom(prev_chunk_bytes, &err)) << err;
    }
  };

  Shape probe_host = ShapeUtil::MakeShape(F32, {4, 512, 512});
  Shape probe_dev = probe_host;
  probe_dev.mutable_layout()->mutable_tiles()->assign({Tile({8, 128})});
  std::shared_ptr<TransposePlan> probe_plan =
      common_client->CreateChunkedTransposePlan(
          /*is_h2d=*/true, probe_host, /*host_byte_strides=*/std::nullopt,
          probe_dev);
  if (probe_plan == nullptr) {
    ASSERT_FALSE(IsTpuId(common_client->platform_id()));
    GTEST_SKIP() << "Backend does not support chunked transpose plans.";
  }
  // Verify plan caching returns the same shared_ptr instance.
  EXPECT_EQ(probe_plan, common_client->CreateChunkedTransposePlan(
                            /*is_h2d=*/true, probe_host,
                            /*host_byte_strides=*/std::nullopt, probe_dev));

  // 1. 2D shapes exceeding 8 MiB (4 * 2 MiB target chunk size) with partial
  // trailing chunks across F32, BF16, and S8 (plus sub-byte S4 falling back to
  // unchunked), including odd (1-byte aligned) host addresses and mixed
  // chunked/unchunked H2D and D2H.
  for (const Shape& shape : {
           ShapeUtil::MakeShape(F32, {2048, 1025}),
           ShapeUtil::MakeShape(BF16, {2048, 2050}),
           ShapeUtil::MakeShape(S8, {2048, 4100}),
           ShapeUtil::MakeShape(S4, {2048, 8200}),
       }) {
    SCOPED_TRACE(shape.ToString(true));
    ASSERT_OK_AND_ASSIGN(Literal src_literal, MakeFakeLiteral(shape));
    for (PjRtClient::HostBufferSemantics semantics :
         {PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes}) {
      ASSERT_OK_AND_ASSIGN(
          std::shared_ptr<Literal> dst_literal,
          RoundTripHostBuffer(client.get(), memory_space,
                              src_literal.untyped_data(), shape,
                              /*byte_strides=*/std::nullopt, semantics));
      EXPECT_TRUE(LiteralTestUtil::Equal(src_literal, *dst_literal));
    }

    if (shape.element_type() != S4) {
      const size_t byte_size = src_literal.size_bytes();
      std::vector<uint8_t> src_storage(byte_size + 8, 0);
      std::vector<uint8_t> dst_storage(byte_size + 8, 0xCD);
      uint8_t* odd_src =
          src_storage.data() +
          ((reinterpret_cast<uintptr_t>(src_storage.data()) & 1) == 0 ? 1 : 0);
      uint8_t* odd_dst =
          dst_storage.data() +
          ((reinterpret_cast<uintptr_t>(dst_storage.data()) & 1) == 0 ? 1 : 0);
      ASSERT_EQ(reinterpret_cast<uintptr_t>(odd_src) & 1, 1u);
      ASSERT_EQ(reinterpret_cast<uintptr_t>(odd_dst) & 1, 1u);
      std::memcpy(odd_src, src_literal.untyped_data(), byte_size);

      ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<PjRtBuffer> odd_buffer,
          client->BufferFromHostBuffer(
              odd_src, shape.element_type(), shape.dimensions(),
              /*byte_strides=*/std::nullopt,
              PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
              /*on_done_with_host_buffer=*/nullptr, memory_space,
              /*device_layout=*/nullptr));
      MutableBorrowingLiteral borrowing_dst(reinterpret_cast<char*>(odd_dst),
                                            shape);
      ASSERT_OK(odd_buffer->ToLiteral(&borrowing_dst).Await());
      EXPECT_EQ(std::memcmp(odd_src, odd_dst, byte_size), 0);
      // Re-read into a fresh literal to verify D2H into `odd_dst` did not
      // corrupt the device buffer.
      ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> verify_literal,
                           odd_buffer->ToLiteral().Await());
      EXPECT_TRUE(LiteralTestUtil::Equal(src_literal, *verify_literal));
    }
  }

  if (IsTpuId(common_client->platform_id())) {
    std::string err;
    Shape mix_shape = ShapeUtil::MakeShape(F32, {2048, 1025});
    ASSERT_OK_AND_ASSIGN(Literal mix_src, MakeFakeLiteral(mix_shape));

    // (a) Chunked H2D (2 MiB chunks) -> unchunked D2H (flag = 0).
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtBuffer> chunked_h2d_buf,
        client->BufferFromHostBuffer(
            mix_src.untyped_data(), mix_shape.element_type(),
            mix_shape.dimensions(), /*byte_strides=*/std::nullopt,
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr, memory_space,
            /*device_layout=*/nullptr));
    ASSERT_OK(chunked_h2d_buf->GetReadyFuture().Await());

    // Verify a tiny positive target chunk size clamps the chunk count to
    // <= 1024 instead of overflowing int or creating millions of chunks.
    ASSERT_TRUE(chunk_flag->ParseFrom("1", &err)) << err;
    std::shared_ptr<TransposePlan> clamped_plan =
        common_client->CreateChunkedTransposePlan(
            /*is_h2d=*/true, mix_shape,
            /*host_byte_strides=*/std::nullopt,
            chunked_h2d_buf->on_device_shape());
    ASSERT_NE(clamped_plan, nullptr);
    EXPECT_GT(clamped_plan->Parallelism(), 1);
    EXPECT_LE(clamped_plan->Parallelism(), 1024);

    ASSERT_TRUE(chunk_flag->ParseFrom("0", &err)) << err;
    EXPECT_EQ(common_client->CreateChunkedTransposePlan(
                  /*is_h2d=*/false, probe_host,
                  /*host_byte_strides=*/std::nullopt, probe_dev),
              nullptr);
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> unchunked_d2h_lit,
                         chunked_h2d_buf->ToLiteral().Await());
    EXPECT_TRUE(LiteralTestUtil::Equal(mix_src, *unchunked_d2h_lit));

    // (b) Unchunked H2D (flag = 0) -> chunked D2H (2 MiB chunks).
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtBuffer> unchunked_h2d_buf,
        client->BufferFromHostBuffer(
            mix_src.untyped_data(), mix_shape.element_type(),
            mix_shape.dimensions(), /*byte_strides=*/std::nullopt,
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr, memory_space,
            /*device_layout=*/nullptr));
    ASSERT_OK(unchunked_h2d_buf->GetReadyFuture().Await());
    ASSERT_TRUE(chunk_flag->ParseFrom(kTargetChunkBytes, &err)) << err;
    EXPECT_NE(common_client->CreateChunkedTransposePlan(
                  /*is_h2d=*/false, probe_host,
                  /*host_byte_strides=*/std::nullopt, probe_dev),
              nullptr);
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> chunked_d2h_lit,
                         unchunked_h2d_buf->ToLiteral().Await());
    EXPECT_TRUE(LiteralTestUtil::Equal(mix_src, *chunked_d2h_lit));
  }

  // 2. Custom host byte strides: column-major and row-padded host buffers
  // across F32 (unpacked) and packed BF16/S8.
  constexpr int64_t kRows = 2048;
  constexpr int64_t kCols = 1536;
  constexpr int64_t kPaddedCols = kCols + 64;
  for (PrimitiveType elem_type : {F32, BF16, S8}) {
    SCOPED_TRACE(PrimitiveType_Name(elem_type));
    const int64_t elem_bytes = ShapeUtil::ByteSizeOfPrimitiveType(elem_type);
    Shape col_major_shape =
        ShapeUtil::MakeShapeWithDenseLayout(elem_type, {kRows, kCols}, {0, 1});
    ASSERT_OK_AND_ASSIGN(Literal col_major_literal,
                         MakeFakeLiteral(col_major_shape));
    Literal expected_literal =
        col_major_literal.Relayout(LayoutUtil::MakeDescendingLayout(2));

    const std::vector<int64_t> col_major_strides = {elem_bytes,
                                                    kRows * elem_bytes};
    ASSERT_OK_AND_ASSIGN(
        std::shared_ptr<Literal> dst_col_major,
        RoundTripHostBuffer(client.get(), memory_space,
                            col_major_literal.untyped_data(),
                            expected_literal.shape(), col_major_strides));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, *dst_col_major));

    std::vector<uint8_t> padded_host(kRows * kPaddedCols * elem_bytes, 0x7f);
    const auto* src_row_major =
        static_cast<const uint8_t*>(expected_literal.untyped_data());
    for (int64_t r = 0; r < kRows; ++r) {
      std::memcpy(&padded_host[r * kPaddedCols * elem_bytes],
                  &src_row_major[r * kCols * elem_bytes], kCols * elem_bytes);
    }
    const std::vector<int64_t> padded_strides = {kPaddedCols * elem_bytes,
                                                 elem_bytes};
    ASSERT_OK_AND_ASSIGN(
        std::shared_ptr<Literal> dst_padded,
        RoundTripHostBuffer(client.get(), memory_space, padded_host.data(),
                            expected_literal.shape(), padded_strides));
    EXPECT_TRUE(LiteralTestUtil::Equal(expected_literal, *dst_padded));

    // Also verify `CreateChunkedTransposePlan` -> `ExecuteChunk` directly on a
    // packed tiled device layout (`T(8,128)` for F32, `T(8,128)(2,1)` for BF16,
    // `T(32,128)(4,1)` for S8) when supported by the backend's packing factor.
    Shape strided_device_shape = expected_literal.shape();
    const int64_t pf = 4 / elem_bytes;
    TileVector* strided_tiles =
        strided_device_shape.mutable_layout()->mutable_tiles();
    strided_tiles->assign({Tile({8 * pf, 128})});
    if (pf > 1) {
      strided_tiles->push_back(Tile({pf, 1}));
    }
    if (common_client->CreateChunkedTransposePlan(
            /*is_h2d=*/true, expected_literal.shape(),
            /*host_byte_strides=*/std::nullopt,
            strided_device_shape) == nullptr) {
      EXPECT_EQ(elem_type, S8);
      continue;
    }
    const size_t strided_dev_bytes = ShapeUtil::ArraySize(strided_device_shape);
    auto run_h2d_chunks =
        [&](const void* src, std::optional<absl::Span<int64_t const>> strides) {
          std::shared_ptr<TransposePlan> plan =
              common_client->CreateChunkedTransposePlan(
                  /*is_h2d=*/true, expected_literal.shape(), strides,
                  strided_device_shape);
          CHECK_NE(plan, nullptr);
          std::vector<uint8_t> out(strided_dev_bytes, 0);
          for (int i = 0; i < plan->Parallelism(); ++i) {
            plan->ExecuteChunk(i, src,
                               out.data() + plan->OutputChunkOffsetBytes(i),
                               /*input_is_global=*/true,
                               /*output_is_global=*/false);
          }
          return out;
        };
    const std::vector<uint8_t> ref_dev = run_h2d_chunks(
        expected_literal.untyped_data(), /*strides=*/std::nullopt);
    EXPECT_EQ(
        run_h2d_chunks(col_major_literal.untyped_data(), col_major_strides),
        ref_dev);
    EXPECT_EQ(run_h2d_chunks(padded_host.data(), padded_strides), ref_dev);
  }

  // 3. 3D and 4D shapes exceeding 2 MiB: host-only byte-for-byte comparison of
  // chunked ExecuteChunk output against Execute and
  // CommonPjRtClient::Linearize across TPU layouts (including partial tiles),
  // plus device round-trips.
  struct TiledLayoutCase {
    Shape shape;
    std::vector<Tile> tiles;
    bool run_device_round_trip = true;
  };
  const std::vector<TiledLayoutCase> layout_cases = {
      // T(8,128) with partial minor tile.
      {ShapeUtil::MakeShape(F32, {4, 512, 1025}), {Tile({8, 128})}},
      // T(8,128) with partial 2nd-minor and minor tiles ({4, 18, 130, 200}).
      {ShapeUtil::MakeShape(F32, {4, 18, 130, 200}), {Tile({8, 128})}},
      // T(8,128)(2,1) with partial minor tile.
      {ShapeUtil::MakeShape(BF16, {4, 16, 128, 257}),
       {Tile({8, 128}), Tile({2, 1})}},
      // T(8,128)(2,1) with partial 2nd-minor and minor tiles ({4, 18, 130,
      // 200}).
      {ShapeUtil::MakeShape(BF16, {4, 18, 130, 200}),
       {Tile({8, 128}), Tile({2, 1})}},
      // T(32,128)(4,1)
      {ShapeUtil::MakeShape(S8, {4, 16, 128, 513}),
       {Tile({32, 128}), Tile({4, 1})},
       /*run_device_round_trip=*/false},
      // Non-default minor_to_major device layouts.
      {ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 512, 1025}, {1, 2, 0}),
       {Tile({8, 128})}},
      {ShapeUtil::MakeShapeWithDenseLayout(BF16, {4, 16, 129, 256},
                                           {2, 3, 0, 1}),
       {Tile({8, 128}), Tile({2, 1})}},
  };

  int partial_tile_cases = 0;
  for (const TiledLayoutCase& tc : layout_cases) {
    Shape host_shape =
        ShapeUtil::MakeShape(tc.shape.element_type(), tc.shape.dimensions());
    const Layout* device_layout =
        tc.shape.layout() != host_shape.layout() ? &tc.shape.layout() : nullptr;
    ASSERT_OK_AND_ASSIGN(Shape device_shape,
                         common_client->MakeDefaultShapeForMemorySpace(
                             memory_space, host_shape, device_layout));
    const int64_t canonical_packing =
        device_shape.layout().tiles().size() >= 2
            ? device_shape.layout().tiles(1).dimension(0)
            : 1;
    device_shape.mutable_layout()->mutable_tiles()->assign(tc.tiles.begin(),
                                                           tc.tiles.end());
    device_shape.mutable_layout()->set_element_size_in_bits(0);
    SCOPED_TRACE(device_shape.ToString(true));

    const int64_t test_packing =
        tc.tiles.size() == 2 ? tc.tiles[1].dimension(0) : 1;
    std::shared_ptr<TransposePlan> lin_plan =
        common_client->CreateChunkedTransposePlan(
            /*is_h2d=*/true, host_shape, /*host_byte_strides=*/std::nullopt,
            device_shape);
    if (test_packing != canonical_packing) {
      EXPECT_EQ(lin_plan, nullptr);
      EXPECT_EQ(common_client->CreateChunkedTransposePlan(
                    /*is_h2d=*/false, host_shape,
                    /*host_byte_strides=*/std::nullopt, device_shape),
                nullptr);
      continue;
    }
    ASSERT_NE(lin_plan, nullptr);
    ASSERT_OK_AND_ASSIGN(Literal src_literal, MakeFakeLiteral(host_shape));
    const int num_chunks = lin_plan->Parallelism();
    EXPECT_GT(num_chunks, 1);
    const size_t array_bytes = ShapeUtil::ArraySize(device_shape);
    EXPECT_EQ(lin_plan->OutputChunkOffsetBytes(num_chunks - 1) +
                  lin_plan->OutputChunkSizeBytes(num_chunks - 1),
              array_bytes);

    // Pre-fill `actual` with non-zero garbage and zero each chunk slot when
    // `InputNumElems() < OutputNumElems()`, matching
    // `PipelinedChunkedTransfer`.
    std::vector<uint8_t> actual(array_bytes, 0xAB);
    const bool zero_h2d_slot =
        lin_plan->InputNumElems() < lin_plan->OutputNumElems();
    if (zero_h2d_slot) {
      ++partial_tile_cases;
    }
    for (int i = 0; i < num_chunks; ++i) {
      uint8_t* slot = actual.data() + lin_plan->OutputChunkOffsetBytes(i);
      if (zero_h2d_slot) {
        std::memset(slot, 0, lin_plan->OutputChunkSizeBytes(i));
      }
      lin_plan->ExecuteChunk(i, src_literal.untyped_data(), slot,
                             /*input_is_global=*/true,
                             /*output_is_global=*/false);
    }
    std::vector<uint8_t> expected_plan(array_bytes, 0);
    lin_plan->Execute(src_literal.untyped_data(), expected_plan.data());
    EXPECT_EQ(actual, expected_plan);

    if (IsTpuId(common_client->platform_id())) {
      struct alignas(64) Aligned64 {
        uint8_t bytes[64];
      };
      std::vector<Aligned64> expected_lin(
          (array_bytes + sizeof(Aligned64) - 1) / sizeof(Aligned64));
      uint8_t* expected_ptr = expected_lin[0].bytes;
      ASSERT_OK(common_client->Linearize(
          absl::MakeSpan(expected_ptr, array_bytes), src_literal.untyped_data(),
          /*byte_strides=*/{}, device_shape, /*dynamic_sizes=*/{},
          memory_space));
      EXPECT_EQ(absl::MakeConstSpan(actual).first(array_bytes),
                absl::MakeConstSpan(expected_ptr, array_bytes));
    }

    std::shared_ptr<TransposePlan> delin_plan =
        common_client->CreateChunkedTransposePlan(
            /*is_h2d=*/false, host_shape, /*host_byte_strides=*/std::nullopt,
            device_shape);
    if (test_packing > 1) {
      EXPECT_EQ(delin_plan, nullptr);
    } else {
      ASSERT_NE(delin_plan, nullptr);
      const int delin_chunks = delin_plan->Parallelism();
      EXPECT_GT(delin_chunks, 1);
      EXPECT_EQ(delin_plan->InputChunkOffsetBytes(delin_chunks - 1) +
                    delin_plan->InputChunkSizeBytes(delin_chunks - 1),
                array_bytes);
      Literal delin_literal(host_shape);
      for (int i = 0; i < delin_chunks; ++i) {
        delin_plan->ExecuteChunk(
            i, actual.data() + delin_plan->InputChunkOffsetBytes(i),
            delin_literal.untyped_data(),
            /*input_is_global=*/false, /*output_is_global=*/true);
      }
      EXPECT_TRUE(LiteralTestUtil::Equal(src_literal, delin_literal));
    }

    if (tc.run_device_round_trip) {
      for (PjRtClient::HostBufferSemantics semantics :
           {PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
            PjRtClient::HostBufferSemantics::
                kImmutableUntilTransferCompletes}) {
        ASSERT_OK_AND_ASSIGN(
            std::shared_ptr<Literal> dst_literal,
            RoundTripHostBuffer(client.get(), memory_space,
                                src_literal.untyped_data(), host_shape,
                                /*byte_strides=*/std::nullopt, semantics,
                                device_layout));
        EXPECT_TRUE(LiteralTestUtil::Equal(src_literal, *dst_literal));
      }
    }
  }
  EXPECT_GE(partial_tile_cases, 4);

  // 4. Fallback conditions: <= 2 MiB tensor, dynamic shape, rank-1 shape with
  // 2D tile, non-128 minor tile, invalid sub-tiles / layout fields, and
  // unaligned sublane dim.
  auto make_bf16_tiled = [&](const Shape& logical,
                             const Layout* layout = nullptr) {
    absl::StatusOr<Shape> dev = common_client->MakeDefaultShapeForMemorySpace(
        memory_space, logical, layout);
    CHECK_OK(dev.status());
    if (dev->layout().tiles().empty()) {
      dev->mutable_layout()->mutable_tiles()->assign(
          {Tile({8, 128}), Tile({2, 1})});
    }
    return *std::move(dev);
  };
  Shape small_host_shape = ShapeUtil::MakeShape(BF16, {256, 4096});  // 2 MiB
  EXPECT_EQ(common_client->CreateChunkedTransposePlan(
                /*is_h2d=*/true, small_host_shape,
                /*host_byte_strides=*/std::nullopt,
                make_bf16_tiled(small_host_shape)),
            nullptr);

  Shape unaligned_sublane = ShapeUtil::MakeShapeWithDenseLayout(
      BF16, {2, 16, 128, 257}, {2, 3, 0, 1});
  Shape host_shape = ShapeUtil::MakeShape(BF16, {2, 16, 128, 257});
  EXPECT_EQ(common_client->CreateChunkedTransposePlan(
                /*is_h2d=*/true, host_shape, /*host_byte_strides=*/std::nullopt,
                make_bf16_tiled(host_shape, &unaligned_sublane.layout())),
            nullptr);

  Shape valid_device_shape = make_bf16_tiled(host_shape);
  auto plan_for_device_shape = [&](auto mutate_fn) {
    Shape s = valid_device_shape;
    mutate_fn(s);
    return common_client->CreateChunkedTransposePlan(
        /*is_h2d=*/true, host_shape, /*host_byte_strides=*/std::nullopt, s);
  };
  EXPECT_EQ(
      plan_for_device_shape([](Shape& s) { s.set_dynamic_dimension(0, true); }),
      nullptr);
  EXPECT_EQ(plan_for_device_shape([](Shape& s) {
              s.mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(32);
            }),
            nullptr);
  EXPECT_EQ(plan_for_device_shape([](Shape& s) {
              s.mutable_layout()->set_tail_padding_alignment_in_elements(256);
            }),
            nullptr);
  EXPECT_EQ(plan_for_device_shape([](Shape& s) {
              s.mutable_layout()->add_split_configs(SplitConfig(0, {1}));
            }),
            nullptr);
  EXPECT_EQ(plan_for_device_shape([](Shape& s) {
              s.mutable_layout()->mutable_tiles()->assign(
                  {Tile({8, 32}), Tile({2, 1})});
            }),
            nullptr);
  EXPECT_EQ(plan_for_device_shape([](Shape& s) {
              s.mutable_layout()->mutable_tiles()->assign(
                  {Tile({8, 128}), Tile({2, 1}), Tile({1, 1})});
            }),
            nullptr);
  EXPECT_EQ(plan_for_device_shape([](Shape& s) {
              s.mutable_layout()->mutable_tiles()->assign(
                  {Tile({8, 128}), Tile({2, 2})});
            }),
            nullptr);
  EXPECT_EQ(plan_for_device_shape([](Shape& s) {
              s.mutable_layout()->mutable_tiles()->assign(
                  {Tile({Tile::kMatchDimension, 128})});
            }),
            nullptr);
  EXPECT_EQ(plan_for_device_shape([](Shape& s) {
              s.mutable_layout()->mutable_tiles()->assign({Tile({8, 128})});
              s.mutable_layout()->set_element_size_in_bits(32);
            }),
            nullptr);

  Shape rank1_host = ShapeUtil::MakeShape(F32, {1024 * 1024});
  Shape rank1_device = rank1_host;
  rank1_device.mutable_layout()->mutable_tiles()->assign({Tile({8, 128})});
  EXPECT_EQ(common_client->CreateChunkedTransposePlan(
                /*is_h2d=*/true, rank1_host, /*host_byte_strides=*/std::nullopt,
                rank1_device),
            nullptr);

  Shape tiled_host_shape = host_shape;
  tiled_host_shape.mutable_layout()->mutable_tiles()->assign({Tile({8, 128})});
  for (bool is_h2d : {true, false}) {
    EXPECT_EQ(common_client->CreateChunkedTransposePlan(
                  is_h2d, tiled_host_shape,
                  /*host_byte_strides=*/std::nullopt, valid_device_shape),
              nullptr);
  }

  Shape mismatched_host_shape = ShapeUtil::MakeShape(F32, {2, 16, 128, 257});
  EXPECT_EQ(common_client->CreateChunkedTransposePlan(
                /*is_h2d=*/false, mismatched_host_shape,
                /*host_byte_strides=*/std::nullopt, valid_device_shape),
            nullptr);

  Shape no_layout_host = host_shape;
  no_layout_host.clear_layout();
  EXPECT_EQ(common_client->CreateChunkedTransposePlan(
                /*is_h2d=*/true, no_layout_host,
                /*host_byte_strides=*/std::nullopt, valid_device_shape),
            nullptr);

  if (IsTpuId(common_client->platform_id())) {
    Shape aligned_dma_device = valid_device_shape;
    aligned_dma_device.mutable_layout()->mutable_tiles()->assign(
        {Tile({2, 128}), Tile({2, 1})});
    EXPECT_NE(common_client->CreateChunkedTransposePlan(
                  /*is_h2d=*/true, host_shape,
                  /*host_byte_strides=*/std::nullopt, aligned_dma_device),
              nullptr);
    Shape unaligned_dma_host = ShapeUtil::MakeShape(BF16, {2, 15, 126, 257});
    Shape unaligned_dma_device = make_bf16_tiled(unaligned_dma_host);
    unaligned_dma_device.mutable_layout()->mutable_tiles()->assign(
        {Tile({2, 128}), Tile({2, 1})});
    EXPECT_EQ(common_client->CreateChunkedTransposePlan(
                  /*is_h2d=*/true, unaligned_dma_host,
                  /*host_byte_strides=*/std::nullopt, unaligned_dma_device),
              nullptr);
  }

  ASSERT_OK_AND_ASSIGN(Literal src_literal, MakeFakeLiteral(host_shape));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<Literal> dst_literal,
      RoundTripHostBuffer(
          client.get(), memory_space, src_literal.untyped_data(), host_shape,
          /*byte_strides=*/std::nullopt,
          PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
          &unaligned_sublane.layout()));
  EXPECT_TRUE(LiteralTestUtil::Equal(src_literal, *dst_literal));
}

}  // namespace
}  // namespace xla

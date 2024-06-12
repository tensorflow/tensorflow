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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo_parser.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/test.h"
#include "xla/tests/literal_test_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class TestClientFactory {
 public:
  void Register(
      std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>()> factory) {
    absl::MutexLock lock(&mu_);
    CHECK(!factory_);
    factory_ = std::move(factory);
  }

  std::function<absl::StatusOr<std::unique_ptr<PjRtClient>>()> Get() const {
    absl::MutexLock lock(&mu_);
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
  return client->Compile(computation, options).value();
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
          client->addressable_devices()[0]));

  ExecuteOptions options;
  options.execution_mode = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executable->Execute({{buffer.get()}}, options));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(auto literal, results[0][0]->ToLiteralSync());

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
          nullptr, client->addressable_devices()[0]));

  ExecuteOptions options;
  options.execution_mode = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executable->Execute({{buffer.get()}}, options));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(auto literal, results[0][0]->ToLiteralSync());

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
                       client->addressable_devices()[0]));

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
  TF_ASSERT_OK_AND_ASSIGN(auto literal, results[0][0]->ToLiteralSync());

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
      auto buffer, client->BufferFromHostBuffer(
                       data.data(), shape.element_type(), shape.dimensions(),
                       /*byte_strides=*/std::nullopt,
                       PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
                       nullptr, client->addressable_devices()[0]));

  ExecuteOptions options;
  options.execution_mode = GetParam();

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executable->Execute({{buffer.get()}}, options));
  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(auto literal, results[0][0]->ToLiteralSync());

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
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer, client->BufferFromHostBuffer(
                       data.data(), shape.element_type(), shape.dimensions(),
                       /*byte_strides=*/std::nullopt,
                       PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
                       nullptr, client->addressable_devices()[0]));

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
          client->addressable_devices()[0]));

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
    TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteralSync());
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
      auto buffer, client->BufferFromHostBuffer(
                       data.data(), shape.element_type(), shape.dimensions(),
                       /*byte_strides=*/std::nullopt,
                       PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
                       nullptr, client->addressable_devices()[0]));

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
        auto literal = results[0][0]->ToLiteralSync().value();
        CHECK(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
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

  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteralSync());
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                     *literal));
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
          client->addressable_devices()[0]));

  auto* device_1 = client->addressable_devices()[1];

  TF_ASSERT_OK_AND_ASSIGN(auto result, buffer->CopyToDevice(device_1));

  TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteralSync());

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
          client->addressable_devices()[0]));

  auto* device_1 = client->addressable_devices()[1];

  constexpr int kNumThreads = 4;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "CopyToDeviceAsync",
                                      kNumThreads);

  constexpr int kConcurrentCopy = 16;
  std::vector<std::unique_ptr<PjRtBuffer>> results(kConcurrentCopy);
  for (int i = 0; i < kConcurrentCopy; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(results[i], buffer->CopyToDevice(device_1));
  }

  // The destructor of TfrtCpuBuffer should wait for outstanding copy.
  buffer.reset();

  for (const auto& result : results) {
    ASSERT_TRUE(result);
    TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteralSync());

    std::vector<int32_t> expected(4, 0);
    EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                       *literal));
  }
}

TEST(PjRtClientTest, CopyToDeviceAsyncExternalCpuOnly) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  ASSERT_GT(client->addressable_devices().size(), 1);

  // Skip non-CPU platforms.
  if (client->platform_id() != CpuId()) return;

  std::vector<int32_t> data(4, 0);
  auto* data_ptr = data.data();
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer,
      client->CreateViewOfDeviceBuffer(
          data_ptr, shape, client->addressable_devices()[0],
          /*on_delete_callback=*/[data = std::move(data)]() mutable {
            data.clear();
            data.shrink_to_fit();
          }));

  auto* device_1 = client->addressable_devices()[1];

  constexpr int kNumThreads = 4;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(),
                                      "CopyToDeviceAsyncExternal", kNumThreads);

  constexpr int kConcurrentCopy = 16;
  std::vector<std::unique_ptr<PjRtBuffer>> results(kConcurrentCopy);
  for (int i = 0; i < kConcurrentCopy; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(results[i], buffer->CopyToDevice(device_1));
  }

  // The destructor of TfrtCpuBuffer should wait for outstanding copy.
  buffer.reset();

  for (const auto& result : results) {
    ASSERT_TRUE(result);
    TF_ASSERT_OK_AND_ASSIGN(auto literal, result->ToLiteralSync());

    std::vector<int32_t> expected(4, 0);
    EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR1<int32_t>(expected),
                                       *literal));
  }
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> MakeFloatBuffer(
    PjRtClient* client, const std::vector<float>& data,
    absl::Span<const int64_t> dimensions) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 2});
  return client->BufferFromHostBuffer(
      data.data(), shape.element_type(), shape.dimensions(),
      /*byte_strides=*/std::nullopt,
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
      client->addressable_devices()[0]);
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
                          client->Compile(xla_computation, {}));

  std::vector<float> data(4, 0);
  TF_ASSERT_OK_AND_ASSIGN(auto buffer0,
                          MakeFloatBuffer(client.get(), data, {2, 2}));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer1,
                          MakeFloatBuffer(client.get(), data, {2, 2}));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer2,
                          MakeFloatBuffer(client.get(), data, {2, 2}));

  {
    auto result = pjrt_executable->Execute(/*argument_handles=*/{{
                                               buffer0.get(),
                                               buffer1.get(),
                                               buffer1.get(),
                                               buffer0.get(),
                                           }},
                                           /*options=*/{});
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
                                           /*options=*/{});
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
                                           /*options=*/{});
    ASSERT_FALSE(result.ok());
    EXPECT_THAT(result.status().message(),
                ::testing::HasSubstr("f(donate(a), a)"));
  }
}

TEST(PjRtClientTest, GetDefaultLayout) {}

}  // namespace
}  // namespace xla

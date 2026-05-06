/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tsl/profiler/lib/profiler_orchestrator.h"

#include <memory>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/test.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace tsl {
namespace profiler {
namespace {

class MockProfiler : public ProfilerInterface {
 public:
  absl::Status Start() override { return absl::OkStatus(); }
  absl::Status Stop() override { return absl::OkStatus(); }
  absl::Status CollectData(tensorflow::profiler::XSpace*) override {
    return absl::OkStatus();
  }
  absl::Status Consume(void* ptr) override {
    consume_called_ = true;
    return absl::OkStatus();
  }
  absl::Status Serialize(void* ptr,
                         tensorflow::profiler::XSpace* output_space) override {
    serialize_called_ = true;
    return absl::OkStatus();
  }

  bool consume_called() const { return consume_called_; }
  bool serialize_called() const { return serialize_called_; }

 private:
  bool consume_called_ = false;
  bool serialize_called_ = false;
};

TEST(ProfilerSessionOrchestratorTest, SimpleLifecycle) {
  ClearRegisteredProfilersForTest();

  static MockProfiler* active_mock = nullptr;

  RegisterProfilerFactory([](const tensorflow::ProfileOptions& options) {
    auto mock = absl::make_unique<MockProfiler>();
    active_mock = mock.get();
    return mock;
  });

  tensorflow::ProfileOptions options = ProfilerSession::DefaultOptions();
  ProfilerSessionOrchestrator orchestrator(options);

  ASSERT_OK(orchestrator.Start());
  auto index_or = orchestrator.Consume();
  ASSERT_OK(index_or.status());
  ASSERT_OK(orchestrator.Serialize(index_or.value()));

  EXPECT_TRUE(active_mock != nullptr);
  if (active_mock) {
    EXPECT_TRUE(active_mock->consume_called());
    EXPECT_TRUE(active_mock->serialize_called());
  }

  ASSERT_OK(orchestrator.Stop());
}

TEST(ProfilerSessionOrchestratorTest, MultipleConsumeAndSelectiveSerialize) {
  ClearRegisteredProfilersForTest();

  RegisterProfilerFactory([](const tensorflow::ProfileOptions& options) {
    return absl::make_unique<MockProfiler>();
  });

  tensorflow::ProfileOptions options = ProfilerSession::DefaultOptions();
  // Using default sizes here
  ProfilerSessionOrchestrator orchestrator(options);

  ASSERT_OK(orchestrator.Start());

  auto index1_or = orchestrator.Consume();
  ASSERT_OK(index1_or.status());

  auto index2_or = orchestrator.Consume();
  ASSERT_OK(index2_or.status());

  EXPECT_NE(index1_or.value(), index2_or.value());

  ASSERT_OK(orchestrator.Serialize(index1_or.value()));
  ASSERT_OK(orchestrator.Serialize(index2_or.value()));

  ASSERT_OK(orchestrator.Stop());
}

TEST(ProfilerSessionOrchestratorTest, ClearConsumeBuffers) {
  ClearRegisteredProfilersForTest();

  RegisterProfilerFactory([](const tensorflow::ProfileOptions& options) {
    return absl::make_unique<MockProfiler>();
  });

  tensorflow::ProfileOptions options = ProfilerSession::DefaultOptions();
  ProfilerSessionOrchestrator orchestrator(options);

  ASSERT_OK(orchestrator.Start());

  auto index1_or = orchestrator.Consume();
  ASSERT_OK(index1_or.status());
  EXPECT_EQ(index1_or.value(), 0);

  orchestrator.ClearConsumeBuffers();

  auto index2_or = orchestrator.Consume();
  ASSERT_OK(index2_or.status());
  EXPECT_EQ(index2_or.value(), 0);  // Should be 0 again after clear!

  ASSERT_OK(orchestrator.Stop());
}

}  // namespace
}  // namespace profiler
}  // namespace tsl

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

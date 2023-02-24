/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark.h"

#include <unistd.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration.pb.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_float_validation_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_nnapi_sl_fake_impl.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/nnapi_sl_fake_impl.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/nnapi/sl/include/SupportLibrary.h"

namespace tflite {
namespace acceleration {
namespace {

TEST(BasicMiniBenchmarkTest, EmptySettings) {
  proto::MinibenchmarkSettings settings_proto;
  flatbuffers::FlatBufferBuilder empty_settings_buffer_;
  const MinibenchmarkSettings* empty_settings =
      ConvertFromProto(settings_proto, &empty_settings_buffer_);
  std::unique_ptr<MiniBenchmark> mb(
      CreateMiniBenchmark(*empty_settings, "ns", "id"));
  mb->TriggerMiniBenchmark();
  const ComputeSettingsT acceleration = mb->GetBestAcceleration();
  EXPECT_EQ(nullptr, acceleration.tflite_settings);
  EXPECT_TRUE(mb->MarkAndGetEventsToLog().empty());
  EXPECT_EQ(-1, mb->NumRemainingAccelerationTests());
}

class MiniBenchmarkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    MiniBenchmarkTestHelper helper;
    should_perform_test_ = helper.should_perform_test();

    if (should_perform_test_) {
      mobilenet_model_path_ = MiniBenchmarkTestHelper::DumpToTempFile(
          "mobilenet_float_with_validation.tflite",
          g_tflite_acceleration_embedded_mobilenet_float_validation_model,
          g_tflite_acceleration_embedded_mobilenet_float_validation_model_len);
    }
  }

  void SetupBenchmark(proto::Delegate delegate, const std::string& model_path,
                      bool reset_storage = true,
                      const nnapi::NnApiSupportLibrary* nnapi_sl = nullptr) {
    proto::MinibenchmarkSettings settings;
    proto::TFLiteSettings* tflite_settings = settings.add_settings_to_test();
    tflite_settings->set_delegate(delegate);
    if ((delegate == proto::Delegate::NNAPI) && nnapi_sl) {
      std::cerr << "Using NNAPI SL\n";
      tflite_settings->mutable_nnapi_settings()->set_support_library_handle(
          reinterpret_cast<int64_t>(nnapi_sl->getFL5()));
    }
    proto::ModelFile* file = settings.mutable_model_file();
    file->set_filename(model_path);
    proto::BenchmarkStoragePaths* paths = settings.mutable_storage_paths();
    paths->set_storage_file_path(::testing::TempDir() + "/storage.fb");
    if (reset_storage) {
      (void)unlink(paths->storage_file_path().c_str());
      // The suffix needs to be same as the one used in
      // MiniBenchmarkImpl::LocalEventStorageFileName
      (void)unlink((paths->storage_file_path() + ".extra.fb").c_str());
    }
    paths->set_data_directory_path(::testing::TempDir());

    if (delegate != proto::Delegate::NONE) {
      // Some of the tests rely on XNNPack beating CPU - need to not apply
      // XNNPack for the CPU variant for the tests to pass.
      proto::TFLiteSettings* cpu_tflite_settings =
          settings.add_settings_to_test();
      cpu_tflite_settings->set_disable_default_delegates(false);
    }

    settings_ = ConvertFromProto(settings, &settings_buffer_);

    mb_ = CreateMiniBenchmark(*settings_, ns_, model_id_);
  }

  void TriggerBenchmark(proto::Delegate delegate, const std::string& model_path,
                        bool reset_storage = true,
                        const nnapi::NnApiSupportLibrary* nnapi_sl = nullptr) {
    SetupBenchmark(delegate, model_path, reset_storage, nnapi_sl);
    mb_->TriggerMiniBenchmark();
  }

  void WaitForValidationCompletion(
      absl::Duration timeout = absl::Seconds(300)) {
    absl::Time deadline = absl::Now() + timeout;
    while (absl::Now() < deadline) {
      if (mb_->NumRemainingAccelerationTests() == 0) return;
      absl::SleepFor(absl::Milliseconds(200));
    }

    // We reach here only when the timeout has been reached w/o validation
    // completing.
    ASSERT_NE(0, mb_->NumRemainingAccelerationTests());
  }

  const std::string ns_ = "org.tensorflow.lite.mini_benchmark.test";
  const std::string model_id_ = "test_minibenchmark_model";

  bool should_perform_test_ = true;

  std::unique_ptr<MiniBenchmark> mb_;
  std::string mobilenet_model_path_;

  flatbuffers::FlatBufferBuilder settings_buffer_;
  // Simply a reference to settings_buffer_ for convenience.
  const MinibenchmarkSettings* settings_;
};

TEST_F(MiniBenchmarkTest, OnlyCPUSettings) {
  if (!should_perform_test_) return;
  SetupBenchmark(proto::Delegate::NONE, mobilenet_model_path_);
  EXPECT_EQ(-1, mb_->NumRemainingAccelerationTests());
  // We haven't triggered the benchmark yet, so a default ComputeSettingsT is
  // expected.
  ComputeSettingsT acceleration = mb_->GetBestAcceleration();
  EXPECT_EQ(nullptr, acceleration.tflite_settings);
  EXPECT_EQ(1, mb_->NumRemainingAccelerationTests());

  mb_->TriggerMiniBenchmark();
  // We just have 1 acceleration test to complete as the default CPU execution.
  WaitForValidationCompletion();
  acceleration = mb_->GetBestAcceleration();
  // As the best is the default CPU execution, the returned acceleration above
  // is still a default ComputeSettingsT.
  EXPECT_EQ(nullptr, acceleration.tflite_settings);
}

TEST_F(MiniBenchmarkTest, RunSuccessfully) {
  if (!should_perform_test_) return;

  TriggerBenchmark(proto::Delegate::XNNPACK, mobilenet_model_path_);

  // We have 2 acceleration tests to complete: one for the default CPU execution
  // and the other for XNNPACK delegate execution.
  WaitForValidationCompletion();
  // Mark existing events to be logged to simplify the logic of checking the
  // best decision event later.
  mb_->MarkAndGetEventsToLog();

  const ComputeSettingsT acceleration1 = mb_->GetBestAcceleration();

  // The 2nd call should return the same acceleration settings.
  const ComputeSettingsT acceleration2 = mb_->GetBestAcceleration();
  EXPECT_EQ(acceleration1, acceleration2);
#ifndef ADDRESS_SANITIZER  // XNNPack is slower under Asan.
  // As we choose mobilenet-v1 float model, XNNPACK delegate should be the best
  // on CPU.
  ASSERT_NE(nullptr, acceleration1.tflite_settings);
  EXPECT_EQ(tflite::Delegate_XNNPACK, acceleration1.tflite_settings->delegate);
#endif  // !ADDRESS_SANITIZER

  EXPECT_EQ(model_id_, acceleration1.model_identifier_for_statistics);
  EXPECT_EQ(ns_, acceleration1.model_namespace_for_statistics);

  // As the best decision event has not been marked as to-be-logged, we should
  // get one best decision event after the call.
  auto events = mb_->MarkAndGetEventsToLog();
  EXPECT_EQ(1, events.size());
  const auto& decision = events.front().best_acceleration_decision;
  EXPECT_NE(nullptr, decision);
#ifndef ADDRESS_SANITIZER  // XNNPack is slower under Asan.
  EXPECT_EQ(tflite::Delegate_XNNPACK,
            decision->min_latency_event->tflite_settings->delegate);
#endif  // !ADDRESS_SANITIZER
}

TEST_F(MiniBenchmarkTest, BestAccelerationEventIsMarkedLoggedAfterRestart) {
  if (!should_perform_test_) return;

  TriggerBenchmark(proto::Delegate::XNNPACK, mobilenet_model_path_);
  // We have 2 acceleration tests to complete: one for the default CPU execution
  // and the other for XNNPACK delegate execution.
  WaitForValidationCompletion();
  // Mark existing events to be logged to simplify the logic of checking the
  // best decision event later.
  mb_->MarkAndGetEventsToLog();
  mb_->GetBestAcceleration();

  // The best acceleration decision event was already persisted to the storage
  // above. So, we could retrieve the best acceleration immediately.
  TriggerBenchmark(proto::Delegate::XNNPACK, mobilenet_model_path_,
                   /*reset_storage=*/false);
  // As all acceleration tests have completed before, we expect no remaining
  // tests to be performed.
  EXPECT_EQ(0, mb_->NumRemainingAccelerationTests());

  const ComputeSettingsT acceleration = mb_->GetBestAcceleration();
#ifndef ADDRESS_SANITIZER  // XNNPack is slower under Asan.
  ASSERT_NE(nullptr, acceleration.tflite_settings);
  // As we choose mobilenet-v1 float model, XNNPACK delegate should be the best
  // on CPU.
  EXPECT_EQ(tflite::Delegate_XNNPACK, acceleration.tflite_settings->delegate);
#endif  // !ADDRESS_SANITIZER
  EXPECT_EQ(model_id_, acceleration.model_identifier_for_statistics);
  EXPECT_EQ(ns_, acceleration.model_namespace_for_statistics);

  // Similar to "RunSuccessfully' test above, the best decision event has not
  // been marked as to-be-logged, we should get one best decision event after
  // the call.
  auto events = mb_->MarkAndGetEventsToLog();
  EXPECT_EQ(1, events.size());
}

TEST_F(MiniBenchmarkTest,
       BestAccelerationEventIsNotReMarkedLoggedAfterRestart) {
  if (!should_perform_test_) return;

  TriggerBenchmark(proto::Delegate::XNNPACK, mobilenet_model_path_);
  // We have 2 acceleration tests to complete: one for the default CPU execution
  // and the other for XNNPACK delegate execution.
  WaitForValidationCompletion();
  mb_->GetBestAcceleration();

  // As the GetBestAcceleration above generates events synchronously, the event
  // will be persisted to the storage after the call, thus no waiting is needed
  // here.
  mb_->MarkAndGetEventsToLog();

  // The best acceleration decision event was already collected above. So, we
  // could retrieve the best acceleration immediately.
  TriggerBenchmark(proto::Delegate::XNNPACK, mobilenet_model_path_,
                   /*reset_storage=*/false);
  mb_->GetBestAcceleration();

  // As we have marked mini-benchmark events to be logged, we will expect
  // empty to-log events.
  EXPECT_TRUE(mb_->MarkAndGetEventsToLog().empty());
}

TEST_F(MiniBenchmarkTest, DelegatePluginNotSupported) {
  if (!should_perform_test_) return;

  // As Hexagon delegate plugin isn't supported in mini-benchmark, we will
  // expect a delegate plugin not-supported error.
  // Also, note that if a supported delegate plugin isn't linked to the this
  // test itself or the ":validator_runner_so_for_tests" target on Android,
  // one will expect a delegate plugin not-found error.
  TriggerBenchmark(proto::Delegate::HEXAGON, mobilenet_model_path_);

  // We have 2 acceleration tests to complete: one for the default CPU execution
  // and the other for HEXAGON delegate not supported.
  WaitForValidationCompletion();

  const ComputeSettingsT acceleration = mb_->GetBestAcceleration();
  // As the best performance is achieved on the default CPU, there's no
  // acceleration settings.
  EXPECT_EQ(nullptr, acceleration.tflite_settings);
  EXPECT_EQ(model_id_, acceleration.model_identifier_for_statistics);
  EXPECT_EQ(ns_, acceleration.model_namespace_for_statistics);

  // Check there is a Hexagon-delegate-not-supported event.
  const auto events = mb_->MarkAndGetEventsToLog();
  bool is_found = false;
  for (const auto& event : events) {
    const auto& t = event.benchmark_event;
    if (t == nullptr) continue;
    if (t->event_type == tflite::BenchmarkEventType_ERROR &&
        t->error->mini_benchmark_error_code ==
            tflite::acceleration::kMinibenchmarkDelegateNotSupported) {
      is_found = true;
      break;
    }
  }
  EXPECT_TRUE(is_found);
}

#ifdef __ANDROID__  // Loading shared libraries only works on Android.
TEST_F(MiniBenchmarkTest, UseNnApiSl) {
  if (!should_perform_test_) return;

  std::string nnapi_sl_path_ = MiniBenchmarkTestHelper::DumpToTempFile(
      "libnnapi_fake.so", g_nnapi_sl_fake_impl, g_nnapi_sl_fake_impl_len);

  std::unique_ptr<const ::tflite::nnapi::NnApiSupportLibrary> nnapi_sl =
      ::tflite::nnapi::loadNnApiSupportLibrary(nnapi_sl_path_);

  ASSERT_TRUE(nnapi_sl);

  TriggerBenchmark(proto::Delegate::NNAPI, mobilenet_model_path_,
                   /*reset_storage=*/true, nnapi_sl.get());

  WaitForValidationCompletion();

  EXPECT_TRUE(tflite::acceleration::WasNnApiSlInvoked());
}
#endif  // __ANDROID__

}  // namespace
}  // namespace acceleration
}  // namespace tflite

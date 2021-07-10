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

#include <fcntl.h>
#ifndef _WIN32
#include <dlfcn.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif  // !_WIN32

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration.pb.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_float_validation_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"

#ifdef __ANDROID__
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_runner_executable.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_validator_runner_so_for_tests.h"
#endif  // __ANDROID__

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
}

class MiniBenchmarkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    should_perform_test_ = true;
#ifdef __ANDROID__
    AndroidInfo android_info;
    auto status = RequestAndroidInfo(&android_info);
    ASSERT_TRUE(status.ok());
    if (android_info.is_emulator) {
      should_perform_test_ = false;
      return;
    }

    WriteFile("librunner_main.so", g_tflite_acceleration_embedded_runner,
              g_tflite_acceleration_embedded_runner_len);
    std::string validator_runner_so_path = WriteFile(
        "libvalidator_runner_so_for_tests.so",
        g_tflite_acceleration_embedded_validator_runner_so_for_tests,
        g_tflite_acceleration_embedded_validator_runner_so_for_tests_len);
    LoadEntryPointModule(validator_runner_so_path);
#endif

    mobilenet_model_path_ = WriteFile(
        "mobilenet_float_with_validation.tflite",
        g_tflite_acceleration_embedded_mobilenet_float_validation_model,
        g_tflite_acceleration_embedded_mobilenet_float_validation_model_len);
  }

  void TriggerBenchmark(proto::Delegate delegate, const std::string& model_path,
                        bool reset_storage = true) {
    proto::MinibenchmarkSettings settings;
    proto::TFLiteSettings* tflite_settings = settings.add_settings_to_test();
    tflite_settings->set_delegate(delegate);
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
    settings_ = ConvertFromProto(settings, &settings_buffer_);

    mb_ = CreateMiniBenchmark(*settings_, ns_, model_id_);
    mb_->TriggerMiniBenchmark();
  }

  int CountBenchmarkEvents(
      const std::vector<tflite::MiniBenchmarkEventT>& events,
      tflite::BenchmarkEventType type) {
    int num_events = 0;
    for (const auto& event : events) {
      if (event.benchmark_event != nullptr &&
          event.benchmark_event->event_type == type) {
        num_events++;
      }
    }
    return num_events;
  }

  std::vector<tflite::MiniBenchmarkEventT> WaitForEvents(
      std::function<bool(const std::vector<tflite::MiniBenchmarkEventT>&)>
          exit_wait,
      absl::Duration timeout = absl::Seconds(300)) {
    std::vector<MiniBenchmarkEventT> events;
    absl::Time deadline = absl::Now() + timeout;
    while (absl::Now() < deadline) {
      auto new_events = mb_->MarkAndGetEventsToLog();
      for (int i = 0; i < new_events.size(); ++i) {
        if (new_events[i].is_log_flushing_event) continue;
        events.emplace_back(std::move(new_events[i]));
      }
      if (exit_wait(events)) break;
      absl::SleepFor(absl::Milliseconds(200));
    }
    return events;
  }

  const std::string ns_ = "org.tensorflow.lite.mini_benchmark.test";
  const std::string model_id_ = "test_minibenchmark_model";

  bool should_perform_test_ = true;

  std::unique_ptr<MiniBenchmark> mb_;
  std::string mobilenet_model_path_;

  flatbuffers::FlatBufferBuilder settings_buffer_;
  // Simply a reference to settings_buffer_ for convenience.
  const MinibenchmarkSettings* settings_;

 private:
  // TODO(b/181571324): Factor out the following as common test helper functions
  // and use them across mini-benchmark-related tests.
  void* LoadEntryPointModule(const std::string& module_path) {
#ifndef _WIN32
    void* module =
        dlopen(module_path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
    EXPECT_NE(nullptr, module) << dlerror();
    return module;
#else   // _WIN32
    return nullptr;
#endif  // !_WIN32
  }

  std::string WriteFile(const std::string& filename, const unsigned char* data,
                        size_t length) {
    std::string dir = ::testing::TempDir() + "tflite/mini_benchmark";
    system((std::string("mkdir -p ") + dir).c_str());
    std::string path = dir + "/" + filename;
    (void)unlink(path.c_str());
    std::string contents(reinterpret_cast<const char*>(data), length);
    std::ofstream f(path, std::ios::binary);
    EXPECT_TRUE(f.is_open());
    f << contents;
    f.close();
    EXPECT_EQ(0, chmod(path.c_str(), 0500));
    return path;
  }
};

TEST_F(MiniBenchmarkTest, RunSuccessfully) {
  if (!should_perform_test_) return;

  TriggerBenchmark(proto::Delegate::XNNPACK, mobilenet_model_path_);

  // We will have at least 2 events: one for the default CPU execution and the
  // other for XNNPACK delegate execution. Additional events might be
  // platform-specific, such as those for failures when trying to set the CPU
  // affinity of the mini-benchmark runner process.
  auto events = WaitForEvents(
      [this](
          const std::vector<tflite::MiniBenchmarkEventT>& intermediate_events) {
        const int num_end_events = CountBenchmarkEvents(
            intermediate_events, tflite::BenchmarkEventType_END);
        return num_end_events == 2;
      });

  const ComputeSettingsT acceleration1 = mb_->GetBestAcceleration();
  EXPECT_NE(nullptr, acceleration1.tflite_settings);

  // The 2nd call should return the same acceleration settings.
  const ComputeSettingsT acceleration2 = mb_->GetBestAcceleration();
  EXPECT_EQ(acceleration1, acceleration2);
  // As we choose mobilenet-v1 float model, XNNPACK delegate should be the best
  // on CPU.
  EXPECT_EQ(tflite::Delegate_XNNPACK, acceleration1.tflite_settings->delegate);

  EXPECT_EQ(model_id_, acceleration1.model_identifier_for_statistics);
  EXPECT_EQ(ns_, acceleration1.model_namespace_for_statistics);

  // As the best decision event has not been marked as to-be-logged, we should
  // get one best decision event after the call.
  events = mb_->MarkAndGetEventsToLog();
  EXPECT_EQ(1, events.size());
  const auto& decision = events.front().best_acceleration_decision;
  EXPECT_NE(nullptr, decision);
  EXPECT_EQ(tflite::Delegate_XNNPACK,
            decision->min_latency_event->tflite_settings->delegate);
}

TEST_F(MiniBenchmarkTest, BestAccelerationEventIsMarkedLoggedAfterRestart) {
  if (!should_perform_test_) return;

  TriggerBenchmark(proto::Delegate::XNNPACK, mobilenet_model_path_);
  // We will have at least 2 events: one for the default CPU execution and the
  // other for XNNPACK delegate execution. Additional events might be
  // platform-specific, such as those for failures when trying to set the CPU
  // affinity of the mini-benchmark runner process.
  WaitForEvents(
      [this](
          const std::vector<tflite::MiniBenchmarkEventT>& intermediate_events) {
        const int num_end_events = CountBenchmarkEvents(
            intermediate_events, tflite::BenchmarkEventType_END);
        return num_end_events == 2;
      });
  mb_->GetBestAcceleration();

  // The best acceleration decision event was already persisted to the storage
  // above. So, we could retrieve the best acceleration immediately.
  TriggerBenchmark(proto::Delegate::XNNPACK, mobilenet_model_path_,
                   /*reset_storage=*/false);
  const ComputeSettingsT acceleration = mb_->GetBestAcceleration();
  // As we choose mobilenet-v1 float model, XNNPACK delegate should be the best
  // on CPU.
  EXPECT_EQ(tflite::Delegate_XNNPACK, acceleration.tflite_settings->delegate);
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
  // We will have at least 2 events: one for the default CPU execution and the
  // other for XNNPACK delegate execution. Additional events might be
  // platform-specific, such as those for failures when trying to set the CPU
  // affinity of the mini-benchmark runner process.
  WaitForEvents(
      [this](
          const std::vector<tflite::MiniBenchmarkEventT>& intermediate_events) {
        const int num_end_events = CountBenchmarkEvents(
            intermediate_events, tflite::BenchmarkEventType_END);
        return num_end_events == 2;
      });
  mb_->GetBestAcceleration();
  // There is no need to use WaitForEvents here but we can just mark
  // mini-benchmark events to be logged because the GetBestAcceleration above
  // generates events synchronously.
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

  // We will have at least 2 events: one for the default CPU execution and the
  // other is for the Hexagon delegate being not supported. Additional events
  // might be platform-specific, such as those for failures when trying to set
  // the CPU affinity of the mini-benchmark runner process.
  auto events = WaitForEvents(
      [this](
          const std::vector<tflite::MiniBenchmarkEventT>& intermediate_events) {
        const int num_end_events =
            CountBenchmarkEvents(intermediate_events,
                                 tflite::BenchmarkEventType_END) +
            CountBenchmarkEvents(intermediate_events,
                                 tflite::BenchmarkEventType_ERROR);
        return num_end_events == 2;
      });

  const ComputeSettingsT acceleration = mb_->GetBestAcceleration();
  // As the best performance is achieved on the default CPU, there's no
  // acceleration settings.
  EXPECT_EQ(nullptr, acceleration.tflite_settings);
  EXPECT_EQ(model_id_, acceleration.model_identifier_for_statistics);
  EXPECT_EQ(ns_, acceleration.model_namespace_for_statistics);

  // Check there is a Hexagon-delegate-not-supported event.
  bool is_found = false;
  for (const auto& event : events) {
    const auto& t = event.benchmark_event;
    if (t == nullptr) continue;
    if (t->event_type == tflite::BenchmarkEventType_ERROR &&
        t->error->exit_code ==
            tflite::acceleration::kMinibenchmarkDelegateNotSupported) {
      is_found = true;
      break;
    }
  }
  EXPECT_TRUE(is_found);
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite

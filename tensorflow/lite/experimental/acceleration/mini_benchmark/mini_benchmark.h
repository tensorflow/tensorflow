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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MINI_BENCHMARK_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MINI_BENCHMARK_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"

namespace tflite {
namespace acceleration {
// Instances are thread-compatible, access from multiple threads must be guarded
// by a mutex.
//
// Caution: The mini-benchmark runs silently in-process on non-Android, rather
// than in a separate process.
class MiniBenchmark {
 public:
  // Get best acceleration based on tests done so far. If no successful tests
  // are found, the best settings are on CPU or if the settings do not contain
  // configurations to test or not all relevant fields are present, the returned
  // ComputeSettingsT will be an object created by the default constructor.
  // Note: if we have successful mini-benchmark run events, the best
  // acceleration configuration will be persisted on the local storage as a new
  // mini-benchmark event.
  virtual ComputeSettingsT GetBestAcceleration() = 0;

  // Trigger the running of tests in the background in a separate thread on
  // Linux, but a separate process on Android. If triggering fails, errors are
  // stored internally.
  //
  // Does nothing if the settings do not contain configurations to test or not
  // all relevant fields are present.
  virtual void TriggerMiniBenchmark() = 0;

  virtual void SetEventTimeoutForTesting(int64_t timeout_us) = 0;

  // Mark mini-benchmark events that have not yet been marked as to be logged,
  // and return these events. This can include errors in triggering the
  // mini-benchmark.
  virtual std::vector<MiniBenchmarkEventT> MarkAndGetEventsToLog() = 0;

  // Get the number of remaining tests that still need to be completed.
  // Note that this method should be only called after calling
  // TriggerMiniBenchmark or GetBestAcceleration where additional
  // mini-benchmark-related setup could be initialized. In short, -1 is returned
  // if the overall mini-benchmark-related setup isn't properly initialized.
  virtual int NumRemainingAccelerationTests() = 0;

  MiniBenchmark() {}
  virtual ~MiniBenchmark() {}

  MiniBenchmark(MiniBenchmark&) = delete;
  MiniBenchmark& operator=(const MiniBenchmark&) = delete;
  MiniBenchmark(MiniBenchmark&&) = delete;
  MiniBenchmark& operator=(const MiniBenchmark&&) = delete;
};

// Instantiate a mini-benchmark. This will return a no-op implementation unless
// the :mini_benchmark_implementation target has been linked in.
std::unique_ptr<MiniBenchmark> CreateMiniBenchmark(
    const MinibenchmarkSettings& settings, const std::string& model_namespace,
    const std::string& model_id);

// A simple registry that allows different mini-benchmark implementations to be
// created by name.
//
// Limitations:
// - Doesn't allow deregistration.
// - Doesn't check for duplication registration.
//
class MinibenchmarkImplementationRegistry {
 public:
  using CreatorFunction = std::function<std::unique_ptr<MiniBenchmark>(
      const MinibenchmarkSettings& /*settings*/,
      const std::string& /*model_namespace*/, const std::string& /*model_id*/)>;

  // Returns a MiniBenchmark registered with `name` or nullptr if no matching
  // mini-benchmark implementation found.
  static std::unique_ptr<MiniBenchmark> CreateByName(
      const std::string& name, const MinibenchmarkSettings& settings,
      const std::string& model_namespace, const std::string& model_id);

  // Struct to be statically allocated for registration.
  struct Register {
    Register(const std::string& name, CreatorFunction creator_function);
  };

 private:
  void RegisterImpl(const std::string& name, CreatorFunction creator_function);
  std::unique_ptr<MiniBenchmark> CreateImpl(
      const std::string& name, const MinibenchmarkSettings& settings,
      const std::string& model_namespace, const std::string& model_id);
  static MinibenchmarkImplementationRegistry* GetSingleton();

  absl::Mutex mutex_;
  std::unordered_map<std::string, CreatorFunction> factories_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace acceleration
}  // namespace tflite

#define TFLITE_REGISTER_MINI_BENCHMARK_FACTORY_FUNCTION(name, f) \
  static auto* g_tflite_mini_benchmark_##name##_ =               \
      new MinibenchmarkImplementationRegistry::Register(#name, f);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MINI_BENCHMARK_H_

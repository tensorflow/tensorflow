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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers

namespace tflite {
namespace acceleration {

namespace {
class NoopMiniBenchmark : public MiniBenchmark {
 public:
  ComputeSettingsT GetBestAcceleration() override { return ComputeSettingsT(); }
  void TriggerMiniBenchmark() override {}
  void SetEventTimeoutForTesting(int64_t) override {}
  std::vector<MiniBenchmarkEventT> MarkAndGetEventsToLog() override {
    return {};
  }
  // We return -1 as this no-op instance doesn't have the overall
  // mini-benchmark-related setup properly initialized.
  int NumRemainingAccelerationTests() override { return -1; }
};
}  // namespace

std::unique_ptr<MiniBenchmark> CreateMiniBenchmark(
    const MinibenchmarkSettings& settings, const std::string& model_namespace,
    const std::string& model_id) {
  absl::StatusOr<std::unique_ptr<MiniBenchmark>> s_or_mb =
      MinibenchmarkImplementationRegistry::CreateByName(
          "Impl", settings, model_namespace, model_id);
  if (!s_or_mb.ok()) {
    return std::unique_ptr<MiniBenchmark>(new NoopMiniBenchmark());
  } else {
    return std::move(*s_or_mb);
  }
}

void MinibenchmarkImplementationRegistry::RegisterImpl(
    const std::string& name, CreatorFunction creator_function) {
  absl::MutexLock lock(&mutex_);
  factories_[name] = creator_function;
}

std::unique_ptr<MiniBenchmark> MinibenchmarkImplementationRegistry::CreateImpl(
    const std::string& name, const MinibenchmarkSettings& settings,
    const std::string& model_namespace, const std::string& model_id) {
  absl::MutexLock lock(&mutex_);
  auto it = factories_.find(name);
  return (it != factories_.end())
             ? it->second(settings, model_namespace, model_id)
             : nullptr;
}

MinibenchmarkImplementationRegistry*
MinibenchmarkImplementationRegistry::GetSingleton() {
  static auto* instance = new MinibenchmarkImplementationRegistry();
  return instance;
}

std::unique_ptr<MiniBenchmark>
MinibenchmarkImplementationRegistry::CreateByName(
    const std::string& name, const MinibenchmarkSettings& settings,
    const std::string& model_namespace, const std::string& model_id) {
  auto* const instance = MinibenchmarkImplementationRegistry::GetSingleton();
  return instance->CreateImpl(name, settings, model_namespace, model_id);
}

MinibenchmarkImplementationRegistry::Register::Register(
    const std::string& name, CreatorFunction creator_function) {
  auto* const instance = MinibenchmarkImplementationRegistry::GetSingleton();
  instance->RegisterImpl(name, creator_function);
}

}  // namespace acceleration
}  // namespace tflite

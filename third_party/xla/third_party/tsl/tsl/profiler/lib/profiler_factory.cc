/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tsl/profiler/lib/profiler_factory.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "tsl/profiler/lib/multi_pass_profiler_controller.h"
#include "tsl/profiler/lib/profiler_controller.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace tsl {
namespace profiler {
namespace {

absl::Mutex mu(absl::kConstInit);
absl::Mutex mu_multipass(absl::kConstInit);

std::vector<ProfilerFactory>* GetFactories() {
  static auto factories = new std::vector<ProfilerFactory>();
  return factories;
}

std::vector<MultiPassProfilerFactory>* GetMultiPassFactories() {
  static auto factories = new std::vector<MultiPassProfilerFactory>();
  return factories;
}

}  // namespace

void RegisterProfilerFactory(ProfilerFactory factory) {
  absl::MutexLock lock(mu);
  GetFactories()->push_back(std::move(factory));
}

void RegisterMultiPassProfilerFactory(MultiPassProfilerFactory factory) {
  absl::MutexLock lock(mu_multipass);
  GetMultiPassFactories()->push_back(std::move(factory));
}

std::vector<std::unique_ptr<profiler::ProfilerInterface>> CreateProfilers(
    const tensorflow::ProfileOptions& options) {
  std::vector<std::unique_ptr<profiler::ProfilerInterface>> result;
  absl::MutexLock lock(mu);
  for (const auto& factory : *GetFactories()) {
    auto profiler = factory(options);
    // A factory might return nullptr based on options.
    if (profiler == nullptr) continue;
    result.emplace_back(
        std::make_unique<ProfilerController>(std::move(profiler)));
  }
  return result;
}

std::vector<std::unique_ptr<MultiPassProfilerInterface>>
CreateMultiPassProfilers(const tensorflow::ProfileOptions& options) {
  std::vector<std::unique_ptr<MultiPassProfilerInterface>> multipass_profilers;
  {
    absl::MutexLock lock(mu_multipass);
    VLOG(3) << "Creating MultiPassProfilers() with "
            << GetMultiPassFactories()->size() << " factories.";
    for (const auto& factory : *GetMultiPassFactories()) {
      auto profiler = factory(options);
      if (profiler == nullptr) {
        continue;
      }
      multipass_profilers.emplace_back(
          std::make_unique<MultiPassProfilerController>(std::move(profiler)));
    }
    VLOG(3) << "Created " << multipass_profilers.size()
            << " multipass profilers.";
  }
  return multipass_profilers;
}

void ClearRegisteredProfilersForTest() {
  absl::MutexLock lock(mu);
  GetFactories()->clear();
}

void ClearRegisteredMultiPassProfilersForTest() {
  absl::MutexLock lock(mu_multipass);
  GetMultiPassFactories()->clear();
}

}  // namespace profiler
}  // namespace tsl

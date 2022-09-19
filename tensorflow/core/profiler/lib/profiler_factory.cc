/* Copyright 2019 The TensorFlow Authors All Rights Reserved.

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
#include "tensorflow/core/profiler/lib/profiler_factory.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/profiler/lib/profiler_controller.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

mutex mu(LINKER_INITIALIZED);

std::vector<ProfilerFactory>* GetFactories() {
  static auto factories = new std::vector<ProfilerFactory>();
  return factories;
}

}  // namespace

void RegisterProfilerFactory(ProfilerFactory factory) {
  mutex_lock lock(mu);
  GetFactories()->push_back(std::move(factory));
}

std::vector<std::unique_ptr<profiler::ProfilerInterface>> CreateProfilers(
    const ProfileOptions& options) {
  std::vector<std::unique_ptr<profiler::ProfilerInterface>> result;
  mutex_lock lock(mu);
  for (const auto& factory : *GetFactories()) {
    auto profiler = factory(options);
    // A factory might return nullptr based on options.
    if (profiler == nullptr) continue;
    result.emplace_back(
        absl::make_unique<ProfilerController>(std::move(profiler)));
  }
  return result;
}

void ClearRegisteredProfilersForTest() {
  mutex_lock lock(mu);
  GetFactories()->clear();
}

}  // namespace profiler
}  // namespace tensorflow

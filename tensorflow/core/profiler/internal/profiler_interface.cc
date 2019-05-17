/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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
#include "tensorflow/core/profiler/internal/profiler_interface.h"

#include "absl/synchronization/mutex.h"

namespace tensorflow {
namespace {
std::vector<ProfilerFactory>* GetFactories() {
  static auto factories = new std::vector<ProfilerFactory>();
  return factories;
}
absl::Mutex* GetMutex() {
  static auto mutex = new absl::Mutex;
  return mutex;
}
}  // namespace

void RegisterProfilerFactory(ProfilerFactory factory) {
  absl::MutexLock lock(GetMutex());
  GetFactories()->push_back(factory);
}

void CreateProfilers(
    const ProfilerContext* context,
    std::vector<std::unique_ptr<profiler::ProfilerInterface>>* result) {
  absl::MutexLock lock(GetMutex());
  for (auto factory : *GetFactories()) {
    if (auto profiler = factory(context)) {
      result->push_back(std::move(profiler));
    }
  }
}
}  // namespace tensorflow

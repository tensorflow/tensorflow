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
#include "tensorflow/core/profiler/internal/profiler_factory.h"

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace profiler {
namespace {

mutex mu(LINKER_INITIALIZED);

std::vector<ProfilerFactory>* GetFactories() TF_EXCLUSIVE_LOCKS_REQUIRED(mu) {
  static auto factories = new std::vector<ProfilerFactory>();
  return factories;
}

}  // namespace

void RegisterProfilerFactory(ProfilerFactory factory) {
  mutex_lock lock(mu);
  GetFactories()->push_back(factory);
}

void CreateProfilers(
    const profiler::ProfilerOptions& options,
    std::vector<std::unique_ptr<profiler::ProfilerInterface>>* result) {
  mutex_lock lock(mu);
  for (auto factory : *GetFactories()) {
    if (auto profiler = factory(options)) {
      result->push_back(std::move(profiler));
    }
  }
}

}  // namespace profiler
}  // namespace tensorflow

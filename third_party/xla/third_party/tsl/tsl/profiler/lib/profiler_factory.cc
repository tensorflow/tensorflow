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
#include "tsl/profiler/lib/profiler_factory.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/synchronization/mutex.h"
#include "tsl/profiler/lib/profiler_controller.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace tsl {
namespace profiler {
namespace {

absl::Mutex mu(absl::kConstInit);

std::vector<ProfilerFactory>* GetFactories() {
  static auto factories = new std::vector<ProfilerFactory>();
  return factories;
}

}  // namespace

void RegisterProfilerFactory(ProfilerFactory factory) {
  absl::MutexLock lock(mu);
  GetFactories()->push_back(std::move(factory));
}

std::vector<std::unique_ptr<profiler::ProfilerInterface>> CreateProfilers(
    const tensorflow::ProfileOptions& options) {
  // Create a copy of options to modify if circular buffer is enabled.
  tensorflow::ProfileOptions options_to_use = options;
  bool circular_buffer_enabled = false;

  // Check if tpu_circular_buffer_tracing is enabled in advanced configuration.
  auto it =
      options.advanced_configuration().find("tpu_circular_buffer_tracing");
  if (it != options.advanced_configuration().end()) {
    circular_buffer_enabled = it->second.bool_value();
  }

  if (circular_buffer_enabled) {
    // Disable other tracers by zeroing their levels in the local options copy.
    options_to_use.set_host_tracer_level(0);
    options_to_use.set_python_tracer_level(0);
  }

  std::vector<std::unique_ptr<profiler::ProfilerInterface>> result;
  absl::MutexLock lock(mu);
  for (const auto& factory : *GetFactories()) {
    auto profiler = factory(options_to_use);
    // A factory might return nullptr based on options.
    if (profiler == nullptr) continue;
    result.emplace_back(
        std::make_unique<ProfilerController>(std::move(profiler)));
  }
  return result;
}

void ClearRegisteredProfilersForTest() {
  absl::MutexLock lock(mu);
  GetFactories()->clear();
}

}  // namespace profiler
}  // namespace tsl

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

#ifndef TENSORFLOW_TSL_PROFILER_LIB_MULTI_PASS_PROFILER_CONTROLLER_H_
#define TENSORFLOW_TSL_PROFILER_LIB_MULTI_PASS_PROFILER_CONTROLLER_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

// Decorator for multi-pass profiler plugins.
//
// Tracks that calls to the underlying multi-pass profiler interface functions
// are made in the expected order: Start, StartPass, [PushRange/PopRange],
// StopPass, ..., Stop, and CollectData. Making calls in an invalid order causes
// them to be aborted.
//
// If a call to the decorated profiler fails, subsequent calls will be aborted
// and no further calls will be forwarded to the underlying profiler.
class MultiPassProfilerController : public MultiPassProfilerInterface {
 public:
  explicit MultiPassProfilerController(
      std::unique_ptr<MultiPassProfilerInterface> profiler);
  ~MultiPassProfilerController() override;

  // MultiPassProfilerInterface methods
  bool NeedMorePasses() override;
  absl::Status StartPass() override;
  absl::Status PushRange(absl::string_view name) override;
  absl::Status PopRange() override;
  absl::Status StopPass() override;

  // ProfilerInterface methods
  absl::Status Start() override;
  absl::Status Stop() override;
  absl::Status CollectData(tensorflow::profiler::XSpace* space) override;

 private:
  enum class MultiPassProfilerState {
    kInit = 0,
    kStarted = 1,
    kPassStarted = 2,
    kPassStopped = 3,
    kStopped = 4,
    kCollectData = 5,
  };

  MultiPassProfilerState state_ = MultiPassProfilerState::kInit;
  std::unique_ptr<MultiPassProfilerInterface> profiler_;
  absl::Status status_;  // result of calls to profiler_
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_MULTI_PASS_PROFILER_CONTROLLER_H_

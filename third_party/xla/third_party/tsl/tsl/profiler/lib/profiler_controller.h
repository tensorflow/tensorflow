/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_TSL_PROFILER_LIB_PROFILER_CONTROLLER_H_
#define TENSORFLOW_TSL_PROFILER_LIB_PROFILER_CONTROLLER_H_

#include <memory>

#include "absl/status/status.h"
#include "xla/tsl/platform/status.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

// Decorator for xprof profiler plugins.
//
// Tracks that calls to the underlying profiler interface functions are made
// in the expected order: Start, Stop and CollectData. Making the calls
// in a different order causes them to be aborted.
//
// Calls made in the right order will be aborted if one of the calls to the
// decorated profiler interface fails, and no more calls will be forwarded to
// the decorated profiler.
class ProfilerController : public ProfilerInterface {
 public:
  explicit ProfilerController(std::unique_ptr<ProfilerInterface> profiler);
  ~ProfilerController() override;

  absl::Status Start() override;

  absl::Status Stop() override;

  absl::Status CollectData(tensorflow::profiler::XSpace* space) override;

 private:
  enum class ProfilerState {
    kInit = 0,
    kStart = 1,
    kStop = 2,
    kCollectData = 3,
  };

  ProfilerState state_ = ProfilerState::kInit;
  std::unique_ptr<ProfilerInterface> profiler_;
  absl::Status status_;  // result of calls to profiler_
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_PROFILER_CONTROLLER_H_
